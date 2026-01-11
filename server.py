
import cv2
import numpy as np
import json
import asyncio
import websockets
import base64
import time
import vision_module  # Module BLIP
import threading      # Để chạy đa luồng
import config
from model_loader import load_all_models
import ai_core as vp  # AI Core của bạn

# --- 1. TẢI (LOAD) TẤT CẢ CÁC MÔ HÌNH (Chỉ 1 lần) ---
print("Đang tải các mô hình AI...")
models = load_all_models()


vision_module.load_model() # Tải model mô tả ảnh vào CPU

print("Server đã sẵn sàng...")

# --- HÀM PHỤ: CHẠY TRONG LUỒNG RIÊNG (THREAD) ---
def caption_worker(frame, question, websocket, loop):
    """
    Hàm này chạy trên CPU ở một luồng riêng biệt.
    Hỗ trợ cả Mô tả ảnh (Caption) và Trả lời câu hỏi (QA).
    """
    try:
        mode_text = f"Câu hỏi: {question}" if question else "Tự mô tả"
        print(f">>> [Thread] Vision đang xử lý... ({mode_text})")
        
        result_text = vision_module.generate_analysis(frame, user_question=question)
        print(f">>> [Thread] Kết quả: {result_text}")

        # 2. Tạo JSON trả về
        response_data = {
            "type": "caption_result", 
            "text": result_text,
            "is_answer": True if question else False # Cờ báo hiệu đây là câu trả lời
        }
        json_output = json.dumps(response_data)

        # 3. Gửi về Client (Thread-safe)
        asyncio.run_coroutine_threadsafe(websocket.send(json_output), loop)

    except Exception as e:
        print(f"Lỗi trong luồng Caption: {e}")

# --- 2. HÀM XỬ LÝ WEBSOCKET (HANDLER) ---
async def handler(websocket):
    print(f"Client đã kết nối từ {websocket.remote_address}")
    
    # Lấy Event Loop hiện tại để truyền cho Thread
    loop = asyncio.get_running_loop()

    sequence = []
    frame_counter = 0
    current_action = "Waiting..."
    confidence = 0.0
    
    # HARDCODE CẢM XÚC: Để "lừa" bên Client là mọi thứ vẫn ổn
    current_emotion = "Neutral"
    
    detected_objects_list = []
    cup_near_face = False
    nose_xy = None
    
    try:
        async for message in websocket:
            # starttime = time.time()
            try:
                # --- PHÂN TÍCH TIN NHẮN ---
                # Cấu trúc: "HEADER,base64_data,[OPTIONAL_QUESTION]"
                parts = message.split(',')
                
                if len(parts) < 2: continue
                    
                header = parts[0] 
                data = parts[1]   

                # --- TRƯỜNG HỢP 1: VISION (Mô tả hoặc Hỏi đáp) ---
                if "CAPTION" in header or "QA" in header:
                    print(f">>> Nhận lệnh Vision: {header}")
                    
                    # Lấy câu hỏi nếu có (cho trường hợp QA)
                    user_question = None
                    if "QA" in header and len(parts) > 2:
                        # Nối lại phần sau đề phòng câu hỏi có dấu phẩy
                        user_question = ",".join(parts[2:])

                    # Giải mã ảnh
                    img_bytes = base64.b64decode(data)
                    img_jpg = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
                    
                    if frame is None: continue

                    # Đẩy sang luồng Vision (Không chặn luồng chính)
                    t = threading.Thread(
                        target=caption_worker, 
                        args=(frame, user_question, websocket, loop)
                    )
                    t.start()
                    
                    # Bỏ qua xử lý Action cho frame này để tiết kiệm tài nguyên
                    continue 

                # --- TRƯỜNG HỢP 2: LUỒNG ACTION BÌNH THƯỜNG (STREAM) ---
                # Giải mã ảnh cho luồng stream
                img_bytes = base64.b64decode(data)
                img_jpg = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
                if frame is None: continue

                # Resize nhẹ để chạy Action cho nhanh
                frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                frame_counter += 1
                
                # 2a. Chạy Pose + LSTM (Mọi frame)
                sequence, current_action, confidence, nose_xy = vp.process_pose_and_action(
                    frame_resized,
                    models["yolo_pose"],
                    models["action_model"],
                    sequence
                )

                # 2b. Chạy Detection (Ngắt quãng - Giả lập mỗi 10 frame chạy 1 lần)
                if frame_counter % 10 == 0: 
                    detected_objects_list, cup_near_face = vp.process_detection(
                        frame_resized,
                        models["yolo_det"],
                        models["object_names"],
                        nose_xy
                    )

                # --- 2c. EMOTION
                if frame_counter % config.EMOTION_INTERVAL == 0:
                    current_emotion = vp.process_emotion(
                        frame,
                        models["face_cascade"],
                        models["emotion_model"],
                        models["emotion_transform"],
                        nose_xy
                    )
                
                # --- 3. LOGIC KẾT HỢP ---
                final_action, final_confidence = vp.apply_custom_logic(
                    current_action, 
                    confidence, 
                    cup_near_face,
                    sequence
                )
                # endtime=time.time()
                # print(f"Xử lý frame {frame_counter} trong {endtime-starttime:.3f} giây")
                # In log kiểm tra
                # if frame_counter % 30 == 0:
                # print(f"Action: {final_action} | Objects: {len(detected_objects_list)}")

                # --- 4. TẠO VÀ GỬI JSON (ACTION) ---
                response_data = {
                    "type": "action_result",
                    "action": final_action,
                    "action_confidence": round(final_confidence, 4),
                    "emotion": current_emotion, # Luôn là "Neutral"
                    "objects": detected_objects_list
                }
                json_output = json.dumps(response_data)
                
                await websocket.send(json_output)

            except Exception as e:
                print(f"Lỗi xử lý frame: {e}")
                continue
            
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {websocket.remote_address} đã ngắt kết nối.")
    except Exception as e:
        print(f"Lỗi bất ngờ xảy ra với client {websocket.remote_address}: {e}")
    finally:
        print(f"Đã dọn dẹp cho client {websocket.remote_address}")

# --- 3. HÀM MAIN ---
async def main():
    print(f"\n--- Đang khởi động Server WebSocket tại ws://{config.SERVER_HOST}:{config.SERVER_PORT} ---")
    async with websockets.serve(handler, config.SERVER_HOST, config.SERVER_PORT,origins=None,max_size=None,ping_interval=None):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nĐã dừng server.")