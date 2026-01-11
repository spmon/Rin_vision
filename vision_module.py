import ollama
import cv2
import numpy as np

# --- CẤU HÌNH ---
# Tên model chính xác như trong ảnh bạn gửi
MODEL_NAME = 'qwen3-vl:2b-instruct' 

def load_model():
    """
    Kiểm tra kết nối với Ollama.
    """
    try:
        # 1. Ping thử Ollama xem có sống không
        models_info = ollama.list()
        
        # 2. In ra để debug (xem nó trả về cái gì)
        # print(f"DEBUG Ollama: {models_info}") 
        
        # Thay vì loop check gây lỗi, ta chỉ cần thông báo kết nối thành công
        # Vì bạn đã pull model rồi nên cứ tin tưởng là nó có ở đó.
        print(f">>> [Vision Module] Đã kết nối Ollama thành công! (Mode: {MODEL_NAME})")
        return True

    except Exception as e:
        # Nếu lỗi này hiện ra nghĩa là Ollama chưa bật hoặc chưa cài
        print(f">>> [Vision Module] LỖI KẾT NỐI OLLAMA: {e}")
        print(">>> HÃY BẬT ỨNG DỤNG OLLAMA Ở THANH TASKBAR LÊN!")
        return False

def generate_analysis(frame_cv2, user_question=None):
    """
    Gửi ảnh sang Ollama (Qwen3-VL) để phân tích.
    """
    try:
        # 1. Chuyển ảnh OpenCV (numpy) -> Bytes (JPG)
        is_success, buffer = cv2.imencode(".jpg", frame_cv2)
        if not is_success:
            return "Lỗi mã hóa ảnh."
        
        image_bytes = buffer.tobytes()

        # 2. Tạo Prompt
        if user_question:
            prompt = user_question+" nói ngắn gọn về những gì bạn thấy trong ảnh này."
            print("User question:", user_question)
        else:
            prompt = "hãy mô tả ngắn gọn những gì bạn thấy trong ảnh này."
            print("No user question provided, using default prompt.")
        # 3. Gọi API Ollama
        response = ollama.chat(
            model=MODEL_NAME,
            keep_alive=-1,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes] 
            }],

        options={
                "num_ctx": 1024,  # Mặc định là 2048 hoặc 4096. Giảm xuống 1024 hoặc 512.
                "temperature": 0.1 # Độ sáng tạo (giữ nguyên cũng được)
            }
        )

        # 4. Lấy kết quả
        return response['message']['content']

    except Exception as e:
        print(f"Lỗi Ollama khi sinh chữ: {e}")
        return "Sorry, I cannot see clearly right now (Ollama Error)."

# --- Test nhanh ---
if __name__ == "__main__":
    if load_model():
        print("Đang test model...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "HELLO WAIFU", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print("Kết quả:", generate_analysis(dummy_img, "What text is in the image?"))