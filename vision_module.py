from ollama import Client # Dùng Client thay vì gọi ollama trực tiếp
import cv2
import numpy as np

# --- CẤU HÌNH ---
MODEL_NAME = 'qwen3-vl:2b-instruct' 
# Địa chỉ trỏ từ Docker ra máy chủ Windows (nơi đang bật app Ollama ở taskbar)
OLLAMA_HOST = 'http://host.docker.internal:11434' 

# Khởi tạo client
client = Client(host=OLLAMA_HOST)

def load_model():
    """
    Kiểm tra kết nối với Ollama qua môi trường Docker.
    """
    try:
        # Dùng client thay vì ollama
        models_info = client.list() 
        print(f">>> [Vision Module] Đã kết nối Ollama thành công tại {OLLAMA_HOST}! (Mode: {MODEL_NAME})")
        return True

    except Exception as e:
        print(f">>> [Vision Module] LỖI KẾT NỐI OLLAMA: {e}")
        print(">>> HÃY KIỂM TRA: 1. Đã bật Ollama ở Taskbar chưa? 2. Mạng Docker có chặn host.docker.internal không?")
        return False

def generate_analysis(frame_cv2, user_question=None):
    """
    Gửi ảnh sang Ollama (Qwen-VL) để phân tích.
    """
    try:
        is_success, buffer = cv2.imencode(".jpg", frame_cv2)
        if not is_success:
            return "Lỗi mã hóa ảnh."
        
        image_bytes = buffer.tobytes()

        if user_question:
            prompt = user_question + " nói ngắn gọn về những gì bạn thấy trong ảnh này."
            print("User question:", user_question)
        else:
            prompt = "hãy mô tả ngắn gọn những gì bạn thấy trong ảnh này."
            print("No user question provided, using default prompt.")
            
        # Gọi API thông qua đối tượng client
        response = client.chat(
            model=MODEL_NAME,
            keep_alive=-1,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes] 
            }],
            options={
                "num_ctx": 1024,  # Hạ xuống 1024 là rất hợp lý để tránh ăn RAM
                "temperature": 0.1
            }
        )

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