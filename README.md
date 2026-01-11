👁️ AI Vision Module - Multimodal Assistant
Đây là module Thị giác cốt lõi trong hệ thống trợ lý ảo đa phương thức. Module đảm nhận vai trò nhận diện hành động, tư thế và biểu cảm khuôn mặt thời gian thực để cung cấp ngữ cảnh cho bộ não (LLM).

🚀 Tính năng kỹ thuật
Action Recognition: Kết hợp YOLOv11-Pose và LSTM để phân tích chuỗi hành động (Action Sequence).

Real-time Processing: Tốc độ xử lý cực cao, nhận diện hành động trong thời gian thực với độ trễ < 1s.

Multi-tasking:

Nhận diện tư thế (Pose Estimation).

Nhận diện hành động (Greeting, Working, Tired, v.v.).

Tiền xử lý dữ liệu hình ảnh cho các module AI khác (BLIP/Vision-Language).

Data Streaming: Truyền kết quả nhận diện qua WebSockets dưới định dạng JSON để đồng bộ với nhân vật Live2D.

🏗️ Kiến trúc Module
Input: Stream video từ Camera thông qua OpenCV.

Detection Layer: YOLOv11 trích xuất tọa độ các điểm mấu chốt (keypoints) trên cơ thể.

Temporal Layer: Mạng LSTM phân tích sự thay đổi tọa độ qua các khung hình để phân loại hành động.

Output: Đẩy dữ liệu qua WebSocket Server (Base64/JSON).

🛠️ Cài đặt (Dành riêng cho Vision Module)
1. Yêu cầu
Python 3.9+

NVIDIA GPU + CUDA (Khuyên dùng để đạt FPS cao).

2. Cài đặt thư viện
Bash

pip install -r requirements.txt