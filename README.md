👁️ Rin AI - Core Vision Module
Đây là "đôi mắt" của hệ thống trợ lý ảo Rin AI, đảm nhận vai trò nhận thức môi trường và hành vi người dùng theo thời gian thực. Module này không chỉ dừng lại ở nhận diện hình ảnh đơn thuần mà là một hệ thống Temporal Action Recognition phức tạp.

🚀 Điểm sáng Kỹ thuật (Technical Highlights)

Kiến trúc Hybrid (YOLOv11 + LSTM): Sử dụng YOLOv11-Pose để trích xuất 17 điểm mấu chốt (keypoints) cơ thể và đưa qua mạng LSTM (Long Short-Term Memory) để phân loại hành động dựa trên chuỗi thời gian (Temporal Analysis).



Xử lý Đa luồng & Độ trễ thấp: Tối ưu hóa pipeline xử lý ảnh để đạt tốc độ nhận diện hành động trong thời gian thực với độ trễ cực thấp (< 1s), đảm bảo sự phản hồi tức thì cho nhân vật Live2D.



Hệ thống Emotion Recognition: Tích hợp mô hình VGG-like (EmotionVGG) để nhận diện biểu cảm khuôn mặt từ dòng stream camera.


Cơ chế Data Streaming: Kết quả nhận diện được đóng gói dưới dạng JSON và truyền qua WebSocket, giúp bộ não LLM và giao diện Live2D đồng bộ hóa trạng thái ngay lập tức.


🏗️ Kiến trúc Module (System Architecture)
Dòng dữ liệu được xử lý qua 3 tầng cốt lõi:


Input Layer: Thu nhận video stream thông qua OpenCV, xử lý tiền định dạng ảnh.



Perception Layer (YOLOv11): Trích xuất tọa độ các điểm mấu chốt (Keypoints) từ khung hình.


Reasoning Layer (LSTM): Phân tích sự biến thiên của tọa độ qua các khung hình để định danh hành động (Greeting, Working, Drinking, v.v.).


📁 Cấu trúc Thư mục (Project Structure)

vision_module.py: Script điều phối chính, kết nối Camera và các module AI.


models/action_lstm.py: Định nghĩa kiến trúc mạng LSTM cho nhận diện hành động.


models/emotion_vgg.py: Mô hình nhận diện cảm xúc khuôn mặt.


train_and_test/: Chứa toàn bộ quy trình từ thu thập dữ liệu thô, tiền xử lý đến huấn luyện mô hình LSTM.



config.py: Quản lý các tham số hệ thống (Confidence, FPS, Path).

🛠️ Cài đặt & Sử dụng
1. Yêu cầu hệ thống
Python 3.12+ 

NVIDIA GPU với hỗ trợ CUDA (Bắt buộc để chạy mượt mà tất cả các module song song).

2. Triển khai
Bash

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Khởi chạy Vision
python vision_server.py
🔗 Credits & Open Source
Emotion Module: Dựa trên kiến trúc của [ https://github.com/anhtuan85/Facial-expression-recognition?tab=readme-ov-file ] và được nhóm tùy chỉnh để tích hợp vào hệ thống Rin AI.


Pose Estimation: Sử dụng Ultralytics YOLOv11.
