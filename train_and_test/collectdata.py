import cv2
import os
import time

# --- CÀI ĐẶT CỦA BẠN ---

# 1. Thư mục gốc để lưu dữ liệu
DATA_PATH = "D:\data_collection"

# 2. Danh sách hành động (tên thư mục)
# Thêm/xóa các hành động bạn muốn
actions = ['idle', 'waving', 'stretching', 'drinking', 'rubbing_eyes']

# 3. Phím nóng tương ứng (chọn phím bạn thích)
# QUAN TRỌNG: Số lượng phím phải bằng số lượng hành động
hotkeys = ['i', 'w', 's', 'd', 'r']

# 4. Thời gian quay mỗi video (giây)
RECORD_TIME = 5.0
# --- HẾT CÀI ĐẶT ---


# Tạo các thư mục nếu chúng chưa tồn tại
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

# Khởi động webcam
cap = cv2.VideoCapture(0) # 0 là webcam mặc định
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit()

# Lấy FPS và kích thước frame của webcam (để lưu video)
webcam_fps = cap.get(cv2.CAP_PROP_FPS)
if webcam_fps == 0: # Một số webcam trả về 0, dùng giá trị dự phòng
    webcam_fps = 30 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec để lưu file .mp4

print("-" * 50)
print(f"Webcam đã bật (FPS: {webcam_fps}, Size: {width}x{height})")
print("Sẵn sàng thu thập dữ liệu...")
print("Nhấn phím tương ứng để bắt đầu quay:")
for i, action in enumerate(actions):
    print(f"  Nhấn '{hotkeys[i]}' cho hành động '{action}'")
print("Nhấn 'q' để thoát.")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame.")
        break
        
    # Lật ngược frame (giống như soi gương) để dễ nhìn
    frame = cv2.flip(frame, 1)

    # Hiển thị hướng dẫn trên màn hình
    cv2.putText(frame, "Nhan phim de quay (q: thoat)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị các phím nóng
    y_pos = 60
    for i, action in enumerate(actions):
        text = f"'{hotkeys[i]}': {action}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30

    cv2.imshow("Data Collection Tool", frame)

    # Chờ phím nhấn
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Đang thoát...")
        break

    # Kiểm tra xem phím có nằm trong danh sách hotkeys không
    if chr(key) in hotkeys:
        action = actions[hotkeys.index(chr(key))]
        action_path = os.path.join(DATA_PATH, action)
        
        # Đếm ngược 3 giây
        print(f"\nChuẩn bị quay '{action}'...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            # Hiển thị đếm ngược trên frame
            temp_frame = frame.copy()
            cv2.putText(temp_frame, str(i), (width // 2 - 50, height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            cv2.imshow("Data Collection Tool", temp_frame)
            cv2.waitKey(1000) # Chờ 1 giây
            
        print("BẮT ĐẦU QUAY!")
        
        # Tạo tên file duy nhất bằng timestamp
        video_name = f"{int(time.time())}.mp4"
        video_path = os.path.join(action_path, video_name)
        
        # Tạo đối tượng VideoWriter
        out = cv2.VideoWriter(video_path, fourcc, webcam_fps, (width, height))
        
        start_time = time.time()
        while (time.time() - start_time) < RECORD_TIME:
            ret, record_frame = cap.read()
            if not ret:
                break
                
            record_frame = cv2.flip(record_frame, 1) # Lật frame trước khi lưu

            # Hiển thị chữ "RECORDING"
            cv2.putText(record_frame, "RECORDING...", (width - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Data Collection Tool", record_frame)
            
            # Ghi frame vào file
            out.write(record_frame)
            
            # Phải có waitKey để imshow hoạt động
            cv2.waitKey(1)

        out.release() # Lưu file video
        print(f"Đã lưu: {video_path}")
        print("-" * 50)

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()