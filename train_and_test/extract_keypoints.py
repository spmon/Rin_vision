import cv2
from ultralytics import YOLO
import numpy as np
import logging
import os
DataSET_PATH = "D:\data_collection"
OUTPUT_PATH = "D:\keypoints_data"
Model_path = "yolo11x-pose.pt"

logging.getLogger("ultralytics").setLevel(logging.WARNING)
print("Đang tải mô hình YOLO...")
model = YOLO(Model_path)
print("Mô hình đã được tải.")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
actions = [d for d in os.listdir(DataSET_PATH) if os.path.isdir(os.path.join(DataSET_PATH, d))]
print(f"Tìm thấy {len(actions)} hành động: {actions}")
for action in actions:
    video_dir = os.path.join(DataSET_PATH, action)
    output_dir = os.path.join(OUTPUT_PATH, action)
    
    # Tạo thư mục con tương ứng trong thư mục đầu ra
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"\n--- Đang xử lý hành động: '{action}' ({len(video_files)} videos) ---")

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        # Tên file đầu ra sẽ giống tên video nhưng đuôi là .npy
        npy_filename = os.path.splitext(video_file)[0] + ".npy"
        npy_path = os.path.join(output_dir, npy_filename)

        # Nếu file .npy đã tồn tại thì bỏ qua (để chạy lại nếu bị ngắt giữa chừng)
        if os.path.exists(npy_path):
            print(f"  [Bỏ qua] {video_file} (Đã xử lý)")
            continue

        print(f"  Đang xử lý: {video_file}...")
        
        cap = cv2.VideoCapture(video_path)
        sequence = [] # Danh sách để chứa keypoints của TẤT CẢ các frame trong video này

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- CHẠY YOLO-POSE ---
            # verbose=False để nó không in ra quá nhiều thông tin
            results = model.predict(frame, verbose=False,conf=0.8)

            # --- TRÍCH XUẤT KEYPOINTS ---
            # Mặc định gán bằng mảng 0 (nếu không tìm thấy người)
            # YOLOv8-pose có 17 keypoints, mỗi điểm có (x, y) -> 34 giá trị
            keypoints_flat = np.zeros(17 * 2) 

            # Kiểm tra xem có phát hiện được người nào không
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                # Lấy keypoints của người đầu tiên được phát hiện (người có độ tự tin cao nhất)
                # results[0].keypoints.xyn trả về tọa độ chuẩn hóa (từ 0 đến 1), rất tốt cho việc huấn luyện
                # Nếu bạn muốn tọa độ pixel thực, dùng .xy thay vì .xyn
                kpts = results[0].keypoints.xyn[0].cpu().numpy()
                
                # Làm phẳng mảng (17x2 -> 34)
                keypoints_flat = kpts.flatten()

            sequence.append(keypoints_flat)

        cap.release()
        
        # Lưu chuỗi keypoints vào file .npy
        # File này sẽ chứa một mảng 2D có kích thước: (số_frame, 34)
        np.save(npy_path, np.array(sequence))

print("\nHoàn thành trích xuất dữ liệu!")


# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import logging

# # --- CÀI ĐẶT ---
# DATA_PATH = "D:\data_collection"
# OUTPUT_PATH = "D:\keypoint_data_normalized" # ĐỔI TÊN THƯ MỤC ĐỂ TRÁNH NHẦM LẪN
# MODEL_PATH = "yolo11x-pose.pt"

# logging.getLogger("ultralytics").setLevel(logging.WARNING)
# model = YOLO(MODEL_PATH)

# if not os.path.exists(OUTPUT_PATH):
#     os.makedirs(OUTPUT_PATH)

# # =========================================
# # HÀM CHUẨN HÓA POSE (MỚI)
# # =========================================
# def normalize_pose(kpts):
#     """
#     Chuẩn hóa keypoints dựa trên vị trí và kích thước của thân trên.
#     kpts: mảng numpy (17, 2) chứa tọa độ (x, y)
#     """
#     try:
#         # 1. Tính toán gốc tọa độ (origin) mới: là điểm giữa 2 vai
#         # kpts[5] = vai trái, kpts[6] = vai phải
#         origin = (kpts[5] + kpts[6]) / 2

#         # 2. Tính toán "thước đo" (scale): là khoảng cách giữa 2 vai
#         scale = np.linalg.norm(kpts[5] - kpts[6])

#         # 3. An toàn: Nếu không thấy vai (scale quá nhỏ), trả về mảng 0
#         if scale < 1e-4:
#             return np.zeros_like(kpts) # Trả về (17, 2) toàn số 0

#         # 4. Thực hiện chuẩn hóa
#         # (Lấy tất cả các điểm trừ đi gốc) / (chia cho thước đo)
#         normalized_kpts = (kpts - origin) / scale
        
#         return normalized_kpts
        
#     except Exception as e:
#         # Nếu có lỗi (ví dụ không phát hiện đủ keypoints), trả về 0
#         return np.zeros_like(kpts)

# # =========================================

# actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
# print(f"Tìm thấy {len(actions)} hành động: {actions}")

# for action in actions:
#     video_dir = os.path.join(DATA_PATH, action)
#     output_dir = os.path.join(OUTPUT_PATH, action)
#     os.makedirs(output_dir, exist_ok=True)
        
#     video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
#     print(f"\n--- Đang xử lý hành động: '{action}' ({len(video_files)} videos) ---")

#     for video_file in video_files:
#         video_path = os.path.join(video_dir, video_file)
#         npy_filename = os.path.splitext(video_file)[0] + ".npy"
#         npy_path = os.path.join(output_dir, npy_filename)

#         if os.path.exists(npy_path):
#             print(f"  [Bỏ qua] {video_file} (Đã xử lý)")
#             continue

#         print(f"  Đang xử lý: {video_file}...")
#         cap = cv2.VideoCapture(video_path)
#         sequence = []

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             results = model.predict(frame, verbose=False, conf=0.5)

#             # Mặc định là mảng 0 (34,)
#             keypoints_flat = np.zeros(17 * 2) 

#             if results[0].keypoints is not None and len(results[0].keypoints) > 0:
#                 # Lấy keypoints (17, 2)
#                 kpts_xy = results[0].keypoints.xyn[0].cpu().numpy()
                
#                 # --- GỌI HÀM CHUẨN HÓA (MỚI) ---
#                 normalized_kpts = normalize_pose(kpts_xy)
                
#                 # Làm phẳng mảng (17, 2) -> (34,)
#                 keypoints_flat = normalized_kpts.flatten()

#             sequence.append(keypoints_flat)

#         cap.release()
#         np.save(npy_path, np.array(sequence))

# print("\nHoàn thành trích xuất dữ liệu ĐÃ CHUẨN HÓA!")