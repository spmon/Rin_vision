import numpy as np

def normalize_pose(kpts):
    """
    Chuẩn hóa Pose dựa trên Torso (Thân người).
    Logic này PHẢI GIỐNG HỆT logic trong prepare_data.py.
    """
    # kpts shape: (17, 2)
    # COCO Keypoints Index:
    # 5: Left Shoulder, 6: Right Shoulder
    # 11: Left Hip, 12: Right Hip

    # 1. Tìm điểm gốc (Trung điểm hông)
    # Nếu hông bị khuất (tọa độ = 0), dùng trung điểm vai để thay thế
    if np.all(kpts[11] == 0) or np.all(kpts[12] == 0):
        origin = (kpts[5] + kpts[6]) / 2 
    else:
        origin = (kpts[11] + kpts[12]) / 2

    # 2. Dịch chuyển toàn bộ cơ thể về gốc (0,0)
    centered_kpts = kpts - origin

    # 3. Tính hệ số tỷ lệ (Scale) dựa trên chiều dài thân (Torso Length)
    shoulder_center = (kpts[5] + kpts[6]) / 2
    hip_center = (kpts[11] + kpts[12]) / 2
    
    # Kiểm tra xem có đủ điểm để tính thân không
    has_shoulders = not (np.all(kpts[5] == 0) or np.all(kpts[6] == 0))
    has_hips = not (np.all(kpts[11] == 0) or np.all(kpts[12] == 0))

    if has_shoulders and has_hips:
        # Ưu tiên 1: Dùng chiều dài lưng (Chuẩn nhất)
        scale = np.linalg.norm(shoulder_center - hip_center)
    elif has_shoulders:
        # Ưu tiên 2: Nếu mất hông, dùng chiều rộng vai * 2
        scale = np.linalg.norm(kpts[5] - kpts[6]) * 2.0
    else:
        # Fallback cuối cùng: Nếu mất hết, giữ nguyên (tránh crash)
        scale = 1.0

    # Tránh chia cho 0 hoặc số quá nhỏ
    if scale < 0.01: 
        scale = 1.0

    # 4. Chia tỷ lệ
    normalized_kpts = centered_kpts / scale
    
    return normalized_kpts