import os
import numpy as np
import warnings

# Tắt cảnh báo NumPy
warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
KEYPOINT_PATH = r"D:\keypoint_data_normalized" 
SEQUENCE_LENGTH = 10 # 10 frame = 1 giây (ở 10 FPS)

# Các cặp chỉ mục đối xứng (COCO Format)
FLIP_PAIRS_LIST = [
    [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]
]

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ TOÁN HỌC (Augmentation & Normalization)
# ==============================================================================

def normalize_pose_torso(kpts):
    """
    Chuẩn hóa Pose dựa trên Torso (Thân người).
    Logic này PHẢI GIỐNG HỆT trong utils/pose_utils.py
    """
    # 1. Tìm điểm gốc (Trung điểm hông)
    if np.all(kpts[11] == 0) or np.all(kpts[12] == 0):
        origin = (kpts[5] + kpts[6]) / 2 # Dùng vai nếu mất hông
    else:
        origin = (kpts[11] + kpts[12]) / 2

    # 2. Dịch chuyển về gốc
    centered_kpts = kpts - origin

    # 3. Tính tỷ lệ (Scale) dựa trên chiều dài thân
    shoulder_center = (kpts[5] + kpts[6]) / 2
    hip_center = (kpts[11] + kpts[12]) / 2
    
    if not (np.all(kpts[5] == 0) or np.all(kpts[11] == 0)):
        scale = np.linalg.norm(shoulder_center - hip_center)
    else:
        scale = np.linalg.norm(kpts[5] - kpts[6]) * 2.0 

    if scale < 0.01: scale = 1.0

    return centered_kpts / scale

def horizontal_flip_sequence(sequence, FLIP_PAIRS):
    """Lật keypoint theo trục ngang."""
    flipped_sequence = np.copy(sequence)
    flipped_sequence[:, 0::2] = 1.0 - flipped_sequence[:, 0::2] 
    for frame_idx in range(flipped_sequence.shape[0]):
        temp_frame = np.copy(flipped_sequence[frame_idx])
        for left_idx, right_idx in FLIP_PAIRS:
            l_x, l_y = 2 * left_idx, 2 * left_idx + 1
            r_x, r_y = 2 * right_idx, 2 * right_idx + 1
            temp_frame[[l_x, l_y, r_x, r_y]] = temp_frame[[r_x, r_y, l_x, l_y]]
        flipped_sequence[frame_idx] = temp_frame
    return flipped_sequence

def random_translate_sequence(sequence, max_shift=0.05):
    """Dịch chuyển vị trí ngẫu nhiên (+/- 5%)."""
    translated_sequence = np.copy(sequence)
    delta_x = np.random.uniform(-max_shift, max_shift)
    delta_y = np.random.uniform(-max_shift, max_shift)
    translated_sequence[:, 0::2] += delta_x
    translated_sequence[:, 1::2] += delta_y
    return np.clip(translated_sequence, -1.5, 1.5)

def random_scale_sequence(sequence, min_scale=0.85, max_scale=1.15):
    """Phóng to/thu nhỏ đồng đều (Giả lập khoảng cách xa/gần)."""
    scale_factor = np.random.uniform(min_scale, max_scale)
    return sequence * scale_factor

def random_body_morph(sequence, max_scale_x=0.2, max_scale_y=0.2):
    """
    Biến đổi hình dạng cơ thể ngẫu nhiên (Giả lập người béo/gầy/cao/thấp).
    Co giãn trục X và Y độc lập.
    """
    morphed_sequence = np.copy(sequence)
    # 1.0 +/- 0.2 -> Từ 0.8 đến 1.2
    scale_x = np.random.uniform(1.0 - max_scale_x, 1.0 + max_scale_x)
    scale_y = np.random.uniform(1.0 - max_scale_y, 1.0 + max_scale_y)
    
    morphed_sequence[:, 0::2] *= scale_x # Co giãn chiều ngang
    morphed_sequence[:, 1::2] *= scale_y # Co giãn chiều dọc
    
    return morphed_sequence

# ==============================================================================
# 2. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
if __name__ == "__main__":
    actions = []
    label_map = {}
    
    # 1. Quét thư mục hành động
    try:
        for action_name in os.listdir(KEYPOINT_PATH):
            if os.path.isdir(os.path.join(KEYPOINT_PATH, action_name)):
                actions.append(action_name)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy đường dẫn {KEYPOINT_PATH}")
        exit()

    actions.sort()
    for i, action in enumerate(actions):
        label_map[action] = i

    print(f"Actions: {actions}")
    sequences, labels = [], []

    # 2. Xử lý dữ liệu
    for action in actions:
        action_path = os.path.join(KEYPOINT_PATH, action)
        npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        print(f"Dang xu ly '{action}' ({len(npy_files)} files)...")
        
        for file_name in npy_files:
            # Load dữ liệu thô (Raw Pixel Coordinates)
            try:
                raw_data = np.load(os.path.join(action_path, file_name))
            except:
                continue
            
            # --- BƯỚC A: CHUẨN HÓA LẠI (TORSO) ---
            normalized_seq_list = []
            for frame in raw_data:
                kpts = frame.reshape(17, 2)
                norm_kpts = normalize_pose_torso(kpts) # Dùng hàm mới
                normalized_seq_list.append(norm_kpts.flatten())
            
            full_normalized_data = np.array(normalized_seq_list)

            # --- BƯỚC B: DOWNSAMPLE (30 FPS -> 10 FPS) ---
            downsampled_data = full_normalized_data[::3] 

            # --- BƯỚC C: CẮT CHUỖI & AUGMENTATION ---
            if len(downsampled_data) >= SEQUENCE_LENGTH:
                 for i in range(len(downsampled_data) - SEQUENCE_LENGTH + 1):
                    # Chuỗi gốc (Base)
                    base_sequence = downsampled_data[i : i + SEQUENCE_LENGTH]

                    # 1. Gốc
                    sequences.append(base_sequence)
                    labels.append(label_map[action])

                    # 2. Gốc + Dịch chuyển (Position Variance)
                    sequences.append(random_translate_sequence(base_sequence))
                    labels.append(label_map[action])

                    # 3. Gốc + Phóng to/nhỏ (Distance Variance)
                    sequences.append(random_scale_sequence(base_sequence))
                    labels.append(label_map[action])

                    # 4. Lật (Angle Variance)
                    flipped_seq = horizontal_flip_sequence(base_sequence, FLIP_PAIRS_LIST)
                    sequences.append(flipped_seq)
                    labels.append(label_map[action])
                    
                    # 5. Lật + Dịch chuyển
                    sequences.append(random_translate_sequence(flipped_seq))
                    labels.append(label_map[action])

                    # 6. Lật + Biến dạng cơ thể (Identity/Shape Variance) <--- QUAN TRỌNG
                    sequences.append(random_body_morph(flipped_seq))
                    labels.append(label_map[action])

    # 3. Lưu file
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64) 

    print(f"\nShape X: {X.shape}")
    print(f"Shape y: {y.shape}")

    np.save('X_data.npy', X)
    np.save('y_data.npy', y)
    np.save('label_map.npy', actions)
    print("\n--- XONG! Dữ liệu đã được chuẩn hóa (Torso) và tăng cường mạnh mẽ. ---")