import numpy as np
import torch
import cv2
import config

# Import các hàm tiện ích
from utils.pose_utils import normalize_pose
from utils.image_utils import preprocess_emotion_image

def is_actually_waving(sequence, threshold=0.2):
    """
    Kiểm tra xem cổ tay có thực sự dao động (lắc) không.
    sequence: np.array shape (30, 34) - Chuỗi 30 frame gần nhất
    threshold: Ngưỡng phương sai (cần tinh chỉnh, 0.02 là khởi điểm tốt)
    """
    # Chỉ số trong mảng phẳng 34 phần tử:
    # Left Wrist (Index 9): x=18
    # Right Wrist (Index 10): x=20
    
    # Lấy dữ liệu 30 frame gần nhất của trục X cổ tay trái và phải
    left_wrist_x_history = sequence[:, 18]
    right_wrist_x_history = sequence[:, 20]
    
    # Tính độ lệch chuẩn (Standard Deviation) - Đo mức độ "rung lắc"
    std_left = np.std(left_wrist_x_history)
    std_right = np.std(right_wrist_x_history)
    print(f"DEBUG DAO ĐỘNG: Left={std_left:.4f} | Right={std_right:.4f} | Threshold={threshold}")
    # Nếu một trong hai tay rung lắc mạnh hơn ngưỡng -> Là vẫy tay
    # (Bạn có thể in std_left, std_right ra để debug xem ngưỡng bao nhiêu là vừa)
    if std_left > threshold or std_right > threshold:
        return True
    
    return False
def process_pose_and_action(frame, yolo_pose, action_model, sequence):
    """Xử lý Pose và dự đoán Hành động từ một frame."""
    pose_results = yolo_pose.predict(frame, verbose=False, conf=0.5)
    
    keypoints_flat = np.zeros(34)
    nose_xy = None
    current_action = "Waiting..."
    confidence = 0.0

    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
        kpts_xyn = pose_results[0].keypoints.xyn[0].cpu().numpy()
        kpts_xy = pose_results[0].keypoints.xy[0].cpu().numpy()
        nose_xy = kpts_xy[0] 
        # Gọi hàm tiện ích
        normalized_kpts = normalize_pose(kpts_xyn)
        keypoints_flat = normalized_kpts.flatten()
    
    sequence.append(keypoints_flat)
    sequence = sequence[-config.SEQUENCE_LENGTH:]
    
    if len(sequence) == config.SEQUENCE_LENGTH and config.LABEL_MAP_ACTION.size > 0:
        seq_np = np.array([sequence], dtype=np.float32)
        input_tensor = torch.from_numpy(seq_np).to(config.DEVICE)
        
        with torch.no_grad():
            output = action_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, predicted_idx = torch.max(probs, 1)
            current_action = config.LABEL_MAP_ACTION[predicted_idx.item()]
            confidence = conf.item()
    
    return sequence, current_action, confidence, nose_xy

def process_detection(frame, yolo_det, object_names, nose_xy):
    """Phát hiện đối tượng và kiểm tra xem có "cốc" gần mặt không."""
    det_results = yolo_det.predict(frame, verbose=False, conf=0.5)
    
    detected_objects_list = []
    cup_near_face = False
    
    for box in det_results[0].boxes:
        class_id = int(box.cls)
        object_name = object_names.get(class_id, "Unknown")
        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
        obj_confidence = float(box.conf)
        
        detected_objects_list.append({
            "name": object_name, "bbox": bbox, "confidence": obj_confidence
        })
        
        if class_id in config.DRINKING_OBJECTS_ID and nose_xy is not None:
            # --- LOGIC MỚI: MỞ RỘNG HITBOX ---
            
            # Lấy chiều cao của cái cốc
            cup_height = bbox[3] - bbox[1]
            
            # Mở rộng vùng kiểm tra lên phía trên khoảng 40% chiều cao cốc
            # (Giả định khoảng cách từ Môi lên Mũi tương đương 30-40% chiều cao cốc)
            extended_y_top = bbox[1] - (cup_height * 0.4)
            
            # Kiểm tra:
            # 1. Mũi nằm trong chiều ngang của cốc (bbox[0] < x < bbox[2])
            # 2. Mũi nằm DƯỚI "đỉnh cốc ảo" (extended_y_top)
            # 3. Mũi nằm TRÊN đáy cốc (bbox[3])
            
            is_horizontal_ok = bbox[0] < nose_xy[0] < bbox[2]
            is_vertical_ok = extended_y_top < nose_xy[1] < bbox[3]
            
            if is_horizontal_ok and is_vertical_ok:
                cup_near_face = True
    
    return detected_objects_list, cup_near_face


def process_emotion(frame, face_cascade, emotion_model, emotion_transform, nose_xy=None):
    """
    Phát hiện khuôn mặt và dự đoán Cảm xúc.
    Đã thêm tham số nose_xy để tối ưu hóa vùng tìm kiếm (ROI).
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []

    # 1. OPTIMIZATION: Nếu có tọa độ mũi, khoanh vùng để tìm mặt nhanh hơn
    if nose_xy is not None:
        nx, ny = int(nose_xy[0]), int(nose_xy[1])
        h_img, w_img = gray_frame.shape
        
        # Tạo vùng ROI quanh mũi (150px mỗi bên)
        roi_size = 150 
        x1 = max(0, nx - roi_size)
        y1 = max(0, ny - roi_size)
        x2 = min(w_img, nx + roi_size)
        y2 = min(h_img, ny + roi_size)
        
        roi_gray = gray_frame[y1:y2, x1:x2]
        
        # Chỉ chạy Haar Cascade trên vùng nhỏ này (Cực nhanh)
        # Giảm minSize xuống 50x50 vì ảnh crop đã sát mặt rồi
        roi_faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        # Chuyển đổi tọa độ từ ROI về tọa độ gốc của ảnh
        for (rx, ry, rw, rh) in roi_faces:
            faces.append((x1 + rx, y1 + ry, rw, rh))

    # 2. FALLBACK: Nếu không có mũi hoặc tìm vùng ROI thất bại -> Quét cả ảnh như cũ
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    # 3. Xử lý khuôn mặt tìm được
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Gọi hàm tiện ích (Code cũ của bạn)
        inputs = preprocess_emotion_image(face_roi, emotion_transform)
        
        if inputs is None:
            return "Error"
            
        try:
            with torch.no_grad():
                outputs = emotion_model(inputs)
                outputs = outputs.view(1, -1, config.NUM_CLASSES_EMOTION).mean(1) # Trung bình 10 crop
                _, predicted = torch.max(outputs, 1)
                return config.EMOTION_LABELS[int(predicted.cpu().numpy())]
                
        except Exception as e:
            # print(f"Lỗi dự đoán cảm xúc: {e}")
            return "Error"
    else:
        return "No Face"

def apply_custom_logic(current_action, confidence, cup_near_face, sequence):
    """
    Logic đè (Override):
    1. Check uống nước (như cũ).
    2. Check vẫy tay giả (MỚI).
    """
    final_action = current_action
    final_confidence = confidence
    
    # --- LOGIC 1: UỐNG NƯỚC
    if (current_action in ['rubbing_eyes', 'waving', 'idle']) and cup_near_face:
        final_action = 'drinking'
        final_confidence = 0.99
        return final_action, final_confidence # Return luôn

    # --- LOGIC 2: CHẶN VẪY TAY GIẢ (False Positive Waving) ---
    if current_action == 'waving':
        # Nếu AI bảo vẫy tay, nhưng tay không lắc -> Chuyển về idle
        if not is_actually_waving(np.array(sequence)):
            final_action = 'idle'

            # Giảm confidence xuống vì AI đang bối rối
            final_confidence = 0.5 
    elif current_action == 'drinking' and not cup_near_face:
        final_action = 'idle'
        final_confidence = confidence * 0.7
        
    return final_action, final_confidence