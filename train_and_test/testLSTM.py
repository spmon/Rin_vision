
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# --- CẤU HÌNH ---
LSTM_MODEL_PATH = 'best_action_model.pth'
POSE_MODEL_PATH = 'yolo11l-pose.pt'
DET_MODEL_PATH = 'yolo11l.pt'
LABEL_MAP = np.load('label_map.npy')

SEQUENCE_LENGTH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Đang sử dụng thiết bị: {DEVICE} ---")
DRINKING_OBJECTS_ID = [39, 40, 41] # cup, wine glass, bottle
DETECTION_INTERVAL = 5 
frame_counter = 0

# --- ĐỊNH NGHĨA LẠI MÔ HÌNH (Đã sửa lỗi AttributeError) ---
class LSTMActionModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=32, num_layers=1, num_classes=len(LABEL_MAP)):
        super(LSTMActionModel, self).__init__()
        # Thêm 2 dòng này để lưu biến
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- HÀM CHUẨN HÓA POSE (Giữ nguyên) ---
def normalize_pose(kpts):
    try:
        origin = (kpts[5] + kpts[6]) / 2
        scale = np.linalg.norm(kpts[5] - kpts[6])
        if scale < 1e-4:
            return np.zeros_like(kpts)
        normalized_kpts = (kpts - origin) / scale
        return normalized_kpts
    except Exception as e:
        return np.zeros_like(kpts)

# --- Tải mô hình (Giữ nguyên) ---
print(f"Đang tải mô hình {LSTM_MODEL_PATH}...")
action_model = LSTMActionModel().to(DEVICE)
action_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
action_model.eval()
print("Đang tải YOLO11 (Pose & Detection)...")
yolo_pose = YOLO(POSE_MODEL_PATH)
yolo_det = YOLO(DET_MODEL_PATH)

# --- Mở webcam (Giữ nguyên) ---
cap = cv2.VideoCapture(0)
sequence = []
current_action = "Waiting..."
confidence = 0.0
# Đổi tên biến:
cup_near_face = False # Biến mới: Cốc/Chai ở gần mặt

print("\n--- BẮT ĐẦU DEMO (Logic Mũi-trong-Cốc) ---")
print("Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    frame_counter += 1

    # --- 1. CHẠY YOLO-POSE (Lấy keypoints và tọa độ MŨI) ---
    pose_results = yolo_pose.predict(frame, verbose=False, conf=0.5)
    
    keypoints_flat = np.zeros(34)
    nose_xy = None # Tọa độ pixel của mũi (ID 0)

    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
        # Lấy keypoints chuẩn hóa (17, 2) cho LSTM
        kpts_xyn = pose_results[0].keypoints.xyn[0].cpu().numpy()
        normalized_kpts = normalize_pose(kpts_xyn)
        keypoints_flat = normalized_kpts.flatten() # (34,)
        
        # Lấy keypoints pixel (17, 2) cho logic Giao nhau
        kpts_xy = pose_results[0].keypoints.xy[0].cpu().numpy()
        nose_xy = kpts_xy[0] # Lấy tọa độ Mũi (x, y)
        
        # (Tùy chọn) Vẽ bộ xương
        for kpt in kpts_xy:
             cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)

    # --- 2. CHẠY YOLO-DETECTION (Logic Mũi-trong-Cốc) ---
    if frame_counter % DETECTION_INTERVAL == 0:
        det_results = yolo_det.predict(frame, verbose=False, conf=0.5, classes=DRINKING_OBJECTS_ID)
        
        cup_near_face = False # Reset trạng thái
        
        # Vòng lặp kiểm tra từng cái cốc/chai được phát hiện
        for box_coords in det_results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box_coords
            
            # Mặc định vẽ hộp màu xanh (vật thể thụ động)
            box_color = (255, 0, 0) # Blue
            
            # KIỂM TRA LOGIC MỚI: Mũi có nằm trong hộp không?
            if nose_xy is not None:
                if (x1 < nose_xy[0] < x2) and (y1 < nose_xy[1] < y2):
                    cup_near_face = True # Xác nhận!
                    box_color = (0, 0, 255) # Đổi thành màu Đỏ (đang tương tác)

            # Vẽ hộp
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(frame, "CUP/BOTTLE", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # --- 3. CẬP NHẬT BỘ NHỚ ĐỆM (Giữ nguyên) ---
    sequence.append(keypoints_flat)
    sequence = sequence[-SEQUENCE_LENGTH:]

    # --- 4. DỰ ĐOÁN LSTM (Đã sửa lỗi Warning) ---
    if len(sequence) == SEQUENCE_LENGTH:
        # Chuyển list -> numpy array -> tensor (cách nhanh)
        seq_np = np.array([sequence], dtype=np.float32)
        input_tensor = torch.from_numpy(seq_np).to(DEVICE)
        
        with torch.no_grad():
            output = action_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, predicted_idx = torch.max(probs, 1)
            current_action = LABEL_MAP[predicted_idx.item()]
            confidence = conf.item()
            
    # --- 5. LOGIC KẾT HỢP (Nâng cấp) ---
    final_action = current_action
    final_confidence = confidence

    # Logic 1: Sửa lỗi "dụi mắt" -> "uống nước"
    if current_action == 'rubbing_eyes' and cup_near_face:
        final_action = 'drinking'
        final_confidence = 0.99 
    elif current_action =='waving' and cup_near_face:
        final_action='drinking'
        final_confidence= 0.99
    elif current_action =='idle' and cup_near_face:
        final_action='drinking'
        final_confidence= 0.99
    # Logic 2: Sửa lỗi "uống nước" -> "dụi mắt"
    # elif current_action == 'drinking' and not cup_near_face:
    #     # Nếu LSTM đoán 'drinking' nhưng không có cốc/chai nào gần mặt
    #     # thì khả năng cao đó là 'rubbing_eyes'
    #     final_action = 'idle'
    #     final_confidence = confidence * 0.7 # Giảm độ tự tin một chút cho an toàn

    # --- 6. HIỂN THỊ KẾT QUẢ (Giữ nguyên) ---
    cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
    text = f"{final_action} ({final_confidence:.1%})"
    color = (0, 255, 0)
    if final_confidence < 0.7:
        color = (0, 255, 255)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow('Action Recognition (Logic Mũi-trong-Cốc)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()