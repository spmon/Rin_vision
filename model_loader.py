# model_loader.py
import torch
import cv2
from ultralytics import YOLO
import torchvision.transforms as transforms
import numpy as np
import config
# Import từ các file model mới
from models.action_lstm import LSTMActionModel
from models.emotion_vgg import VGG

def load_all_models():
    """
    Tải tất cả các mô hình và bộ xử lý cần thiết.
    Trả về một dictionary chứa các model đã được tải.
    """
    print(f"--- Đang sử dụng thiết bị: {config.DEVICE} ---")
    
    models = {}

    # 1. Tải mô hình LSTM (Hành động)
    print("Đang tải mô hình LSTM (Hành động)...")
    action_model = LSTMActionModel().to(config.DEVICE)
    action_model.load_state_dict(torch.load(config.LSTM_MODEL_PATH, map_location=config.DEVICE))
    action_model.eval()
    models["action_model"] = action_model

    # 2. Tải mô hình Cảm xúc (VGG)
    print(f"Đang tải mô hình cảm xúc {config.EMOTION_MODEL_PATH}...")
    emotion_model = VGG("VGG19").to(config.DEVICE)
    try:
        checkpoint = torch.load(config.EMOTION_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        emotion_model.load_state_dict(checkpoint["model_weights"])
        emotion_model.eval()
        print(f"Đã tải thành công mô hình cảm xúc (Epoch {checkpoint['epoch']})")
        models["emotion_model"] = emotion_model
    except Exception as e:
        print(f"LỖI: Không thể tải mô hình cảm xúc: {e}")
        exit()

    # 3. Tải YOLO (Pose & Detection)
    print("Đang tải YOLO (Pose & Detection)...")
    models["yolo_pose"] = YOLO(config.POSE_MODEL_PATH)
    models["yolo_det"] = YOLO(config.DET_MODEL_PATH)
    models["object_names"] = models["yolo_det"].names

    # 4. Tải Haar Cascade
    print("Đang tải Haar Cascade (Phát hiện mặt)...")
    face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)
    if face_cascade.empty():
        print(f"LỖI: Không thể tải file {config.CASCADE_PATH}")
        exit()
    models["face_cascade"] = face_cascade

    # 5. Tải bộ xử lý ảnh cho model Cảm xúc
    print("Đang tải bộ xử lý ảnh...")
    crop_size = 44 # Kích thước này phải khớp với lúc train model VGG
    emotion_transform = transforms.Compose([
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    models["emotion_transform"] = emotion_transform

    print("\n--- TẤT CẢ MÔ HÌNH ĐÃ SẴN SÀNG! ---")
    print("Warm-up GPU...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Chạy thử Pose (Nặng)
    models["yolo_pose"].predict(dummy_frame, verbose=False)
    
    # Chạy thử Emotion (VGG19)
    # Cần một tensor đầu vào đúng kích thước (dummy tensor)
    dummy_input = torch.zeros(10, 1, 48, 48).to(config.DEVICE)
    with torch.no_grad():
        models["emotion_model"](dummy_input)

    print("GPU đã nóng máy, sẵn sàng chiến đấu!")
    return models