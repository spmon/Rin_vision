# config.py
import torch
import numpy as np

# --- CẤU HÌNH ---
LSTM_MODEL_PATH = 'best_action_model2.pth'
POSE_MODEL_PATH = 'yolo11l-pose.pt'
DET_MODEL_PATH = 'yolo11l.pt'
EMOTION_MODEL_PATH = 'model_state.pth.tar'
CASCADE_PATH = "haarcascade_frontalface_default.xml" 

try:
    LABEL_MAP_ACTION = np.load('label_map.npy')
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'label_map.npy'. Sẽ sử dụng mảng rỗng.")
    LABEL_MAP_ACTION = np.array([])

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES_EMOTION = 7

# --- THAM SỐ XỬ LÝ ---
SEQUENCE_LENGTH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRINKING_OBJECTS_ID = [39, 40, 41]
DETECTION_INTERVAL = 5
EMOTION_INTERVAL = 15

# Cấu hình server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080