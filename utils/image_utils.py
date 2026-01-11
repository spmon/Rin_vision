# utils/image_utils.py
import cv2
import numpy as np
import torch
from PIL import Image
import config

def preprocess_emotion_image(face_roi, emotion_transform):
    """
    Tiền xử lý ROI khuôn mặt cho model nhận diện cảm xúc.
    Bao gồm: chuyển xám, resize, TenCrop, và chuyển sang Tensor.
    """
    try:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (48, 48))
        face_pil = Image.fromarray(np.uint8(face_gray))
        
        inputs = emotion_transform(face_pil)
        ncrops, c, ht, wt = np.shape(inputs)
        inputs = inputs.view(-1, c, ht, wt).to(config.DEVICE)
        return inputs
    except Exception as e:
        # print(f"Lỗi tiền xử lý ảnh cảm xúc: {e}")
        return None