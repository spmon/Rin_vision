import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- CẤU HÌNH HUẤN LUYỆN (ĐÃ TỐI ƯU CÔNG SUẤT) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Đang sử dụng thiết bị: {DEVICE} ---")

EPOCHS = 60 
BATCH_SIZE = 64 
LEARNING_RATE = 0.001 
HIDDEN_SIZE = 32 # <-- GIẢM TỪ 64 XUỐNG 32
NUM_LAYERS = 1   # <-- GIẢM TỪ 2 XUỐNG 1

# ĐƯỜNG DẪN FILE
X_PATH = 'X_data.npy'
y_PATH = 'y_data.npy'
LABEL_MAP_PATH = 'label_map.npy'

# =========================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# =========================================
if not os.path.exists(X_PATH):
    print(f"LỖI: Không tìm thấy file {X_PATH}. Hãy chạy prepare_data.py trước!")
    exit()

print("Đang tải dữ liệu...")
X = np.load(X_PATH)
y = np.load(y_PATH)
actions = np.load(LABEL_MAP_PATH)
NUM_CLASSES = len(actions)
INPUT_SIZE = X.shape[2] 

print(f"Dữ liệu đầu vào (X): {X.shape}")
print(f"Nhãn (y): {y.shape}")
print(f"Các hành động: {actions} (Tổng cộng: {NUM_CLASSES})")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# =========================================
# 2. ĐỊNH NGHĨA MÔ HÌNH LSTM
# =========================================
class LSTMActionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMActionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=0.7) # <-- TĂNG DROPOUT
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

model = LSTMActionModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
print("\n--- Cấu trúc mô hình ---")
print(model)

criterion = nn.CrossEntropyLoss()
# <-- THÊM WEIGHT DECAY (L2 REGULARIZATION)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 

# =========================================
# 3. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# =========================================
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
for epoch in range(EPOCHS):
    # --- Giai đoạn Train ---
    model.train() 
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # <-- THÊM GAUSSIAN NOISE (DẠY CHỐNG NHIỄU)
        if model.training: 
            noise = torch.randn_like(inputs) * 0.01 
            inputs = inputs + noise.to(inputs.device)
        
        # 1. Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 2. Backward pass và cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Tính toán thống kê
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # --- Giai đoạn Validation ---
    model.eval() 
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # --- Tổng kết Epoch ---
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = train_correct / train_total
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = val_correct / val_total
    
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    print(f'Epoch [{epoch+1}/{EPOCHS}] | '
          f'Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2%}'
          f' | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2%}')

    # Lưu mô hình tốt nhất
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'best_action_model2.pth')

print("\n--- HUẤN LUYỆN HOÀN TẤT! ---")
print(f"Độ chính xác tốt nhất trên tập Val: {best_val_acc:.2%}")
print("Mô hình đã được lưu vào 'best_action_model.pth'")

# =========================================
# 4. VẼ BIỂU ĐỒ (Tùy chọn)
# =========================================
try:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Độ chính xác (Accuracy)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Độ lỗi (Loss)')
    plt.legend()
    plt.savefig('training_history.png') 
    print("Đã lưu biểu đồ vào 'training_history.png'")
    plt.show() 
except:
    pass
model.load_state_dict(torch.load('best_action_model2.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# 2. Tính toán Matrix
cm = confusion_matrix(all_labels, all_preds)

# 3. Vẽ Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Ground Truth)')
plt.xticks(rotation=45)
plt.tight_layout()

# Lưu ảnh
plt.savefig('confusion_matrix.png')
print("Đã lưu 'confusion_matrix.png'")
plt.show()

# 4. In báo cáo chi tiết (Precision, Recall, F1)
print("\n--- BÁO CÁO CHI TIẾT ---")
print(classification_report(all_labels, all_preds, target_names=actions))