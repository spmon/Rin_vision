
import shutil
import random
import yaml
from tqdm import tqdm

# ⚙️ Cấu hình
COCO_PATH = r"D:\Yolo_Datasets\coco"  # Đường dẫn coco chuẩn
TARGET_PATH = r"D:\testyolo\coco_filtered"  # Nơi lưu ảnh/label sau khi tách
CLASSES_TO_KEEP = ["person","backpack","handbag","tie","suitcase","bottle","cup","chair","laptop","mouse","remote","keyboard","cell phone","book","clock"]

# 📖 Đọc file YAML gốc để lấy tên class
# [SỬA] Dùng thư viện yaml để đọc file, an toàn hơn
yaml_path = r"D:\testyolo\.venv\Lib\site-packages\ultralytics\cfg\datasets\coco.yaml"
try:
    with open(yaml_path, "r", encoding="utf-8") as f:
        coco_data = yaml.safe_load(f)
    original_names = coco_data['names']
    if isinstance(original_names, dict): # Xử lý nếu 'names' là dict
        original_names = [original_names[i] for i in sorted(original_names.keys())]
except Exception as e:
    print(f"LỖI: Không thể đọc file YAML: {e}"); exit()

# 🔢 Lấy id của các class cần giữ
keep_ids = [i for i, n in enumerate(original_names) if n in CLASSES_TO_KEEP]
print(f"Các class cần giữ: {CLASSES_TO_KEEP} → ID gốc {keep_ids}")

# ⭐ TẠO MAP ÁNH XẠ ID (GỐC -> MỚI)
id_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
print(f"Ánh xạ ID (gốc -> mới): {id_map}")

# 📦 Tạo thư mục đầu ra
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(TARGET_PATH, sub), exist_ok=True)

# ---------------------------------------------------------------------------
# [SỬA] 1. THÊM HÀM CHUYỂN ĐỔI (SEG -> BBOX)
# ---------------------------------------------------------------------------
def convert_segmentation_to_bbox(points):
    """
    Tính toán hộp vuông (x_center, y_center, width, height)
    từ một danh sách các điểm đường viền (x1, y1, x2, y2, ...)
    """
    # Tách các tọa độ x và y
    x_coords = [float(points[i]) for i in range(0, len(points), 2)]
    y_coords = [float(points[i]) for i in range(1, len(points), 2)]
    
    # Tìm min/max
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # Chuyển đổi (x_min, y_min, x_max, y_max) sang (x_center, y_center, w, h)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

# ---------------------------------------------------------------------------
# 📂 Duyệt label và copy file
def filter_labels(label_dir, img_dir, out_label_dir, out_img_dir):
    total, kept = 0, 0
    if not os.path.exists(label_dir):
        print(f"[LỖI] Không tìm thấy thư mục: {label_dir}")
        return
        
    for file in tqdm(os.listdir(label_dir), desc=f"Đang xử lý {os.path.basename(label_dir)}"):
        if not file.endswith(".txt"):
            continue
        total += 1
        label_path = os.path.join(label_dir, file)
        
        with open(label_path, "r") as f:
            lines = f.readlines()

        # [SỬA] 2. THAY ĐỔI LOGIC LỌC
        new_label_content = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: # Cần ít nhất (id x1 y1)
                continue
            
            try:
                old_id = int(parts[0])
            except ValueError:
                continue
            
            # Nếu ID gốc nằm trong danh sách cần giữ
            if old_id in id_map:
                new_id = id_map[old_id] # Lấy ID mới
                
                # Lấy tất cả các điểm tọa độ (bỏ qua id)
                segmentation_points = parts[1:]
                
                # Chuyển đổi đường viền -> hộp vuông
                x_c, y_c, w, h = convert_segmentation_to_bbox(segmentation_points)
                
                # Tạo dòng mới cho file Bounding Box
                new_line = f"{new_id} {x_c} {y_c} {w} {h}"
                new_label_content.append(new_line)

        if new_label_content: # Chỉ copy ảnh/nhãn nếu còn object
            kept += 1
            # Ghi lại file label với nội dung BBOX
            with open(os.path.join(out_label_dir, file), "w") as f:
                f.write('\n'.join(new_label_content))
                
            # Copy ảnh tương ứng
            img_file_jpg = file.replace(".txt", ".jpg")
            src_img_jpg = os.path.join(img_dir, img_file_jpg)
            
            if os.path.exists(src_img_jpg):
                shutil.copy(src_img_jpg, os.path.join(out_img_dir, img_file_jpg))
            else:
                # Thử tìm .png hoặc .jpeg
                img_file_png = file.replace(".txt", ".png")
                src_img_png = os.path.join(img_dir, img_file_png)
                if os.path.exists(src_img_png):
                    shutil.copy(src_img_png, os.path.join(out_img_dir, img_file_png))
                else:
                    img_file_jpeg = file.replace(".txt", ".jpeg")
                    src_img_jpeg = os.path.join(img_dir, img_file_jpeg)
                    if os.path.exists(src_img_jpeg):
                        shutil.copy(src_img_jpeg, os.path.join(out_img_dir, img_file_jpeg))
                    else:
                        print(f"[⚠️] Không tìm thấy ảnh: {img_file_jpg} (hoặc .png, .jpeg)")
                
    print(f"Đã xử lý {total} file label, giữ lại {kept} file có class hợp lệ.")

# ---------------------------------------------------------------------------
# [SỬA] 3. CHẠY LỌC CHO CẢ TRAIN VÀ VAL
# ---------------------------------------------------------------------------
print("Đang lọc và chuyển đổi bộ train...")
filter_labels(os.path.join(COCO_PATH, "labels", "train2017"),
              os.path.join(COCO_PATH, "images", "train2017"),
              os.path.join(TARGET_PATH, "labels", "train"),
              os.path.join(TARGET_PATH, "images", "train"))

print("\nĐang lọc và chuyển đổi bộ val...")
filter_labels(os.path.join(COCO_PATH, "labels", "val2017"),
              os.path.join(COCO_PATH, "images", "val2017"),
              os.path.join(TARGET_PATH, "labels", "val"),
              os.path.join(TARGET_PATH, "images", "val"))

# ---------------------------------------------------------------------------
# [SỬA] 4. TẠO FILE YAML MỚI
# ---------------------------------------------------------------------------
new_yaml_path = os.path.join(TARGET_PATH, 'filtered_data.yaml')
new_yaml_data = {
    'path': os.path.abspath(TARGET_PATH), # Đường dẫn tuyệt đối
    'train': 'images/train',
    'val': 'images/val', # Trỏ đến thư mục val đã lọc
    'nc': len(CLASSES_TO_KEEP),
    'names': CLASSES_TO_KEEP
}

with open(new_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(new_yaml_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

print(f"\n✅ Hoàn tất! Dữ liệu BBOX đã được lọc và chia tại: {TARGET_PATH}")
print(f"➡️ File config để huấn luyện: {new_yaml_path}")