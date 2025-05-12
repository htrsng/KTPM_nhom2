from PIL import Image, ImageEnhance
import cv2
import os
import shutil

# Thư mục đầu vào và đầu ra
input_dir = "data"
output_dir = "processed_dataset"
size = (640, 640)  # Kích thước chuẩn cho YOLOv8

def enhance_image(image):
    """Tăng độ tương phản và độ sắc nét cho ảnh"""
    # Chuyển PIL Image sang OpenCV để xử lý
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_np)
    l = clahe.apply(l)
    img_np = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2BGR)
    
    # Chuyển lại sang PIL Image
    image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    return image

def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        output_path = os.path.join(output_dir, item)

        if os.path.isdir(item_path):
            process_folder(item_path, output_path)
        elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif')):
            try:
                # Mở ảnh
                img = Image.open(item_path).convert("RGB")
                original_size = img.size
                
                # Tăng chất lượng
                img = enhance_image(img)
                
                # Resize ảnh
                img = img.resize(size, Image.LANCZOS)
                
                # Lưu ảnh
                if item.lower().endswith(('.jpg', '.jpeg')):
                    img.save(output_path, quality=95)
                else:
                    img.save(output_path)
                
                # Sao chép file nhãn (nếu có)
                label_path = os.path.splitext(item_path)[0] + '.txt'
                label_output_path = os.path.splitext(output_path)[0] + '.txt'
                if os.path.exists(label_path):
                    shutil.copy(label_path, label_output_path)
                    # Lưu ý: Cập nhật tọa độ nhãn sẽ thực hiện sau khi gắn nhãn
            except Exception as e:
                print(f"Lỗi xử lý ảnh {item} tại {item_path}: {e}")
        elif item.lower().endswith('.txt'):
            shutil.copy(item_path, output_path)

# Chạy xử lý
import numpy as np
process_folder(input_dir, output_dir)
print("Tiền xử lý ảnh hoàn tất!")