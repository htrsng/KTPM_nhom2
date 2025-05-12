from PIL import Image
import os
import shutil

# Thư mục đầu vào và đầu ra
input_dir = "data"
output_dir = "processed_dataset"
size = (640, 640)  # Kích thước phù hợp với YOLOv8 và LabelImg (có thể điều chỉnh)

def update_labels(label_path, original_size, new_size):
    """Cập nhật tọa độ nhãn trong file .txt theo kích thước mới"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        orig_w, orig_h = original_size
        new_w, new_h = new_size
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # Đảm bảo định dạng YOLO (class, x_center, y_center, width, height)
                continue
            class_id, x_center, y_center, width, height = map(float, parts[:5])
            
            # Cập nhật tọa độ theo tỷ lệ mới
            x_center = x_center * (new_w / orig_w)
            y_center = y_center * (new_h / orig_h)
            width = width * (new_w / orig_w)
            height = height * (new_h / orig_h)
            
            new_lines.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Lưu file nhãn mới
        with open(label_path, 'w') as f:
            f.writelines(new_lines)
    except Exception as e:
        print(f"Lỗi khi cập nhật nhãn {label_path}: {e}")

def process_folder(input_dir, output_dir):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        output_path = os.path.join(output_dir, item)

        if os.path.isdir(item_path):
            # Xử lý đệ quy thư mục con
            process_folder(item_path, output_path)
        elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif')):
            try:
                # Mở và xử lý ảnh
                img = Image.open(item_path).convert("RGB")
                original_size = img.size  # Lưu kích thước gốc
                img = img.resize(size, Image.LANCZOS)  # Sử dụng LANCZOS để giữ độ sắc nét
                
                # Lưu ảnh với chất lượng cao
                if item.lower().endswith(('.jpg', '.jpeg')):
                    img.save(output_path, quality=95)  # Chất lượng cao cho JPEG
                else:
                    img.save(output_path)  # Giữ nguyên định dạng gốc cho PNG, WebP, v.v.
                
                # Kiểm tra và cập nhật file nhãn (nếu có)
                label_path = os.path.splitext(item_path)[0] + '.txt'
                label_output_path = os.path.splitext(output_path)[0] + '.txt'
                if os.path.exists(label_path):
                    shutil.copy(label_path, label_output_path)  # Sao chép file nhãn
                    update_labels(label_output_path, original_size, size)  # Cập nhật tọa độ nhãn
                
            except Exception as e:
                print(f"Lỗi xử lý ảnh {item} tại {item_path}: {e}")
        elif item.lower().endswith('.txt'):
            # Sao chép file nhãn nếu không phải ảnh
            shutil.copy(item_path, output_path)

# Chạy xử lý
process_folder(input_dir, output_dir)
print("Xử lý hoàn tất!")