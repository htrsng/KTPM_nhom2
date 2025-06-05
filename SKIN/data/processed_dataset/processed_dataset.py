from PIL import Image, ImageEnhance
import cv2
import os
import shutil
import numpy as np

input_dir = "D:/KTPM_nhom2/SKIN/data" 
output_dir = "D:/KTPM_nhom2/SKIN/processed_dataset"  
size = (640, 640) 

def enhance_image(image):
    """Tăng độ tương phản và độ sắc nét cho ảnh"""
    # Chuyển PIL Image sang OpenCV
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    

    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 5, 5, 7, 21)
    
    # Tăng độ tương phản với CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)) 
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_np)
    l = clahe.apply(l)
    img_np = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2BGR)
    
   
    alpha = 1.5  # Hệ số tương phản 
    beta = 10    # Hệ số sáng 
    img_np = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    
    # Tăng độ sắc nét với Unsharp Mask
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    img_np = cv2.addWeighted(img_np, 1.2, blurred, -0.2, 0)
    
    # Chuyển lại sang PIL Image
    image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    
    # Tăng độ sắc nét nhẹ bằng PIL
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    return image

def normalize_labels(label_path, original_size, new_size):
    """Chuẩn hóa tọa độ nhãn sau khi resize ảnh"""
    if not os.path.exists(label_path):
        return
    
    width_ratio = new_size[0] / original_size[0]
    height_ratio = new_size[1] / original_size[1]
    
    lines = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Dòng không hợp lệ trong {label_path}: {line.strip()}")
                continue
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:5])
            # Chuẩn hóa tọa độ
            x_center *= width_ratio
            y_center *= height_ratio
            width *= width_ratio
            height *= height_ratio
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        with open(label_path, 'w') as f:
            f.writelines(new_lines)
    except Exception as e:
        print(f"Lỗi xử lý nhãn {label_path}: {e}")

def process_folder(input_dir, output_dir):
    """Xử lý ảnh và nhãn trong thư mục"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        output_path = os.path.join(output_dir, os.path.splitext(item)[0] + '.jpg')

        if os.path.isdir(item_path):
            process_folder(item_path, os.path.join(output_dir, item))
        elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                # Mở ảnh
                img = Image.open(item_path).convert("RGB")
                original_size = img.size
                
                # Chuyển sang OpenCV để resize
                img_np = np.array(img)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # Resize với OpenCV
                if original_size[0] > size[0] or original_size[1] > size[1]:
                    img_np = cv2.resize(img_np, size, interpolation=cv2.INTER_AREA)
                else:
                    img_np = cv2.resize(img_np, size, interpolation=cv2.INTER_CUBIC)
                # Chuyển lại sang PIL
                img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                
                # Tăng chất lượng
                img = enhance_image(img)
                
                # Lưu ảnh
                img.save(output_path, 'JPEG', quality=100)
                print(f"Đã xử lý: {item} -> {os.path.basename(output_path)}")
                
                # Sao chép và chuẩn hóa file nhãn
                label_path = os.path.splitext(item_path)[0] + '.txt'
                label_output_path = os.path.splitext(output_path)[0] + '.txt'
                if os.path.exists(label_path):
                    shutil.copy(label_path, label_output_path)
                    normalize_labels(label_output_path, original_size, size)
                    print(f"Đã sao chép và chuẩn hóa nhãn: {os.path.basename(label_path)}")
            except Exception as e:
                print(f"Lỗi xử lý ảnh {item} tại {item_path}: {e}")
        elif item.lower().endswith('.txt'):
            shutil.copy(item_path, os.path.join(output_dir, item))
            print(f"Đã sao chép nhãn: {item}")

# Chạy xử lý
process_folder(input_dir, output_dir)
print("Tiền xử lý ảnh hoàn tất!")