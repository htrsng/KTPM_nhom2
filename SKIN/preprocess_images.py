from PIL import Image, ImageEnhance
import cv2
import os
import shutil
import numpy as np

input_dir = "data"
output_dir = "processed_dataset"
target_size = (512, 512)  

def enhance_image(image):
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Làm sáng ảnh
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l = clahe.apply(l)
    img_lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # Sharpen nhẹ (dùng kernel dịu hơn)
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    img_np = cv2.filter2D(img_np, -1, kernel)

    # Chuyển lại sang PIL
    image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    return image

def letterbox_image(image, target_size):
    image = image.convert('RGB')
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_resized = image.resize((nw, nh), Image.LANCZOS)
    new_image = Image.new("RGB", target_size, (128, 128, 128))  
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image, (iw, ih), (nw, nh), ((w - nw) // 2, (h - nh) // 2)

def update_labels(label_path, orig_size, resized_info):
    
    try:
        iw, ih = orig_size
        nw, nh = resized_info[1]
        pad_x, pad_y = resized_info[2]
        scale_x = nw / iw
        scale_y = nh / ih

        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, x, y, w, h = map(float, parts)
            # Chuyển đổi tọa độ từ ảnh gốc sang ảnh mới
            x = x * iw * scale_x + pad_x
            y = y * ih * scale_y + pad_y
            w = w * iw * scale_x
            h = h * ih * scale_y
          # chuẩn hóa tọa độ 
            x /= target_size[0] 
            y /= target_size[1]
            w /= target_size[0]
            h /= target_size[1]
            new_lines.append(f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        with open(label_path, 'w') as f:
            f.writelines(new_lines)

    except Exception as e:
        print(f"Lỗi khi cập nhật nhãn {label_path}: {e}")

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
                img = Image.open(item_path).convert("RGB")
                img = enhance_image(img)
                new_img, orig_size, resized_size, padding = letterbox_image(img, target_size)
                new_img.save(output_path, quality=95)

                label_path = os.path.splitext(item_path)[0] + '.txt'
                label_output_path = os.path.splitext(output_path)[0] + '.txt'
                if os.path.exists(label_path):
                    shutil.copy(label_path, label_output_path)
                    update_labels(label_output_path, orig_size, (orig_size, resized_size, padding))
            except Exception as e:
                print(f"Lỗi xử lý ảnh {item} tại {item_path}: {e}")
        elif item.lower().endswith('.txt'):
            shutil.copy(item_path, output_path)

# Chạy
process_folder(input_dir, output_dir)
print("✅ Tiền xử lý ảnh hoàn tất.")
