from PIL import Image
import os

input_dir = "data"
output_dir = "processed_dataset"
size = (224, 224)

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
                img = img.resize(size)
                img.save(output_path)
            except Exception as e:
                print(f"Lỗi xử lý ảnh {item} tại {item_path}: {e}")

process_folder(input_dir, output_dir)
