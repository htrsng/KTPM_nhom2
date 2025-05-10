import os

image_dir = "processed_dataset"
label_dir = "labels"

# Lặp qua từng thư mục class (acne, lochanlong, sactoda)
for class_name in os.listdir(image_dir):
    image_class_dir = os.path.join(image_dir, class_name)
    label_class_dir = os.path.join(label_dir, class_name)

    # Bỏ qua nếu không phải thư mục
    if not os.path.isdir(image_class_dir):
        continue

    for image_file in os.listdir(image_class_dir):
        # Tạo đường dẫn đầy đủ cho ảnh
        image_path = os.path.join(image_class_dir, image_file)

        # Tạo tên file nhãn tương ứng (thay phần mở rộng bằng .txt)
        base_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(label_class_dir, base_name + '.txt')

        # Kiểm tra xem file nhãn có tồn tại không
        if not os.path.exists(label_path):
            print(f"[!] Missing label for {image_path}")
            continue  # Bỏ qua nếu không có nhãn

        # ✅ Ở đây bạn có thể xử lý ảnh & nhãn như bình thường
        print(f"[OK] Found: {image_path} + {label_path}")
