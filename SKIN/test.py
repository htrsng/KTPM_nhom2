import os
import shutil
import random
from collections import defaultdict

# Thư mục chứa ảnh và nhãn đã tiền xử lý
input_images_dir = "processed_dataset"  # Thư mục chứa ảnh
input_labels_dir = "labelimg"  # Thư mục chứa nhãn
output_dir = "dataset"  # Thư mục đầu ra

# Tỷ lệ chia dữ liệu
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def get_class_instances(labels_dir):
    """Đếm số instances của mỗi lớp từ các file nhãn"""
    class_counts = defaultdict(int)
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
    return class_counts

def split_data(images_dir, labels_dir, output_dir):
    # Tạo thư mục đầu ra
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)  # Xáo trộn ngẫu nhiên

    # Tính số lượng ảnh cho mỗi tập
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count  # Đảm bảo tổng bằng 100%

    # Chia ảnh
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]

    # Sao chép ảnh và nhãn
    for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img in images:
            # Sao chép ảnh
            shutil.copy(os.path.join(images_dir, img), os.path.join(output_dir, 'images', split, img))
            # Sao chép nhãn
            label_file = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(os.path.join(labels_dir, label_file)):
                shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, 'labels', split, label_file))

    print(f"Đã chia dữ liệu: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

# Kiểm tra số instances mỗi lớp
class_counts = get_class_instances(input_labels_dir)
print("Số instances mỗi lớp:", class_counts)

# Chia dữ liệu
split_data(input_images_dir, input_labels_dir, output_dir)
print("Tổ chức dữ liệu hoàn tất!")