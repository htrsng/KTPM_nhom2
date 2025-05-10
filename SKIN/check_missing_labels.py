import os

# Thư mục chứa ảnh và nhãn
image_dir = r'D:\KTPM_nhom2\SKIN\dataset\images\train'
label_dir = r'D:\KTPM_nhom2\SKIN\dataset\labels\train'

# Lấy danh sách tất cả ảnh (đuôi .jpg, .png, ...)
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Kiểm tra từng ảnh xem có file nhãn tương ứng không
missing_labels = []
for img_file in image_files:
    label_file = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(label_dir, label_file)
    if not os.path.exists(label_path):
        missing_labels.append(img_file)

# Kết quả
if missing_labels:
    print(f"[⚠] {len(missing_labels)} ảnh bị thiếu nhãn:")
    for file in missing_labels:
        print(f" - {file}")
else:
    print("[✅] Tất cả ảnh đều có file nhãn tương ứng.")
