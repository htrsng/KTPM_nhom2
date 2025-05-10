import os

label_dir = "D:/KTPM_nhom2/SKIN/dataset/labels/train"

for fname in os.listdir(label_dir):
    path = os.path.join(label_dir, fname)
    with open(path, 'r') as f:
        lines = f.readlines()

    valid_lines = []
    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError("Không đủ 5 giá trị")
            cls_id = int(parts[0])
            if not (0 <= cls_id <= 2):
                raise ValueError("class_id > 2")
            floats = list(map(float, parts[1:]))
            if not all(0 <= x <= 1 for x in floats):
                raise ValueError("giá trị tọa độ không nằm trong [0,1]")
            valid_lines.append(line)
        except Exception as e:
            print(f"[⚠] Lỗi ở file {fname}: {line.strip()} ({e})")

    if valid_lines:
        with open(path, 'w') as f:
            f.writelines(valid_lines)
    else:
        print(f"[🗑] Xóa file {fname} vì không còn dòng hợp lệ")
        os.remove(path)
