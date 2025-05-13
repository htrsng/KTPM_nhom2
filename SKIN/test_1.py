
import os

labels_path = 'D:/KTPM_nhom2/SKIN/labelimg'  
for file in os.listdir(labels_path):
    if file.endswith('.txt'):
        full_path = os.path.join(labels_path, file)
        with open(full_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                parts = line.strip().split()
                if not parts:
                    continue
                if not parts[0].isdigit():
                    print(f"Lỗi tại {file} - dòng {i+1}: '{line.strip()}'")
