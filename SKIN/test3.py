import os
import shutil

source_dir = r"D:\KTPM_nhom2\SKIN"
target_dir = r"D:\KTPM_nhom2\SKIN\dataset\labels\train"

os.makedirs(target_dir, exist_ok=True)

moved = 0
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.txt'):
            full_path = os.path.join(root, file)
            new_path = os.path.join(target_dir, file)
            if full_path != new_path:  # Tránh sao chép chính nó
                try:
                    shutil.move(full_path, new_path)
                    print(f"[✔] Đã di chuyển: {file}")
                    moved += 1
                except Exception as e:
                    print(f"[!] Không thể di chuyển {file}: {e}")

print(f"\n✅ Hoàn tất! Đã di chuyển {moved} file .txt vào {target_dir}")
