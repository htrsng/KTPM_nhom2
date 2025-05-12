from ultralytics import YOLO
import os

# Load model
model = YOLO('runs/detect/train/weights/best.pt')

# Đường dẫn ảnh hoặc thư mục ảnh
source = 'test_img'

# Tạo thư mục lưu kết quả nếu chưa có
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Dự đoán
results = model.predict(source=source, conf=0.1, save=True, save_txt=True, project=output_dir, name='predict')

print(f"Kết quả đã lưu vào thư mục: {output_dir}/predict")
