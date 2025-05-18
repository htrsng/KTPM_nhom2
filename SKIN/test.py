from ultralytics import YOLO
import os

# Đường dẫn đến mô hình
MODEL_PATH = r"D:\KTPM_nhom2\SKIN\models\best.pt"

# Đường dẫn đến hình ảnh kiểm tra (thay đổi thành hình ảnh của bạn)
IMAGE_PATH = r"D:\KTPM_nhom2\SKIN\test_images"  # Thay bằng đường dẫn thực tế

# Đường dẫn để lưu kết quả (nếu có)
OUTPUT_DIR = r"D:\KTPM_nhom2\SKIN\test_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def test_model():
    try:
        # Tải mô hình
        model = YOLO(MODEL_PATH)
        print("Model labels:", model.names)

        # Kiểm tra xem file hình ảnh có tồn tại không
        if not os.path.exists(IMAGE_PATH):
            print(f"Error: Image file {IMAGE_PATH} does not exist")
            return

        # Chạy suy luận
        results = model.predict(IMAGE_PATH, save=True, save_dir=OUTPUT_DIR, verbose=True)

        # In kết quả
        for result in results:
            print(f"\nDetections: {len(result.boxes)}")
            if len(result.boxes) == 0:
                print("No detections found.")
            else:
                for box in result.boxes:
                    label = result.names[int(box.cls)]
                    confidence = float(box.conf)
                    print(f"Detected {label} with confidence {confidence:.2f}")
        
        print(f"Results saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_model()