from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import numpy as np
import uuid
import re
import cv2
import albumentations as A
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the model
MODEL_PATH = r"D:\KTPM_nhom2\SKIN\models\best.pt"

# Load YOLOv8 model
try:
    logger.info("Loading YOLO model")
    model = YOLO(MODEL_PATH)
    logger.info("Model labels: %s", model.names)
    logger.info(model.info())
except Exception as e:
    logger.error("Error loading model: %s", e)
    raise

# Directory to save uploaded images and results
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
PREDICT_FOLDER = os.path.join(RESULT_FOLDER, "predict")
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, PREDICT_FOLDER]:
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            logger.info("Created folder: %s", folder)
        except Exception as e:
            logger.error("Error creating folder %s: %s", folder, e)
            raise
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["PREDICT_FOLDER"] = PREDICT_FOLDER

# Scoring weights for skin issues
ISSUE_WEIGHTS = {
    "acne": 0.5,
    "pores": 0.3,
    "pigment": 0.2
}

def check_image_quality(img):
    """Check image sharpness using Laplacian variance."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > 100
    except Exception as e:
        logger.warning("Error checking image quality: %s", e)
        return False

def preprocess_image(image_stream):
    """Preprocess image: resize and lightly normalize."""
    try:
        if not image_stream or image_stream.filename == "":
            raise ValueError("Empty image stream or no file provided")
        
        ext = os.path.splitext(image_stream.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            raise ValueError("Only JPEG or PNG images are supported")
        
        img_array = np.frombuffer(image_stream.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format or corrupted file")
        
        if not check_image_quality(img):
            logger.warning("Input image is blurry")
        
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        
        aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        ])
        img = aug(image=img)["image"]
        
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    except Exception as e:
        logger.error("Image preprocessing failed: %s", e)
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def calculate_skin_score(predictions):
    """Calculate skin score based on model predictions."""
    total_penalty = 0
    issue_counts = {"acne": 0, "pores": 0, "pigment": 0}
    issue_confidences = {"acne": [], "pores": [], "pigment": []}
    
    for pred in predictions:
        logger.info("Prediction boxes: %d", len(pred.boxes))
        for box in pred.boxes:
            label = pred.names[int(box.cls)]
            confidence = float(box.conf)
            weight = ISSUE_WEIGHTS.get(label, 0.3)
            total_penalty += confidence * weight
            issue_counts[label] += 1
            issue_confidences[label].append(confidence)
            logger.info("Detected %s with confidence %.2f, penalty %.2f", label, confidence, confidence * weight)
    
    score = max(0, 10 - total_penalty * 5)
    logger.info("Calculated score: %.1f", score)
    return round(score, 1), issue_counts, issue_confidences

def suggest_improvements(issue_counts, issue_confidences):
    """Suggest improvements based on skin issues."""
    improvements = {"acne": [], "pores": [], "pigment": []}
    
    if issue_counts["acne"] > 0:
        avg_conf = sum(issue_confidences["acne"]) / len(issue_confidences["acne"]) if issue_confidences["acne"] else 0
        if issue_counts["acne"] > 2 or avg_conf > 0.5:
            improvements["acne"] = [
                "Thăm bác sĩ da liễu để được tư vấn chuyên sâu.",
                "Sử dụng kem trị mụn chứa BHA hoặc retinoid.",
                "Rửa mặt 2 lần/ngày, tránh chạm tay lên mặt."
            ]
        else:
            improvements["acne"] = [
                "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ.",
                "Dùng kem trị mụn không kê đơn (benzoyl peroxide).",
                "Giữ da sạch và đủ ẩm."
            ]
    
    if issue_counts["pores"] > 0:
        avg_conf = sum(issue_confidences["pores"]) / len(issue_confidences["pores"]) if issue_confidences["pores"] else 0
        if issue_counts["pores"] > 2 or avg_conf > 0.5:
            improvements["pores"] = [
                "Sử dụng mặt nạ đất sét 1-2 lần/tuần.",
                "Duy trì độ ẩm với kem dưỡng không gây bít tắc.",
                "Tẩy tế bào chết định kỳ với AHA."
            ]
        else:
            improvements["pores"] = [
                "Tẩy tế bào chết 1-2 lần/tuần với sản phẩm dịu nhẹ.",
                "Dùng toner se khít lỗ chân lông.",
                "Rửa mặt kỹ để loại bỏ dầu thừa."
            ]
    
    if issue_counts["pigment"] > 0:
        avg_conf = sum(issue_confidences["pigment"]) / len(issue_confidences["pigment"]) if issue_confidences["pigment"] else 0
        if issue_counts["pigment"] > 2 or avg_conf > 0.5:
            improvements["pigment"] = [
                "Sử dụng serum vitamin C hoặc niacinamide hàng ngày.",
                "Dùng kem chống nắng SPF 50+ và che chắn kỹ khi ra ngoài.",
                "Tham khảo peel da hóa học với bác sĩ da liễu."
            ]
        else:
            improvements["pigment"] = [
                "Dùng kem chống nắng SPF 30+ mỗi ngày.",
                "Sử dụng serum vitamin C để làm sáng da.",
                "Tránh tiếp xúc trực tiếp với ánh nắng."
            ]
    
    return improvements

def suggest_products(issue_counts):
    """Suggest products based on skin issues."""
    products = {"acne": [], "pores": [], "pigment": []}
    
    if issue_counts["acne"] > 0:
        products["acne"] = [
            "Cetaphil Gentle Cleanser",
            "Differin Gel (Adapalene 0.1%)",
            "La Roche-Posay Effaclar Duo"
        ]
    if issue_counts["pores"] > 0:
        products["pores"] = [
            "Paula’s Choice Skin Perfecting 2% BHA Liquid",
            "Innisfree Super Volcanic Pore Clay Mask",
            "The Ordinary Niacinamide 10% + Zinc 1%"
        ]
    if issue_counts["pigment"] > 0:
        products["pigment"] = [
            "The Ordinary Vitamin C Suspension 23%",
            "La Roche-Posay Anthelios SPF 50 Sunscreen",
            "Skinceuticals C E Ferulic Serum"
        ]
    
    return products

def sanitize_filename(filename):
    """Sanitize filename by replacing spaces and special characters."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

@app.route("/")
def index():
    logger.info("Serving index.html")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received /predict request")
    logger.debug("Files: %s", request.files)
    
    if not all(side in request.files for side in ["left", "right", "front"]):
        logger.error("Missing one or more images")
        return jsonify({"error": "Vui lòng tải lên cả ba ảnh (trái, phải, chính diện)"}), 400

    images = {
        "left": request.files["left"],
        "right": request.files["right"],
        "front": request.files["front"]
    }

    image_paths = {}
    filenames = {}
    result_filenames = {}
    results = {}

    try:
        for side, image in images.items():
            logger.info("Processing %s image", side)
            if not image or image.filename == "":
                logger.error("No %s image uploaded", side)
                return jsonify({"error": f"Không có ảnh {side} được tải lên"}), 400

            # Preprocess image
            logger.debug("Preprocessing %s image", side)
            processed_image = preprocess_image(image)
            
            # Sanitize filename and add unique identifier
            ext = '.jpg'
            sanitized_filename = sanitize_filename(os.path.splitext(image.filename)[0])
            filename = f"{side}_{uuid.uuid4().hex}_{sanitized_filename}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            # Save image
            try:
                with open(filepath, 'wb') as f:
                    f.write(processed_image)
                logger.info("Saved %s image to %s", side, filepath)
            except PermissionError as e:
                logger.error("Permission error saving %s: %s", filepath, e)
                return jsonify({"error": f"Không có quyền ghi vào {filepath}"}), 500

            image_paths[side] = filepath
            filenames[side] = filename

            # Run YOLO inference
            logger.info("Running YOLO prediction for %s", side)
            try:
                result = model.predict(
                    filepath,
                    save=True,
                    project=app.config["RESULT_FOLDER"],
                    name="predict",
                    exist_ok=True,
                    conf=0.2,
                    augment=True,
                    verbose=True
                )
                results[side] = result[0]
                logger.info("Results for %s: %d detections", side, len(result[0].boxes))
            except Exception as e:
                logger.error("YOLO prediction error for %s: %s", side, e)
                return jsonify({"error": f"Dự đoán YOLO thất bại cho {side}: {str(e)}"}), 500

            # Verify result file
            result_filepath = os.path.join(app.config["PREDICT_FOLDER"], filename)
            if not os.path.exists(result_filepath):
                logger.error("Result file %s not found", result_filepath)
                return jsonify({"error": f"Kết quả YOLO cho {side} không được lưu"}), 500
            result_filenames[side] = filename

        # Calculate skin score
        logger.info("Calculating skin score")
        combined_predictions = [results["left"], results["right"], results["front"]]
        skin_score, issue_counts, issue_confidences = calculate_skin_score(combined_predictions)

        # Suggest improvements and products
        logger.info("Generating suggestions")
        improvements = suggest_improvements(issue_counts, issue_confidences)
        products = suggest_products(issue_counts)

        # Prepare response
        response = {
            "score": skin_score,
            "analysis": {
                "left": [{"label": results["left"].names[int(box.cls)], "confidence": float(box.conf)} for box in results["left"].boxes] or [{"label": "Không phát hiện", "confidence": 0.0}],
                "right": [{"label": results["right"].names[int(box.cls)], "confidence": float(box.conf)} for box in results["right"].boxes] or [{"label": "Không phát hiện", "confidence": 0.0}],
                "front": [{"label": results["front"].names[int(box.cls)], "confidence": float(box.conf)} for box in results["front"].boxes] or [{"label": "Không phát hiện", "confidence": 0.0}]
            },
            "image_urls": {
                "left": f"/static/results/predict/{result_filenames['left']}",
                "right": f"/static/results/predict/{result_filenames['right']}",
                "front": f"/static/results/predict/{result_filenames['front']}"
            },
            "improvements": improvements,
            "products": products
        }

        logger.info("Response prepared: %s", response)
        return jsonify(response)
    except ValueError as e:
        logger.error("Value error: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": f"Dự đoán thất bại: {str(e)}"}), 500

@app.route("/cleanup", methods=["POST"])
def cleanup():
    """Clean up old files in uploads and results folders."""
    logger.info("Received /cleanup request")
    try:
        for folder in [app.config["UPLOAD_FOLDER"], app.config["PREDICT_FOLDER"]]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info("Deleted %s", file_path)
                    except (PermissionError, OSError) as e:
                        logger.warning("Error deleting %s: %s", file_path, e)
                        continue
        logger.info("Cleanup completed")
        return jsonify({"message": "Đã dọn dẹp các file cũ"})
    except Exception as e:
        logger.error("Cleanup error: %s", e)
        return jsonify({"error": f"Dọn dẹp thất bại: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)