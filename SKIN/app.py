from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import numpy as np
import uuid
import re

app = Flask(__name__)

# Path to the model
MODEL_PATH = r"D:\KTPM_nhom2\SKIN\models\best.pt"

# Load YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
    print("Model labels:", model.names)
    print(model.info())
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Directory to save uploaded images and results
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"  # Thư mục cho ảnh kết quả với bounding boxes
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Scoring weights for skin issues
ISSUE_WEIGHTS = {
    "acne": 0.5,
    "pores": 0.3,
    "pigment": 0.2
}

def calculate_skin_score(predictions):
    """Calculate skin score based on model predictions."""
    total_penalty = 0
    for pred in predictions:
        print(f"Prediction boxes: {len(pred.boxes)}")
        for box in pred.boxes:
            label = pred.names[int(box.cls)]
            confidence = float(box.conf)
            weight = ISSUE_WEIGHTS.get(label, 0.3)
            total_penalty += confidence * weight
            print(f"Detected {label} with confidence {confidence:.2f}, penalty {confidence * weight:.2f}")
    
    score = max(0, 10 - total_penalty * 5)
    print(f"Calculated score: {score}")
    return round(score, 1)

def sanitize_filename(filename):
    """Sanitize filename by replacing spaces and special characters."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("Received files:", request.files)
    if "left" not in request.files or "right" not in request.files or "front" not in request.files:
        return jsonify({"error": "Please upload all three images (left, right, front)"}), 400

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
            if image.filename == "":
                return jsonify({"error": f"No {side} image uploaded"}), 400

            # Sanitize filename and add unique identifier
            ext = os.path.splitext(image.filename)[1]
            sanitized_filename = sanitize_filename(os.path.splitext(image.filename)[0])
            filename = f"{side}_{uuid.uuid4().hex}_{sanitized_filename}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(filepath)
            image_paths[side] = filepath
            filenames[side] = filename

            # Run YOLO inference and save result with bounding boxes
            result = model.predict(
                filepath,
                save=True,
                project=app.config["RESULT_FOLDER"],
                name="predict",
                exist_ok=True,
                verbose=True
            )
            results[side] = result
            print(f"Results for {side}: {len(result[0].boxes)} detections")

            # Get result filename (YOLO saves with same name in result folder)
            result_filenames[side] = filename

        # Calculate skin score
        combined_predictions = [results["left"][0], results["right"][0], results["front"][0]]
        skin_score = calculate_skin_score(combined_predictions)

        # Prepare response
        response = {
            "score": skin_score,
            "analysis": {
                "left": [{"label": results["left"][0].names[int(box.cls)], "confidence": float(box.conf)} for box in results["left"][0].boxes] or [{"label": "No detections", "confidence": 0.0}],
                "right": [{"label": results["right"][0].names[int(box.cls)], "confidence": float(box.conf)} for box in results["right"][0].boxes] or [{"label": "No detections", "confidence": 0.0}],
                "front": [{"label": results["front"][0].names[int(box.cls)], "confidence": float(box.conf)} for box in results["front"][0].boxes] or [{"label": "No detections", "confidence": 0.0}]
            },
            "image_urls": {
                "left": f"/static/results/predict/{result_filenames['left']}",
                "right": f"/static/results/predict/{result_filenames['right']}",
                "front": f"/static/results/predict/{result_filenames['front']}"
            }
        }

        return jsonify(response)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)