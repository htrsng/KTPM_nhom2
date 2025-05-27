from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import numpy as np
import uuid
import re
import cv2
import albumentations as A

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
RESULT_FOLDER = "static/results"
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

def check_image_quality(img):
    """Check image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > 100  # Threshold for sharpness

def preprocess_image(image_stream):
    """Preprocess image: resize and lightly normalize."""
    try:
        # Read image from stream
        img_array = np.frombuffer(image_stream.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format or corrupted file")
        
        # Check image quality
        if not check_image_quality(img):
            print("Warning: Input image is blurry")
        
        # Resize to 512x512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply light augmentation
        aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        ])
        img = aug(image=img)["image"]
        
        # Save processed image
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

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

            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Sanitize filename and add unique identifier
            ext = '.jpg'
            sanitized_filename = sanitize_filename(os.path.splitext(image.filename)[0])
            filename = f"{side}_{uuid.uuid4().hex}_{sanitized_filename}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            # Save processed image
            with open(filepath, 'wb') as f:
                f.write(processed_image)
            image_paths[side] = filepath
            filenames[side] = filename

            # Run YOLO inference and save result with bounding boxes
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
            results[side] = result
            print(f"Results for {side}: {len(result[0].boxes)} detections")

            # Get result filename
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
    except ValueError as e:
        print(f"Value error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/cleanup", methods=["POST"])
def cleanup():
    """Clean up old files in uploads and results folders."""
    try:
        for folder in [app.config["UPLOAD_FOLDER"], os.path.join(app.config["RESULT_FOLDER"], "predict")]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        return jsonify({"message": "Cleaned up old files"})
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False)
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phân tích da - SKIN</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
        }
        .hidden {
            display: none;
        }
        .galaxy-shadow {
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Header -->
    <header class="bg-gradient-to-r from-cyan-100 to-white text-gray-800 py-6 galaxy-shadow">
        <div class="container mx-auto text-center">
            <h1 class="text-3xl font-bold">Phân tích tình trạng da</h1>
            <p class="mt-2">Tải lên ảnh khuôn mặt để nhận phân tích và gợi ý chăm sóc da</p>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto py-8 px-4">
        <!-- Upload Form -->
        <section class="bg-white rounded-lg p-6 mb-8 galaxy-shadow border border-gray-100">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Ảnh trái</label>
                        <input type="file" name="left" accept="image/*" class="mt-1 block w-full border-gray-200 rounded-md shadow-sm focus:ring-blue-200 focus:border-blue-200">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Ảnh phải</label>
                        <input type="file" name="right" accept="image/*" class="mt-1 block w-full border-gray-200 rounded-md shadow-sm focus:ring-blue-200 focus:border-blue-200">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Ảnh chính diện</label>
                        <input type="file" name="front" accept="image/*" class="mt-1 block w-full border-gray-200 rounded-md shadow-sm focus:ring-blue-200 focus:border-blue-200">
                    </div>
                </div>
                <div class="mt-4 flex justify-center space-x-4">
                    <button type="submit" class="bg-blue-300 text-white px-6 py-2 rounded-md hover:bg-blue-400">Phân tích</button>
                    <button type="button" id="cleanupButton" class="bg-blue-200 text-white px-6 py-2 rounded-md hover:bg-blue-300">Dọn dẹp file</button>
                </div>
            </form>
            <div id="loading" class="hidden mt-4 text-center">
                <svg class="animate-spin h-8 w-8 text-blue-300 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p class="mt-2 text-gray-600">Đang phân tích...</p>
            </div>
        </section>

        <!-- Results -->
        <section id="results" class="hidden">
            <!-- Skin Score -->
            <div class="bg-white rounded-lg p-6 mb-8 text-center galaxy-shadow border border-gray-100">
                <h2 class="text-2xl font-semibold text-gray-800">Điểm da</h2>
                <p id="skinScore" class="text-4xl font-bold text-blue-300 mt-2">0/10</p>
            </div>

            <!-- Analysis -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <!-- Left -->
                <div class="bg-white rounded-lg p-6 galaxy-shadow border border-gray-100">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Ảnh trái</h3>
                    <img id="leftImage" src="" alt="Ảnh trái" class="w-full h-64 object-cover rounded-md mb-4 hover:scale-105 transition-transform">
                    <ul id="leftAnalysis" class="list-disc pl-5 text-gray-600"></ul>
                </div>
                <!-- Right -->
                <div class="bg-white rounded-lg p-6 galaxy-shadow border border-gray-100">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Ảnh phải</h3>
                    <img id="rightImage" src="" alt="Ảnh phải" class="w-full h-64 object-cover rounded-md mb-4 hover:scale-105 transition-transform">
                    <ul id="rightAnalysis" class="list-disc pl-5 text-gray-600"></ul>
                </div>
                <!-- Front -->
                <div class="bg-white rounded-lg p-6 galaxy-shadow border border-gray-100">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Ảnh chính diện</h3>
                    <img id="frontImage" src="" alt="Ảnh chính diện" class="w-full h-64 object-cover rounded-md mb-4 hover:scale-105 transition-transform">
                    <ul id="frontAnalysis" class="list-disc pl-5 text-gray-600"></ul>
                </div>
            </div>

            <!-- Improvements -->
            <div class="bg-white rounded-lg p-6 mb-8 galaxy-shadow border border-gray-100">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Phương pháp cải thiện</h2>
                <div id="improvements" class="text-gray-600"></div>
            </div>

            <!-- Products -->
            <div class="bg-white rounded-lg p-6 galaxy-shadow border border-gray-100">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Sản phẩm gợi ý</h2>
                <div id="products" class="text-gray-600"></div>
            </div>
        </section>
    </main>

    <script>
        // Handle form submission
        document.getElementById("uploadForm").addEventListener("submit", async function (event) {
            event.preventDefault();
            document.getElementById("loading").classList.remove("hidden");
            document.getElementById("results").classList.add("hidden");

            const formData = new FormData(this);
            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();
                console.log("Response JSON:", result);

                if (response.ok) {
                    document.getElementById("loading").classList.add("hidden");
                    document.getElementById("results").classList.remove("hidden");

                    // Display score
                    document.getElementById("skinScore").textContent = `${result.score}/10`;

                    // Display analysis
                    ["left", "right", "front"].forEach(side => {
                        const analysisList = document.getElementById(`${side}Analysis`);
                        const imgElement = document.getElementById(`${side}Image`);
                        analysisList.innerHTML = "";
                        imgElement.src = result.image_urls[side] + "?t=" + new Date().getTime();

                        if (result.analysis[side][0].label === "No detections") {
                            const li = document.createElement("li");
                            li.textContent = "Không phát hiện vấn đề da nào.";
                            analysisList.appendChild(li);
                        } else {
                            result.analysis[side].forEach(item => {
                                const li = document.createElement("li");
                                li.textContent = `${item.label}: ${(item.confidence * 100).toFixed(1)}%`;
                                analysisList.appendChild(li);
                            });
                        }
                    });

                    // Display improvements
                    const improvementsDiv = document.getElementById("improvements");
                    improvementsDiv.innerHTML = "";
                    ["acne", "pores", "pigment"].forEach(issue => {
                        if (result.improvements[issue].length > 0) {
                            const h4 = document.createElement("h4");
                            h4.className = "text-lg font-medium text-gray-700 mt-4";
                            h4.textContent = issue.charAt(0).toUpperCase() + issue.slice(1);
                            improvementsDiv.appendChild(h4);
                            const ul = document.createElement("ul");
                            ul.className = "list-disc pl-5";
                            result.improvements[issue].forEach(item => {
                                const li = document.createElement("li");
                                li.textContent = item;
                                ul.appendChild(li);
                            });
                            improvementsDiv.appendChild(ul);
                        }
                    });

                    // Display products
                    const productsDiv = document.getElementById("products");
                    productsDiv.innerHTML = "";
                    ["acne", "pores", "pigment"].forEach(issue => {
                        if (result.products[issue].length > 0) {
                            const h4 = document.createElement("h4");
                            h4.className = "text-lg font-medium text-gray-700 mt-4";
                            h4.textContent = issue.charAt(0).toUpperCase() + issue.slice(1);
                            productsDiv.appendChild(h4);
                            const ul = document.createElement("ul");
                            ul.className = "list-disc pl-5";
                            result.products[issue].forEach(item => {
                                const li = document.createElement("li");
                                li.textContent = item;
                                ul.appendChild(li);
                            });
                            productsDiv.appendChild(ul);
                        }
                    });
                } else {
                    document.getElementById("loading").classList.add("hidden");
                    alert("Lỗi: " + result.error);
                }
            } catch (error) {
                document.getElementById("loading").classList.add("hidden");
                console.error("Error:", error);
                alert("Đã xảy ra lỗi khi xử lý yêu cầu.");
            }
        });

        // Handle cleanup
        document.getElementById("cleanupButton").addEventListener("click", async function () {
            try {
                const response = await fetch("/cleanup", {
                    method: "POST"
                });
                const result = await response.json();
                if (response.ok) {
                    alert(result.message);
                } else {
                    alert("Lỗi: " + result.error);
                }
            } catch (error) {
                console.error("Error:", error);
                alert("Lỗi khi dọn dẹp file.");
            }
        });
    </script>
</body>
</html>