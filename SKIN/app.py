from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import numpy as np
import uuid
import re
import cv2
import albumentations as A
import logging
import json

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Load YOLOv8 model
try:
    logger.info("Loading YOLO model")
    model = YOLO(MODEL_PATH)
    logger.info("Model labels: %s", model.names)
except Exception as e:
    logger.error("Error loading model: %s", e)
    raise

# Directory to save uploaded images and results
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
PREDICT_FOLDER = os.path.join(RESULT_FOLDER, "predict")
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, PREDICT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info("Created folder: %s", folder)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["PREDICT_FOLDER"] = PREDICT_FOLDER

# Scoring weights for skin issues
ISSUE_WEIGHTS = {
    "acne": 0.5,
    "pores": 0.3,
    "pigment": 0.2
}

# Questions for skin type diagnosis
QUESTIONS = [
    {
        "id": "q1",
        "text": "Sau khi rửa mặt sạch 30 phút, bạn cảm thấy da mặt thế nào?",
        "options": [
            {"text": "Da căng và khô ráp, hơi ngứa hoặc có vết nứt nhỏ", "skin_type": "da khô"},
            {"text": "Da bóng dầu nhiều, nhất là vùng chữ T (trán, mũi, cằm)", "skin_type": "da dầu"},
            {"text": "Vùng chữ T hơi bóng, nhưng hai bên má thì khô", "skin_type": "da hỗn hợp thiên dầu"},
            {"text": "Vùng chữ T bóng dầu, hai bên má bình thường", "skin_type": "da hỗn hợp thiên khô"},
            {"text": "Da không khô cũng không nhờn, khá ổn định", "skin_type": "da thường"},
            {"text": "Da hơi nóng, rát hoặc ngứa sau khi dùng sữa rửa mặt", "skin_type": "da nhạy cảm"}
        ]
    },
    {
        "id": "q2",
        "text": "Lỗ chân lông của bạn như thế nào?",
        "options": [
            {"text": "Nhỏ, khó thấy", "skin_type": "da khô"},
            {"text": "To rõ, đặc biệt ở vùng chữ T", "skin_type": "da dầu"},
            {"text": "To ở vùng chữ T, nhỏ ở hai bên má", "skin_type": "da hỗn hợp thiên dầu"},
            {"text": "Nhỏ ở vùng má, vùng chữ T thì hơi to", "skin_type": "da hỗn hợp thiên khô"},
            {"text": "Lỗ chân lông đều và không nổi bật", "skin_type": "da thường"},
            {"text": "Da dễ nổi đỏ, kích ứng vùng có lỗ chân lông", "skin_type": "da nhạy cảm"}
        ]
    },
    {
        "id": "q3",
        "text": "Bạn thường xuyên gặp vấn đề gì trên da?",
        "options": [
            {"text": "Bong tróc, nứt nẻ", "skin_type": "da khô"},
            {"text": "Nhờn bóng, mụn đầu đen, mụn viêm", "skin_type": "da dầu"},
            {"text": "Mụn đầu đen vùng chữ T, da khô vùng má", "skin_type": "da hỗn hợp thiên dầu"},
            {"text": "Mụn đầu trắng và khô nhẹ ở má", "skin_type": "da hỗn hợp thiên khô"},
            {"text": "Ít gặp vấn đề, thỉnh thoảng nổi mụn nhẹ", "skin_type": "da thường"},
            {"text": "Ngứa, đỏ, châm chích, đặc biệt khi thay đổi mỹ phẩm", "skin_type": "da nhạy cảm"}
        ]
    },
    {
        "id": "q4",
        "text": "Bạn cảm thấy da thay đổi thế nào trong ngày?",
        "options": [
            {"text": "Càng về chiều càng khô, khó chịu", "skin_type": "da khô"},
            {"text": "Dầu nhiều rõ rệt sau vài tiếng", "skin_type": "da dầu"},
            {"text": "Ban đầu khô, nhưng vùng chữ T bắt đầu đổ dầu sau vài tiếng", "skin_type": "da hỗn hợp thiên dầu"},
            {"text": "Khô vùng má, nhưng chữ T vẫn ra ít dầu", "skin_type": "da hỗn hợp thiên khô"},
            {"text": "Gần như không thay đổi, ổn định cả ngày", "skin_type": "da thường"},
            {"text": "Thay đổi liên tục, dễ bị kích ứng khi trời nóng, lạnh hoặc khi makeup", "skin_type": "da nhạy cảm"}
        ]
    },
    {
        "id": "q5",
        "text": "Khi sử dụng mỹ phẩm lạ, bạn thường gặp tình trạng nào?",
        "options": [
            {"text": "Không có vấn đề gì", "skin_type": "da thường"},
            {"text": "Dầu nhiều hơn, dễ nổi mụn", "skin_type": "da dầu"},
            {"text": "Châm chích nhẹ vùng da khô", "skin_type": "da khô"},
            {"text": "Mụn đầu đen vùng mũi, khô má", "skin_type": "da hỗn hợp thiên dầu"},
            {"text": "Căng nhẹ nhưng không kích ứng", "skin_type": "da hỗn hợp thiên khô"},
            {"text": "Rát, đỏ, ngứa hoặc nổi mụn li ti", "skin_type": "da nhạy cảm"}
        ]
    }
]

# Products for recommendation
PRODUCTS = [
    {"name": "Cetaphil Gentle Cleanser", "type": "sữa rửa mặt", "price": 250000, "description": "Làm sạch dịu nhẹ, không gây khô da.", "skin_types": ["da thường", "da khô", "da nhạy cảm"]},
    {"name": "La Roche-Posay Effaclar Gel", "type": "sữa rửa mặt", "price": 350000, "description": "Kiểm soát dầu, giảm mụn.", "skin_types": ["da dầu", "da hỗn hợp thiên dầu"]},
    {"name": "Bioderma Sensibio H2O", "type": "tẩy trang", "price": 400000, "description": "Tẩy trang dịu nhẹ cho da nhạy cảm.", "skin_types": ["da nhạy cảm", "da khô"]},
    {"name": "Garnier Micellar Water", "type": "tẩy trang", "price": 200000, "description": "Tẩy sạch dầu thừa, phù hợp da dầu.", "skin_types": ["da dầu", "da hỗn hợp thiên dầu"]},
    {"name": "Anessa Perfect UV SPF50+", "type": "kem chống nắng", "price": 500000, "description": "Chống nắng mạnh, không nhờn.", "skin_types": ["da thường", "da dầu", "da hỗn hợp thiên dầu"]},
    {"name": "La Roche-Posay Anthelios SPF50", "type": "kem chống nắng", "price": 450000, "description": "Chống nắng cho da nhạy cảm.", "skin_types": ["da nhạy cảm", "da khô"]},
    {"name": "The Ordinary Niacinamide 10%", "type": "serum", "price": 300000, "description": "Giảm dầu, thu nhỏ lỗ chân lông.", "skin_types": ["da dầu", "da hỗn hợp thiên dầu"]},
    {"name": "CeraVe Hydrating Serum", "type": "serum", "price": 400000, "description": "Dưỡng ẩm sâu, phục hồi da.", "skin_types": ["da khô", "da nhạy cảm", "da thường"]}
]

def check_image_quality(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > 100
    except Exception as e:
        logger.warning("Error checking image quality: %s", e)
        return False

def preprocess_image(image_stream):
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
    total_penalty = 0
    issue_counts = {"acne": 0, "pores": 0, "pigment": 0}
    issue_confidences = {"acne": [], "pores": [], "pigment": []}
    for pred in predictions:
        for box in pred.boxes:
            label = pred.names[int(box.cls)]
            confidence = float(box.conf)
            weight = ISSUE_WEIGHTS.get(label, 0.3)
            total_penalty += confidence * weight
            issue_counts[label] += 1
            issue_confidences[label].append(confidence)
    score = max(0, 10 - total_penalty * 5)
    return round(score, 1), issue_counts, issue_confidences

def suggest_improvements(issue_counts, issue_confidences):
    improvements = {"mụn": [], "lỗ chân lông to": [], "thâm/nám": []}
    if issue_counts["acne"] > 0:
        avg_conf = sum(issue_confidences["acne"]) / len(issue_confidences["acne"]) if issue_confidences["acne"] else 0
        improvements["mụn"] = [
            "Giữ da mặt sạch, rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ.",
            "Tránh sờ tay lên mặt và không tự ý nặn mụn.",
            "Sử dụng kem trị mụn không kê đơn (chứa benzoyl peroxide hoặc salicylic acid)."
        ] if avg_conf <= 0.5 else [
            "Thăm khám bác sĩ da liễu để được tư vấn chuyên sâu.",
            "Sử dụng sản phẩm trị mụn chứa BHA hoặc retinoid theo hướng dẫn của chuyên gia.",
            "Hạn chế ăn đồ cay nóng, dầu mỡ và uống đủ nước."
        ]
    if issue_counts["pores"] > 0:
        improvements["lỗ chân lông to"] = [
            "Tẩy tế bào chết 1-2 lần/tuần với sản phẩm dịu nhẹ phù hợp da.",
            "Sử dụng toner/nước hoa hồng giúp se khít lỗ chân lông.",
            "Luôn tẩy trang sạch sẽ và giữ da thông thoáng.",
            "Hạn chế trang điểm dày và tránh thức khuya."
        ]
    if issue_counts["pigment"] > 0:
        improvements["thâm/nám"] = [
            "Dùng kem chống nắng SPF 30+ mỗi ngày, thoa lại sau mỗi 2-3 tiếng.",
            "Sử dụng serum vitamin C hoặc sản phẩm làm sáng da có nguồn gốc rõ ràng.",
            "Tránh nắng kỹ, đội mũ/nón và đeo khẩu trang khi ra ngoài."
        ]
    return improvements

def suggest_products(issue_counts):
    products = {"mụn": [], "lỗ chân lông to": [], "thâm/nám": []}
    if issue_counts["acne"] > 0:
        products["mụn"] = [
            "Sữa rửa mặt Cetaphil Gentle Cleanser (dịu nhẹ cho da mụn)",
            "Gel trị mụn Differin (adapalene 0.1%)",
            "Kem trị mụn La Roche-Posay Effaclar Duo+",
            "Miếng dán mụn Some By Mi Clear Spot Patch"
        ]
    if issue_counts["pores"] > 0:
        products["lỗ chân lông to"] = [
            "Toner Paula’s Choice Skin Perfecting 2% BHA (giúp làm sạch sâu lỗ chân lông)",
            "Mặt nạ đất sét Innisfree Super Volcanic Pore Clay Mask",
            "Toner The Ordinary Glycolic Acid 7% (dùng 1-2 lần/tuần)",
            "Serum The Ordinary Niacinamide 10% + Zinc 1% (hỗ trợ thu nhỏ lỗ chân lông)"
        ]
    if issue_counts["pigment"] > 0:
        products["thâm/nám"] = [
            "Serum The Ordinary Vitamin C 23% + HA Spheres 2%",
            "Kem chống nắng La Roche-Posay Anthelios SPF 50+",
            "Serum Melano CC (vitamin C Nhật Bản)",
            "Kem dưỡng sáng da Kiehl’s Clearly Corrective Dark Spot Solution"
        ]
    return products

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

@app.route("/")
def index():
    logger.info("Serving index.html")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received /predict request")
    if not all(side in request.files for side in ["left", "right", "front"]):
        return jsonify({"error": "Vui lòng tải lên cả ba ảnh"}), 400

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
            if not image or image.filename == "":
                return jsonify({"error": f"Không có ảnh {side}"}), 400
            processed_image = preprocess_image(image)
            ext = '.jpg'
            sanitized_filename = sanitize_filename(os.path.splitext(image.filename)[0])
            filename = f"{side}_{uuid.uuid4().hex}_{sanitized_filename}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(filepath, 'wb') as f:
                f.write(processed_image)
            image_paths[side] = filepath
            filenames[side] = filename

            result = model.predict(
                filepath,
                save=True,
                project=app.config["RESULT_FOLDER"],
                name="predict",
                exist_ok=True,
                conf=0.2,
                augment=True
            )
            results[side] = result[0]
            result_filepath = os.path.join(app.config["PREDICT_FOLDER"], filename)
            if not os.path.exists(result_filepath):
                return jsonify({"error": f"Kết quả YOLO cho {side} không được lưu"}), 500
            result_filenames[side] = filename

        skin_score, issue_counts, issue_confidences = calculate_skin_score([results["left"], results["right"], results["front"]])
        improvements = suggest_improvements(issue_counts, issue_confidences)
        products = suggest_products(issue_counts)

        response = {
            "score": skin_score,
            "analysis": {
                "left": [
                    {"label": box_label, "confidence": float(box.conf)}
                    for box in results["left"].boxes
                    if (box_label := results["left"].names[int(box.cls)]) not in ["pigment", "pores"]
                ] or [{"label": "Không phát hiện", "confidence": 0.0}],
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
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": f"Dự đoán thất bại: {str(e)}"}), 500

@app.route("/skin_type", methods=["GET", "POST"])
def skin_type():
    if request.method == "GET":
        return render_template("index.html", show_questions=True, questions=QUESTIONS)
    elif request.method == "POST":
        try:
            answers = request.form.to_dict()
            skin_type_counts = {
                "da dầu": 0,
                "da khô": 0,
                "da hỗn hợp thiên dầu": 0,
                "da hỗn hợp thiên khô": 0,
                "da thường": 0,
                "da nhạy cảm": 0
            }
            for qid in ["q1", "q2", "q3", "q4", "q5"]:
                if qid not in answers:
                    return jsonify({"error": f"Thiếu câu trả lời cho câu {qid}"}), 400
                for question in QUESTIONS:
                    if question["id"] == qid:
                        for option in question["options"]:
                            if option["text"] == answers[qid]:
                                skin_type_counts[option["skin_type"]] += 1
                                break
            max_count = max(skin_type_counts.values())
            if max_count == 0:
                return jsonify({"error": "Không xác định được loại da"}), 400
            skin_type = max(skin_type_counts, key=lambda k: (skin_type_counts[k], ["da nhạy cảm", "da hỗn hợp thiên dầu", "da hỗn hợp thiên khô", "da dầu", "da khô", "da thường"].index(k)))
            return jsonify({"skin_type": skin_type})
        except Exception as e:
            logger.error("Skin type error: %s", e)
            return jsonify({"error": f"Lỗi xác định loại da: {str(e)}"}), 500

@app.route("/recommend_products/<skin_type>")
def recommend_products(skin_type):
    filtered_products = [p for p in PRODUCTS if skin_type in p["skin_types"]]
    sort_order = request.args.get("sort", "low_to_high")
    filtered_products.sort(key=lambda x: x["price"], reverse=(sort_order == "high_to_low"))
    return render_template("index.html", show_products=True, skin_type=skin_type, products=filtered_products)

@app.route("/cleanup", methods=["POST"])
def cleanup():
    try:
        for folder in [app.config["UPLOAD_FOLDER"], app.config["PREDICT_FOLDER"]]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        return jsonify({"message": "Đã dọn dẹp các file cũ"})
    except Exception as e:
        logger.error("Cleanup error: %s", e)
        return jsonify({"error": f"Dọn dẹp thất bại: {str(e)}"}), 500

@app.route("/hotro")
def hotro():
    return render_template("hotro.html")

@app.route("/cosmetic/<skin_type>/<page>")
def cosmetic_page(skin_type, page):
    return render_template(f"cosmetic/{skin_type}/{page}")

if __name__ == "__main__":
    logger.info("Starting Flask server")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)