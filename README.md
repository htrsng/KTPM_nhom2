# Sức Khỏe & Làm Đẹp - Web App 🌸

![Project Status](https://img.shields.io/badge/status-completed-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8+-yellow)

Một ứng dụng web hiện đại giúp bạn chăm sóc **làn da**, **vóc dáng**, và **tinh thần** với giao diện thân thiện, trải nghiệm cá nhân hóa, và công nghệ AI tiên tiến. Đây là giải pháp **all-in-one** để nâng tầm vẻ đẹp và sức khỏe của bạn!
### 🎥 Demo sản phẩm
- "📺 **Video Demo :** https://www.youtube.com/watch?v=I6bRZ0dXn58 "
- 🔗 **GitHub Repository**: https://github.com/htrsng/KTPM_nhom2

---

## ✨ Giới thiệu

**Sức Khỏe & Làm Đẹp** là ứng dụng web dành cho những ai muốn:
- **Cải thiện làn da** với phân tích AI và gợi ý sản phẩm phù hợp.
- **Hoàn thiện vóc dáng** qua tính toán calo và phong cách thời trang.
- **Chia sẻ cảm xúc** với chatbot thân thiện, hỗ trợ tâm lý.

**Điểm nổi bật**: Tích hợp toàn diện chăm sóc da, ngoại hình, và tinh thần trong một nền tảng duy nhất, mang đến trải nghiệm mượt mà và trực quan.

> [📸 Placeholder: Thêm ảnh giao diện chính tại đây]

---

## 🚀 Tính năng chính

| Tính năng        | Mô tả                                                                 |
|------------------|----------------------------------------------------------------------|
| **Mỹ phẩm**      | Upload 3 ảnh khuôn mặt để phân tích da bằng AI YOLOv8. Nhận điểm số, gợi ý cải thiện, và sản phẩm phù hợp. |
| **Vóc dáng**     | Tính toán TDEE/BMR, gợi ý phối đồ và màu sắc cá nhân dựa trên số đo. |
| **Trò chuyện**   | Chatbot thân thiện, đổi màu giao diện theo cảm xúc (vui, buồn, tức giận). |

---

## 🎯 Mục tiêu dài hạn

- 🌐 Mở rộng sang ứng dụng di động.
- 🤖 Nâng cấp AI phân tích da và chatbot để hiểu tâm trạng người dùng tốt hơn.
- 🗄️ Tích hợp cơ sở dữ liệu (MongoDB/SQLite) để quản lý dữ liệu động.
---

## 📊 Trạng thái dự án

Dự án đã **hoàn thiện** và sẵn sàng sử dụng. Các tính năng mới sẽ được phát triển trong tương lai để nâng cao trải nghiệm.

---

## 🛠️ Yêu cầu hệ thống

### Phần mềm
- **Python**: 3.8 hoặc cao hơn
- **Trình duyệt**: Chrome, Firefox, Safari, Edge (phiên bản mới nhất)
- **Dependencies**:
  ```bash
  pip install flask ultralytics opencv-python numpy albumentations python-dotenv gunicorn
---

### Phần cứng 
- **RAM** : Tối thiểu 8GB (khuyến nghị 16GB)
- **CPU**: Đa nhân (GPU NVIDIA với CUDA để tăng tốc YOLOv8)
- **Dung lượng**: ~6MB cho file best.pt và không gian cho thư mục static
---

### Mô hình AI
- File best.pt (YOLOv8, ~6MB) nằm tại SKIN/models/best.pt, đã train sẵn để phân tích da.
- **Lưu ý:** Đường dẫn đến file best.pt được hard-coded trong app.py. Đảm bảo file tồn tại tại SKIN/models/best.pt trước khi chạy ứng dụng.
---

### Cơ sở dữ liệu : 
- Hiện sử dụng dữ liệu tĩnh (PRODUCTS, QUESTIONS). Kế hoạch tích hợp MongoDB/SQLite trong tương lai.
---

### 📥 Cài đặt 

**1. Clone repository**
```bash
git clone https://github.com/htrsng/KTPM_nhom2.git
```
 **2. Cài đặt dependencies**
```bash 
pip install flask ultralytics opencv-python numpy albumentations python-dotenv gunicorn
```
- nếu gặp lỗi cài thêm 
```bash 
pip install torch 
``` 
**3. Chạy ứng dụng**
- Di chuyển vào thư mục SKIN : 
```bash
cd SKIN 
```
- Chạy server Flask : 
```bash 
python app.py 
``` 
- Trong môi trường production :
```bash
gunicorn --bind 127.0.0.1:5000 -w 4 SKIN.app:app
```
- Chạy tổng quan : 
   Mở file index.html trong thư mục gốc (KTPM_nhom2/index.html) bằng Live Server (VD: extension Live Server trong VS Code).
--- 

### 🚀 Sử dụng : 
## 1 . Truy cập ứng dụng
  - Mở [repo-root]/index.html với Live Server để truy cập trang tổng quan.
  - Mở trình duyệt tại http://127.0.0.1:5000 (module Mỹ phẩm).
  - Truy cập các module khác:
      - Vóc dáng: [repo-root]/VocDang/index.html
      - Trò chuyện: [repo-root]/TroChuyen/index.html
## 2. Tính năng Mỹ phẩm:
  - Upload 3 ảnh khuôn mặt (JPG/PNG) .
  - Trả lời khảo sát loại da để nhận gợi ý sản phẩm.
  - Xem kết quả: điểm số da, vấn đề (mụn, lỗ chân lông, thâm/nám), gợi ý cải thiện.
## 3. Tính năng Vóc dáng:
  - Truy cập [repo-root]/VocDang/index.html.
  - Tính TDEE/BMR tại VocDang/modules/weight/tdee.html.
  - Nhập số đo cơ thể tại VocDang/modules/bodyshape/shape.html để nhận gợi ý phối đồ.
  - Khảo sát màu sắc cá nhân tại VocDang/modules/skintone/tone.html.

## 4. Tính năng Trò chuyện:
   - Truy cập [repo-root]/TroChuyen/index.html 
   - Chọn trạng thái cảm xúc để tương tác với chatbot.

### 📂 Cấu trúc thư mục 
```text
KTPM_nhom2/
├── index.html               # Trang tổng quan
├── SKIN/                    # Module Mỹ phẩm
│   ├── app.py               # Backend Flask
│   ├── models/              # Mô hình YOLOv8
│   │   └── best.pt          # File mô hình (~6MB)
│   ├── static/              # File tĩnh
│   │   ├── cosmetic/        # CSS, JS cho mỹ phẩm
│   │   ├── img_danhaycam/   # Hình ảnh sản phẩm da nhạy cảm
│   │   ├── img_dadau/       # Hình ảnh sản phẩm da dầu
│   │   ├── img_danthuong/   # Hình ảnh sản phẩm da thường
│   │   ├── img_dahonhop/    # Hình ảnh sản phẩm da hỗn hợp
│   │   ├── uploads/         # Ảnh người dùng upload
│   │   └── results/predict/ # Kết quả YOLOv8
│   └── templates/           # HTML giao diện
│       ├── cosmetic/        # HTML cho từng loại da
│       ├── index.html       # Trang chính module SKIN
│       └── hotro.html       # Trang chatbot
├── VocDang/                 # Module Vóc dáng
│   ├── data/                # Dữ liệu tĩnh
│   │   ├── palettes.json    # Màu sắc cá nhân
│   │   ├── questions.json   # Câu hỏi khảo sát
│   │   └── styles.json      # Phong cách phối đồ
│   ├── modules/             # Tính năng con
│   │   ├── bodyshape/       # Phân tích số đo
│   │   ├── skintone/        # Phân tích màu sắc
│   │   └── weight/          # Tính TDEE/BMR
│   ├── style.css            # CSS giao diện
│   ├── index.html           # Trang chính module VocDang
│   └── script.js            # JS module VocDang
├── TroChuyen/               # Module Trò chuyện
│   └── index.html           # Trang chính chatbot
└── README.md                # Tài liệu dự án
``` 
--- 
### 💻Công nghệ sử dụng 
- **Backend:** Python, Flask  
- **Frontend:** HTML5, JavaScript, Tailwind CSS  
- **AI:** YOLOv8 (phân tích da)  
- **Xử lý ảnh:** OpenCV, Albumentations  
- **Dữ liệu tĩnh:** JSON (`palettes.json`, `questions.json`, `styles.json`)

--- 

### 🤝 Đóng góp 
Chúng tôi hoan nghênh mọi đóng góp ! Để tham gia : 
- **1.Fork repository**: https://github.com/htrsng/KTPM_nhom2
- **2. Tạo branch :** 
 ```bash 
 git checkout -b feature/ten-tinh-nang
 ```
- **3. Commit thay đổi :** 
 ```bash 
 git commit -m "Mô tả thay đổi"
 ```
- **4. Push branch :** 
 ```bash 
 git push origin feature/ten-tinh-nang
 ```
- **5. Tạo Pull Request trên GitHub.**
   - **Quy tắc code**: Tuân thủ PEP8 (Python) và Prettier (JS/CSS).
   - ***Báo lỗi**: Mở issue trên GitHub.
--- 
### 📬 Nhóm phát triển

| Họ tên                | Mã số sinh viên | Vai trò     | Email                  | GitHub            |
|-----------------------|------------------|-------------|-------------------------|--------------------|
| Nguyễn Thị Huyền Trang | 23010181         | Phát triển | 23010181@st.phenikaa-uni.edu.vn      | [htrsng](https://github.com/htrsng) |
| Trần Xuân Thành        | 23010160         | Phát triển | 23010160@st.phenikaa-uni.edu.vn      | [Kaitszo](https://github.com/Kaitszo) |







