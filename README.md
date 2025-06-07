
Sức Khỏe & Làm Đẹp - Web App
Một trang web hiện đại giúp người dùng chăm sóc bản thân toàn diện từ làn da, vóc dáng đến tinh thần, với giao diện đẹp mắt, trải nghiệm cá nhân hóa, và tích hợp các công cụ thông minh sử dụng AI. Dự án cung cấp một giải pháp "all-in-one" cho những ai muốn cải thiện ngoại hình và sức khỏe tinh thần.
Mô tả
Sức Khỏe & Làm Đẹp là một ứng dụng web dành cho những người quan tâm đến mỹ phẩm, cải thiện vóc dáng, tìm kiếm phong cách và màu sắc cá nhân phù hợp, cũng như chia sẻ cảm xúc qua chatbot để có những ngày vui vẻ hơn. Điểm nổi bật của dự án là sự kết hợp toàn diện các tính năng chăm sóc da, ngoại hình, và hỗ trợ tâm lý, mang đến trải nghiệm liền mạch và thân thiện.
Tính năng chính

Mỹ phẩm:

Người dùng upload 3 ảnh khuôn mặt (trái, phải, chính diện) để phân tích các vấn đề da như mụn, lỗ chân lông, thâm/nám bằng mô hình AI YOLOv8.
Ảnh được xử lý (resize 512x512, kiểm tra độ nét, tăng sáng/tương phản) trước khi phân tích.
Hệ thống chấm điểm da, đưa ra gợi ý cải thiện và sản phẩm phù hợp.
Khảo sát xác định loại da (da dầu, da khô, da hỗn hợp, da thường, da nhạy cảm) để gợi ý sản phẩm tối ưu.


Vóc dáng:

Tính toán nhu cầu calo (TDEE/BMR) dựa trên số đo, chiều cao, cân nặng để hỗ trợ mục tiêu tăng cân, giảm cân, hoặc duy trì cân nặng.
Gợi ý phong cách phối đồ dựa trên số đo cơ thể.
Khảo sát màu sắc cá nhân để giúp người dùng tự tin hơn với phong cách phù hợp.


Trò chuyện & Tâm sự:

Khu vực chat chia sẻ cảm xúc với giao diện đổi màu theo trạng thái (vui, buồn, tức giận, muốn tâm sự).
Chatbot tự động trả lời, tạo cảm giác thân thiện và hỗ trợ tâm lý.



Mục tiêu dài hạn

Mở rộng sang ứng dụng di động.
Cải thiện mô hình AI phân tích da để đạt độ chính xác cao hơn.
Nâng cấp hệ thống AI chatbot để nắm bắt tâm trạng và trạng thái người dùng tốt hơn.

Trạng thái dự án
Dự án đã hoàn thiện và sẵn sàng sử dụng. Các tính năng mới có thể được phát triển trong tương lai để tăng cường trải nghiệm người dùng.
Yêu cầu hệ thống

Phần mềm:

Python 3.8 hoặc cao hơn.
Trình duyệt: Google Chrome, Firefox, Safari, hoặc Microsoft Edge (phiên bản mới nhất).
Dependencies:pip install flask ultralytics opencv-python numpy albumentations python-dotenv gunicorn




Phần cứng:

RAM: Tối thiểu 8GB (khuyến nghị 16GB để xử lý ảnh và mô hình YOLOv8 hiệu quả).
CPU: Đa nhân (GPU NVIDIA với CUDA khuyến nghị nếu muốn tăng tốc phân tích YOLOv8).
Dung lượng lưu trữ: Khoảng 6MB cho file mô hình best.pt và không gian cho thư mục static/uploads, static/results.


Mô hình YOLOv8:

File mô hình best.pt được bao gồm trong repository tại SKIN/models/best.pt. Kích thước file: khoảng 6MB. Mô hình đã được train sẵn và người dùng có thể sử dụng trực tiếp để phân tích da.


Cơ sở dữ liệu:

Hiện tại, dự án sử dụng dữ liệu tĩnh (PRODUCTS và QUESTIONS) trong mã nguồn. Trong tương lai, dự án có kế hoạch tích hợp cơ sở dữ liệu (như MongoDB hoặc SQLite) để quản lý dữ liệu động.



Cài đặt

Clone repository:
git clone https://github.com/htrsng/KTPM_nhom2.git


Cài đặt dependencies:
pip install flask ultralytics opencv-python numpy albumentations python-dotenv gunicorn


Cấu hình biến môi trường:

Sao chép file .env.example thành .env:cp .env.example .env


Cập nhật file .env với các giá trị phù hợp:MODEL_PATH=SKIN/models/best.pt
PORT=5000
FLASK_ENV=development




Chạy ứng dụng:

Di chuyển vào thư mục SKIN:cd SKIN


Chạy server Flask:python app.py


Trong môi trường production, sử dụng:gunicorn --bind 127.0.0.1:5000 -w 4 SKIN.app:app





Sử dụng

Truy cập ứng dụng:

Mở trình duyệt và truy cập http://127.0.0.1:5000 để vào module Mỹ phẩm.
Truy cập trang tổng quan tại [repo-root]/index.html hoặc các module khác:
Vóc dáng: [repo-root]/VocDang/index.html
Trò chuyện: [repo-root]/TroChuyen/index.html




Tính năng Mỹ phẩm:

Upload 3 ảnh khuôn mặt (trái, phải, chính diện, định dạng JPG/PNG) qua form tại http://127.0.0.1:5000.
Trả lời khảo sát loại da (qua endpoint /skin_type) để nhận gợi ý sản phẩm phù hợp.
Xem kết quả phân tích da, bao gồm điểm số, vấn đề da (mụn, lỗ chân lông, thâm/nám), gợi ý cải thiện, và danh sách sản phẩm.


Tính năng Vóc dáng:

Truy cập [repo-root]/VocDang/index.html.
Nhập số đo, chiều cao, cân nặng qua các form trong VocDang/modules/bodyshape/shape.html để nhận gợi ý phối đồ.
Thực hiện khảo sát màu sắc cá nhân trong VocDang/modules/skintone/tone.html.
Tính toán nhu cầu calo (TDEE/BMR) qua VocDang/modules/weight/tdee.html.


Tính năng Trò chuyện:

Truy cập [repo-root]/TroChuyen/index.html hoặc http://127.0.0.1:5000/hotro.
Chọn trạng thái cảm xúc (vui, buồn, tức giận, v.v.) để tương tác với chatbot.
Chatbot trả lời tự động, hỗ trợ chia sẻ cảm xúc và tạo trải nghiệm thân thiện.



Cấu trúc thư mục
Dự án được tổ chức thành ba module chính: SKIN (Mỹ phẩm), VocDang (Vóc dáng), và TroChuyen (Trò chuyện). Dưới đây là cấu trúc thư mục:
KTPM_nhom2/
│
├── index.html               # Trang tổng quan của ứng dụng
├── SKIN/                    # Module Mỹ phẩm
│   ├── app.py               # Backend Flask xử lý phân tích da và API
│   ├── models/              # Thư mục chứa mô hình YOLOv8
│   │   └── best.pt          # File mô hình YOLOv8 (6MB)
│   ├── static/              # File tĩnh (CSS, JS, hình ảnh)
│   │   ├── cosmetic/        # CSS và JS cho giao diện mỹ phẩm
│   │   │   ├── css/styles.css
│   │   │   └── js/main.js
│   │   ├── img_danhaycam/   # Hình ảnh sản phẩm cho da nhạy cảm
│   │   ├── img_dadau/       # Hình ảnh sản phẩm cho da dầu
│   │   ├── img_danthuong/   # Hình ảnh sản phẩm cho da thường
│   │   ├── img_dahonhop/    # Hình ảnh sản phẩm cho da hỗn hợp
│   │   ├── uploads/         # Thư mục lưu ảnh người dùng upload
│   │   └── results/         # Thư mục lưu kết quả phân tích YOLOv8
│   │       └── predict/
│   └── templates/           # File HTML giao diện
│       ├── cosmetic/        # HTML cho từng loại da
│       │   ├── danhaycam/   # HTML cho da nhạy cảm
│       │   ├── dandau/      # HTML cho da dầu
│       │   ├── danthuong/   # HTML cho da thường
│       │   └── danhonhop/   # HTML cho da hỗn hợp
│       ├── index.html       # Trang chính của module SKIN
│       └── hotro.html       # Trang giao diện chatbot
├── VocDang/                 # Module Vóc dáng
│   ├── data/                # Dữ liệu tĩnh
│   │   ├── palettes.json    # Dữ liệu màu sắc cá nhân
│   │   ├── questions.json   # Câu hỏi khảo sát
│   │   └── styles.json      # Dữ liệu phong cách phối đồ
│   ├── modules/             # Các tính năng con
│   │   ├── bodyshape/       # Tính năng phân tích số đo cơ thể
│   │   ├── skintone/        # Tính năng phân tích màu sắc cá nhân
│   │   └── weight/          # Tính năng tính toán TDEE/BMR
│   ├── style.css            # CSS cho giao diện
│   ├── index.html           # Trang chính của module VocDang
│   └── script.js            # JS cho module VocDang
├── TroChuyen/               # Module Trò chuyện
│   └── index.html           # Trang chính của module chatbot
└── README.md                # Tài liệu hướng dẫn dự án

Công nghệ sử dụng

Backend: Python, Flask (xử lý logic, phân tích ảnh AI, API, render template).
Frontend: HTML5, JavaScript, Tailwind CSS (giao diện responsive, hiệu ứng animation và gradient động).
AI: Mô hình YOLOv8 tự train để phân tích da (file best.pt).
Xử lý ảnh: OpenCV, Albumentations (resize 512x512, kiểm tra độ nét, tăng sáng/tương phản).
Dữ liệu tĩnh: JSON (palettes.json, questions.json, styles.json) cho module Vóc dáng.

Đóng góp
Chúng tôi hoan nghênh mọi đóng góp để cải thiện dự án! Để đóng góp:

Fork repository tại https://github.com/htrsng/KTPM_nhom2.
Tạo branch mới:git checkout -b feature/ten-tinh-nang


Commit thay đổi:git commit -m "Mô tả thay đổi"


Push lên branch:git push origin feature/ten-tinh-nang


Tạo Pull Request trên GitHub.


Quy tắc code: Vui lòng tuân thủ các tiêu chuẩn như PEP8 cho Python và sử dụng công cụ linting (như Prettier) cho JavaScript/CSS nếu có.
Báo lỗi: Vui lòng mở issue trên GitHub để báo lỗi hoặc đề xuất tính năng.

Giấy phép
Dự án được phát hành theo MIT License.
Liên hệ

Email: [Vui lòng liên hệ nhóm phát triển để biết thêm chi tiết]
GitHub: [https://github.com/htrsng]

