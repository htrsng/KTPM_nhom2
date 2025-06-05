#  Sức Khỏe & Làm Đẹp - Web App

Một trang web hiện đại giúp người dùng chăm sóc bản thân toàn diện từ **làn da**, **vóc dáng** đến **tinh thần**, với giao diện đẹp mắt, cá nhân hóa trải nghiệm và tích hợp các công cụ hữu ích.

## 🚀 Tính năng chính

### 1.  Mỹ Phẩm
- Người dùng upload 3 ảnh khuôn mặt (trái, phải, chính diện) qua form (không bắt buộc chụp trực tiếp camera).
- Sử dụng mô hình AI YOLOv8 tự train (file best.pt) để nhận diện các vấn đề trên da như mụn, lỗ chân lông, sắc tố.
- Ảnh đầu vào được xử lý trước khi phân tích: resize về 512x512, kiểm tra độ nét, tăng cường dữ liệu nhẹ (brightness/contrast).
- Chấm điểm các tổn thương trên da, đưa ra gợi ý phương pháp và sản phẩm phù hợp.
- Cho người dùng làm khảo sát để phân loại loại da (da nhạy cảm, da thường, da dầu, da hỗn hợp). Từ đó đưa ra các sản phẩm hỗ trợ tốt nhất .

### 2.  Vóc Dáng
- Xác định mong muốn người dùng: tăng cân, giảm cân hoặc duy trì cân nặng. Thu thập thông tin cá nhân (số đo, chiều cao, cân nặng) để tính toán nhu cầu calo (TDEE, BMR).Rồi đưa ra các gợi ý calo phù hợp trong ngày .
- Yêu cầu người dùng cung cấp số đo của bản thân . Gợi ý cách phối đồ cho từng dáng người .
- Đưa ra khảo sát để cung cấp màu sắc cá nhân phù hợp với người dùng . Giusp người dùng tự tin hơn trong cách tìm ra trang phục phù hợp

### 3. 💬 Trò Chuyện & Tâm Sự
- Khu vực chat chia sẻ cảm xúc.
- Tùy vào cảm xúc hiện tại người dùng có thể chọn trạng thái phù hợp để có thể nói chuyện choa đổi cùng phần trò chuyện.
- Giao diện có hiệu ứng đổi màu theo trạng thái cảm xúc (vui, buồn, tức giận, muốn tâm sự).
- Chatbot trả lời tự động hỗ trợ tâm lý, tạo cảm giác thân thiện và gần gũi.

##  Công nghệ sử dụng

- Backend: Python với Flask framework xử lý logic, phân tích ảnh AI, API và render template.
- Frontend: HTML5, JavaScript, Tailwind CSS thiết kế giao diện hiện đại, responsive.
- Hiệu ứng: Animation và gradient động tạo trải nghiệm bắt mắt, mượt mà.
- AI: Mô hình YOLOv8 tự train dùng nhận diện các vấn đề da.
- Ảnh: Xử lý ảnh đầu vào (resize, kiểm tra nét, tăng sáng/tương phản) trước khi phân tích

