import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import os
import torch.nn as nn

# Tạo lại kiến trúc model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 lớp: acne, lochanlong, sactoda

# Load trọng số đã huấn luyện
if not os.path.exists("skin_model.pth"):
    st.error("Không tìm thấy tệp mô hình 'skin_model.pth'. Vui lòng kiểm tra lại.")
else:
    model.load_state_dict(torch.load("skin_model.pth", map_location=torch.device('cpu')))
    model.eval()

# Các class
classes = ['acne', 'lochanlong', 'sactoda']

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("🧴 Chấm điểm tình trạng da")
st.write("📷 Bạn có thể **chụp ảnh** hoặc **tải ảnh lên** để phân tích:")

# Tùy chọn: Chụp ảnh hoặc tải ảnh
option = st.radio("Chọn phương thức nhập ảnh:", ["Chụp ảnh", "Tải ảnh lên"])

if option == "Chụp ảnh":
    # Chụp ảnh bằng camera
    captured_image = st.camera_input("Chụp ảnh trực tiếp")
    if captured_image is not None:
        image = Image.open(captured_image)
    else:
        image = None
elif option == "Tải ảnh lên":
    # Tải ảnh từ máy
    uploaded_file = st.file_uploader("Tải ảnh từ máy", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        image = None
else:
    image = None

# Xử lý ảnh
if image is not None:
    st.image(image, caption='Ảnh bạn đã chọn', use_column_width=True)

    # Tiền xử lý
    input_tensor = transform(image).unsqueeze(0)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        scores = torch.softmax(output[0], dim=0)

    # Hiển thị kết quả
    st.subheader("📊 Kết quả phân tích da:")
    for i in range(len(classes)):
        score = scores[i].item()
        st.write(f"🔹 {classes[i]}: {score*10:.2f}/10")
else:
    st.warning("Vui lòng chọn phương thức nhập ảnh để phân tích.")
