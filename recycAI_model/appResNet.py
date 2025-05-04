import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# Başlık
st.title("♻️ RecycAI - ResNet18 ile Akıllı Atık Tanıma")

# Atık sınıfları
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Atık türü - kutu eşlemesi
WASTE_BIN_MAP = {
    "plastic": "♻️ Sarı kutu",
    "glass": "🟢 Yeşil kutu",
    "paper": "🔵 Mavi kutu",
    "metal": "⚪ Gri kutu",
    "cardboard": "🔵 Mavi kutu",
    "trash": "🚮 Genel çöp"
}

# Görüntü dönüşümleri (ResNet + ImageNet uyumlu)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Model yükleyici
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("resnet18_waste.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Görsel yükleme
uploaded_file = st.file_uploader("Bir atık fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

# ✅ Kutu stili ve animasyon (renkli kutu gösterimi)
def display_bin_info(label):
    if label == "plastic":
        st.markdown("### ♻️ Sarı Kutu", unsafe_allow_html=True)
        st.markdown('<div style="background-color:#f9e79f;padding:20px;border-radius:10px;">Plastik atıklar buraya atılmalı.</div>', unsafe_allow_html=True)
    elif label == "glass":
        st.markdown("### 🟢 Yeşil Kutu", unsafe_allow_html=True)
        st.markdown('<div style="background-color:#d4efdf;padding:20px;border-radius:10px;">Cam atıklar bu kutuya atılır.</div>', unsafe_allow_html=True)
    elif label == "paper" or label == "cardboard":
        st.markdown("### 🔵 Mavi Kutu", unsafe_allow_html=True)
        st.markdown('<div style="background-color:#d6eaf8;padding:20px;border-radius:10px;">Kağıt ve kartonlar bu kutuya atılmalı.</div>', unsafe_allow_html=True)
    elif label == "metal":
        st.markdown("### ⚪ Gri Kutu", unsafe_allow_html=True)
        st.markdown('<div style="background-color:#ebedef;padding:20px;border-radius:10px;">Metal atıklar bu kutuya atılır.</div>', unsafe_allow_html=True)
    elif label == "trash":
        st.markdown("### 🚮 Genel Çöp", unsafe_allow_html=True)
        st.markdown('<div style="background-color:#f5b7b1;padding:20px;border-radius:10px;">Bu atık geri dönüştürülemez. Genel çöp kutusuna atılmalı.</div>', unsafe_allow_html=True)
    else:
        st.markdown("⚠️ Kutusu bulunamadı.")

# Görsel varsa tahmin yap
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen fotoğraf", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        predicted_label = CLASSES[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    # Tahmin sonucu
    st.success(f"🧠 Tahmin: **{predicted_label}** (%{confidence * 100:.2f} güven)")

    # Kutu önerisi
    waste_bin = WASTE_BIN_MAP.get(predicted_label, "⚠️ Belirsiz")
    st.info(f"📦 Bu atık **{waste_bin}** kutusuna atılmalıdır.")

    # ✅ Animasyonlu kutu göster
    display_bin_info(predicted_label)
