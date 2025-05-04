import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Başlık
st.title("♻️ RecycAI - Akıllı Atık Tanıma")

# Sınıf isimleri
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Görüntü dönüştürme (224x224 için)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Model sınıfı (224x224 uyumlu mimari)
class WasteClassifierCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(WasteClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),  # 🔧 Burayı değiştirdik!
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Modeli yükle
@st.cache_resource
def load_model():
    model = WasteClassifierCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("waste_classifier.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Görsel yükleme
uploaded_file = st.file_uploader("📷 Bir atık fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Yüklenen fotoğraf", use_column_width=True)

    # Tahmin et
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = CLASSES[predicted.item()]
        st.success(f"🧠 Tahmin edilen atık türü: **{predicted_class}**")
