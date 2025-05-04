import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Model sınıfı
class WasteClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(WasteClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),  # 🔧 Güncellendi!
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Sınıflar
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Görsel dönüşüm
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Modeli yükle
model = WasteClassifierCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load("waste_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Görsel yolu (değiştirerek farklı test yapabilirsin)
image_path = "dataset/trashnet/glass/glass1.jpg"

# Görseli yükle ve hazırla
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Tahmin
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"🧠 Tahmin edilen atık türü: {CLASSES[predicted.item()]}")
