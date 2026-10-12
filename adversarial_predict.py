from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. CNN Model
# ============================================================

class bird_drone(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# 3. Load Adversarially Trained Model
# ============================================================

model = bird_drone().to(device)

model.load_state_dict(
    torch.load(
        "bird_drone_robust_model.pth",
        map_location=device
    )
)

model.eval()


# ============================================================
# 4. Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================================
# 5. Select Image
# ============================================================

image_path = "39.jpg"


image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


# ============================================================
# 6. Prediction
# ============================================================

with torch.no_grad():

    output = model(image)

    probabilities = F.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


confidence = confidence.item()
predicted = predicted.item()


# ============================================================
# 7. Classes
# ============================================================

classes = [
    "bird",
    "drone"
]


# ============================================================
# 8. Display Result
# ============================================================

print()
print("==============================")
print("BIRD vs DRONE PREDICTION")
print("==============================")

print(
    "Prediction:",
    classes[predicted]
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)

print("==============================")