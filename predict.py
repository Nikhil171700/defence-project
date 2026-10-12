from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

import torch.nn.functional as F

class bird_drone(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = bird_drone()
model.load_state_dict(torch.load("bird_drone_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


image_path = "39.jpg"   

image = Image.open(image_path).convert("RGB")
image = transform(image)

image = image.unsqueeze(0)
'''

print(train.class_to_idx)

probabilities = torch.softmax(output, dim=1)

print(probabilities)

predicted = torch.argmax(probabilities, dim=1)

print(predicted.item())'''
with torch.no_grad():

    output = model(image)
  
    probabilities = F.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)

    confidence = confidence.item()
    predicted = predicted.item()


classes = ["bird", "drone"]

if confidence > 0.50:
    print("Prediction:", classes[predicted])
    print("Confidence:", confidence * 100, "%")
else:
    print("Unknown object detected")
