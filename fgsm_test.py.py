from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Transform
# --------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# --------------------
# Test Dataset
# --------------------
test_dataset = ImageFolder(
    "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# --------------------
# CNN Model
# --------------------
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

# --------------------
# Load Model
# --------------------
model = bird_drone()
model.load_state_dict(
    torch.load("bird_drone_model.pth")
)

model.eval()

# --------------------
# FGSM Settings
# --------------------
epsilon = 0.03

correct = 0
total = 0

# --------------------
# FGSM Test Loop
# --------------------
for images, labels in test_loader:

    images.requires_grad = True

    outputs = model(images)

    _, initial_pred = torch.max(outputs, 1)

    # Skip if already wrong
    if initial_pred.item() != labels.item():
        continue

    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()

    loss.backward()

    data_grad = images.grad.data

    perturbed_image = (
        images +
        epsilon * data_grad.sign()
    )

    perturbed_image = torch.clamp(
        perturbed_image,
        0,
        1
    )

    adv_output = model(perturbed_image)

    _, final_pred = torch.max(
        adv_output,
        1
    )

    total += 1

    if final_pred.item() == labels.item():
        correct += 1

fgsm_accuracy = 100 * correct / total

print(
    f"FGSM Accuracy: {fgsm_accuracy:.2f}%"
)