from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# 2. Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================================
# 3. Test Dataset
# ============================================================

test_dataset = ImageFolder(
    "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))


# ============================================================
# 4. CNN Model
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
# 5. Load Robust Model
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
# 6. Clean Accuracy
# ============================================================

correct = 0
total = 0

for images, labels in test_loader:

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

    total += labels.size(0)

    correct += (predicted == labels).sum().item()


clean_accuracy = 100 * correct / total


print()
print("Clean Accuracy =", clean_accuracy)


# ============================================================
# 7. FGSM Settings
# ============================================================

epsilon = 0.03

correct = 0
total = 0


# ============================================================
# 8. FGSM Testing
# ============================================================

for images, labels in test_loader:

    images = images.to(device)
    labels = labels.to(device)

    # Enable gradient
    images.requires_grad = True

    outputs = model(images)

    # Original prediction
    _, initial_pred = torch.max(outputs, 1)

    # Only attack images that were originally correct
    correct_mask = initial_pred == labels

    if correct_mask.sum().item() == 0:
        continue

    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()

    loss.backward()

    data_grad = images.grad.data

    # Generate FGSM images
    perturbed_images = (
        images +
        epsilon * data_grad.sign()
    )

    perturbed_images = torch.clamp(
        perturbed_images,
        0,
        1
    )

    # Test adversarial images
    with torch.no_grad():

        adv_outputs = model(perturbed_images)

        _, final_pred = torch.max(
            adv_outputs,
            1
        )

    # Count only originally correct images
    final_pred = final_pred[correct_mask]
    labels_correct = labels[correct_mask]

    total += labels_correct.size(0)

    correct += (
        final_pred == labels_correct
    ).sum().item()


fgsm_accuracy = 100 * correct / total


print("FGSM Accuracy =", fgsm_accuracy)

print()
print("===================================")
print("FINAL ADVERSARIAL MODEL RESULTS")
print("===================================")
print(f"Clean Accuracy : {clean_accuracy:.2f}%")
print(f"FGSM Accuracy  : {fgsm_accuracy:.2f}%")
print("===================================")