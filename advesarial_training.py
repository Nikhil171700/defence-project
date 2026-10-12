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
# 2. Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================================
# 3. Training Dataset
# ============================================================

train_dataset = ImageFolder(
    "train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


print("Classes:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))


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
# 5. Create Model
# ============================================================

model = bird_drone().to(device)


# ============================================================
# 6. Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 7. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 8. FGSM Settings
# ============================================================

epsilon = 0.03


# ============================================================
# 9. Adversarial Training
# ============================================================

epochs = 10


for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------------------------
        # Step 1: Generate adversarial examples
        # ----------------------------------------------------

        images.requires_grad = True

        outputs = model(images)

        clean_loss = criterion(outputs, labels)

        model.zero_grad()

        clean_loss.backward()

        data_grad = images.grad.data

        # FGSM attack
        adversarial_images = (
            images +
            epsilon * data_grad.sign()
        )

        # Keep pixel values between 0 and 1
        adversarial_images = torch.clamp(
            adversarial_images,
            0,
            1
        )

        # ----------------------------------------------------
        # Step 2: Train using adversarial images
        # ----------------------------------------------------

        optimizer.zero_grad()

        adversarial_outputs = model(adversarial_images)

        adversarial_loss = criterion(
            adversarial_outputs,
            labels
        )

        adversarial_loss.backward()

        optimizer.step()

        running_loss += adversarial_loss.item()


    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss = {running_loss:.4f}"
    )


# ============================================================
# 10. Save Robust Model
# ============================================================

torch.save(
    model.state_dict(),
    "bird_drone_robust_model.pth"
)


print()
print("Adversarial training completed!")
print("Model saved as: bird_drone_robust_model.pth")