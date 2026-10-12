from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
'''
Resize
Original image
600 × 800
↓
128 × 128
Every image becomes the same size.
Neural networks require equal-sized images.
ToTensor()
Images are normally stored as
Pixel values
0–255
Example
255 120  30
200 100  10
Neural networks prefer numbers between
0 and 1
After
ToTensor()
255 → 1.0
128 → 0.50
0   → 0.0
It also changes the image into a PyTorch Tensor
'''
# Datasets
train_dataset = ImageFolder("train", transform=transform)
test_dataset = ImageFolder("test", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Model
class bird_drone(nn.Module):#nn.Module is the base class for all neural networks in PyTorch
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),#Rectified Linear Unit.
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
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

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, Loss = {running_loss:.4f}"
    )
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs,1)#return maximum values[4.8,3.5,2.1] and maximum index we take only index[1,0,0]

        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        #.item() converts the result from a PyTorch tensor to a normal Python integer.
accuracy = 100 * correct / total

print("Accuracy =", accuracy)
torch.save(model.state_dict(), "bird_drone_model.pth")

'''
A CNN cannot understand:
 
bird1.jpg
drone1.jpg

It needs:

Image + Label

For example:

bird image  -> 0
drone image -> 1

ImageFolder automatically creates those (image, label) pairs.

What is ImageFolder?

ImageFolder is a ready-made PyTorch class that:

Reads images from folders.
Automatically assigns labels.
Creates a dataset that can be used for training a CNN.

Instead of writing hundreds of lines yourself, PyTorch does it for you.
1. torchvision

torchvision is a library that comes with PyTorch and is specially made for computer vision (images and videos).

Think of it like this:

PyTorch
│
├── torch          → tensors, neural networks, training
│
└── torchvision    → images, datasets, transforms

If your project is:

Bird vs Drone Classification

you will use torchvision a lot.

2. datasets

Inside torchvision, there is a module called datasets.

torchvision.datasets

contains ready-made dataset classes.

Its job is:

Load images
Load labels
Organize data
Give data to the CNN
You are importing the ImageFolder class from the datasets module.
'''
'''
Starting from a 128×128 RGB image:

Layer	Output Shape
Input	128 × 128 × 3
Conv1	128 × 128 × 16
Pool1	64 × 64 × 16
Conv2	64 × 64 × 32
Pool2	32 × 32 × 32
Conv3	32 × 32 × 64
Pool3	16 × 16 × 64

64 = feature maps
16 × 16 = size of each feature map
'''