import cv2
import os    #It provides functions to work with files and folders
from PIL import Image #This statement imports the Image module from the PIL (Python Imaging Library) package
from sklearn.model_selection import train_test_split
import shutil

bird_folder = "bird"
drone_folder="drone" 

image_bird = os.listdir(bird_folder)[0]  #This returnsthe data as  list of all files and folders inside bird_folder
image_drone=os.listdir(drone_folder)[0]
img = cv2.imread(os.path.join(bird_folder, image_bird))  #This combines the folder path and filename safely.
img1 = cv2.imread(os.path.join(drone_folder, image_drone))
#cv2.imshow("Bird Image", img)
#cv2.imshow("drone image",img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print("Drone:", len(os.listdir("drone")))
print("Bird:", len(os.listdir("bird")))


folder = "bird"
#
for file in os.listdir(folder):
    path = os.path.join(folder, file)

    try:
        img = Image.open(path)
        img.verify() #This checks whether the image file is actually valid

    except:
        print("Corrupted:", path)
        os.remove(path)

folder = "drone"

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    try:
        img = Image.open(path)
        img.verify()

    except:
        print("Corrupted:", path)
        os.remove(path)
print("Drone:", len(os.listdir("drone")))
print("Bird:", len(os.listdir("bird")))

image_bird = os.listdir(bird_folder)
train_bird,test_bird=train_test_split(image_bird,test_size=0.2,random_state=42)
print(len(train_bird))
print(len(test_bird)) 

image_drone = os.listdir(drone_folder)
train_drone,test_drone=train_test_split(image_drone,test_size=0.2,random_state=42)
print(len(train_drone))
print(len(test_drone)) 

# Create folders
folders = [
    "train/bird",
    "train/drone",
    "test/bird",
    "test/drone"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)#Make directories

# Copy bird images
for img in train_bird:
    shutil.copy(
        os.path.join(bird_folder, img),
        os.path.join("train/bird", img)
    )

for img in test_bird:
    shutil.copy(
        os.path.join(bird_folder, img),
        os.path.join("test/bird", img)
    )

# Copy drone images
for img in train_drone:
    shutil.copy(
        os.path.join(drone_folder, img),
        os.path.join("train/drone", img)
    )

for img in test_drone:
    shutil.copy(
        os.path.join(drone_folder, img),
        os.path.join("test/drone", img)
    )

print("Dataset organized successfully!")

'''
What is train_bird?

It's just a Python list of filenames.

Example:

train_bird = [
    "bird1.jpg",
    "bird2.jpg",
    "bird3.jpg"
]

And:

test_bird = [
    "bird10.jpg",
    "bird11.jpg"
]

These are not images and not PyTorch datasets.

What PyTorch needs

A CNN cannot train on:

["bird1.jpg", "bird2.jpg", "bird3.jpg"]

It needs:

Image Pixels + Label

Example:

bird1.jpg  → image array → label=0
bird2.jpg  → image array → label=0
drone1.jpg → image array → label=1
Why people create train/test folders

Because ImageFolder automatically does all the work.

Example:

train/
│
├── bird/
│   ├── bird1.jpg
│   ├── bird2.jpg
│
└── drone/
    ├── drone1.jpg
    ├── drone2.jpg

Then:

from torchvision.datasets import ImageFolder

train_dataset = ImageFolder("train")

PyTorch automatically understands:

bird  → class 0
drone → class 1

What is shutil?

shutil stands for:

Shell Utilities

It is a Python module used for file and folder operations such as:

Copying files
Moving files
Deleting folders
Copying entire folders
'''