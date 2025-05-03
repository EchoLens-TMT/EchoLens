import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),         # Resize to fixed size
    transforms.ToTensor(),                 # Convert to tensor (C, H, W)
    transforms.Normalize(                  # Normalize using ImageNet means & stds
        mean=[0.485, 0.456, 0.406],        # RGB mean
        std=[0.229, 0.224, 0.225]          # RGB std
    )
])

# Custom dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name  # return name to track, for now

# Create dataset and dataloader
dataset = ImageCaptionDataset(image_folder=r"D:\DIT\First Sem\Computer Vision\EchoLens\DataSet\Images", transform=image_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                        
# Function to denormalize for visualization
def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).numpy()  # Change to HWC
    img = std * img + mean  # denormalize
    img = np.clip(img, 0, 1)
    return img

# Get a batch of images
data_iter = iter(dataloader)
images, image_names = next(data_iter)

# Plot the first few images in the batch
plt.figure(figsize=(12, 6))
for i in range(6):  # show first 6 images
    plt.subplot(2, 3, i+1)
    img = denormalize(images[i])
    plt.imshow(img)
    plt.title(image_names[i])
    plt.axis('off')

plt.tight_layout()
plt.show()



class ImageLoader():
    def __init__(self, image_data_path, caption_file):
        self.image_folder = image_data_path
        self.caption_file = caption_file

    def read_images(self):
        img = cv2.imread


img = cv2.imread(r"D:\DIT\First Sem\Computer Vision\EchoLense\DataSet\Images\47871819_db55ac4699.jpg")

cv2.imshow('My Image', img)

cv2.waitKey(0)

cv2.destroyAllWindows()