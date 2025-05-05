import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

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

# 2. Loader class
class ImageLoader:
    def __init__(self, image_folder, transform, batch_size=32):
        self.dataset = ImageCaptionDataset(image_folder, transform)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def get_dataloader(self):
        return self.dataloader

    def denormalize(self, img_tensor):
        """
        Converts a normalized image tensor back to viewable format (H, W, C) for visualization.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img_tensor.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    def visualize_batch(self, n=6):
        """
        Visualizes first 'n' images from the batch.
        """
        data_iter = iter(self.dataloader)
        images, image_names = next(data_iter)

        plt.figure(figsize=(12, 6))
        for i in range(min(n, len(images))):
            plt.subplot(2, 3, i+1)
            img = self.denormalize(images[i])
            plt.imshow(img)
            plt.title(image_names[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()