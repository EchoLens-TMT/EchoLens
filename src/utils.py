# from PIL import Image
# import os
# import torch
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return transform(image)

# def load_captions(csv_path):
#     import pandas as pd
#     df = pd.read_csv(csv_path)
#     return df