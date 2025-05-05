from torchvision import transforms
from imageLoader import ImageCaptionDataset, ImageLoader
from encoder import CNNEncoder
import torch

if __name__ == "__main__":
    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),         # Resize to fixed size
        transforms.ToTensor(),                 # Convert to tensor (C, H, W)
        transforms.Normalize(                  # Normalize using ImageNet means & stds
            mean=[0.485, 0.456, 0.406],        # RGB mean
            std=[0.229, 0.224, 0.225]          # RGB std
        )
    ])

    image_folder = r"D:\DIT\First Sem\Computer Vision\EchoLens\DataSet\Images"

    dataset = ImageCaptionDataset(image_folder, transform=image_transforms)
    loader = ImageLoader(image_folder, image_transforms)

    # Visualize a batch of images
    loader.visualize_batch()

    encoder = CNNEncoder()
    dummy_input = torch.randn(32, 3, 224, 224)  # batch of 32 RGB images
    output = encoder(dummy_input)
    print(output.shape)  # Expected: torch.Size([32, 256])