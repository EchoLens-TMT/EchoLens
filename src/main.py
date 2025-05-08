from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import os
from tqdm import tqdm

from imageLoader import ImageCaptionDataset, ImageLoader
from encoder import CNNEncoder
from vocabulary import Vocabulary
from decoder import DecoderRNN

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

    image_folder = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images"
    captions_dataset = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/captions.txt"

    # dataset = ImageCaptionDataset(image_folder, transform=image_transforms)
    # loader = ImageLoader(image_folder, image_transforms)

    # # Visualize a batch of images
    # loader.visualize_batch()

    with open(str(captions_dataset), "r", encoding="utf-8") as f:
        captions = [str(line.strip().lower().split(',')[1]) for line in f.readlines()]

    # 2. Initialize and build vocab
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(captions)

    ###################################################################
    # Training 
    ###################################################################

    # -------- CONFIGURATION --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_epochs = 25
    batch_size = 32
    learning_rate = 1e-3
    freq_threshold = 5

    # -------- SETUP DATA & VOCAB --------
    image_loader = ImageLoader(image_folder, captions_dataset, transform=image_transforms, batch_size=batch_size)
    dataloader = image_loader.get_dataloader()

    # -------- MODEL INIT --------
    encoder = CNNEncoder().to(device)
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1).to(device)

    # -------- LOSS & OPTIMIZER --------
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    params = list(decoder.parameters()) + list(encoder.fc2.parameters()) + list(encoder.bn2.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    # -------- TRAINING LOOP --------
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        loop = tqdm(dataloader, leave=True)
        for imgs, imgs_name, captions in loop:
            imgs = imgs.to(device)
            # Convert raw captions into numerical format
            tokenized_captions = [vocab.numericalize(caption) for caption in captions]

            caption_lengths = [len(cap) for cap in tokenized_captions]
            max_len = max(caption_lengths)
            padded_captions = [cap + [vocab.word2idx["<pad>"]] * (max_len - len(cap)) for cap in tokenized_captions]

            targets = torch.tensor(padded_captions).to(device)

            # Forward pass
            features = encoder(imgs)
            outputs = decoder(features, targets[:, :-1])

            # Loss
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # Save models
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
    