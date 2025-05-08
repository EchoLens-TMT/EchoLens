import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

from imageLoader import ImageLoader
from encoder import CNNEncoder
from decoder import DecoderRNN
from vocabulary import Vocabulary

import os
from tqdm import tqdm

# -------- CONFIGURATION --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_folder = "path/to/images"
caption_file = "path/to/captions.txt"  # if needed
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
freq_threshold = 5

# -------- SETUP DATA & VOCAB --------
image_loader = ImageLoader(image_folder, transform=image_transforms, batch_size=batch_size)
dataloader = image_loader.get_dataloader()

# Dummy: Build vocab from captions list
captions_dataset = ["a man riding a horse", "a dog jumping over a hurdle"] * 20000  # Simulated
vocab = Vocabulary(freq_threshold)
vocab.build_vocabulary(captions_dataset)

# -------- MODEL INIT --------
encoder = CNNEncoder(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1).to(device)

# -------- LOSS & OPTIMIZER --------
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

# -------- TRAINING LOOP --------
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0

    loop = tqdm(dataloader, leave=True)
    for imgs, captions in loop:
        imgs = imgs.to(device)
        # Convert raw captions into numerical format
        tokenized_captions = [[vocab.stoi["<START>"]] + [vocab.stoi.get(token, vocab.stoi["<UNK>"]) for token in caption.lower().split()] + [vocab.stoi["<END>"]] for caption in captions]

        caption_lengths = [len(cap) for cap in tokenized_captions]
        max_len = max(caption_lengths)
        padded_captions = [cap + [vocab.stoi["<PAD>"]] * (max_len - len(cap)) for cap in tokenized_captions]

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
