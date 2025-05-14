import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from caption_model import ImageCaptioningModel
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import os
from torch.nn.utils.rnn import pad_sequence
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import pickle


class CaptionDataset(Dataset):
    def __init__(self, csv_file, img_dir, vocab, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        caption = self.df.iloc[idx, 1].lower().split()

        caption_tokens = [self.vocab['<start>']] + [self.vocab.get(word, self.vocab['<unk>']) for word in caption] + [self.vocab['<end>']]
        caption_tensor = torch.tensor(caption_tokens)

        if self.transform:
            image = self.transform(image)

        return image, caption_tensor

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>'])
    return images, captions


def build_vocab(captions, threshold=1):
    counter = Counter()
    for caption in captions:
        counter.update(caption.lower().split())
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = {word: idx+4 for idx, word in enumerate(words)}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    return vocab

# Load captions
df = pd.read_csv("/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/DataSet/captions.csv")
captions = df['caption'].tolist()
vocab = build_vocab(captions)
# Save vocabulary object
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
vocab_size = len(vocab)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CaptionDataset("/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/DataSet/captions.csv", "/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/DataSet/Images", vocab, transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ImageCaptioningModel(vocab_size, embed_size=256, hidden_size=512).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader,leave=True)
    for images, captions in loop:
        # images, captions = zip(*batch)
        # images, captions = batch
        # images = torch.stack(images).to(device)
        images = images.to(device)
        # captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>']).to(device)
        captions = captions.to(device)

        # outputs = model(images, captions[:, :-1])
        # loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        outputs = model(images, captions[:, :-1])  # model expects input without <end>
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))  # target is without <start>


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print("Output shape:", outputs.shape)         # (batch_size, T, vocab_size)
        # print("Target shape:", captions[:, 1:].shape) # (batch_size, T)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# ✅ Save model after all epochs are done
torch.save(model.state_dict(), "caption_model.pth")

