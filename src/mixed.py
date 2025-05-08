# import cv2

# def __init__(self, image_data_path, caption_file):
#         self.image_folder = image_data_path
#         self.caption_file = caption_file

# def read_images(self):
#     img = cv2.imread


# img = cv2.imread(r"D:\DIT\First Sem\Computer Vision\EchoLense\DataSet\Images\47871819_db55ac4699.jpg")

# cv2.imshow('My Image', img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()
# import torch as tf
# print(tf.__version__)

import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Dummy: Build vocab from captions list
captions_dataset = ["a man riding a horse", "a dog jumping over a hurdle"] * 20000  # Simulated
freq_threshold = 3
vocab = Vocabulary(freq_threshold)
vocab.build_vocabulary(captions_dataset)


encoder = CNNEncoder()
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab))

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

encoder.eval()
decoder.eval()
encoder.to(device)
decoder.to(device)

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    return image.to(device)

def generate_caption(image_tensor, encoder, decoder, vocab, max_length=20):
    with torch.no_grad():
        features = encoder(image_tensor)
        inputs = features.unsqueeze(1)  # [1, 1, embed_size]

        caption = []
        word = torch.tensor([vocab.word2idx["<start>"]]).to(device)
        states = None

        for _ in range(max_length):
            embeddings = decoder.embed(word).unsqueeze(1)  # [1, 1, embed_size]
            hiddens, states = decoder.lstm(embeddings, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)

            word = predicted
            predicted_word = vocab.idx2word[predicted.item()]

            if predicted_word == "<end>":
                break
            caption.append(predicted_word)

    return " ".join(caption)

image_tensor = preprocess_image("/mnt/c/Users/Tushar Garg/Downloads/woman-3432069_1280.jpg")
caption = generate_caption(image_tensor, encoder, decoder, vocab)
print("Generated Caption:", caption)
