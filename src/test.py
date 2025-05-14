import torch
from PIL import Image
from torchvision import transforms
from caption_model import ImageCaptioningModel
from tts import speak
# from vocab import Vocabulary
import pickle

# Load vocab
# vocab = {...}  # Load same vocab as used in training
# inv_vocab = {v: k for k, v in vocab.items()}

# Load the saved vocabulary object
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# inv_vocab = {v: k for k, v in vocab.word2idx.items()}
inv_vocab = {v: k for k, v in vocab.items()}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ImageCaptioningModel(len(vocab), 256, 512)
model.load_state_dict(torch.load("caption_model.pth"))
# model.load_state_dict(torch.load("caption_model.py"))
model.eval().to(device)

def generate_caption(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.cnn(image)
        input_id = torch.tensor([[vocab['<start>']]]).to(device)
        caption = []

        for _ in range(20):
            embeddings = model.embed(input_id)
            if len(caption) == 0:
                embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
            hiddens, _ = model.lstm(embeddings)
            outputs = model.fc(hiddens[:, -1, :])
            _, predicted = outputs.max(1)

            word = inv_vocab[predicted.item()]
            if word == '<end>':
                break
            caption.append(word)
            input_id = torch.cat([input_id, predicted.unsqueeze(0)], dim=1)

    result = ' '.join(caption)
    print("Caption:", result)
    speak(result)

# generate_caption("/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/src/image/mQcKcyt3Nb25vMYBSbb47aQ9Kw.jpg")
generate_caption("/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/src/image/pexels-souvenirpixels-414612.jpg")
