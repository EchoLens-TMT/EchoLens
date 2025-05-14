import torch
import torch.nn as nn
import torchvision.models as models

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ImageCaptioningModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    # def forward(self, images, captions):
    #     features = self.cnn(images)
    #     embeddings = self.embed(captions)
    #     embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    #     hiddens, _ = self.lstm(embeddings)
    #     outputs = self.fc(hiddens)
    #     return outputs

    def forward(self, images, captions):
        features = self.cnn(images)  # (B, embed_size)
        features = features.unsqueeze(1)  # (B, 1, embed_size)

        embeddings = self.embed(captions)  # (B, T, embed_size)

        # Concatenate features at the start
        embeddings = torch.cat((features, embeddings), 1)  # (B, T+1, embed_size)

        hiddens, _ = self.lstm(embeddings)  # (B, T+1, hidden_size)

        # Drop the output corresponding to the image (1st token)
        outputs = self.fc(hiddens[:, 1:, :])  # (B, T, vocab_size)

        return outputs
