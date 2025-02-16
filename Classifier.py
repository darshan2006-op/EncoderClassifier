import torch 
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder

class Classifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes=1, device="cuda"):
        super().__init__()
        self.encoder = Encoder(vocab_size, num_heads, d_model, num_layers, device)
        self.layers = nn.Sequential(nn.Linear(d_model, num_classes), nn.Softmax())

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim = 1)
        x = self.layers(x)
        return x