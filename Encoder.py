import torch
import math
import torch.nn as nn
import torch.nn.functional as F 
from Embeddings import Embedder, PositionEncoder
from LayerNorm import LayerNorm, PositionalFeedForwardNetwork, ResidualConnection
from Attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, dff):
        super().__init__()
        self.num_heads = num_heads
        self.dmodel = d_model   

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.res1 = ResidualConnection()
        self.res2 = ResidualConnection()

        self.pwff = PositionalFeedForwardNetwork(d_model, dff)

    def forward(self, x):
        res = x
        x = self.mha(x)
        x = self.res1(x,res)

        res = x
        x = self.pwff(x)
        x = self.res2(x, res)

        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, num_head, d_model, num_layers, dff, device):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(num_head, d_model, dff) for _ in range(num_layers)])
        self.embedder = Embedder(vocab_size, d_model)
        self.position_encoder = PositionEncoder(device)

    def forward(self, x):
        x = self.embedder(x)
        x = self.position_encoder(x)
        return self.layers(x)