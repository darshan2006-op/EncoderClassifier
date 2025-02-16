import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedder = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedder(x) * math.sqrt(self.embed_dim)
    
class PositionEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.d = device
        self.register_buffer("pe", None)

    def gen_pe(self, seq_len, dmodel):
        self.pe = torch.zeros(seq_len, dmodel)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        dt = torch.arange(0, dmodel, 2)
        dt = torch.exp((dt/dmodel) * (-math.log(10_000)))
        angle_term = pos * dt
        self.pe[:,0::2] = torch.sin(angle_term)
        self.pe[:,1::2] = torch.cos(angle_term)
        self.pe = self.pe.unsqueeze(0).to(self.d)

    def forward(self, x):
        _, seq_len, dmodel = x.shape
        self.gen_pe(seq_len, dmodel)
        return x + self.pe
