import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Parameter(torch.ones(1))
        self.a2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True, unbiased=False)
        p =  (x - mean) / (sd + 1e-7)
        return self.a1 * p + self.a2

class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = LayerNorm()
        self.do = nn.Dropout(0.2)

    def forward(self, x1, x2):
        return self.do(self.ln(x1 + x2))
    
class PositionalFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.l = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dff, d_model))
    
    def forward(self, x: torch.Tensor):
        return self.l(x)
    
