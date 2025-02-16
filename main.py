from Encoder import Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

enc = Encoder(10, 8, 32, 2)
inp = torch.tensor([[2,3,7,8], [4, 8, 7, 5]])
print(enc(inp))