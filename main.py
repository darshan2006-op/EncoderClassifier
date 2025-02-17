from Classifier import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

enc = Classifier(len(10), 100, 4, 6, 50)
inp = torch.tensor([[2,3,7,8], [4, 8, 7, 5]])
print(enc(inp))