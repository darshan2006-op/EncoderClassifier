import torch
import math
import torch.nn as nn
import torch.nn.functional as F 

def sacalar_attention(k:torch.Tensor,q:torch.Tensor,v:torch.Tensor, dmodel, mask=None):
    logits = (torch.matmul(q, k.transpose(-2,-1))) / torch.sqrt(torch.tensor(dmodel))
    if mask is not None:
        logits += mask
    att_w = F.softmax(logits, dim=-1)
    output = torch.matmul(att_w, v)
    return output, att_w

class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = dmodel
        self.W_k = nn.Linear(dmodel, dmodel, bias=False)
        self.W_q = nn.Linear(dmodel, dmodel, bias=False)
        self.W_v = nn.Linear(dmodel, dmodel, bias=False)
        self.W_o = nn.Linear(dmodel, dmodel, bias=False)

    def split_heads(self, x:torch.Tensor):
        b, s, e = x.shape
        return x.view(b,s,self.num_heads, e // self.num_heads).transpose(1,2)
    
    def combine_heads(self, x:torch.Tensor):
        b,nh,s,e = x.shape
        return x.transpose(1,2).contiguous().view(b, s, nh * e)
    
    def forward(self, x):
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        out, att = sacalar_attention(k,q,v,self.d_model)
        out = self.combine_heads(out)
        out = self.W_o(out)
        return out
