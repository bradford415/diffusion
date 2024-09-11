import torch
from torch import nn


class Attention(nn.Module):
    """TODO
    
    """
    def __init__(self, dropout = 0.0):
        """TODO"""
        super().__init__()
        self.dropout = dropout
        
        
class SelfAttention(nn.Module):
    """TODO"""
    
    def __init__(self, dim, heads, dim_head, num_mem_kv):
        super().__init__()
        self.heads = heads
        
    def forward(self, q, k, v):
        """TODO"""
        
        q_len, k_len = q.shape[-1], k.shape[-1]
        # Start here and implement attention https://github.com/sooftware/attentions/blob/master/attentions.py
        scale = 
        torch.matmul
        
        