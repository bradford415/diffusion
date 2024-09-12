from typing import Optional

import torch
from torch import nn
from torch.nn import function as F


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
        """dim"""
        super().__init__()
        self.heads = heads
        scale
        
    def forward(self, q, k, v):
        """TODO
        q: queries (batch_size, q_len, d_model)
        k: keys (batch_size, q_len, d_model)
        v: vectors (batch_size, q_len, d_model)
        
        """
        
        q_len, k_len = q.shape[-1], k.shape[-1]
        # Start here and implement attention https://github.com/sooftware/attentions/blob/master/attentions.py
        
        torch.matmul
        
        
class MultiheadAttention(nn.Module):
    """TODO"""
    
    def __init__(self, input_dim, embed_dim, num_heads):
        """TODO
        
        Args:
            embed_dim: Total dimension of the model; embed_dim will be split across
                       num_heads (attention_dim // num_heads) 
            num_heads: Number of attention heads; each head will have dimension of attention_dim // num_heads 
        """
        
        assert embed_dim % num_heads == 0, "The number of heads should be divisble by the attenion_dim"
        head_dim = embed_dim // num_heads
        
        self.to_qkv = nn.Linear(input_dim, head_dim*3, bias=False)
        
        self.heads = nn.ModuleList(
            [Attention(head_dim, head_size, block_size) for _ in range(num_heads)]
        )
        
        self.projection = nn.Linear(input_dim, input_dim)
        
    def forward(self, input):
        
        q, k, v = self.to_qkv(input).chunk(3)
        
        
    
        
        
class Attention(nn.Module):
    """Scaled dot product attention with an optional mask"""
    
    def __init__(self, attention_dim: int):
        """Initialize attention module
        
        Args:
            attention_dim: dimension of the attention TODO clarify
        """
        super().__init__()
        
        ############# START HERE USE QKV PROJECTIONS LIKE IN NANOGPT; also need to see if lucid rains uses multi head or just attention
        
        # Used to scale the qk dot product
        self.sqrt_dim = torch.sqrt(attention_dim)
        
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """T
        
        Args:
            input: TODO
            mask: Tensor containing indices to be masked
            
        Returns:
           The context vectors (batch_size, seq_len, d_model)
        """
        # (batch_size, q_len, k_len)
        scores = torch.matmul(q, k.transpose(1,2)) / self.sqrt_dim
        
        # Mask attention indices if mask is provided
        if mask is not None:
            scores.masked_fill_(mask.view(scores.size()), -float('Inf'))
        
        attention = F.softmax(scores, dim=-1)
        
        # Considered the context vectors because it's a weighted sum of the attention scores;
        # this gives a `context` value about an input's location
        context = torch.matmul(attention, v) # (batch_size, v_len, d_model)
        return context
        
        
        
        
        