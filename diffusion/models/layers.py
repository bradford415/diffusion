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
    
    def __init__(self, embed_dim, num_heads):
        """TODO
        
        Args:
            embed_dim: Total dimension of the model; embed_dim will be split across
                       num_heads (attention_dim // num_heads)  after it's projected
            num_heads: Number of attention heads; each head will have dimension of attention_dim // num_heads 
        """
        assert embed_dim % num_heads == 0, "The number of heads should be divisble by the attenion_dim"

        self.num_heads = num_heads

        head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # self.heads = nn.ModuleList(
        #     [Attention(head_dim, head_size, block_size) for _ in range(num_heads)]
        # )
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """"Forward pass through Multiheaded Attention; 
        for self-attention the queries, keys, and values should be the same
        
        Args:
            queries: Input tensor to compute the attention of
            keys: Input tensor to compute the attention of
            values: Input tensor to compute the context of; for self-attention this should be the same
                    as q & v
        """
        ## TODO START HERE
        query = self.q_proj(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        
        q, k, v = self.to_qkv(input).chunk(3)
        
        
class Attention(nn.Module):
    """Scaled dot product attention with an optional mask"""
    
    def __init__(self, attention_dim: int):
        """Initialize attention module
        
        Args:
            attention_dim: dimension of the attention TODO clarify
        """
        super().__init__()
        
        # Used to scale the qk dot product
        self.sqrt_dim = torch.sqrt(attention_dim)
        
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """T
        
        Args:
            input: The input
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
        
        
        
        
        