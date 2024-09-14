from typing import Optional

import torch
from torch import nn
from torch.nn import function as F


class MultiheadAttention(nn.Module):
    """TODO"""

    def __init__(self, embed_dim, num_heads):
        """TODO

        Args:
            embed_dim: Total dimension of the model; embed_dim will be split across
                       num_heads (attention_dim // num_heads)  after it's projected
            num_heads: Number of attention heads; each head will have dimension of attention_dim // num_heads
        
        Returns:
            Linear projected attention values (batch, seq_len, embed_dim)
        """
        assert (
            embed_dim % num_heads == 0
        ), "The number of heads should be divisble by the attenion_dim"

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = Attention()

        self.linear_proj = nn.linear(embed_dim, embed_dim, bias=False)

        # self.heads = nn.ModuleList(
        #     [Attention(head_dim, head_size, block_size) for _ in range(num_heads)]
        # )

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """ "Forward pass through Multiheaded Attention;
        for self-attention the queries, keys, and values should be the same

        Args:
            queries: Input tensor to compute the attention of
            keys: Input tensor to compute the attention of
            values: Input tensor to compute the context of; for self-attention this should be the same
                    as q & v
        """
        # Linearly project q, k, & v (batch, seq_len, embed_dim)
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        # Split into heads (batch, num_heads, seq_len, head_dim)
        query_heads = queries.view(
            queries.shape[0], queries.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_heads = keys.view(
            keys.shape[0], keys.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_heads = values.view(
            values.shape, values.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention on all heads
        attention = self.attention(query_heads, key_heads, value_heads)

        # Combine all the heads together (concatenation step);
        # (b, heads, seq, head_dim) -> (b, seq, heads, head_dim) -> (b, seq, heads*head_dim)
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(attention.shape[0], -1, self.num_heads * self.head_dim)
        )

        # Final linear projection of MHA
        attention_proj = self.linear_proj(attention)

        return attention_proj


class Attention(nn.Module):
    """Scaled dot product attention with an optional mask; typically this
    is used in MHA where q, k, v have already been linearly projected and split
    into multiple heads
    """

    def __init__(self):
        """Initialize attention module"""
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention on q, k, & v

        q, k, v should be a minimum of 3 dimensions such as (batch, seq_len, embed_dim)
        or (batch, num_heads, seq_len, head_dim); attention will be computed on the last 2 dims

        Args:
            queries: Input tensor to compute the attention of
            keys: Input tensor to compute the attention of
            values: Input tensor to compute the context of; for self-attention this should be the same
                    as q & v
            mask: Optional tensor containing indices to be masked; typically used in decoders for NLP

        Returns:
           The context vectors (batch_size, seq_len, d_model)
        """

        # Used to scale the qk dot product
        sqrt_dim = torch.sqrt(k.shape[-1])

        # (batch_size, num_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt_dim

        # Mask attention indices if mask is provided; softmax will set -inf to 0
        if mask is not None:
            scores.masked_fill_(mask.view(scores.size()), -float("Inf"))

        attention = F.softmax(scores, dim=-1)

        # Considered the context vectors because it's a weighted sum of the attention scores;
        # this gives a `context` value about an input's location
        context = torch.matmul(attention, v)  # (batch_size, num_heads, v_len, head_dim)
        return context


class AttentionFeatureMaps(nn.Module):
    # START HERE try to implement as shown here
    # https://github.com/w86763777/pytorch-ddpm/blob/master/diffusion.py
    """Scaled dot product attention with an optional mask; typically this
    is used in MHA where q, k, v have already been linearly projected and split
    into multiple heads
    """

    def __init__(self, attention_dim: int):
        """Initialize attention module

        Args:
            attention_dim: Dimension of the attention
        """
        super().__init__()

        # Used to scale the qk dot product
        self.sqrt_dim = torch.sqrt(attention_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention on q, k, & v

        q, k, v should be a minimum of 3 dimensions such as (batch, seq_len, embed_dim)
        or (batch, num_heads, seq_len, head_dim); attention will be computed on the last 2 dims

        Args:
            queries: Input tensor to compute the attention of
            keys: Input tensor to compute the attention of
            values: Input tensor to compute the context of; for self-attention this should be the same
                    as q & v
            mask: Optional tensor containing indices to be masked; typically used in decoders for NLP

        Returns:
           The context vectors (batch_size, seq_len, d_model)
        """
        # (batch_size, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_dim

        # Mask attention indices if mask is provided; softmax will set -inf to 0
        if mask is not None:
            scores.masked_fill_(mask.view(scores.size()), -float("Inf"))

        attention = F.softmax(scores, dim=-1)

        # Considered the context vectors because it's a weighted sum of the attention scores;
        # this gives a `context` value about an input's location
        context = torch.matmul(attention, v)  # (batch_size, v_len, d_model)
        return context