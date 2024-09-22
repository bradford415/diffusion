from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    """TODO

    Implementation based on: https://github.com/hkproj/pytorch-transformer/blob/3beddd2604bfa5e0fbd53f52361bd887dc027a8c/model.py#L83
    """

    def __init__(self, embed_dim, num_heads):
        """TODO

        Args:
            embed_dim: Total dimension of the model; embed_dim will be split across
                       num_heads (embed_dim // num_heads)  after it's projected
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
    """Scaled dot product attention with an optional mask

    Typically this is used in MHA where q, k, v have already been linearly
    projected and split into multiple heads; this can be used for sequences
    and feature maps (h, w) but the feature maps features (pixels) should be
    flattened
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


class MultiheadedAttentionFM(nn.Module):
    """Multiheaded Attention with feature maps

    This is essentially the same as MHA but the h,w features of the
    feature maps need to be flattened first

    This implementation is based on: https://github.com/w86763777/pytorch-ddpm/blob/f804ccbd58a758b07f79a3b9ecdfb1beb67258f6/model.py#L78
    """

    def __init__(self, embed_ch, num_heads):
        """TODO

        Args:
            embed_ch: Total channels of the attention model; embed_ch will be split across
                       num_heads (embed_ch // num_heads) after it's projected
            num_heads: Number of attention heads; each head will have dimension
                       of embed_ch // num_heads

        Returns:
            Linear projected attention values (batch, seq_len, embed_dim)
        """
        assert (
            embed_ch % num_heads == 0
        ), "The number of heads should be divisble by the attenion_dim"

        self.num_heads = num_heads

        self.head_dim = embed_ch // num_heads

        self.group_norm = nn.GroupNorm(32, embed_ch)

        # NOTE: might need to make bias true?
        self.q_proj = nn.Conv2d(
            embed_ch, embed_ch, kernel_size=1, stride=1, padding=1, bias=False
        )
        self.k_proj = nn.Conv2d(
            embed_ch, embed_ch, kernel_size=1, stride=1, padding=1, bias=False
        )
        self.v_proj = nn.Conv2d(
            embed_ch, embed_ch, kernel_size=1, stride=1, padding=1, bias=False
        )

        self.attention = Attention()

        self.final_proj = nn.Conv2d(embed_ch, embed_ch, bias=False)

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
        # Project q, k, & v (batch, embed_ch, height, width)
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        batch, embed_ch, height, width = queries.shape

        # Flatten spatial features;
        # (batch, height, width, embed_ch) -> (batch, height*width, embed_ch)
        queries = queries.permute(0, 2, 3, 1).view(batch, height * width, embed_ch)
        keys = keys.permute(0, 2, 3, 1).view(batch, height * width, embed_ch)
        values = values.permute(0, 2, 3, 1).view(batch, height * width, embed_ch)

        # Split into multiple heads (batch, num_heads, features, head_dim)
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
        # (b, heads, features, head_dim) -> (b, features, heads, head_dim) -> (b, features, heads*head_dim)
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(attention.shape[0], -1, self.num_heads * self.head_dim)
        )

        # Convert features back to spatial dimensions
        # (b, height, width, embed_ch) -> (b, embed_ch, height, width)
        attention = attention.view(batch, height, width, embed_ch).permute(0, 3, 1, 2)

        # Final projection of MHA
        attention_proj = self.final_proj(attention)

        return attention_proj


class ResnetBlock(nn.Module):
    """TODO

    Implementation based on: https://github.com/w86763777/pytorch-ddpm/blob/f804ccbd58a758b07f79a3b9ecdfb1beb67258f6/model.py#L116
    """

    def __init__(self, in_ch, out_ch, time_emb_dim=None, dropout=0.0):
        """TODO"""
        super().__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch * 2),  # nn.Linear(time_emb_dim, out_ch)
            )

        # NOTE: It looks like the original implementation reverses the order i.e., conv2d last: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L45
        #       but many implementations are written differently and more often Conv2d seems to be first
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.GroupNorm(32, in_ch),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.GroupNorm(32, out_ch),
            nn.Dropout(dropout),
        )

        # Whether to add a point-wise conv skip connection or a regular skip connection;
        # in the default parameters the point-wise conv is applied at the end of each unet level
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        """Forward pass through ResNetBlock with time embeddings

        Args:
            x: Feature maps (B, C, H, W)
            time_emb: (B, time_dim) TODO: Verify this shape
        """

        x = self.block1(x)

        # Add projected time embeddings to the feature maps ;
        # (b, out_ch, h, w) + (b, out_ch, 1, 1)
        # this broadcasts the value of the time_emb accross the feature map h & w
        x += self.time_proj(time_emb)[:, :, None, None]

        x = self.block2(x)

        return x


class Downsample(nn.Module):
    """Downsample feature map; this is used at the last layer of each unet encoder level"""

    def __init__(self, in_ch: torch.Tensor, out_ch: torch.Tensor):
        """Initialize downsample module

        Args:
            in_ch: The number of channels in the input feature map
            out_ch: The number of output channels after convolution
        """
        super().__init__()

        # Downsample by factor of 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Downsample the feature map"""
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """Upsample feature maps; this is used at the last layer of each unet decoder level"""

    def __init__(self, in_ch: torch.Tensor, out_ch: torch.Tensor):
        """Initialize upsample module"""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """TODO"""
        # Upsample by a factor of 2 with nearest neighbors
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x
