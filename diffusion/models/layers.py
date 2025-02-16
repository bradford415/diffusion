from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class MultiheadAttention(nn.Module):
    """TODO

    Implementation based on: https://github.com/hkproj/pytorch-transformer/blob/3beddd2604bfa5e0fbd53f52361bd887dc027a8c/model.py#L83
    """

    def __init__(self, embed_dim, num_heads):
        """TODO

        NOTE: some MHA implementations allow you to specify the input dimensions to qkv and have
        another parameter for full dimension of attention i.e., the qkv projection dim

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
        """Forward pass through Multiheaded Attention;
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
        or (batch, num_heads, seq_len, head_dim) for mha; attention will be computed on the last 2 dims

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
        sqrt_dim = torch.sqrt(torch.tensor(k.shape[-1]))

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
    """Multiheaded Self-Attention with feature maps; this is self attention because the forward() only takes 1 input so
    it would be performing attention on itself; for cross-attention (multiple inputs), simply create another module
    which allows different q, k, v; the reason its not done here is because it just worked better for the use case
    and makes the code a little cleaner (:

    This is essentially the same as MHA but the h, w features of the
    feature maps need to be flattened first **idk what I was writing here**

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
        super().__init__()

        assert (
            embed_ch % num_heads == 0
        ), "The number of heads should be divisble by the attenion_dim"

        self.num_heads = num_heads

        self.head_dim = embed_ch // num_heads

        self.group_norm = nn.GroupNorm(32, embed_ch)

        # NOTE: might need to make bias true? I don't think so though
        self.q_proj = nn.Conv2d(
            embed_ch, embed_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.k_proj = nn.Conv2d(
            embed_ch, embed_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.v_proj = nn.Conv2d(
            embed_ch, embed_ch, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.attention = Attention()

        self.final_proj = nn.Conv2d(embed_ch, embed_ch, kernel_size=1, bias=False)

        # self.heads = nn.ModuleList(
        #     [Attention(head_dim, head_size, block_size) for _ in range(num_heads)]
        # )

    def forward(self, input: torch.Tensor):
        """ "Forward pass through Multiheaded Attention;
        for self-attention the queries, keys, and values should be the same

        Args:
            input: Input tensor to compute self-attention on (b, c, h, w)
        """
        norm_input = self.group_norm(input)

        # Project q, k, & v (batch, embed_ch, height, width)
        queries = self.q_proj(norm_input)
        keys = self.k_proj(norm_input)
        values = self.v_proj(norm_input)

        batch, embed_ch, height, width = queries.shape

        # Flatten spatial features;
        # (batch, height, width, embed_ch) -> (batch, height*width, embed_ch)
        queries = queries.permute(0, 2, 3, 1).view(batch, height * width, embed_ch)
        keys = keys.permute(0, 2, 3, 1).view(batch, height * width, embed_ch)
        values = values.permute(0, 2, 3, 1).view(batch, height * width, embed_ch)

        # Split into embeddings into multiple heads; features = height*width
        # (b, features, num_heads, head_dim) -> (b, num_heads, features, head_dim)
        query_heads = queries.view(
            queries.shape[0], queries.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_heads = keys.view(
            keys.shape[0], keys.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_heads = values.view(
            values.shape[0], values.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention on all heads (b, heads, features, head_dim)
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

        assert attention_proj.shape == input.shape

        return attention_proj + input


class ResBlock(nn.Module):
    """Resnet-based block with 2 convolutions, time-embeddings, and a skip connection.

    This block is used in DDPM and latent diffusion, specifically in the unet and the encoder
    component of the autoencoder; unets typically use the time embedding but the encoder does not

    Implementation based on: https://github.com/w86763777/pytorch-ddpm/blob/f804ccbd58a758b07f79a3b9ecdfb1beb67258f6/model.py#L116
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = None, dropout: float = 0.0, eps = 1e-6):
        """TODO
        Args:
            in_ch: TODO
            out_ch:
            time_emb_dim: dimension size of the input time embedding; this is determined
                        at the start of unet when the timestep is intially projected
            eps: a very small value added to the denominator for numerical stability
        """
        super().__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch),  # nn.Linear(time_emb_dim, out_ch)
            )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

        # Whether to use point-wise conv skip connection or a regular skip connection;
        # in the default parameters the point-wise conv is applied at the end of each unet level
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb = None):
        """Forward pass through ResBlock with time embeddings

        Args:
            x: Feature maps (b, c, h, w)
            time_emb: (b, time_dim) TODO: Verify this shape
        """

        h = self.block1(x)

        # Add projected time embeddings to the feature maps; 
        # (b, out_ch, h, w) + (b, out_ch, 1, 1)
        # this broadcasts the value of the time_emb accross the feature map h & w
        if self.time_proj is not None:
            h += self.time_proj(time_emb)[:, :, None, None]

        h = self.block2(h)
        h = h + self.shortcut(x)

        return h


class Downsample(nn.Module):
    """Downsample feature map; this is used at the last layer of each unet encoder level"""

    def __init__(self, ch: torch.Tensor):
        """Initialize downsample module

        Args:
            in_ch: The number of channels in the input feature map
            out_ch: The number of output channels after convolution
        """
        super().__init__()

        # Downsample by factor of 2
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Downsample the feature map"""
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """Upsample feature maps; this is used at the last layer of each unet decoder level"""

    def __init__(self, ch: torch.Tensor):
        """Initialize upsample module"""
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """TODO"""
        # Upsample by a factor of 2 with nearest neighbors
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class AttnBlock(nn.Module):
    """Directly from github code"""

    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class TimestepEmbedSequential(nn.Sequential):
    """Sequential block for modules with different inputs

    Different modules are wrapped in this class because they have different
    forward() signatures and this class calls them accordingly; for example, ResBlock
    takes a feature map and a time embedding input but Attention only takes the feature map.
    This class calls them accordingly

    Based on: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/model/unet.py#L185
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, MultiheadedAttentionFM):
                x = layer(x)  # layer(x, cond)
            else:
                x = layer(x)
        return x


def init_weights(module):
    """I added this but did not seem to change much"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
