import copy
import math
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from random import random

import torch
import torch.nn.functional as F
from torch import einsum, nn
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import utils

from diffusion.models.positional import SinusoidalPosEmb


class Unet(Module):
    """Unet model to be trained for diffusion during the reverse process

    This is the only training that is performed in diffusion.
    """

    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        learned_variance=False,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=False,  # defaults to full attention only for inner most layer
        flash_attn=False,
    ):
        """Initialize UNet model

        Args:
            dim:
            dim_mults: Multiplier for dim which sets the number of channels in the unet model
            attn_heads: Number of heads in mult-head attntion
            all_attn: Whether to apply attention to all unet levels; if False only use attention on the last level;


        """
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = dim

        # Initial 7x7 conv in ResNet
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=7, padding=3)

        # Create the channels of the model
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        
        # Pair the downsampled channels with the upsampled channels [(down[0], up[1]) ...]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Dimension of the noise time embeddings
        time_dim = dim * 4

        # Initialize postional embeddings for the noise timesteps
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
        fourier_dim = dim

        # Positional embedding module
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        num_stages = len(dim_mults)
        
        # Apply attention to all layers if True or only last layer if False
        if full_attn:
            attn_layers = (True,) * num_stages
        else:
            attn_layers = ((False,) * num_stages - 1, True)

        attn_heads = (attn_heads,) * num_stages
        attn_dim_head = (attn_dim_head,) * num_stages

        assert len(attn_layers) == len(dim_mults)

        # prepare blocks

        ######### START HERE, REMOVE PARTIAL AND UNDERSTAND RESNET
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_attn else None

            attention

            self.downs.append(
                ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_klass(
                            dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(
            mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(
                ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_klass(
                            dim_out,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = channels * (1 if not learned_variance else 2)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
