import copy
import math
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from random import random

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T

from diffusion.models.positional import SinusoidalPosEmb
from diffusion.models.layers import (
    MultiheadedAttentionFM,
    ResnetBlock,
    Downsample,
    Upsample,
)


class Unet(nn.Module):
    """Unet model to be trained for diffusion during the reverse process

    This is the only training that is performed in diffusion.
    """

    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        image_ch=3,
        self_condition=False,
        learned_variance=False,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_ch=128,
        attn_heads=4,
        attn_levels=[False, False, False, True],
        flash_attn=False,
    ):
        """Initialize UNet model

        Args:
            dim:
            dim_mults: Multiplier for dim which sets the number of channels in the unet model
            img_channels: Number of channels in the original image and output image; rgb=3 grayscale=1
            attn_ch: Total channels for MHA; embed_ch will be split across
                       num_heads (attn_ch // num_heads) after it's projected
            attn_heads: Number of heads in mult-head attntion
            attn_levels: The levels of UNet to perform attention after; the default parameters apply attention
                         only to the last level (i.e., before the middle layers)
        """
        super().__init__()

        # determine dimensions

        self.channels = image_ch
        self.self_condition = self_condition
        input_channels = image_ch * 1  # (2 if self_condition else 1)

        init_dim = dim

        # Initial 7x7 conv in ResNet
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=7, padding=3)

        # Create the channels of the model (5,)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        # Pair the channels for each ResnetBlock
        # Ex: dims = [64, 64, 128, 256, 512] -> in_out_ch = [(64, 64), ..., (256, 512)]
        in_out_ch = list(zip(dims[:-1], dims[1:]))

        # Dimension of the noise time embeddings
        time_dim = dim * 4

        # Initialize postional embeddings for the noise timesteps
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        # Positional embedding module
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Number of Unet levels
        num_stages = len(dim_mults)

        if len(attn_levels) != num_stages:
            raise ValueError(
                "Length of attn_levels should be the same as the length of dim_mults"
            )

        # TODO
        attn_heads = (attn_heads,) * num_stages
        attn_chh = (attn_chh,) * num_stages

        # Create Unet layers
        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])
        num_resolutions = len(in_out_ch)

        # Initialize the encoder layers of unet (downsampling)
        for unet_layer, (
            (ch_in, ch_out),
            layer_attn_heads,
            layer_attn_chh,
            attn,
        ) in enumerate(zip(in_out_ch, attn_heads, attn_chh, attn_levels)):
            # Whether to perform attention for the current Unet level;
            # the default parameters only use attention for the last Unet level (before the middle layers)
            level_layers = nn.ModuleList(
                [
                    ResnetBlock(ch_in, ch_in),
                    ResnetBlock(ch_in, ch_in),
                    MultiheadedAttentionFM(
                        embed_ch=layer_attn_chh, heads=layer_attn_heads
                    )
                    if attn
                    else nn.Identity(),
                ]
            )

            # Downsample by a factor of 2 if not the last unet level
            if unet_layer != (num_resolutions - 1):
                level_layers.append(Downsample(ch_in, ch_out))
            else:
                level_layers.append(nn.Conv2d(ch_in, ch_out, 3, padding=1))

            # Append the level to the list of downsample levels
            self.down_layers.append(level_layers)

        # Initialize the middle layers of unet
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim)
        self.mid_attn = MultiheadedAttentionFM(
            heads=attn_heads[-1], dim_head=attn_chh[-1]
        )
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim)

        # Reverse unet level parameters for the upsampling process
        in_out_ch, attn_heads, attn_chh, attn_levels = (
            in_out_ch[::-1],
            attn_heads[::-1],
            attn_chh[::-1],
            attn_levels[::-1],
        )

        # Initialize the decoder layers of unet (upsampling)
        for unet_layer, (
            (ch_in, ch_out),
            layer_attn_heads,
            layer_attn_ch,
            attn,
        ) in enumerate(zip(in_out_ch, attn_heads, attn_ch, attn_levels)):
            level_layers = nn.ModuleList(
                [
                    # Mid layers use ch_out and down layers use ch_in therefore we need
                    # ch_out+ch_in channels for channel-wise concatenation
                    ResnetBlock(ch_out + ch_in, ch_out),
                    ResnetBlock(ch_out + ch_in, ch_out),
                    MultiheadedAttentionFM(
                        ch_out,
                        dim_head=layer_attn_ch,
                        heads=layer_attn_heads,
                    )
                    if attn
                    else nn.Identity(),
                ]
            )

            # Upsample feature maps by a factor of 2 if not the last unet decoder level
            if unet_layer != (num_resolutions - 1):
                level_layers.append(Upsample(ch_in, ch_out))
            else:
                level_layers.append(nn.Conv2d(ch_out, ch_in, 3, padding=1))

            self.up_layers.append(level_layers)

        self.out_ch = image_ch * 1  # (1 if not learned_variance else 2)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim)

        # Final convolution to return to rgb channels
        self.final_conv = nn.Conv2d(init_dim, self.out_ch, 1)

    def forward(self, x, time, x_self_cond=None):
        """Forward pass of Unet

        Args:
            x: Preprocessed image input to unet (b, c, h, w)
            time: Noise time embeddings (b, dim) TODO verify this shape
        """

        # TODO: Maybe put input image divisble check

        x = self.init_conv(x)
        r = x.clone()

        # Project the noise time embedding (b, time_dim)
        time = self.time_mlp(time)

        feature_maps = [x]

        # Encoder layers
        for block1, block2, attn, downsample in self.down_layers:
            x = block1(x, time)

            # Appends the feature maps so they can be concatenated during upsampling
            feature_maps.append(x)

            x = block2(x, time)
            x = attn(x) + x
            feature_maps.append(x)

            x = downsample(x)

        # Middle of Unet
        x = self.mid_block1(x, time)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time)

        # Decoder layers
        for block1, block2, attn, upsample in self.self.up_layers:
            x = torch.cat((x, feature_maps.pop()), dim=1)
            x = block1(x, time)

            x = torch.cat((x, feature_maps.pop()), dim=1)
            x = block2(x, time)
            x = attn(x) + x

            x = upsample(x)

        # TODO: remove assert once I know it passes
        assert torch.allclose(feature_maps[0] == r)

        x = torch.cat((x, feature_maps.pop()), dim=1)
        x = self.final_res_block(x, time)
        assert len(feature_maps) == 0

        # Restore original input channels
        return self.final_conv(x)
