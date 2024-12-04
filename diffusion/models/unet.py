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

from diffusion.models.layers import (AttnBlock, Downsample,
                                     MultiheadedAttentionFM, ResBlock,
                                     TimestepEmbedSequential, Upsample)
from diffusion.models.positional import SinusoidalPosEmb


class Unet(nn.Module):
    """Unet model to be trained for diffusion during the reverse process

    In DDPM the unet model is used to predict the noise, epsilon, added to an image
    s

    This is the only training that is performed in diffusion.
    """

    def __init__(
        self,
        ch=64,
        ch_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        image_ch=3,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_ch=128,
        attn_heads=4,
        attn_levels=[False, True, False, False],
    ):
        """Initialize UNet model

        The goal of unet in the ddpm paper is to predict the noise (epsilon) added to the image
        shown on line 4 in algorithm 1; eps ~ N(0, I) where N is the normal distribution and I
        is the variance based on the variance schedule (I'm pretty sure).

        Args:
            dim: base dimension for the unet model; this will be multiplied by dim_mults
            dim_mults: Multiplier for dim which sets the number of channels in the unet model
            num_res_blocks: number of residual blocks per unet level; this is typically 2
            img_ch: Number of channels in the original image and output image; rgb=3 grayscale=1
            sinusoidal_pos_emb_theta: constant used in transfomrmer paper for postional embeddings
            dropout: probability of a tensor element to be zeroed during training;
                     lucidrains uses 0.0 i.e., no dropout
            attn_ch: Total channels for MHA; attn_ch will be split across
                       num_heads (attn_ch // num_heads) after it's projected
            attn_heads: Number of heads in mult-head attntion
            attn_levels: The levels of UNet to perform attention after; the default parameters apply attention
                         only to the last level (i.e., before the middle layers)
        """
        super().__init__()

        # determine dimensions
        self.channels = image_ch
        input_channels = image_ch * 1  # (2 if self_condition else 1)

        init_ch = ch

        # Initial 7x7 conv in ResNet
        self.init_conv = nn.Conv2d(input_channels, init_ch, kernel_size=7, padding=3)
        # self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=3, padding=1)

        # Create the channels of the model (5,)
        dims = [init_ch, *map(lambda m: ch * m, ch_mults)]

        # Pair the channels for each ResnetBlock
        # Ex: dims = [64, 64, 128, 256, 512] -> in_out_ch = [(64, 64), ..., (256, 512)]
        in_out_ch = list(zip(dims[:-1], dims[1:]))

        # Dimension of the noise time embeddings
        time_dim = ch * 4

        # Initialize postional embeddings for the noise timesteps
        sinu_pos_emb = SinusoidalPosEmb(ch, theta=sinusoidal_pos_emb_theta)

        # Positional embedding module
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Number of Unet levels
        num_stages = len(ch_mults)

        if len(attn_levels) != num_stages:
            raise ValueError(
                "Length of attn_levels should be the same as the length of dim_mults"
            )

        # Attention heads and ch for each level; attention is only used in the unet level
        # if attn_levels i
        attn_heads = (attn_heads,) * num_stages
        attn_ch = (attn_ch,) * num_stages

        num_resolutions = len(in_out_ch)

        # keep track of each output channel during downsampling to concat during upsampling
        # Initialize the encoder layers of unet (downsampling)
        # NOTE: the parallelism here looks nice however it's very sensitive to architecture changes and I would not recommend it in the future
        # for level, (
        #     (ch_in, ch_out),
        #     layer_attn_heads,
        #     layer_attn_ch,  # currently this isn't used; attn_dim is just the output_dim of the ResBlocks
        #     attn,
        # ) in enumerate(zip(in_out_ch, attn_heads, attn_ch, attn_levels)):
        self.down_layers = nn.ModuleList()
        chs_down = [ch]
        curr_ch = ch
        for level, mult in enumerate(ch_mults):
            # Whether to perform attention for the current Unet level;
            # the default parameters only use attention for the last Unet level (before the middle layers)
            ch_out = mult * ch
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(curr_ch, ch_out, time_emb_dim=time_dim, dropout=dropout),
                    MultiheadedAttentionFM(
                        embed_ch=ch_out, num_heads=attn_heads[level]
                    ),
                ]  # TODO might need to make attention conditional
                self.down_layers.append(TimestepEmbedSequential(*layers))
                curr_ch = ch_out
                chs_down.append(curr_ch)

            # level_layers = nn.ModuleList(
            #     [

            #         ResBlock(ch_out, ch_out, time_emb_dim=time_dim, dropout=dropout),
            #         (
            #             # NOTE: this implementation uses the output of resnetblock as the dimension for attnetion;
            #             #       the lucidrains implementation has a seperate parameter to control the attention dim
            #             MultiheadedAttentionFM(
            #                 embed_ch=ch_out, num_heads=layer_attn_heads
            #             )
            #             # AttnBlock(in_ch=ch_in)
            #             if attn
            #             else nn.Identity()
            #         ),
            #     ]
            # )

            # Downsample by a factor of 2 if not the last unet level
            if level != (num_resolutions - 1):
                self.down_layers.append(TimestepEmbedSequential(Downsample(curr_ch)))
                chs_down.append(curr_ch)
            # else:
            #     self.down_layers.append(
            #         #    nn.Identity()
            #         TimestepEmbedSequential(nn.Conv2d(curr_ch, curr_ch, 3, padding=1))
            #     )
            #     chs_down.append(curr_ch)

            # Append the level to the list of downsample levels
            # self.down_layers.append(level_layers)

        # Initialize the middle layers of unet
        mid_dim = curr_ch  # dims[-1]
        self.mid_block1 = ResBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, dropout=dropout
        )
        self.mid_attn = MultiheadedAttentionFM(
            embed_ch=mid_dim, num_heads=attn_heads[-1]
        )
        # self.mid_attn = AttnBlock(in_ch=mid_dim)
        self.mid_block2 = ResBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, dropout=dropout
        )

        # Reverse unet level parameters for the upsampling process
        in_out_ch, attn_heads, attn_ch, attn_levels = (
            in_out_ch[::-1],
            attn_heads[::-1],
            attn_ch[::-1],
            attn_levels[::-1],
        )

        # # Initialize the decoder layers of unet (upsampling)
        # for level, (
        #     (ch_in, ch_out),
        #     layer_attn_heads,
        #     layer_attn_ch,
        #     attn,
        # ) in enumerate(zip(in_out_ch, attn_heads, attn_ch, attn_levels)):
        self.up_layers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mults))):
            ch_out = mult * ch
            for res_index in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        curr_ch + chs_down.pop(),
                        ch_out,
                        time_emb_dim=time_dim,
                        dropout=dropout,
                    ),
                    MultiheadedAttentionFM(
                        embed_ch=ch_out, num_heads=attn_heads[level]
                    ),
                ]  # TODO: might have to make attention conditional
                curr_ch = ch_out
            # level_layers = nn.ModuleList(
            #     [
            #         # Mid layers use ch_out and down layers use ch_in therefore we need ch_out+ch_in channels
            #         # for channel-wise concatenation; the concatenation is done in forward()
            #         ResBlock(
            #             ch_out + ch_in,
            #             ch_out,
            #             time_emb_dim=time_dim,
            #             dropout=dropout
            #             # ch_in + ch_in, ch_in, time_emb_dim=time_dim, dropout=dropout
            #         ),
            #         MultiheadedAttentionFM(
            #             embed_ch=ch_out,
            #             num_heads=layer_attn_heads,
            #         ),
            #         ResBlock(
            #             ch_out + ch_in,
            #             ch_out,
            #             time_emb_dim=time_dim,
            #             dropout=dropout
            #             # ch_in + ch_in, ch_in, time_emb_dim=time_dim, dropout=dropout
            #         ),
            #         (
            #             MultiheadedAttentionFM(
            #                 embed_ch=ch_out,
            #                 num_heads=layer_attn_heads,
            #             )
            #             # AttnBlock(in_ch=ch_out)
            #             if attn
            #             else nn.Identity()
            #         ),
            #     ]
            # )

                # Upsample feature maps by a factor of 2 if not the last unet decoder level
                if level != 0 and res_index == num_res_blocks:
                    layers.append(Upsample(ch_out))
                    # level_layers.append(Upsample(ch_in, ch_in))
                # else:
                #     layers.append(
                #         # nn.Identity()
                #         nn.Conv2d(ch_out, ch_out, 3, padding=1)
                #     )
                self.up_layers.append(TimestepEmbedSequential(*layers))
                

            # self.up_layers.append(level_layers)

        # Ensure all of the downsampling featuremaps were concatenated
        assert len(chs_down) == 0

        self.out_ch = image_ch * 1
        self.final_res_block = ResBlock(
            ch, ch, time_emb_dim=time_dim, dropout=dropout
        )

        # Final convolution to return to rgb channels
        self.final_conv = nn.Conv2d(ch, self.out_ch, 1)

    def forward(self, x, time):
        """Forward pass of Unet

        Args:
            x: Preprocessed image input to unet (b, c, h, w)
            time: Noise time embeddings (b, time_dim)
        """

        # TODO: Maybe put input image divisble check

        x = self.init_conv(x)

        # Project the noise time embedding (b, time_dim)
        time = self.time_mlp(time)

        feature_maps = [x]
        for block in self.down_layers:
            x = block(x, time)
            feature_maps.append(x)

        # # Encoder layers
        # for block1, attn1, block2, attn2, downsample in self.down_layers:
        #     x = block1(x, time)

        #     x = attn1(x)

        #     # Append the feature maps so they can be concatenated during upsampling
        #     feature_maps.append(x)

        #     x = block2(x, time)

        #     x = attn2(x)  # residual is added in the attention module

        #     feature_maps.append(x)

        #     x = downsample(x)

        # Middle of Unet
        x = self.mid_block1(x, time)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time)

        # # Decoder layers
        # for block1, attn1, block2, attn2, upsample in self.up_layers:
        #     x = torch.cat((x, feature_maps.pop()), dim=1)

        #     x = block1(x, time)

        #     x = attn1(x)

        #     x = torch.cat((x, feature_maps.pop()), dim=1)

        #     x = block2(x, time)
        #     x = attn2(x)

        #     x = upsample(x)
        for module in self.up_layers:
            x = torch.cat((x, feature_maps.pop()), dim=1)
            x = module(x, time)

        #x = torch.cat((x, feature_maps.pop()), dim=1)
        x = self.final_res_block(x, time)
        assert len(feature_maps) == 0

        # Restore original input channels
        return self.final_conv(x)
