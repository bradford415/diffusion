import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Create sinusoidal position embeddings to for a noise timestep"""

    def __init__(self, dim, theta=10000):
        """Initialize the positional embedding module

        Args:
            dim: Dimension of the position embedding
            theta:
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, timestep):
        """Calculates the positional embedding for a specific position

        Implements the formula from "Attention is all you need"; however, the standard
        equation is not very efficient to compute due to the nested loops. We can use
        logarithmic properties to make this more efficient to compute in pytorch

        This article explains it well: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6

        Args:
            timestep: The position/timestep to encode (B,); this will be randomly sampled
                      from [0, num_timesteps]

        Return:
            (B, dim)
        """
        device = timestep.device
        half_dim = self.dim // 2

        # Generate all the divisors of the embeddings at once
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Calculate the position embeddings; this works since the propety above moves the denominator
        # to the numerator (-emb) so we can multiply by the timestep
        emb = timestep[:, None] * emb[None, :]  # (1, dim/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (1, dim)
        return emb
