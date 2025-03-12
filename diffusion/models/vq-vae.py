from torch import nn

from diffusion.models.layers import ResBlock, Downsample, AttnBlock


class VQModel(nn.module):
    """Latent diffusion autoencoder based on VQ-VAE (Vector Quantized Variational Autoencoder)

    Used to encode the input images to a lower-dimensional latent space and then decode them
    back to the original image space after performing diffusion.
    """

    def __init__(self, embed_dim, n_embed):
        """Initialize the VQ Autoencoder

        Args:
            embed_dim: TODO
            n_embed: TODO
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        pass


class Encoder(nn.Module):
    """Encoder module for the VQ-VAE model

    NOTE: no attention is used in the encoder downsampling based on this configuration:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/celebahq-ldm-vq-4.yaml#L55
    """

    def __init__(
        self,
        *,
        in_channels: int,
        resolution,
        ch: int,
        out_ch: int,
        ch_mult: list[int] = [1, 2, 4, 8],
        num_res_blocks,
        z_channels,
        attn_resoluion,
        dropout=0.0,
        resample_with_conv=True
    ):
        """TODO

        Args:
            in_channels: number of channels in the input image (e.g., 3 for RGB)
            resolution: TODO
            ch: TODO
            ch_mult: multiplier for ch to determine the number of channels in subsequent layers
            num_res_blocks: number of residual blocks in each resolution
            z_channels: number of channels in the latent/embedding space;
                        NOTE: z_channels is hardcoded to multiply by 2 at the end of this module
                              based on the implementation, so z_channels will actually be z_channels * 2

        """
        super().__init__()
        self.ch = ch
        num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Prepend a 1 to the ch_mult allowing us to loop over the resolutions in a nice way
        ch_mult = [1] + ch_mult
        channels_list = [ch * mult for mult in ch_mult]

        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        # Stores the main encoder blocks
        self.down = nn.ModuleList()

        for res_i in range(num_resolutions):
            # TODO comment
            res_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                res_blocks.append(
                    ResBlock(in_ch=ch, out_ch=ch * ch_mult[res_i]), dropout=0.0
                )
                ch = ch * ch_mult[res_i]

            # Create the resolution block module
            down = nn.Module()
            down.block = res_blocks

            # Downsample at the end of each resolution block except the last
            if res_i != num_resolutions - 1:
                down.downsample = Downsample(ch)
            else:
                down.downsample = nn.Identity()

            self.down.append(down)

        # Final ResBlocks with attention; ch is the output channels after downsampling
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(ch, ch, dropout=0.0)
        self.mid.atten_1 = AttnBlock(ch)
        self.mid.block_2 = ResBlock(ch, ch, dropout=0.0)

        # Embed the feature maps to (b, 2*z_channels, h, w)
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )  # can make a wrapper fn if the num_groups needs to change
        self.conv_out = nn.Conv2d(
            ch, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

        ####### START HERE, implement forward

    def forward(self, x):
        """TODO"""

        pass
