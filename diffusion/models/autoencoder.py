import torch
from torch import nn

from diffusion.models.layers import AttnBlock, Downsample, ResBlock, Upsample


class AutoencoderKL(nn.module):
    """Latent diffusion autoencoder based on KL-divergence VAE; maps to continous latent space

    Used to encode the input images to a lower-dimensional, continuous latent space and then 
    decode them back to the original image space after performing diffusion.

    NOTE: TODO write briefly that this module uses 'soft'quantization but is still using
          the continous latent space

    KL-divergence VAE is a type of variational autoencoder that uses the KL-divergence
    loss to minimize the distance between the latent distribution and the prior distribution.

    """

    def __init__(
        self, *, embed_dim: int, n_embed: int, z_ch: int, double_z: bool = False
    ):
        """Initialize the KL VAE

        Args:
            embed_dim: TODO
            n_embed: TODO
            z_channels: number of channels in the latent/embedding space
            double_z: whether to double the number of channels in the latent space
        """
        super().__init__()

        assert double_z, "double_z should be True for the KL VAE"

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Conv to map from the emb space to the quantized emb space moments (mean & log variance);
        # for a probability distribution, 1st moment = mean 2nd moment = variance;
        # https://en.wikipedia.org/wiki/Moment_(mathematics);
        # emb_space = continuous space, quantized_emb_space = discrete space
        self.quant_conv = nn.Conv2d(2 * z_ch, 2 * embed_dim, kernel_size=1, stride=1)

        # Conv to map from quantized space back to embedding space
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, kernel_size=1, stride=1)

    def encode(self, img: torch.Tensor) -> "DiagonalGaussianDistribution":
        """Encode the image to a lower-dimensional latent space using the Encoder() module

        Args:
            img: the input image to encode to a latent space (b, c, h, w)
        """
        # Encode the image to a lower-dimensional latent space (b, c, h, w)
        z = self.encoder(img)

        # Get the moments of the quantized embedding space; mean and log variance
        moments = self.quant_conv(z)

        posterior = DiagonalGaussianDistribution(moments)
        
        return posterior
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent vector back to the image space using the Decoder() module

        Args:
            z: the latent vector (b, emb_ch, h, w); TODO maybe make shape more specific

        Returns:
            the reconstructed image from the latent space (b, c, h, w) where c=3 for rgb
        """
        #  Map to the embedding space from the quantized representation
        z = self.post_quant_conv(z)

        # Return to the image space
        img = self.decoder(z)
        
        return img

    def forward(self, x):
        pass


class VQModel(nn.module):
    """Latent diffusion autoencoder based on VQ-VAE (Vector Quantized Variational Autoencoder);
    maps to a discrete latent space

    This works by the encoder mapping images to a continuous latent representation, and, instead of
    using this continuous representation directly, the model matches each latent vector to its closest
    entry in the codebook (using nearest-neighbor lookup). This process quantizes the latent 
    representation into discrete indices referring to entries in the codebook. This codebook 
    is learned during training to optimize the recontruction quality. Finally. the decoder
    reconstructs theimage from the selected codebook vectors instead of using continuous latent
    values.


    Used to encode the input images to a lower-dimensional latent space and then decode them
    back to the original image space after performing diffusion.
    """

    def __init__(
        self, *, embed_dim: int, n_embed: int, z_ch: int, double_z: bool = False
    ):
        """Initialize the VQ Autoencoder

        Args:
            embed_dim: TODO
            n_embed: TODO
            z_channels: number of channels in the latent/embedding space
            double_z: whether to double the number of channels in the latent space
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Conv to map from the emb space to the quantized emb space moments (mean & log variance);
        # for a probability distribution, 1st moment = mean 2nd moment = variance;
        # https://en.wikipedia.org/wiki/Moment_(mathematics);
        # emb_space = continuous space, quantized_emb_space = discrete space
        self.quant_conv = nn.Conv2d(z_ch, embed_dim, kernel_size=1, stride=1)

        # Conv to map from quantized space back to embedding space
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, kernel_size=1, stride=1)

    def encode(self, img: torch.Tensor):
        """Encode the image to a lower-dimensional latent space using the Encoder() module

        Args:
            img: the input image to encode to a latent space (b, c, h, w)
        """
        # Encode the image to a lower-dimensional latent space (b, c, h, w)
        z = self.encoder(img)

        # GEt the moments of the quantized embedding space
        moments = self.quant_conv(z)
        
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError


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
        z_channels: int,
        double_z: bool,
        attn_resoluion,
        dropout=0.0,
        resample_with_conv=True
    ):
        """TODO

        Args:
            in_channels: number of channels in the input image (e.g., 3 for RGB)
            resolution: TODO
            ch: base channels to be multiplied by ch_mult
            ch_mult: multiplier for ch to determine the number of channels in subsequent layers
            num_res_blocks: number of residual blocks in each resolution
            z_channels: number of channels in the latent/embedding space;
            double_z: whether to double the number of channels in the latent space; this is typically True for
                      the KL autoencoder but False for the VQ autoencoder
            attn_resolution: TODO
            dropout: the dropout probability for each ResBlock; this is typically 0.0 for the encoder

        """
        super().__init__()
        self.ch = ch
        num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Double the latent chs if double_z is True
        z_channels = z_channels * 2 if double_z else z_channels

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
                    ResBlock(in_ch=ch, out_ch=channels_list[res_i + 1], dropout=dropout)
                )
                ch = channels_list[res_i + 1]

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
        self.mid.block_1 = ResBlock(ch, ch, dropout=dropout)
        self.mid.atten_1 = AttnBlock(ch)
        self.mid.block_2 = ResBlock(ch, ch, dropout=dropout)

        # Embed the feature maps to (b, 2*z_channels, h, w)
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=ch, eps=1e-6, affine=True
        )  # can make a wrapper fn if the num_groups needs to change
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv2d(
            ch, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, img: torch.Tensor):
        """Encodes the image into a lower dimensional latent space

        Args:
            img: the input image to encode to a latent space (b, c, h, w)
        """

        x = self.conv_in(img)

        # Loop through down blocks at every resolution; each down block has res_blocks and a downsample
        for down in self.down:

            for res_block in down.block:
                x = res_block(x)

            x = down.downsample(x)

        # Final encoder ResNet blocks w/ attention; no downsampling
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = self.activation(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """Decoder module for the VQ-VAE model

    Used to upsample the embedding
    """

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: list[int] = [1, 2, 4, 8],
        num_res_blocks: int,
        out_ch: int,
        z_channels: int,
        dropout: float = 0.0
    ):
        """Initializes the decoder module

        Args:
            ch: base channels to be multiplied by ch_mult
            ch_mult: multiplier for ch to determine the number of channels in subsequent layers
            num_res_blocks: number of residual blocks in each resolution
            out_ch: the number of channels in the output image (e.g., 3 for RGB)
            z_channels: number of channels in the latent/embedding space;
            attn_resolution: TODO
            dropout: the dropout probability for each ResBlock; this is typically 0.0 for the encoder

        """
        super().__init__()
        self.ch = ch
        num_resolutions = len(ch_mult)

        # NOTE: We don't need to prepend a 1 like in the Encoder
        channel_list = [m * ch for m in ch_mult]

        # The starting ch will begin in reverse order from the encoder
        ch = channel_list[-1]

        self.conv_in = nn.Conv2d(z_channels, ch, kernel_size=3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(ch, ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(ch)
        self.mid.block_2 = ResBlock(ch, ch, dropout=dropout)

        # Stores the main decoder blocks
        self.up = nn.ModuleList()

        for res_i in reversed(range(num_resolutions)):

            res_blocks = nn.ModuleList()
            # Decoder has an additional ResBlock at each resolution
            for _ in range(num_res_blocks + 1):
                res_blocks.append(
                    ResBlock(in_ch=ch, out_ch=channel_list[res_i], dropout=dropout)
                )
                ch = channel_list[res_i]

            up = nn.Module()
            up.block = res_blocks

            # Upsample at the end of each resolution except the last
            if res_i != 0:
                up.upsample = Upsample(scale_factor=2, mode="nearest")
            else:
                up.upsample = nn.Identity()

            # Prepend to be consistent with the checkpoint; I think this means the weights
            # layers are named in increasing order starting with the lowest res, since we prepend
            # we need to loop through these modules in reverse order
            # TODO: understand why this is necessary
            self.up.insert(0, up)

        # Map to image space; i.e., 3 channels for RGB
        self.activation = nn.SiLU()
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=ch, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z_emb):
        """Decode the latent vector back to the image space

        TODO: understand if its the denoised latent vector or the noise latent vector
              or something else

        Args:
            z_emb: the denoised latent vector; TODO: verify this is accurate and add a little more
        """

        x = self.conv_in(z_emb)

        # ResNet blocks w/ attention
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Loop through up blocks for each resolution; up
        for up in reversed(self.up):
            for res_block in up.block:
                x = res_block(x)

            x = up.upsample(x)

        # Normalie and map to image space
        x = self.norm_out(x)
        x = self.activation(x)
        x = self.conv_out(x)

        return x


class DiagonalGaussianDistribution(nn.Module):
    """TODO"""

    def __init__(self, parameters: torch.Tensor):
        """Initialize the DiagonalGaussianDistribution module

        A diagonal gaussian distribution is used mostly for computational efficiency;
        this assumes no correlation between latent variables.

        parameters: an embeded feature map which represents the mean and log variance of the distribution;
                    the feature maps are of shape (b, 2 * z_channels, h, w) where the first
                    half of the z_chs are the means and the second half are the log variances
        """
        super().__init__()
        
        # Extract the means and log variances from the parameters; clip the log variances
        self.mean, log_variance = torch.chunk(parameters, 2, dim=1)
        self.log_variance = log_variance.clamp(-30.0, 20.0)

        # Compute the std dev; reminder std_dev = variance^0.5 and torch.exp gets rid of the log
        self.std = torch.exp(0.5 * self.log_variance)

    def sample(self) -> torch.Tensor:
        """Sample from a gaussian distribution with the learned mean and std

        Creates a random tensor sampled from a standard normal distribution and scales it by the
        learned std dev and adds the learned mean; because a standard normal distribution has 
        std_dev=1 and mean=0, the multiplication scales the std_dev by 1 * self.std
        and addition adds the mean by 0 + self.mean

        This method implements the reparameterization trick used in VAEs and latent diffusion
        z=μ+σ⋅ϵ,ϵ~N(0,I) where I represents the identity matrix which indicates that the
        covariance structure is diagonal and independent.
         
        retuns:
            shape: (b, z_channels, h, w)
        """
        return self.mean + self.std * torch.randn_like(self.std)
