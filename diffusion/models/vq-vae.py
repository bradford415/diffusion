from torch import nn


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
    """Encoder module for the VQ-VAE model"""
    
    def __init__(self, in_channels, resolution, z_channels, ch, out_ch, ch_mult = (1, 2, 4, 8), num_res_blocks, attn_resoluion, dropout=0.0, resample_with_conv=True):
        """TODO
        
        Args:
        
        """
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        # STTTTTTAAAARRRTTTTTTT HEEEEEEEEERRREEEEE
        
        
    def forward(self, x):
        """TODO"""
        pass