from .ddpm import DDPM
from .unet import Unet

from .autoencoder import AutoencoderKL

vae_map = {"autoencoder_kl": AutoencoderKL}