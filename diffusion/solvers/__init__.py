import torch

from .config import *
from .schedulers import *

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

solver_configs = {"": kl_vae_f_4_d_8_lsun_bedroom}
