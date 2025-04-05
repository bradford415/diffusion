import torch

from .config import *
from .schedulers import *

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,
}

solver_configs = {"resnet50_imagenet": kl_vae_f_4_d_8_lsun_bedroom}
