import datetime
import logging
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from diffusion.data.cifar import build_cifar
from diffusion.evaluate import sample
from diffusion.models import DDPM, Unet
from diffusion.models.layers import init_weights
from diffusion.trainer import Trainer
from diffusion.utils import reproduce

model_map: Dict[str, Any] = {"unet": Unet}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path: str = None):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: TODO: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    model_name = model_config["model_name"]

    # Initialize paths
    output_path = (
        Path(base_config["output_path"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )

    # NOTE: initially I was trying to make resuming training from the same directory as the checkpoint file
    #       but there was a lot of edge cases and making a new output_dir seems easiest
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    log.info("Initializing...\n")

    log.info("writing outputs to %s", str(output_path))

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))
    else:
        log.info("Using CPU")

    # Extract the train arguments from base config
    sample_args = base_config["sampling"]

    # Initalize models
    denoise_model = Unet(**model_config["model_params"][model_name]).to(device)
    
    diffusion_model = DDPM(
        denoise_model, device=device, **model_config["model_params"]["ddpm"]
    ).to(device)

    # Assign the image size to generate'
    img_size = base_config["sample_size"]
    if isinstance(img_size, int):
        img_size = [img_size, img_size]
    diffusion_model.image_size = img_size

    # Compute and log the number of params in the model
    reproduce.model_info(diffusion_model)

    # Save configuration files
    reproduce.save_configs(
        config_dicts=[base_config, model_config],
        save_names=["base_config.yaml", "model_config.yaml"],
        output_path=output_path / "reproduce",
    )

    sampling_args = {
        "ema_model": diffusion_model,
        "output_dir": output_path,
        "device": device,
        **sample_args,
    }
    sample(**sampling_args)


if __name__ == "__main__":
    Fire(main)
