import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn

from diffusion.data.transforms import to_numpy_image
from diffusion.visualize import save_images_mpl, save_images_pil

log = logging.getLogger(__name__)


@torch.inference_mode
def sample(
    ema_model: nn.Module,
    output_dir: str,
    checkpoint_path: str,
    num_samples: int = 16,
    sample_batch_size: int = 4,
    device=torch.device("cpu"),
):
    """Denoises pure noise to generate and save images

    Args:
        ema_model: model to sample from; this will be the same as the diffusion model
                   the only difference is that the ema weights will be loaded into it
    """
    step = load_model(
        checkpoint_path=checkpoint_path, ema_model=ema_model, device=device
    )
    ema_model.eval()

    # TODO
    gen_images_output = Path(output_dir) / "samples"
    gen_images_output.mkdir(parents=True, exist_ok=True)

    # Split the number of samples to generate into a list of batches
    sample_batch_sizes = num_samples_to_batches(num_samples, sample_batch_size)
    log.info(
        "Generating %d images using the following batch sizes: %s",
        num_samples,
        sample_batch_sizes,
    )
    generated_images = []
    for index, batch_size in enumerate(sample_batch_sizes):
        log.info("Processing batch %d/%d", index + 1, len(sample_batch_sizes))
        generated_images.append(ema_model.sample_generation(batch_size=batch_size))

    all_images = torch.cat(generated_images, dim=0)

    all_images = to_numpy_image(all_images)

    # for index, image_set in enumerate(generated_images):
    save_images_mpl(
        all_images,
        num_samples**0.5,
        str(gen_images_output / "generated_images.png"),
    )
    
    save_images_pil(
        all_images,
        str(gen_images_output),
    )


def load_model(
    checkpoint_path: str,
    diffusion_model: nn.Module = None,
    optimizer: nn.Module = None,
    ema_model: nn.Module = None,
    device=torch.device("cpu"),
):
    """Load the ddpm model to resume training or generate new images from

    Args:
        checkpoint_path: path to the weights file to resume training from
        diffusion_model: the diffusion model being trained
        optimizer: the optimizer used during training
        ema_model: ema which is used for the sampling process
        current_step: the current step the training is on when
                        the model is saved
    """
    # Load the torch weights
    weights = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # load the state dictionaries for the necessary training modules
    if diffusion_model is not None:
        diffusion_model.load_state_dict(weights["model"])
    if optimizer is not None:
        optimizer.load_state_dict(weights["optimizer"])
    if ema_model is not None:
        ema_model.load_state_dict(weights["ema_model"])
    start_step = weights["step"]

    return start_step

def num_samples_to_batches(num_samples: int, batch_size, ):
    """Create a list of batch sizes and the remaining batch size at the last index;
    this is useful to pass the number of eval samples by batch

    Example: num_samples = 25 and batch_size = 16 -> [16, 9]

    Args:
        num_samples: number of samples to generate images of
    """
    groups = num_samples // batch_size
    remainder = num_samples % batch_size
    batch_arr = [batch_size] * groups
    if remainder > 0:
        batch_arr.append(remainder)
    return batch_arr