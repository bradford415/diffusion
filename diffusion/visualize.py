from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image


def save_images_mpl(generated_images: np.ndarray, sqrt_num: int, save_name: str):
    """TODO"""
    sqrt_num = int(sqrt_num)
    fig, axs = plt.subplots(sqrt_num, sqrt_num)

    for index, ax in enumerate(axs.flat):
        ax.imshow(generated_images[index])
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.47, hspace=0.05)
    fig.savefig(str(save_name), bbox_inches="tight", dpi=300)
    plt.close()


def save_images_pil(generated_images: np.ndarray, save_name: str):
    """TODO"""
    for index, image in enumerate(generated_images):
        pil_img = Image.fromarray(image)
        pil_img.save(Path(save_name) / f"gen_image_pil_{index}.png")
