import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image


def save_gen_images(generated_images: np.ndarray, sqrt_num: int, save_name: str):
    """TODO"""
    # image_grid = (make_grid(generated_images) + 1) / 2
    # save_image(image_grid, save_name)
    sqrt_num = int(sqrt_num)
    fig, axs = plt.subplots(sqrt_num, sqrt_num)

    for index, ax in enumerate(axs.flat):
        ax.imshow(generated_images[index])
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.47, hspace=0.05)
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    plt.close()
