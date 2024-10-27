import matplotlib.pyplot as plt
import numpy as np


def save_gen_images(generated_images: np.ndarray, sqrt_num: int, save_name: str):
    """TODO"""
    sqrt_num = int(sqrt_num)
    fig, axs = plt.subplots(sqrt_num, sqrt_num)

    for index, ax in enumerate(axs.flat):
        ax.imshow(generated_images[index])
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.47, hspace=0.05)
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()
