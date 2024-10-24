import matplotlib.pyplot as plt


def save_gen_images(generated_images, sqrt_num, save_name):
    """TODO"""
    fig, axs = plt.subplots(sqrt_num, sqrt_num)

    for index, ax in enumerate(axs.flat):
        ax.imshow(generated_images[index])
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.47, hspace=0.05)
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()
