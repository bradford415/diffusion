# CIFAR 10/100 dataset copied from here: https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
# Very slightly modified to keep consistent with the API I have been using
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from torchvision import transforms as T

from diffusion.data.transforms import Unnormalize

unnormalize = T.Compose(
    [
        Unnormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def make_cifar_transforms(
    dataset_split,
    orig_size: Tuple[int, int] = [32, 32],
    resize_size: Union[int, Tuple] = None,
    crop_size: Union[int, Tuple, None] = None,
    horizontal_flip=0.5,
):
    """Initialize transforms for the cifar dataset

    Args:
        dataset_split: which dataset split to use; `train` or `val`
        orig_size: tuple, or int if square image, of the orginal height and width
        resize_size: Image size to resize the image to (h, w); if scalar, the smaller
                     image dimension will be resized to this value keeping the aspect ratio;
                     if tuple both dimensions will be resized to this value but the aspect ratio
                     will most likely not be maintained
        crop_size: Center crop size; for the default case resize and crop size are the same, therefore,
                   the center crop will have no effect; I believe this is here just in case the resize is
                   different
        horizontal_flip: Probability of the image being horiztonally flip; use 0.0 to disable horizontal flipping


    """
    if resize_size is None:
        resize_size = orig_size

    # The default case uses resize and crop size as the same value
    if crop_size is None:
        crop_size = orig_size

    # Convert to tensor and normalize between [-1,1];
    # NOTE: this normalization is the same as the original implementation (i.e., img * 2 -1)
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # For now, the train and test transforms are the same
    if dataset_split == "train":
        # resize and center crop are none by default
        return T.Compose(
            [
                T.Resize(resize_size),
                T.RandomHorizontalFlip(p=horizontal_flip),
                # T.CenterCrop(crop_size),
                normalize,
            ]
        )
    elif dataset_split == "test":
        return normalize
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")
