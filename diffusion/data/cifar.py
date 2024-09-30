# CIFAR 10/100 dataset copied from here: https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
# Very slightly modified to keep consistent with the API I have been using
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from diffusion.data.transforms import UnNormalize

# TODO: find a better place to put this (maybe return with make_cifa_transforms?)
reverse_transforms = T.Compose(
    [
        UnNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        T.Lambda(lambda t: t * 255.0),  # [0,1] -> [0, 255]
        T.Lambda(lambda t: t.numpy().astype(np.uint8)),
        T.toPILImage(),
    ]
)


def make_cifar_transforms(
    dataset_split,
    resize_size: Union[int, Tuple] = 128,
    crop_size: Union[int, Tuple, None] = None,
    horizontal_flip=0.5,
):
    """Initialize transforms for the cifar dataset

    Args:
        dataset_split: which dataset split to use; `train` or `val`
        resize_size: Image size to resize the image to (h, w); if scalar, the smaller
                     image dimension will be resized to this value keeping the aspect ratio;
                     if tuple both dimensions will be resized to this value but the aspect ratio
                     will most likely not be maintained
        crop_size: Center crop size; for the default case resize and crop size are the same, therefore,
                   the center crop will have no effect; I believe this is here just in case the resize is
                   different
        horizontal_flip: Probability of the image being horiztonally flip; use 0.0 to disable horizontal flipping


    """
    # The default case uses resize and crop size as the same value
    if crop_size is None:
        crop_size = resize_size

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
        return T.compose(
            [
                T.Resize(resize_size),
                T.RandomHorizontalFlip(p=horizontal_flip),
                T.CenterCrop(crop_size),
                normalize,
            ]
        )
    elif dataset_split == "test":
        return normalize
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


def build_cifar(
    dataset_name: str, dataset_split: str, root: str = "../", debug_mode: bool = False
) -> Union[CIFAR10, CIFAR100]:
    """Initialize the cifar 10 or 100 dataset

    Args:
        split: which dataset split to use; `train` or `val`
        root: full path to the dataset root; for the cifar dataset this is the location the
              dataset will be downloaded initially
        debug_mode: Whether to build the dataset in debug mode; if true, this only uses a few samples
                    to quickly run the code
    """
    dataset_root = Path(root)

    # Create the data augmentation transforms
    data_transforms = make_cifar_transforms(dataset_split)

    dataset_args = {
        "root": dataset_root,
        "train": dataset_split == "train",
        "download": True,
        "transform": data_transforms,
    }
    if dataset_name == "cifar10":
        dataset = CIFAR10(**dataset_args)
    elif dataset_name == CIFAR100:
        dataset = CIFAR100(**dataset_args)
    else:
        raise ValueError("Dataset not recognized, must be cifar10 or cifar100")

    # TODO: manipulate dataset for debug mode

    return dataset
