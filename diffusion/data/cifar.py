# CIFAR 10/100 dataset copied from here: https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
# Very slightly modified to keep consistent with the API I have been using
from pathlib import Path
from typing import Union

import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def make_cifar_transforms(dataset_split):
    """Initialize transforms for the coco dataset

    These transforms are based on torchvision transforms but are overrided in data/transforms.py
    This allows for slight modifications in the the transform

    Args:
        dataset_split: which dataset split to use; `train` or `val`

    """
    # Convert to tensor and normalize between [-1,1]
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    # For now, the train and test transforms are the same
    if dataset_split == "train":
        return normalize
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
