from pathlib import Path
from typing import Union

from torchvision.datasets import CIFAR10, CIFAR100

from diffusion.data.cifar import make_cifar_transforms
from diffusion.data.lsun import LSUNBedrooms, build_lsun_transforms

dataset_map = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "lsun_bedroom": LSUNBedrooms,
}


def create_dataset(
    dataset_name: str, split: str, dev_mode: bool = False, **dataset_params: dict
):
    """Builds the desired dataset by handling the dataset specific parameters

    Args:
        dataset_name: the name of the dataset to create
        split: the dataset split to use; `train` or `val`
        dev_mode:
        dataset_params: the parameters specific to the dataset
    """
    if dataset_name == "cifar10":
        dataset = _build_cifar(dataset_name, split, **dataset_params)
    if dataset_name == "lsun_bedroom":
        dataset = _build_lsun(dataset_name, split, dev_mode=dev_mode, **dataset_params)
    elif dataset_name not in dataset_map:
        raise ValueError(f"dataset {dataset_name} not recognized.")

    return dataset


def _build_lsun(
    dataset_name: str,
    dataset_split: str,
    root: str,
    size: int = 256,
    dev_mode: bool = False,
):
    """Initialize the lsun dataset

    Args:
        dataset_name: the name of the dataset to create
        split: which dataset split to use; `train` or `val`
        size: size of the input images; images are first center-cropped, then resized if needed
    """
    transforms = build_lsun_transforms(dataset_split=dataset_split, size=size)

    dataset = dataset_map[dataset_name](
        dataset_split, root=root, transforms=transforms, dev_mode=dev_mode
    )

    return dataset


def _build_cifar(
    dataset_name: str,
    dataset_split: str,
    root: str = "../",
    orig_size=32,
    resize_size=None,
    debug_mode: bool = False,
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
    data_transforms = make_cifar_transforms(
        dataset_split, orig_size=orig_size, resize_size=resize_size
    )

    dataset_args = {
        "root": dataset_root,
        "train": dataset_split == "train",
        "download": True,
        "transform": data_transforms,
    }

    dataset_map[dataset_name]

    # TODO: manipulate dataset for debug mode

    return dataset
