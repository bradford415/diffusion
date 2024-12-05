# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
import glob
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CelebFacesA(Dataset):
    """TODO.

    Dataset can be found here https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.
    """

    def __init__(
        self, dataset_root: str, dataset_split: str = "train", transforms: T = None
    ):
        """Initialize the CelebFacesA Dataset

        Args:
            dataset_root: Path to the dataset root
            dataset_split: which dataset split to use; `train`, `val`, `test`
        """
        self._transforms = transforms
        self._images = _get_file_paths(dataset_root, dataset_split)

    def __getitem__(self, index) -> Image:
        """Retrieve and preprocess samples from the dataset"""

        _image = Image.open(self._images[index]).convert("RGB")

        # Preprocess the input data before passing it to the model
        if self._transforms is not None:
            _image = self._transforms(_image)

        return _image

    def __len__(self):
        """Returns dataset length"""
        return len(self._images)


def _make_celebfacesa_transforms_old(dataset_split: str) -> T:
    """Initialize transforms for the CelebFacesA dataset.

    Args:
        dataset_split: which dataset split to use; `train`, `val`, `test`

    """
    # Normalize data into the range [-1, 1]; specified in DCGAN paper
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img_size = (64, 64)

    if dataset_split == "train":
        return T.Compose(
            [
                T.Resize(img_size),
                T.CenterCrop(
                    img_size
                ),  # this doesn't do anything if the img_size are the same
                normalize,
            ]
        )

    if dataset_split == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {dataset_split}")


def make_celebfacesa_transforms(
    dataset_split,
    orig_size: tuple[int, int] = [256, 256],
    resize_size: Union[int, tuple] = None,
    crop_size: Union[int, tuple, None] = None,
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


def _get_file_paths(dataset_root: str, split: str):
    """TODO

    Args:
        dataset_root: Path to the dataset root dir
        split: `train`, `val`, or `test`
    """
    dataset_path = Path(dataset_root)

    if split == "train":
        images_root = dataset_path / "*"

    image_paths = glob.glob(f"{images_root}")
    return image_paths


def build_CelebFacesA(
    dataset_root: str,
    dataset_split: str = "train",
) -> Dataset:
    """Initializes the CelebFacesA dataset class

    Args:
        root: Full path to the dataset root
        split: which dataset split to use; `train`, `val`, or `test`
    """
    coco_root = Path(dataset_root)

    # Create the data augmentation transforms
    data_transforms = _make_celebfacesa_transforms(dataset_split)

    dataset = CelebFacesA(
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        transforms=data_transforms,
    )

    return dataset