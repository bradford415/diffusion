import os
from pathlib import Path
from typing import Union

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class LSUNBase(Dataset):
    """Parent class for the LSUN datasets"""

    def __init__(
        self,
        root: str,
        split_txt_file: str = "",
        transforms: T = None,
        dev_mode: bool = False,
    ):
        """Initialize the base class for the LSUN datasets
        Args:
            root: the root directory of the dataset
            split_txt_file: relative path, from the data root, to the split text file from the latent diffusion repo found here:
                            https://github.com/CompVis/latent-diffusion/tree/main?tab=readme-ov-file#lsun
            transforms: transformations to apply to the images

        Args:
            TODO
        """
        self.data_root = root
        self.data_paths = Path(root) / split_txt_file

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()

        # Crate a dictionary with keys: rel file_path and abs file_path
        # and values: a list of the corresponding file paths
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }

        self.transforms = transforms

        if dev_mode:
            self.image_paths = self.image_paths[:16]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        """Retrieve and preprocess an image from the dataset"""
        # TODO Comment and make this more readable
        example = dict((k, self.labels[k][i]) for k in self.labels)
        _image = Image.open(example["file_path_"]).convert("RGB")

        # Preprocess the input data before passing it to the model
        if self._transforms is not None:
            _image = self._transforms(_image)

        # TODO: see if this dict is necessary
        example["image"] = _image
        return example


class LSUNBedrooms(LSUNBase):
    """LSUN bedrooms dataset class"""

    def __init__(self, split: str, **base_kwargs):
        """Initialize the LSUN bedrooms dataset"""
        if split == "train":
            split_txt_file = "lsun/bedrooms_train.txt"
        elif split == "val":
            split_txt_file = "lsun/bedrooms_val.txt"
        else:
            raise ValueError(f"unknown dataset split {split}")

        super().__init__(
            split_txt_file=split_txt_file,
            **base_kwargs,
        )


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="data/lsun/church_outdoor_train.txt",
            data_root="data/lsun/churches",
            **kwargs,
        )


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/lsun/church_outdoor_val.txt",
            root="data/lsun/churches",
            flip_p=flip_p,
            **kwargs,
        )


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs
        )


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/lsun/cat_val.txt",
            data_root="data/lsun/cats",
            flip_p=flip_p,
            **kwargs,
        )


def build_lsun_transforms(
    dataset_split,
    size: Union[int, tuple] = 256,
    horizontal_flip=0.5,
    interpolation="bicubic",
):
    """Initialize transforms for the lsun dataset

    Args:
        dataset_split: the dataset split to use; `train` or `val`
        size: desired image size for training and generation (h, w); if scalar, the smaller
                     image dimension will be resized to this value keeping the aspect ratio;
                     if tuple both dimensions will be resized to this value but the aspect ratio
                     will most likely not be maintained
        horizontal_flip: probability of the image being horiztonally flip; use 0.0 to disable horizontal flipping
        interpolation: interpolation method to use for resizing


    """
    _interpolation = {
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
    }[interpolation]

    # Convert to tensor and normalize between [-1,1];
    # NOTE: this normalization is the same as the original implementation (i.e., img // 127.5 - 1)
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Crop first to maintain aspect ratio if possible; many images in lsun bedroom are 256 x n where n > 256
    if dataset_split == "train":
        return T.Compose(
            [
                T.CenterCrop(size),
                T.Resize(size),
                T.RandomHorizontalFlip(p=horizontal_flip),
                normalize,
            ]
        )
    elif dataset_split == "val":
        return [
            T.CenterCrop(size),
            T.Resize(size, _interpolation),
            normalize,
        ]
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")
