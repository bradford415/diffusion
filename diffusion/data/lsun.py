import os
from typing import Union

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class LSUNBase(Dataset):
    def __init__(
        self,
        split: str,
        split_txt_file: str,
        dataset_root: str,
        interpolation="bicubic",
    ):
        """TODO
        
        Args:
            TODO
        """
        self.data_paths = split_txt_file
        self.data_root = dataset_root

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }

        _interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.transforms = build_lsun_transforms(split, size=256, interpolation=_interpolation)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        (
            h,
            w,
        ) = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


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
            data_root="data/lsun/churches",
            flip_p=flip_p,
            **kwargs,
        )


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **base_kwargs):
        super().__init__(
            txt_file="data/lsun/bedrooms_train.txt",
            dataset_root="data/lsun/bedrooms",
            **base_kwargs,
        )


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/lsun/bedrooms_val.txt",
            data_root="data/lsun/bedrooms",
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
    size: Union[int, tuple] = None,
    horizontal_flip=0.5,
):
    """Initialize transforms for the lsun dataset

    Args:
        dataset_split: the dataset split to use; `train` or `val`
        size: desired image size for training and generation (h, w); if scalar, the smaller
                     image dimension will be resized to this value keeping the aspect ratio;
                     if tuple both dimensions will be resized to this value but the aspect ratio
                     will most likely not be maintained
        horizontal_flip: probability of the image being horiztonally flip; use 0.0 to disable horizontal flipping


    """
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
                T.Resize(size),
                normalize,
            ]
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")
