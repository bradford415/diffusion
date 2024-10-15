# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# The primary use of overwriting the transforms is to handle
# the bounding box transformations as well
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL.Image import Image as PILImage


class Unnormalize(object):
    """Unormalize a tensor that normalized by torchvision.transforms.Normalize

    Normalize subtracts mean and divides by std dev so to Unnormalize we need to
    multiply by the std dev and add the mean
    """

    def __init__(self, mean: List[float], std: List[float], inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor: Tensor image of size (C, H, W) to be unnormalized.
        Returns:
            Tensor: Normalized image.
        """
        if not self.inplace:
            tensor = tensor.clone()

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
