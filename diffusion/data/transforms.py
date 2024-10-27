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


class Unnormalize:
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
        This code is largely based on: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L905
        Args:
            tensor: Tensor image of size (B, C, H, W) or (C, H, W) to be unnormalized.
        Returns:
            Tensor: Unormalized image.
        """
        
        if tensor.ndim < 3:
            raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
            )
            
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        
        if not self.inplace:
            tensor = tensor.clone()
            
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        # Modifies in place
        tensor.mul_(std).add_(mean)
        # The normalize code -> t.sub_(m).div_(s)

        return tensor
    
# TODO: find a better place to put this (maybe return with make_cifa_transforms?)
reverse_transforms = T.Compose(
    [
        Unnormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1] -> [0, 1]
        T.Lambda(lambda t: t.permute(0, 2, 3, 1)),  # BCHW to BHWC
        T.Lambda(lambda t: t * 255.0),  # [0, 1] -> [0, 255]
        T.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
        #T.ToPILImage(),
    ]
)

