from torchvision.datasets import CIFAR10, CIFAR100

from .lsun import LSUNBedrooms

dataset_map = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "lsun_bedroom": LSUNBedrooms,
}
