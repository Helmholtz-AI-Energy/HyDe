from typing import Tuple

import numpy as np
import torch.nn as nn

from ..lowlevel import logging, utils

logger = logging.get_logger()

__all__ = [
    "crop_center",
    "MultipleWeightedLosses",
    "SAMLoss",
]


def crop_center(image: np.ndarray, crop_size: Tuple):
    """
    Crop the center of an image to make uniform sizes before saving.
    Expected shape for image: [..., bands, rows, cols]

    Parameters
    ----------
    image
    crop_size

    Returns
    -------

    """
    sl = [
        slice(None),
    ] * image.ndim
    strow = image.shape[-2] // 2 - (crop_size[-2] // 2)
    stcol = image.shape[-1] // 2 - (crop_size[-1] // 2)
    sl[-2] = slice(strow, strow + crop_size[-2])
    sl[-1] = slice(stcol, stcol + crop_size[-1])
    return image[sl]


class MultipleWeightedLosses(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleWeightedLosses, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return "weight={}".format(self.weight)


class SAMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return utils.sam(input, target).mean()
