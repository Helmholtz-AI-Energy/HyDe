# There are functions for creating a train and validation iterator.

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .. import general_nn_utils

__all__ = ["ICVLDataset"]


class ICVLDataset(Dataset):
    def __init__(
        self,
        datadir,
        transform,
        target_transform=None,
        common_transforms=None,
        val=False,
        net2d=False,
    ):
        super(ICVLDataset, self).__init__()
        datadir = Path(datadir)
        self.files = [datadir / f for f in os.listdir(datadir)]
        self.base_transforms = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        general_nn_utils.RandRot90Transform(),
                        transforms.RandomVerticalFlip(p=0.5),
                    ],
                    p=0.75,
                ),
                transforms.RandomCrop((1024, 1024)),
            ]
        )

        self.transform = transform
        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.length = len(self.files)
        # if
        self.val = val

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.tensor(np.load(self.files[idx]), dtype=torch.float).unsqueeze(0)
        if not self.val:
            img = self.base_transforms(img)
        if self.common_transforms is not None:
            img = self.common_transforms(img)
        target = img.clone()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
