# There are functions for creating a train and validation iterator.

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms

from ...lowlevel import logging, utils
from .. import general_nn_utils

logger = logging.get_logger()

__all__ = ["ICVLDataset"]


class ICVLDataset(Dataset):
    def __init__(
        self,
        datadir,
        crop_size=(512, 512),
        target_transform=None,
        common_transforms=None,
        easy_transform=None,
        medium_transform=None,
        last_transform=None,
        val=False,
        net2d=False,
    ):
        super(ICVLDataset, self).__init__()
        datadir = Path(datadir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".npy")]
        if dist.is_initialized():
            random.shuffle(self.files)

        # load all the data at the top
        # first = list(np.load(self.files[0]).shape) #.transpose((1, 2, 0)).shape)
        # first.insert(0, len(self.files))
        # print("first", first)

        self.loadfrom = []  # np.zeros(first, dtype=np.float32)
        for c, f in enumerate(self.files):
            # print(f, np.load(f).shape)
            loaded, _ = utils.normalize(torch.tensor(np.asarray(np.load(f), dtype=np.float32)))
            self.loadfrom.append(loaded)

        self.loadfrom = tuple(self.loadfrom)

        self.base_transforms = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.RandomApply(
                    [
                        general_nn_utils.RandRot90Transform(),
                        transforms.RandomVerticalFlip(p=0.5),
                    ],
                    p=0.75,
                ),
                transforms.RandomCrop(crop_size),
            ]
        )
        self.stage = 0

        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.length = len(self.files)

        self.easy_transform = easy_transform
        self.medium_transform = medium_transform
        self.last_transform = last_transform

        self.val = val

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # logger.info(f'loading file: {self.files[idx]}')
        # return None
        # img = np.load(self.files[idx]).transpose((2, 3, 1))
        img = self.loadfrom[idx].unsqueeze(0)
        # print(img.shape)
        if not self.val:
            img = self.base_transforms(img)

        if self.common_transforms is not None:
            img = self.common_transforms(img)
        target = img.clone().detach()
        # logger.info("after clone")

        if self.val or (self.stage == 0 and self.easy_transform is not None):
            img = self.easy_transform(img)
        elif self.stage == 1 and self.medium_transform is not None:
            img = self.medium_transform(img)
        else:
            img = self.last_transform(img)

        # logger.info("after transform")
        if self.target_transform is not None:
            target = self.target_transform(target)
        # logger.info("after target transform")

        return img, target
