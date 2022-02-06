# There are functions for creating a train and validation iterator.

import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms

from .. import general_nn_utils
from ...lowlevel import logging
logger = logging.get_logger()

__all__ = ["ICVLDataset"]


class ICVLDataset(Dataset):
    def __init__(
        self,
        datadir,
        transform,
        crop_size=(256, 256),
        target_transform=None,
        common_transforms=None,
        val=False,
        net2d=False,
    ):
        super(ICVLDataset, self).__init__()
        datadir = Path(datadir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".npy")]
        if dist.is_initialized():
            random.shuffle(self.files)

        # load all the data at the top
        #first = list(np.load(self.files[0]).shape) #.transpose((1, 2, 0)).shape)
        #first.insert(0, len(self.files))
        #print("first", first)

        self.loadfrom = [] #np.zeros(first, dtype=np.float32)
        for c, f in enumerate(self.files):
            #print(f, np.load(f).shape)
            self.loadfrom.append(np.asarray(np.load(f), dtype=np.float32))  #.transpose((1, 2, 0))
        self.loadfrom = tuple(self.loadfrom)

        self.base_transforms = transforms.Compose(
            [
                #transforms.ToTensor(),
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

        self.transform = transform
        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.length = len(self.files)
        self.val_transform = transforms.Compose(
            [
                #transforms.ToTensor(),
                transforms.RandomCrop(crop_size),
            ]
        )
        # if
        self.val = val

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #logger.info(f'loading file: {self.files[idx]}')
        #return None
        #img = np.load(self.files[idx]).transpose((2, 3, 1))
         
        img = torch.tensor(self.loadfrom[idx], dtype=torch.float).unsqueeze(0)
        #print(img.shape)
        if not self.val:
            img = self.base_transforms(img)
        else:
            img = self.val_transform(img)
        #logger.info("after base")
        
        # make img 5 dimensional...???
        # TODO: bfloat16??
        #img = img.unsqueeze(0).to(torch.float)
        
        if self.common_transforms is not None:
            img = self.common_transforms(img)
        target = img.clone().detach()
        #logger.info("after clone")
        if self.transform is not None:
            img = self.transform(img)
        #logger.info("after transform")
        if self.target_transform is not None:
            target = self.target_transform(target)
        #logger.info("after target transform")

        img = img.to(torch.bfloat16)
        target = img.to(torch.bfloat16)

        return img, target
