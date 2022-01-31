# There are functions for creating a train and validation iterator.

import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

__all__ = ["GeneralImageDataset", "HsiToTensor", "LoadMatHSI", "LoadMatKey", "MatDataFromFolder"]


class GeneralImageDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(GeneralImageDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class HsiToTensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """

    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        return img.float()


class LoadMatKey(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, mat):
        item = mat[self.key][:].transpose((2, 0, 1))
        return item.astype(np.float32)


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform

    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key][:].transpose((2, 0, 1)))
            gt = self.transform(mat[self.gt_key][:].transpose((2, 0, 1)))
        else:
            input = mat[self.input_key][:].transpose((2, 0, 1))
            gt = mat[self.gt_key][:].transpose((2, 0, 1))
        # input = torch.from_numpy(input[None]).float()
        input = torch.from_numpy(input).float()
        # gt = torch.from_numpy(gt[None]).float()  # for 3D net
        gt = torch.from_numpy(gt).float()

        return input, gt


class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""

    def __init__(self, data_dir, load=loadmat, suffix="mat", fns=None, size=None):
        super(MatDataFromFolder, self).__init__()
        if fns is not None:
            self.filenames = [os.path.join(data_dir, fn) for fn in fns]
        else:
            self.filenames = [
                os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn.endswith(suffix)
            ]

        self.load = load

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):
        mat = self.load(self.filenames[index])
        return mat

    def __len__(self):
        return len(self.filenames)


# ======================= old code below ==============================================


# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)


# Define Transforms
# this will do ONLY ONE of the given transforms
# TODO: use random choice instead of this
# class SequentialSelect(object):
#     pass
#
#
# Define Datasets
# def get_train_valid_loader(
#     dataset,
#     batch_size,
#     train_transform=None,
#     valid_transform=None,
#     valid_size=None,
#     shuffle=True,
#     verbose=False,
#     num_workers=1,
#     pin_memory=False,
# ):
#     """
#     Utility function for loading and returning train and valid
#     multi-process iterators over any pytorch dataset. A sample
#     of the images can be optionally displayed.
#     If using CUDA, num_workers should be set to 1 and pin_memory to True.
#     Params
#     ------
#     - dataset: full dataset which contains training and validation data
#     - batch_size: how many samples per batch to load. (train, val)
#     - train_transform/valid_transform: callable function
#       applied to each sample of dataset. default: transforms.ToTensor().
#     - valid_size: should be a integer in the range [1, len(dataset)].
#     - shuffle: whether to shuffle the train/validation indices.
#     - verbose: display the verbose information of dataset.
#     - num_workers: number of subprocesses to use when loading the dataset.
#     - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
#       True if using GPU.
#     Returns
#     -------
#     - train_loader: training set iterator.
#     - valid_loader: validation set iterator.
#     """
#     error_msg = "[!] valid_size should be an integer in the range [1, %d]." % (len(dataset))
#     if not valid_size:
#         valid_size = int(0.1 * len(dataset))
#     if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
#         raise TypeError(error_msg)
#
#     # define transform
#     default_transform = lambda item: item  # identity maping
#     train_transform = train_transform or default_transform
#     valid_transform = valid_transform or default_transform
#
#     # generate train/val datasets
#     partitions = {"Train": len(dataset) - valid_size, "Valid": valid_size}
#
#     train_dataset = TransformDataset(
#         SplitDataset(dataset, partitions, initial_partition="Train"), train_transform
#     )
#
#     valid_dataset = TransformDataset(
#         SplitDataset(dataset, partitions, initial_partition="Valid"), valid_transform
#     )
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size[0],
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )
#
#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=batch_size[1],
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )
#
#     return (train_loader, valid_loader)
#
#
# def get_train_valid_dataset(dataset, valid_size=None):
#     error_msg = "[!] valid_size should be an integer in the range [1, %d]." % (len(dataset))
#     if not valid_size:
#         valid_size = int(0.1 * len(dataset))
#     if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
#         raise TypeError(error_msg)
#
#     # generate train/val datasets
#     partitions = {"Train": len(dataset) - valid_size, "Valid": valid_size}
#
#     train_dataset = SplitDataset(dataset, partitions, initial_partition="Train")
#     valid_dataset = SplitDataset(dataset, partitions, initial_partition="Valid")
#
#     return (train_dataset, valid_dataset)
