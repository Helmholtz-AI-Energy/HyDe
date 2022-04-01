# There are functions for creating a train and validation iterator.
import os
import random
import urllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import shuffle

import h5py
import numpy as np
import requests
import torch
import torch.distributed as dist
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from ...lowlevel import logging, utils
from . import transforms as hyde_transforms

logger = logging.get_logger()

__all__ = ["download_ICVL_data", "ICVLDataset"]


def download_ICVL_data(target_dir, num_threads=5, matkey="rad", save_dtype=np.float32):
    """
    Download the ICVL dataset from http://icvl.cs.bgu.ac.il/hyperspectral/
    Expected data size: ~45 GB when saved as float32 (default)

    Parameters
    ----------
    target_dir: str, Path
        place to put the downloaded data
    num_threads: int, optional
        number of threads to use for downloading
        default: 5
    matkey: str, optional
        the key to use when getting the data out of the mat file
        default: rad
    save_dtype: np.dtype, optional
        the datatype to save the data as
        default: np.float32
    """
    url = "http://icvl.cs.bgu.ac.il/hyperspectral/"
    ext = "mat"
    target_dir = Path(target_dir)

    def listFD(url, ext=""):
        page = requests.get(url).text
        # print(page)
        soup = BeautifulSoup(page, "html.parser")
        ret = [node.get("href") for node in soup.find_all("a") if node.get("href").endswith(ext)]
        shuffle(ret)
        return ret

    # load one test file -> test gaussian
    test_files = []
    hyde_dir = Path(__file__[: __file__.find("src")])
    with open(hyde_dir / "src/hyde/nn/ICVL_test.txt", "r") as f:
        for fn in f.readlines():
            test_files.append(fn[:-1])

    # make test and train dirs
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    def load_n_save(furl, fpath):
        urllib.request.urlretrieve(furl, fpath)
        with h5py.File(fpath, "r") as f:
            try:
                img_clean = f[matkey][:].astype(save_dtype)
            except Exception as e:
                logger.warning(f"FAILURE: URL: {furl}\nfile: {fpath}\n{e}")
                return
        np.save(fpath, img_clean)
        return
        # logger.info(f"finished file: {fpath}")

    # print(test_files)
    logger.debug("Expected data size: ~90.2 GB uncompressed")
    logger.info("Starting download of data... this might take some time...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        retl = []
        for file in listFD(url, ext):
            sv_name = file.split("/")[-1]
            targ = test_dir if sv_name in test_files else train_dir
            if not os.path.isfile(targ / sv_name):
                ret = executor.submit(load_n_save, file, targ / sv_name)
                retl.append(ret)
        for r in tqdm(retl):
            r.result()
    logger.info("finished download")


class ICVLDataset(Dataset):
    """
    The torch Dataset to use for training a NN on the ICVL dataset.
    To download the data please use the `download_ICVL_data` function.

    NOTE: this will automatically do random rotations and vertical flips on both the network
    input and label

    Parameters
    ----------
    datadir: str
        location of the data to be loaded
    crop_size: tuple, optional
        the size to crop the image to. Many datasets have non-uniform data
        default: (512, 512)
    target_transform: optional
        transforms to perform on the label
    common_transforms: optional
        transforms to apply to both the network input and the label
    transform: optional
        transforms to apply to the network input
    val: bool, optional
        flag indicating if this is the validation set
    """

    def __init__(
        self,
        datadir,
        crop_size=(512, 512),
        target_transform=None,
        common_transforms=None,
        transform=None,
        val=False,
        band_norm=True,
    ):
        super(ICVLDataset, self).__init__()
        datadir = Path(datadir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".npy")]
        if dist.is_initialized():
            random.shuffle(self.files)

        # load all the data at the top
        self.loadfrom = []  # np.zeros(first, dtype=np.float32)
        self.band_norm = band_norm
        for c, f in enumerate(self.files):
            # the images are already in [bands, height, width]
            # loaded, _ = utils.normalize(
            #     torch.tensor(np.load(f), dtype=torch.float32), by_band=band_norm, band_dim=0
            # )
            loaded = torch.tensor(np.load(f), dtype=torch.float32)
            self.loadfrom.append(loaded)

        self.loadfrom = tuple(self.loadfrom)

        if not val:
            self.base_transforms = transforms.Compose(
                [
                    # transforms.CenterCrop(crop_size),
                    # transforms.RandomCrop(crop_size),
                    transforms.RandomResizedCrop(
                        crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
                    ),
                    hyde_transforms.RandomBandPerm(10),
                    hyde_transforms.RandChoice(
                        [
                            hyde_transforms.RandRot90Transform(),
                            transforms.RandomVerticalFlip(p=0.9),
                            transforms.RandomAffine(
                                degrees=180,
                                # scale=(0.1, 10), # old (0.1, 3)
                                shear=20,
                            ),
                            transforms.RandomHorizontalFlip(p=0.9),
                            transforms.RandomPerspective(p=0.88),
                        ],
                        p=None,  # 0.5,
                        combos=True,
                    ),
                ]
            )
        else:
            self.base_transforms = transforms.CenterCrop(crop_size)  # RandomCrop(crop_size)

        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.length = len(self.files)

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.loadfrom[idx].unsqueeze(0)

        img = self.base_transforms(img)

        if self.common_transforms is not None:
            img = self.common_transforms(img)
        target = img.clone().detach()

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.to(dtype=torch.float)
        target = target.to(dtype=torch.float)
        # norm after noise
        img, consts = utils.normalize(img, by_band=self.band_norm, band_dim=-3)
        target = utils.normalize_w_consts(target, consts, -3)
        return img, target
