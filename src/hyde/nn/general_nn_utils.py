import os
import random
import urllib
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import requests
import torch.nn as nn
import torchvision.transforms.functional as TF
from bs4 import BeautifulSoup

from ..lowlevel import add_noise, logging, utils

logger = logging.get_logger()

from tqdm import tqdm

__all__ = [
    "download_ICVL_data",
    "AddGaussianNoise",
    "AddGaussianNoiseBlind",
    "AddNoiseComplex",
    "AddNoiseDeadline",
    "AddNoiseImpulse",
    "AddNoiseMixed",
    "AddNoiseNonIIDdB",
    "AddNoiseStripe",
    "crop_center",
    "lmdb_data_2_volume",
    "MultipleWeightedLosses",
    "RandRot90Transform",
]


def download_ICVL_data(target_dir, num_threads=5, matkey="rad"):

    url = "http://icvl.cs.bgu.ac.il/hyperspectral/"
    ext = "mat"
    target_dir = Path(target_dir)

    def listFD(url, ext=""):
        page = requests.get(url).text
        # print(page)
        soup = BeautifulSoup(page, "html.parser")
        return [node.get("href") for node in soup.find_all("a") if node.get("href").endswith(ext)]

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
            img_clean = f[matkey][:].astype(np.float32)
        np.save(fpath, img_clean)

    # print(test_files)
    logger.debug("Expected data size: ~90.2 GB uncompressed")
    logger.info("Starting download of data... this might take some time...")
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
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


class AddGaussianNoise(object):
    """
    Add Gaussian white noise to a torch.Tensor. The *power* of the noise is controlled
    by the `noise_pow` parameter. This value is in dB

    Parameters
    ----------
    signal: torch.Tensor
    noise_pow: int, float
    scale_factor: float, optional
        The noise added is relative to the initial signal strength. if the signal is normalized,
        then the final values should be scaled by the same max value
    verbose: bool, optional
        print out how much noise was added

    Returns
    -------
    noisy_signal: torch.Tensor
    """

    """add gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigma_db, scale_factor=255.0):
        self.sigma_db = sigma_db
        self.scale_factor = scale_factor

    def __call__(self, img):
        # noise = np.random.randn(*img.shape) * self.sigma_ratio
        return add_noise.add_noise_db(
            img, self.sigma_db, scale_factor=self.scale_factor, verbose=False
        )


class AddGaussianNoiseBlind(object):
    """
    add blind gaussian noise to the given NUMPY array (B,H,W)

    Parameters
    ----------
    max_sigma_db: int, float
        the maximum noise power level in dB
    min_sigma_db: int, float, optional
        the minimum noise power level in dB
        default: 0
    """

    def __init__(self, max_sigma_db, min_sigma_db=0, scale_factor=255.0):
        self.max_sigma_db = max_sigma_db
        self.min_sigma_db = min_sigma_db
        self.scale_factor = scale_factor

    def __call__(self, img):
        # noise = np.random.randn(*img.shape) * self.sigmas[next(self.pos)]
        # return img + noise
        return add_noise.add_gaussian_noise_blind(
            img, self.min_sigma_db, self.max_sigma_db, self.scale_factor
        )


class AddNoiseNonIIDdB(object):
    """
    add non-iid gaussian noise to the given numpy array (B,H,W)

    Parameters
    ----------
    max_power: int, float, optional
        the maximum noise power level that can be used during noise generation
        unit: dB
        default: 40
    """

    def __init__(self, max_power=40, scale_factor=255.0):
        self.scale_factor = scale_factor
        self.max_power = max_power

    def __call__(self, img):
        return add_noise.add_non_iid_noise_db(
            img, max_power=self.max_power, scale_factor=self.scale_factor
        )


class AddNoiseMixed(object):
    """
    add mixed noise to the given numpy array (B,H,W)

    Parameters
    ----------
    noise_bank: List
        noisemakers (e.g. AddNoiseImpulse)
    num_bands: List
        number of bands which is corrupted by each item in noise_bank
    """

    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        b = img.shape[0]
        all_bands = np.random.permutation(range(b))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * b))
            bands = all_bands[pos : pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class _AddNoiseImpulse(object):
    """
    add impulse noise to the given numpy array (B,H,W)
    """

    def __init__(self, amounts, s_vs_p=0.5):
        # s_vs_p is a probability and not a noise amount (its fine as is)
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[i, ...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class _AddNoiseStripe(object):
    """
    add stripe noise to the given numpy array (B,H,W)
    """

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(
            np.floor(self.min_amount * W), np.floor(self.max_amount * W), len(bands)
        )
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class _AddNoiseDeadline(object):
    """
    add deadline noise to the given numpy array (B,H,W)
    """

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(
            np.ceil(self.min_amount * W), np.ceil(self.max_amount * W), len(bands)
        )
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):
        super().__init__(noise_bank=[_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])], num_bands=[0.33333])


class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        super().__init__(noise_bank=[_AddNoiseStripe(0.05, 0.15)], num_bands=[0.33333])


class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        super().__init__(noise_bank=[_AddNoiseDeadline(0.05, 0.15)], num_bands=[0.33333])


class AddNoiseComplex(AddNoiseMixed):
    def __init__(self):
        super().__init__(
            noise_bank=[
                _AddNoiseStripe(0.05, 0.15),
                _AddNoiseDeadline(0.05, 0.15),
                _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7]),
            ],
            num_bands=[0.33333, 0.33333, 0.33333],
        )


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


def lmdb_data_2_volume(
    data: np.ndarray, ksizes: Union[Tuple, List], strides: [Tuple, List]
) -> np.ndarray:
    """
    Construct Volumes from Original High Dimensional (D) Data
    This function is intended to be used with LMBD dataset creation.

    Parameters
    ----------
    data : np.ndarray
    ksizes : tuple, list
        sizes to get
    strides : tuple, list

    Returns
    -------
    volumes : np.ndarray

    References
    ----------
    https://github.com/Vandermode/QRNN3D/blob/master/utility/util.py
    """
    dshape = data.shape

    def pat_num(l, k, s):  # noqa: E741
        return np.floor((l - k) / s) + 1

    ttl_pat_num = 1
    for i in range(len(ksizes)):
        ttl_pat_num = ttl_pat_num * pat_num(dshape[i], ksizes[i], strides[i])

    vol = np.zeros([int(ttl_pat_num)] + ksizes)
    # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple(
            [slice(key, -ksizes[i] + key + 1 or None, strides[i]) for i, key in enumerate(s)]
        )
        vol[s1] = np.reshape(data[s2], (-1,))

    return vol


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


class RandRot90Transform:
    """Rotate by one of the given angles."""

    def __init__(self, angles=None):
        if angles is None:
            angles = [-90, 0, 90, 180]
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
