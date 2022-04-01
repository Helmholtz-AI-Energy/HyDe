import random
from itertools import combinations
from typing import Iterable, List, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF

from ...lowlevel import add_noise, logging, utils

logger = logging.get_logger()

__all__ = [
    "AddGaussianNoise",
    "AddGaussianNoiseBlind",
    # "AddNoiseComplex",
    "AddNoiseDeadline",
    "AddNoiseImpulse",
    "AddNoiseMixed",
    "AddNoiseNonIIDdB",
    "AddNoiseStripe",
    "GaussianSNRLevel",
    "GaussianBlindSNRLevel",
    "RandChoice",
    "RandomBandPerm",
    "RandRot90Transform",
]


class GaussianSNRLevel(object):
    """
    Add Gaussian white noise to a torch.Tensor. The *power* of the noise is controlled
    by the `noise_pow` parameter. This value is in dB

    Parameters
    ----------
    sigma_db: int, float
    scale_factor: float, optional
        The noise added is relative to the initial signal strength. if the signal is normalized,
        then the final values should be scaled by the same max value
    """

    def __init__(self, sigma_db, scale_factor=1):
        self.sigma_db = sigma_db
        self.scale_factor = scale_factor

    def __call__(self, img):
        return add_noise.add_noise_to_db_level(img, self.sigma_db)


class GaussianBlindSNRLevel(object):
    """
    Add Gaussian white noise to a torch.Tensor. The *power* of the noise is controlled
    by the `noise_pow` parameter. This value is in dB

    Parameters
    ----------
    sigma_db: int, float
    scale_factor: float, optional
        The noise added is relative to the initial signal strength. if the signal is normalized,
        then the final values should be scaled by the same max value
    """

    def __init__(self, max_sigma_db, min_sigma_db=0, scale_factor=1):
        self.max_sigma_db = max_sigma_db
        self.min_sigma_db = min_sigma_db
        self.diff = max_sigma_db - min_sigma_db
        self.scale_factor = scale_factor

    def __call__(self, img):
        db = self.diff * np.random.random() + self.min_sigma_db
        return add_noise.add_noise_to_db_level(img, db)


class AddGaussianNoise(object):
    """
    Add Gaussian white noise to a torch.Tensor. The *power* of the noise is controlled
    by the `noise_pow` parameter. This value is in dB

    Parameters
    ----------
    sigma_db: int, float
    scale_factor: float, optional
        The noise added is relative to the initial signal strength. if the signal is normalized,
        then the final values should be scaled by the same max value
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
        b = img.shape[-3]
        all_bands = np.random.permutation(range(b))

        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * b))
            bands = all_bands[pos : pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class AddNoiseImpulse(object):
    """
    add impulse noise to the given numpy array (B,H,W)
    """

    def __init__(self, amounts=(0.1, 0.3, 0.5, 0.7), s_vs_p=0.5, bands=0.333333, band_dim=-3):
        # s_vs_p is a probability and not a noise amount (its fine as is)
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p
        self.bands = bands
        self.band_dim = band_dim

    def __call__(self, img):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        return add_noise.add_noise_impulse(
            img,
            self.bands,
            amounts=self.amounts,
            salt_vs_pepper=self.s_vs_p,
            band_dim=self.band_dim,
        )


class AddNoiseStripe(object):
    """
    add stripe noise to the given numpy array (B,H,W)
    """

    def __init__(self, min_amount=0.05, max_amount=0.15, bands=0.3333333, band_dim=-3):
        # noise_bank=[_AddNoiseStripe(0.05, 0.15)], num_bands=[0.33333]
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.bands = bands
        self.band_dim = band_dim

    def __call__(self, img):
        return add_noise.add_noise_stripe(
            img,
            self.bands,
            min_amount=self.min_amount,
            max_amount=self.max_amount,
            band_dim=self.band_dim,
        )


class AddNoiseDeadline(object):
    """
    add deadline noise to the given numpy array (B,H,W)
    """

    def __init__(self, min_amount=0.05, max_amount=0.15, bands=0.3333333, band_dim=-3):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.bands = bands
        self.band_dim = band_dim

    def __call__(self, img):
        return add_noise.add_noise_deadline(
            img,
            self.bands,
            min_amount=self.min_amount,
            max_amount=self.max_amount,
            band_dim=self.band_dim,
        )


class RandChoice:
    """
    Randomly choose between multiple transforms

    Parameters
    ----------
    transforms: iterable
        iterable with a number of initialized transformation
    p: int, iterable, optional
        probability of getting any of the transforms.
        if `p` is a single value, the same `p` will be used for each transform
    combos: bool, optional
        if True, apply combinations of transforms.
        If this is True, then the `p` value is ignored.
        defaul: False
    """

    def __init__(self, transforms: Iterable, p: Union[int, Iterable] = None, combos: bool = False):
        self.transforms = transforms
        self.p = p
        if isinstance(p, (float)):
            self.p = [p for _ in transforms]
        # this will apply equal probability to each transform
        self.combos = []
        for i in range(1, len(self.transforms) + 1):
            self.combos.extend(list(combinations(self.transforms, i)))
        self.use_combos = combos

    def __call__(self, x, *args):
        if self.use_combos:
            trfms = self.combos
        else:
            trfms = self.transforms

        if self.p is None:
            # in this case, select 1 element of the list to call
            # equal probability that it is no transform
            i = random.randrange(len(trfms) + 1)
            if i == len(trfms):
                return x

            if not self.use_combos:
                return random.choice(self.transforms)(x, *args)

            sel_trfm = random.choice(trfms)
            for tf in sel_trfm:
                x = tf(x, *args)
            return x
            # return random.choice(self.transforms)(x, *args)
        else:
            return random.choices(self.transforms, weights=self.p)[0](x, *args)


class RandomBandPerm(object):
    """
    Get a random collection of consecutive bands from an HSI
    """

    # crop an image in 3 dimensions, (also crop the bands)
    def __init__(self, bands=10):
        self.bands = bands

    def __call__(self, image):
        # assume that the image is larger than the crop size
        # order: [1, bands, height, width]
        # bands = torch.randperm(image.shape[-3], device=image.device)
        st = torch.randint(image.shape[-3] - self.bands, (1,)).item()
        # bands = bands[: self.bands]
        sl = [
            slice(None),
        ] * image.ndim
        sl[-3] = slice(st, st + self.bands)
        return image[sl]


class RandRot90Transform:
    """Rotate by one of the given angles."""

    def __init__(self, angles=None):
        if angles is None:
            angles = [-90, 0, 90, 180]
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
