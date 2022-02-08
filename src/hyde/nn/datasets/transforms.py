import random
from typing import List

import numpy as np
import torchvision.transforms.functional as TF

from ...lowlevel import add_noise, logging

logger = logging.get_logger()

__all__ = [
    "AddGaussianNoise",
    "AddGaussianNoiseBlind",
    "AddNoiseComplex",
    "AddNoiseDeadline",
    "AddNoiseImpulse",
    "AddNoiseMixed",
    "AddNoiseNonIIDdB",
    "AddNoiseStripe",
    "RandRot90Transform",
]


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


class RandRot90Transform:
    """Rotate by one of the given angles."""

    def __init__(self, angles=None):
        if angles is None:
            angles = [-90, 0, 90, 180]
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
