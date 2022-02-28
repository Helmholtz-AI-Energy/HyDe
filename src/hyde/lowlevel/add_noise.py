from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np
import torch

from . import utils
from .logging import get_logger

logger = get_logger()

__all__ = [
    "add_noise_db",
    "add_noise_std",
    "add_noise_to_db_level",
    "add_simulated_lines",
    "add_noise_deadline",
    "add_noise_stripe",
    "add_noise_on_bands",
    "add_noise_impulse",
    "add_noise_salt_n_pepper",
    "add_non_iid_noise_db",
    "add_gaussian_noise_blind",
]


def add_noise_std(
    image: torch.Tensor, sigma: float, noise_type: str = "additive", iid: bool = True
) -> torch.Tensor:
    """
    Add noise based on a standard deviation. Options for noise include additive (gaussian) noise with
    or without i.i.d. and poisson.
    Poisson will add a standard 15 dB noise to the data and ignores the sigma value.

    Parameters
    ----------
    image: torch.Tensor
        signal on which to add noise
    sigma: float
        standard deviation of the noise distribution to be added to the signal
        NOTE: the 'standard' values of these are [0.10, 0.08, 0.06, 0.04, 0.02]
    noise_type: str
        options: [additive, poisson]
        toggle between noise types
        default: additive
    iid: bool
        if the noise added should be i.i.d. or not.
        default: True

    Returns
    -------
    noisy_image: torch.Tensor
    """
    if noise_type == "additive" and iid:
        noise = sigma * torch.randn_like(image)
        img_noisy = image + noise
    elif noise_type == "additive" and not iid:
        sigma = torch.rand(image.shape[-1], dtype=image.dtype, device=image.device) * 0.1
        noise = torch.randn_like(image)
        for band in range(image.shape[-1]):
            noise[:, :, band] *= sigma[band]
        img_noisy = image + noise
    elif noise_type == "poisson":
        img_wN = image
        snr_db = 15
        snr_set = torch.exp(
            snr_db * torch.log(torch.tensor(10, device=image.device, dtype=image.dtype)) / 10
        )

        rc = image.shape[0] * image.shape[1]
        bands = image.shape[-1]
        img_wn_noisy = torch.zeros((bands, rc), dtype=image.dtype, device=image.device)
        for i in range(bands):
            img_wntmp = img_wN[:, :, i].unsqueeze(0)
            img_wntmp[img_wntmp <= 0] = 0
            # factor = snr_set/( sum(img_wNtmp.^2)/sum(img_wNtmp) );
            factor = snr_set / ((img_wntmp ** 2).sum() / img_wntmp.sum())
            # img_wN_scale(i,1:N) = factor*img_wNtmp;
            # img_wN_scale[i] = factor * img_wNtmp
            # % Generates Poisson random samples
            # img_wN_noisy(i,1:N) = poissrnd(factor*img_wNtmp);
            img_wn_noisy[i] = torch.poisson(factor * img_wntmp)
        img_noisy = img_wn_noisy.T.reshape(image.shape)
    else:
        raise ValueError(f"noise type must be one of [poissson, additive], currently: {noise_type}")
    return img_noisy


def add_noise_db(
    signal: torch.Tensor, noise_pow: Union[int, float], scale_factor: float = 1, verbose=False
) -> torch.Tensor:
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
    noise_to_add = 10 ** (noise_pow / 20) / scale_factor
    try:
        noise = torch.randn_like(signal) * noise_to_add
        # if verbose:
        #     print(f"Added Noise [dB]: {10 * torch.log10(torch.mean(torch.pow(noise, 2)))}")
    except TypeError:  # numpy.ndarray  todo: raise statement?
        noise = np.random.normal(scale=noise_to_add, size=signal.shape)

    if verbose:
        logger.info(f"Added Noise [dB]: {(noise * scale_factor).pow(2).mean().log10() * 10}")
    else:
        logger.debug(f"Added Noise [dB]: {(noise * scale_factor).pow(2).mean().log10() * 10}")

    return noise + signal


def add_noise_to_db_level(signal: torch.Tensor, target_power: float) -> torch.Tensor:
    """
    Convert a signal to a given SNR ratio (in dB)

    Parameters
    ----------
    signal: torch.Tensor
    target_power: float
        target SNR level in dB

    Returns
    -------
    noisy signal : torch.Tensor
    """
    # get current snr
    offset = torch.randn_like(signal)
    offset_db = torch.log10(offset.std()) * 20
    curr_snr, _ = utils.snr(signal + offset, signal)
    # NOTE: this assumes that the signal is clean
    noise_to_add = curr_snr - target_power + offset_db
    # print("noise_to_add", noise_to_add)
    return add_noise_db(signal, noise_to_add, scale_factor=1)


def add_simulated_lines(signal: torch.Tensor, bands=(9, 15)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove random(-ish) veritical sections from specified bands

    Parameters
    ----------
    signal: torch.Tensor
    bands: tuple
        bands to put stripe into
        must be 3 dimensions!

    Returns
    -------
    signal_with_lines: torch.Tensor
        input signal with the simulated lines
    mask: torch.Tensor
        mask indicates where there is data. i.e. 1 -> good data, 0 -> no data
    """
    mask = torch.ones_like(signal)
    bands_strp = list(range(*bands))
    dtp, dev = signal.dtype, signal.device
    first = True
    for ib in bands_strp:
        if first:
            loc_strp = torch.ceil(torch.rand(20, dtype=dtp, device=dev) * signal.shape[1])
            loc_strp = torch.cat([loc_strp, torch.arange(40, 51, dtype=dtp, device=dev)], dim=0)
            loc_strp = torch.cat([loc_strp, torch.arange(10, 41, dtype=dtp, device=dev)], dim=0)
            loc_strp = loc_strp.to(torch.long)
            first = False
        signal[:, loc_strp, ib] = 0
        mask[:, loc_strp, ib] = 0

    return signal, mask


def add_gaussian_noise_blind(
    signal: torch.Tensor, min_db: float, max_db: float, scale_factor: float = 1
) -> torch.Tensor:
    """
    Add blind Gaussian (additive) noise to a tensor. The noise generated will be a random dB level
    between `min_db` and `max_db`.

    Parameters
    ----------
    signal: torch.Tensor
        single on which to add noise
    min_db: float, optional
        minimum amount of noise to add unit: dB
    max_db: float, optional
        minimum amount of noise to add unit: dB
    scale_factor: float, optional
        Scaling factor by which to scale the generated noise.

    Returns
    -------
    image + random noise: torch.Tensor
    """
    noise_db = ((max_db - min_db) * np.random.rand()) + min_db
    return add_noise_db(signal, noise_db, verbose=False, scale_factor=scale_factor)


def add_non_iid_noise_db(signal: torch.Tensor, max_power, scale_factor=1.0) -> torch.Tensor:
    """
    Add non-IID noise to a signal

    Parameters
    ----------
    signal: torch.Tensor
        signal to which noise will be added
    max_power: float
        maximum dB value of the added noise
    scale_factor: float
        Scaling factor by which to scale the generated noise.

    Returns
    -------
    image + non-iid noise: torch.Tensor
    """
    try:
        bwsigmas = torch.rand(signal.shape[0], 1, 1) * max_power
        bwsigmas = bwsigmas.to(device=signal.device, dtype=signal.dtype)
        nz = torch.randn(*signal.shape, device=signal.device, dtype=signal.dtype)
        noise = nz * bwsigmas / scale_factor
    except TypeError:  # torch version
        bwsigmas = np.random.rand(signal.shape[0], 1, 1) * max_power
        noise = np.random.randn(*signal.shape) * bwsigmas / scale_factor

    return signal + noise


def add_noise_on_bands(
    signal: torch.Tensor,
    bands: float,
    noise_fn: Callable,
    noise_fn_args: Dict = None,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Add noise to a random percentage of bands
    The percentage of bands which are effected is determined by `bands`.

    Parameters
    ----------
    signal: torch.Tensor
    bands: float
        the percentage of bands upon which noise will be applied
    noise_fn: Callable
        the function to generate noise
    noise_fn_args: dict, optional
        the arguments for the `noise_fn`, if any
    band_dim: int, optional
        index (int) which indicates which dimension of the signal is the band dimension
    inplace: bool, optional
        if true (default) overwrite the input signal with the result

    Returns
    -------
    noisy image: torch.Tensor
    """
    if noise_fn_args is None:
        noise_fn_args = dict()

    # note: if this is called multiple times, it may add noise to the same bands
    b = signal.shape[band_dim]
    try:
        all_bands = torch.randperm(b, device=signal.device)
    except (TypeError, AttributeError):
        all_bands = np.random.permutation(range(b))

    pos = 0
    if 0 < bands <= 1:
        bands = int(bands * b)
    bands = all_bands[pos : pos + bands]
    pos += bands
    sig = signal if inplace else signal.clone()
    img = noise_fn(sig, bands, band_dim=band_dim, **noise_fn_args)
    return img


def add_noise_impulse(
    signal: torch.Tensor,
    bands: float,
    amounts: Iterable = (0.1, 0.3, 0.5, 0.7),
    salt_vs_pepper: float = 0.5,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Add noise to a random percentage of bands.
    Impulse noise is sharp noise like clicks and pops. it is also referred to as salt and pepper noise.
    The percentage of bands which are effected is determined by `bands`.

    Parameters
    ----------
    signal: torch.Tensor
    bands: float
        the percentage of bands upon which noise will be applied
    amounts: iterable, optional
        the amount of noise to add. This will be chosen from randomly
    salt_vs_pepper: float, optional
        ratio of salt to pepper noise
    band_dim: int, optional
        index (int) which indicates which dimension of the signal is the band dimension
    inplace: bool, optional
        if true (default) overwrite the input signal with the result

    Returns
    -------
    noisy image: torch.Tensor
    """

    return add_noise_on_bands(
        signal,
        bands,
        __add_noise_impulse,
        noise_fn_args={
            "amounts": amounts,
            "salt_vs_pepper": salt_vs_pepper,
        },
        band_dim=band_dim,
        inplace=inplace,
    )


add_noise_salt_n_pepper = add_noise_impulse
add_noise_salt_n_pepper.__doc__ = "See `add_noise_impulse`"


def __add_noise_impulse(
    signal: torch.Tensor,
    bands: Iterable,
    amounts: Iterable,
    salt_vs_pepper: float,
    band_dim: int,
) -> torch.Tensor:
    # add impulse noise to a tensor, for more info see `add_noise_impulse`
    # note: this is intended to be used with the `add_noise_on_bands` function
    sl = [
        slice(None),
    ] * signal.ndim
    # bwamounts = torch.tensor(amounts)[torch.randint(high=len(amounts), shape=(len(bands),), device=signal.device)]
    bwamounts = [amounts[i.item()] for i in torch.randint(high=len(amounts), size=(len(bands),))]
    for i, amount in zip(bands, bwamounts):
        # out = signal[i, ...]
        sl[band_dim] = i
        out = signal[sl]
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=tuple(out.shape), p=[p, 1 - p])
        flipped = torch.from_numpy(flipped).to(signal.device)
        salted = np.random.choice([True, False], size=tuple(out.shape), p=[q, 1 - q])
        salted = torch.from_numpy(salted).to(signal.device)
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        signal[sl] = out

    return signal


def add_noise_stripe(
    signal: torch.Tensor,
    bands: float = 0.3333333,
    min_amount: float = 0.05,
    max_amount: float = 0.15,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Add noise to random vertical stripes in random bands.
    the noise added is a uniform distribution.
    The percentage of bands which are effected is determined by `bands`.

    Parameters
    ----------
    signal: torch.Tensor
    bands: float
        the percentage of bands upon which noise will be applied
    min_amount: float
        minimum number of stripes to modify
    max_amount: float
        maximum number of stripes to modify
    band_dim: int, optional
    inplace: bool, optional

    Returns
    -------
    noisy image: torch.Tensor
    """
    return add_noise_on_bands(
        signal,
        bands,
        __add_noise_stripe,
        noise_fn_args={
            "min_amount": min_amount,
            "max_amount": max_amount,
        },
        band_dim=band_dim,
        inplace=inplace,
    )


def __add_noise_stripe(
    signal: torch.Tensor,
    bands: Iterable,
    min_amount: float,
    max_amount: float,
    band_dim: int,
) -> torch.Tensor:
    # add random uniform noise to random stripes on random bands
    # ASSUMPTION: format is either [batch, band, h, w] or [h, w, band]
    logger.debug(
        "The add_noise_strip function assumes that the image format is "
        "[batch, band, h, w], [band, h, w], or [h, w, band]"
    )
    if band_dim == -1:
        _, w, _ = signal.shape[-3], signal.shape[-2], signal.shape[-1]
        w_dim = -2
    else:
        _, _, w = signal.shape[-3], signal.shape[-2], signal.shape[-1]
        w_dim = -1
    # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]

    num_stripe = np.random.randint(int(min_amount * w), int(max_amount * w), len(bands))
    sl = [
        slice(None),
    ] * signal.ndim
    for i, n in zip(bands, num_stripe):
        loc = torch.randperm(w)
        loc = loc[:n]
        sl[band_dim] = i
        sl[w_dim] = loc
        stripe = torch.rand_like(loc, dtype=signal.dtype, device=signal.device)
        signal[sl] -= stripe * 0.5 - 0.25

    return signal


def add_noise_deadline(
    signal: torch.Tensor,
    bands: float,
    min_amount: float = 0.05,
    max_amount: float = 0.15,
    band_dim: int = -1,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Add deadline noise. This will make random veritcal lines 0 on random bands.
    The percentage of bands which are effected is determined by `bands`.

    Parameters
    ----------
    signal: torch.Tensor
    bands: float
        the percentage of bands upon which noise will be applied
    min_amount: float
        minimum number of stripes to modify
    max_amount: float
        maximum number of stripes to modify
    band_dim: int, optional
    inplace: bool, optional

    Returns
    -------
    noisy image: torch.Tensor
    """
    return add_noise_on_bands(
        signal,
        bands,
        __add_noise_deadline,
        noise_fn_args={
            "min_amount": min_amount,
            "max_amount": max_amount,
        },
        band_dim=band_dim,
        inplace=inplace,
    )


def __add_noise_deadline(signal, bands, min_amount, max_amount, band_dim=-1):
    logger.debug(
        "The add_noise_deadline function assumes that the image format is "
        "[batch, band, h, w], [band, h, w], or [h, w, band]"
    )
    if band_dim == -1:
        _, w, _ = signal.shape[-3], signal.shape[-2], signal.shape[-1]
        w_dim = -2
    else:
        _, _, w = signal.shape[-3], signal.shape[-2], signal.shape[-1]
        w_dim = -1
    # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
    num_deadline = np.random.randint(int(min_amount * w), int(max_amount * w), len(bands))

    sl = [
        slice(None),
    ] * signal.ndim
    for i, n in zip(bands, num_deadline):
        loc = torch.randperm(w, device=signal.device)
        loc = loc[:n]
        sl[band_dim] = i
        sl[w_dim] = loc
        signal[sl] *= 0

    return signal
