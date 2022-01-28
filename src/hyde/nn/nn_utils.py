from itertools import product
from typing import List, Tuple, Union

import numpy as np
import torch

__all__ = [
    "crop_center",
    "lmdb_data_2_volume",
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
