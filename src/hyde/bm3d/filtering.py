"""
These functions are based heavily on https://github.com/Ryanshuai/BM3D_py/
"""
import math
from typing import Tuple

import torch

from .utils import hadamard

__all__ = ["hadamard_transform", "ht_filtering_hadamard", "wiener_filtering_hadamard"]


def hadamard_transform(vec):
    """
    Perform a hadamard transform on a vector

    Parameters
    ----------
    vec: torch.Tensor

    Returns
    -------
    transformed vector
    """
    n = vec.shape[-1]
    h_mat = hadamard(n)
    v_h = vec @ h_mat
    return v_h


def ht_filtering_hadamard(
    group_3d: torch.Tensor, sigma: float, lambda_hard: float, do_weight: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    hard threshold filtering after hadamard transform

    Parameters
    ----------
    group_3d: torch.Tensor
        contains the 3D block for a reference patch
    sigma: float
        value of the noise estimate
    lambda_hard: float
        value of thresholding
    do_weight: bool
        if true process the weighting, otherwise do nothing

    Returns
    -------
    group_3d : torch.Tensor
        filtered and threshold 3D block for a reference patch
    weight : torch.Tensor
        weight for the threshold
    """
    one = torch.tensor(1.0, dtype=group_3d.dtype, device=group_3d.device)
    zed = torch.tensor(0.0, dtype=group_3d.dtype, device=group_3d.device)

    nsx_r = group_3d.shape[-1]
    coef_norm = math.sqrt(nsx_r)
    coef = 1.0 / nsx_r

    group_3d_h = hadamard_transform(group_3d)

    # hard threshold filtering in this block
    th = lambda_hard * sigma * coef_norm
    th_3d = torch.where(torch.abs(group_3d_h) > th, 1, 0)
    weight = torch.sum(th_3d)

    group_3d_h = torch.where(torch.abs(group_3d_h) > th, group_3d_h, zed)

    group_3d = hadamard_transform(group_3d_h)

    group_3d *= coef
    if do_weight:
        weight = 1.0 / (sigma * sigma * weight) if weight > 0.0 else one

    return group_3d, weight


@torch.jit.script
def wiener_filtering_hadamard(
    group_3d_img: torch.Tensor,
    group_3d_est: torch.Tensor,
    sigma: float,
    do_weight: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    wiener_filtering after hadamard transform

    Parameters
    ----------
    group_3d_img : torch.Tensor
        contains the 3D block for a reference patch for the base image
    group_3d_est : torch.Tensor
        contains the 3D block for a reference patch for the image after the first step of BM3D
    sigma: float
        value of the noise estimate
    do_weight: bool
        if true process the weighting, otherwise do nothing

    Returns
    -------
    group_3d : torch.Tensor
        filtered and threshold 3D block for a reference patch
    weight : torch.Tensor
        weight for the threshold
    """
    assert group_3d_img.shape == group_3d_est.shape
    nsx_r = group_3d_img.shape[-1]
    coef = 1.0 / nsx_r

    group_3d_img_h = hadamard_transform(group_3d_img)  # along nSx_r axis
    group_3d_est_h = hadamard_transform(group_3d_est)

    # wiener filtering in this block
    value = torch.pow(group_3d_est_h, 2) * coef
    value /= value + sigma * sigma
    group_3d_est_h = group_3d_img_h * value * coef
    weight = torch.sum(value)

    group_3d_est = hadamard_transform(group_3d_est_h)

    if do_weight:
        one = torch.tensor(1.0, dtype=group_3d_img.dtype, device=group_3d_img.device)
        weight = 1.0 / (sigma * sigma * weight) if weight > 0.0 else one

    return group_3d_est, weight
