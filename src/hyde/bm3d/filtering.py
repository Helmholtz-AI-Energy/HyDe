import math
from typing import Tuple

import torch

from .utils import hadamard

__all__ = ["hadamard_transform", "ht_filtering_hadamard", "wiener_filtering_hadamard"]


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n)  # .astype(np.float64)
    v_h = vec @ h_mat
    return v_h


def ht_filtering_hadamard(
    group_3D: torch.Tensor, sigma: float, lambdaHard3D, doWeight: bool
) -> Tuple[torch.Tensor, torch.Tensor]:  # group_3D shape=(n*n, nSx_r)
    """
    :hard threshold filtering after hadamard transform
    :param group_3D:
    :param sigma:
    :param lambdaHard3D:
    :param doWeight:
    :return:
    """
    nSx_r = group_3D.shape[-1]
    coef_norm = math.sqrt(nSx_r)
    coef = 1.0 / nSx_r

    group_3D_h = hadamard_transform(group_3D)

    # hard threshold filtering in this block
    T = lambdaHard3D * sigma * coef_norm
    T_3D = torch.where(torch.abs(group_3D_h) > T, 1, 0)
    weight = torch.sum(T_3D)
    # print((torch.abs(group_3D_h) > T).dtype)
    # todo: device setting
    group_3D_h = torch.where(torch.abs(group_3D_h) > T, group_3D_h, torch.tensor(0.0))

    group_3D = hadamard_transform(group_3D_h)

    group_3D *= coef
    if doWeight:
        one = torch.tensor(1.0, dtype=group_3D.dtype, device=group_3D.device)
        weight = 1.0 / (sigma * sigma * weight) if weight > 0.0 else one

    return group_3D, weight


@torch.jit.script
def wiener_filtering_hadamard(
    group_3D_img: torch.Tensor, group_3D_est: torch.Tensor, sigma: float, doWeight: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :wiener_filtering after hadamard transform
    :param group_3D_img:
    :param group_3D_est:
    :param sigma:
    :param doWeight:
    :return:
    """
    assert group_3D_img.shape == group_3D_est.shape
    nSx_r = group_3D_img.shape[-1]
    coef = 1.0 / nSx_r

    group_3D_img_h = hadamard_transform(group_3D_img)  # along nSx_r axis
    group_3D_est_h = hadamard_transform(group_3D_est)

    # wiener filtering in this block
    value = torch.pow(group_3D_est_h, 2) * coef
    value /= value + sigma * sigma
    group_3D_est_h = group_3D_img_h * value * coef
    weight = torch.sum(value)

    group_3D_est = hadamard_transform(group_3D_est_h)

    if doWeight:
        one = torch.tensor(1.0, dtype=group_3D_img.dtype, device=group_3D_img.device)
        weight = 1.0 / (sigma * sigma * weight) if weight > 0.0 else one

    return group_3D_est, weight
