# import numpy as np
from typing import Tuple

import torch
from torch.nn.functional import pad

from .. import dct
from ..utils import symmetric_pad
from . import utils
from .filtering import ht_filtering_hadamard, wiener_filtering_hadamard

__all__ = ["bm3d"]


def bm3d(
    noisy_im,
    sigma,
    n_H,
    k_H,
    N_H,
    p_H,
    tauMatch_H,
    useSD_H,
    tau_2D_H,
    lambda3D_H,
    n_W,
    k_W,
    N_W,
    p_W,
    tauMatch_W,
    useSD_W,
    tau_2D_W,
):
    k_H = 8 if (tau_2D_H == "BIOR" or sigma < 40.0) else k_H
    k_W = 8 if (tau_2D_W == "BIOR" or sigma < 40.0) else k_W

    noisy_im_p = symmetric_pad(noisy_im, n_H)
    img_basic = _bm3d_1st_step(
        sigma=sigma,
        img_noisy=noisy_im_p,
        nHard=n_H,
        kHard=k_H,
        NHard=N_H,
        pHard=p_H,
        lambda_hard_3d=lambda3D_H,
        tau_match=tauMatch_H,
        use_sd=useSD_H,
        tau_2d=tau_2D_H,
    )
    img_basic = img_basic[n_H:-n_H, n_H:-n_H]

    assert not torch.any(torch.isnan(img_basic))
    img_basic_p = symmetric_pad(img_basic, n_W)
    noisy_im_p = symmetric_pad(noisy_im, n_W)
    img_denoised = _bm3d_2nd_step(
        sigma=sigma,
        img_noisy=noisy_im_p,
        img_basic=img_basic_p,
        n_wien=n_W,
        k_wien=k_W,
        NWien=N_W,
        pWien=p_W,
        tau_match=tauMatch_W,
        use_sd=useSD_W,
        tau_2d=tau_2D_W,
    )
    img_denoised = img_denoised[n_W:-n_W, n_W:-n_W]

    return img_basic, img_denoised


def _bm3d_1st_step(
    sigma,
    img_noisy,
    nHard,
    kHard,
    NHard,
    pHard,
    lambda_hard_3d,
    tau_match,
    use_sd,
    tau_2d,
):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = utils.ind_initialize(height - kHard + 1, nHard, pHard)
    column_ind = utils.ind_initialize(width - kHard + 1, nHard, pHard)

    kaiser_window = utils.get_kaiser_window(kHard)
    ri_rj_n__ni_nj, threshold_count = utils.precompute_BM(
        img_noisy, kHW=kHard, NHW=NHard, nHW=nHard, tauMatch=tau_match
    )
    group_len = int(torch.sum(threshold_count))
    group_3d_table = torch.zeros((group_len, kHard, kHard))
    weight_table = torch.zeros((height, width))
    all_patches = utils.image2patches(img_noisy, kHard, kHard)

    if tau_2d == "DCT":
        fre_all_patches = dct.dct_2d(all_patches)
    else:  # 'BIOR'
        fre_all_patches = utils.bior_2d_forward(all_patches)

    acc_pointer = 0

    for i_r in row_ind:
        for j_r in column_ind:
            nsx_r = threshold_count[i_r, j_r].item()
            group_3d = utils.build_3D_group(
                fre_all_patches, ri_rj_n__ni_nj[i_r, j_r], nsx_r
            )
            group_3d, weight = ht_filtering_hadamard(
                group_3d, sigma, lambda_hard_3d, not use_sd
            )
            group_3d = group_3d.permute((2, 0, 1))
            group_3d_table[acc_pointer : acc_pointer + nsx_r] = group_3d
            acc_pointer += nsx_r

            if use_sd:
                weight = utils.sd_weighting(group_3d)

            weight_table[i_r, j_r] = weight

    if tau_2d == "DCT":
        group_3d_table = dct.idct_2d(group_3d_table)
    else:  # 'BIOR'
        group_3d_table = utils.bior_2d_reverse(group_3d_table)

    # aggregation part
    numerator = torch.zeros_like(
        img_noisy, dtype=img_noisy.dtype, device=img_noisy.device
    )
    denominator = torch.zeros(
        (img_noisy.shape[0] - 2 * nHard, img_noisy.shape[1] - 2 * nHard),
        dtype=img_noisy.dtype,
        device=img_noisy.device,
    )
    denominator = pad(denominator, (nHard, nHard, nHard, nHard), "constant", 1.0)

    img_basic = __agg_loop(
        row_ind=row_ind,
        column_ind=column_ind,
        threshold_count=threshold_count,
        ri_rj_n__ni_nj=ri_rj_n__ni_nj,
        group_table=group_3d_table,
        weight_table=weight_table,
        numerator=numerator,
        sliding_len=kHard,
        kaiser_window=kaiser_window,
        denominator=denominator,
    )
    return img_basic


@torch.jit.script
def __agg_loop(
    row_ind: torch.Tensor,
    column_ind: torch.Tensor,
    threshold_count: torch.Tensor,
    ri_rj_n__ni_nj: torch.Tensor,
    group_table: torch.Tensor,
    weight_table: torch.Tensor,
    numerator: torch.Tensor,
    sliding_len: int,
    kaiser_window: torch.Tensor,
    denominator: torch.Tensor,
):
    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nsx_r = threshold_count[i_r, j_r]
            n_ni_nj = ri_rj_n__ni_nj[i_r, j_r]
            group = group_table[acc_pointer : acc_pointer + nsx_r]
            acc_pointer += nsx_r
            weight = weight_table[i_r, j_r]

            for n in range(int(nsx_r.item())):
                ni = n_ni_nj[n][0]
                nj = n_ni_nj[n][1]
                patch = group[n]

                numerator[ni : ni + sliding_len, nj : nj + sliding_len] += (
                    patch * kaiser_window * weight
                )
                denominator[ni : ni + sliding_len, nj : nj + sliding_len] += (
                    kaiser_window * weight
                )

    return numerator / denominator


@torch.jit.script
def __step2_weight_calc(
    threshold_count: torch.Tensor,
    i_r: torch.Tensor,
    j_r: torch.Tensor,
    fre_noisy_patches: torch.Tensor,
    ri_rj_n__ni_nj: torch.Tensor,
    fre_basic_patches: torch.Tensor,
    group_3d_table: torch.Tensor,
    sigma: float,
    use_sd: bool,
    acc_pointer: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    nsx_r = threshold_count[i_r, j_r]
    group_3d_img = utils.build_3D_group(
        fre_noisy_patches, ri_rj_n__ni_nj[i_r, j_r], nsx_r
    )
    group_3d_est = utils.build_3D_group(
        fre_basic_patches, ri_rj_n__ni_nj[i_r, j_r], nsx_r
    )
    group_3d, weight = wiener_filtering_hadamard(
        group_3d_img, group_3d_est, sigma, not use_sd
    )
    group_3d = group_3d.permute((2, 0, 1))

    group_3d_table[acc_pointer : acc_pointer + nsx_r] = group_3d

    if use_sd:
        weight = utils.sd_weighting(group_3d)

    return weight, nsx_r, group_3d_table


def _bm3d_2nd_step(
    sigma, img_noisy, img_basic, n_wien, k_wien, NWien, pWien, tau_match, use_sd, tau_2d
):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = utils.ind_initialize(height - k_wien + 1, n_wien, pWien)
    column_ind = utils.ind_initialize(width - k_wien + 1, n_wien, pWien)

    kaiser_window = utils.get_kaiser_window(k_wien)
    ri_rj_n__ni_nj, threshold_count = utils.precompute_BM(
        img_basic, kHW=k_wien, NHW=NWien, nHW=n_wien, tauMatch=tau_match
    )
    group_len = int(torch.sum(threshold_count))
    group_3d_table = torch.zeros(
        (group_len, k_wien, k_wien), device=img_noisy.device, dtype=img_noisy.dtype
    )
    weight_table = torch.zeros(
        (height, width), device=img_noisy.device, dtype=img_noisy.dtype
    )

    noisy_patches = utils.image2patches(img_noisy, k_wien, k_wien)
    basic_patches = utils.image2patches(img_basic, k_wien, k_wien)
    if tau_2d == "DCT":
        fre_noisy_patches = dct.dct_2d(noisy_patches, "ortho")
        fre_basic_patches = dct.dct_2d(basic_patches, "ortho")
    else:  # 'BIOR'
        fre_noisy_patches = utils.bior_2d_forward(noisy_patches)
        fre_basic_patches = utils.bior_2d_forward(basic_patches)

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            weight, nsx_r, group_3d_table = __step2_weight_calc(
                threshold_count,
                i_r,
                j_r,
                fre_noisy_patches,
                ri_rj_n__ni_nj,
                fre_basic_patches,
                group_3d_table,
                sigma,
                use_sd,
                acc_pointer,
            )
            acc_pointer += nsx_r
            weight_table[i_r, j_r] = weight

    if tau_2d == "DCT":
        group_3d_table = dct.idct_2d(group_3d_table, "ortho")
    else:  # 'BIOR'
        group_3d_table = utils.bior_2d_reverse(group_3d_table)

    # aggregation part
    numerator = torch.zeros_like(
        img_noisy, device=img_noisy.device, dtype=img_noisy.dtype
    )
    denominator = torch.zeros(
        (img_noisy.shape[0] - 2 * n_wien, img_noisy.shape[1] - 2 * n_wien),
        device=img_noisy.device,
        dtype=img_noisy.dtype,
    )
    denominator = pad(denominator, (n_wien, n_wien, n_wien, n_wien), "constant", 1.0)

    img_basic = __agg_loop(
        row_ind=row_ind,
        column_ind=column_ind,
        threshold_count=threshold_count,
        ri_rj_n__ni_nj=ri_rj_n__ni_nj,
        group_table=group_3d_table,
        weight_table=weight_table,
        numerator=numerator,
        sliding_len=k_wien,
        kaiser_window=kaiser_window,
        denominator=denominator,
    )
    return img_basic
