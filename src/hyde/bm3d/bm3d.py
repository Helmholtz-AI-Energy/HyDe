# import numpy as np
import math
from typing import Tuple

import pytorch_wavelets as twave
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
    k_H = 8 if (tau_2D_H == "BIOR" or sigma < 40.0) else 12
    k_W = 8 if (tau_2D_W == "BIOR" or sigma < 40.0) else 12

    noisy_im_p = symmetric_pad(noisy_im, n_H)
    img_basic = _bm3d_1st_step(
        sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H
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
        tauMatch=tauMatch_W,
        useSD=useSD_W,
        tau_2D=tau_2D_W,
    )
    img_denoised = img_denoised[n_W:-n_W, n_W:-n_W]

    return img_basic, img_denoised


def _bm3d_1st_step(
    sigma, img_noisy, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatch, useSD, tau_2D
):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = utils.ind_initialize(height - kHard + 1, nHard, pHard)
    column_ind = utils.ind_initialize(width - kHard + 1, nHard, pHard)

    kaiser_window = utils.get_kaiser_window(kHard)
    ri_rj_N__ni_nj, threshold_count = utils.precompute_BM(
        img_noisy, kHW=kHard, NHW=NHard, nHW=nHard, tauMatch=tauMatch
    )
    group_len = int(torch.sum(threshold_count))
    group_3D_table = torch.zeros((group_len, kHard, kHard))
    weight_table = torch.zeros((height, width))
    all_patches = utils.image2patches(img_noisy, kHard, kHard)  # i_j_ipatch_jpatch__v

    if tau_2D == "DCT":
        fre_all_patches = dct.dct_2d(all_patches)
    else:  # 'BIOR'
        fre_all_patches = utils.bior_2d_forward(all_patches)

    acc_pointer = 0
    # print(fre_all_patches[0, 0, 0])
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r].item()
            group_3D = utils.build_3D_group(
                fre_all_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r
            )
            group_3D, weight = ht_filtering_hadamard(
                group_3D, sigma, lambdaHard3D, not useSD
            )
            group_3D = group_3D.permute((2, 0, 1))
            # print(i_r, j_r, group_3D[3, 3, :5])
            group_3D_table[acc_pointer : acc_pointer + nSx_r] = group_3D
            acc_pointer += nSx_r

            if useSD:
                weight = utils.sd_weighting(group_3D)

            weight_table[i_r, j_r] = weight
    # print(group_3D_table[0, :7, :7])
    if tau_2D == "DCT":
        group_3D_table = dct.idct_2d(group_3D_table)
    else:  # 'BIOR'
        group_3D_table = utils.bior_2d_reverse(group_3D_table)

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
    # acc_pointer = 0
    #
    # for i_r in row_ind:
    #     for j_r in column_ind:
    #         nSx_r = threshold_count[i_r, j_r]
    #         N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
    #         group_3D = group_3D_table[acc_pointer : acc_pointer + nSx_r]
    #         acc_pointer += nSx_r
    #         weight = weight_table[i_r, j_r]
    #
    #         for n in range(nSx_r):
    #             ni, nj = N_ni_nj[n]
    #             patch = group_3D[n]
    #
    #             numerator[ni : ni + kHard, nj : nj + kHard] += (
    #                 patch * kaiserWindow * weight
    #             )
    #             denominator[ni : ni + kHard, nj : nj + kHard] += kaiserWindow * weight
    #
    # img_basic = numerator / denominator
    img_basic = __agg_loop(
        row_ind=row_ind,
        column_ind=column_ind,
        threshold_count=threshold_count,
        ri_rj_N__ni_nj=ri_rj_N__ni_nj,
        group_table=group_3D_table,
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
    ri_rj_N__ni_nj: torch.Tensor,
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
            nSx_r = threshold_count[i_r, j_r]
            N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
            group = group_table[acc_pointer : acc_pointer + nSx_r]
            acc_pointer += nSx_r
            weight = weight_table[i_r, j_r]

            for n in range(int(nSx_r.item())):
                ni = N_ni_nj[n][0]
                nj = N_ni_nj[n][1]
                patch = group[n]

                numerator[ni : ni + sliding_len, nj : nj + sliding_len] += (
                    patch * kaiser_window * weight
                )
                denominator[ni : ni + sliding_len, nj : nj + sliding_len] += (
                    kaiser_window * weight
                )

    return numerator / denominator


@torch.jit.script
def __loop_fn(
    threshold_count: torch.Tensor,
    i_r: torch.Tensor,
    j_r: torch.Tensor,
    fre_noisy_patches: torch.Tensor,
    ri_rj_N__ni_nj: torch.Tensor,
    fre_basic_patches: torch.Tensor,
    group_3D_table: torch.Tensor,
    sigma: float,
    useSD: bool,
    acc_pointer: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    nSx_r = threshold_count[i_r, j_r]
    group_3D_img = utils.build_3D_group(
        fre_noisy_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r
    )
    group_3D_est = utils.build_3D_group(
        fre_basic_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r
    )
    group_3D, weight = wiener_filtering_hadamard(
        group_3D_img, group_3D_est, sigma, not useSD
    )
    group_3D = group_3D.permute((2, 0, 1))

    group_3D_table[acc_pointer : acc_pointer + nSx_r] = group_3D

    if useSD:
        weight = utils.sd_weighting(group_3D)

    return weight, nSx_r, group_3D_table


def _bm3d_2nd_step(
    sigma, img_noisy, img_basic, n_wien, k_wien, NWien, pWien, tauMatch, useSD, tau_2D
):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = utils.ind_initialize(height - k_wien + 1, n_wien, pWien)
    column_ind = utils.ind_initialize(width - k_wien + 1, n_wien, pWien)

    kaiser_window = utils.get_kaiser_window(k_wien)
    ri_rj_N__ni_nj, threshold_count = utils.precompute_BM(
        img_basic, kHW=k_wien, NHW=NWien, nHW=n_wien, tauMatch=tauMatch
    )
    group_len = int(torch.sum(threshold_count))
    group_3d_table = torch.zeros(
        (group_len, k_wien, k_wien), device=img_noisy.device, dtype=img_noisy.dtype
    )
    weight_table = torch.zeros(
        (height, width), device=img_noisy.device, dtype=img_noisy.dtype
    )

    noisy_patches = utils.image2patches(
        img_noisy, k_wien, k_wien
    )  # i_j_ipatch_jpatch__v
    basic_patches = utils.image2patches(
        img_basic, k_wien, k_wien
    )  # i_j_ipatch_jpatch__v
    if tau_2D == "DCT":
        fre_noisy_patches = dct.dct_2d(noisy_patches, "ortho")
        fre_basic_patches = dct.dct_2d(basic_patches, "ortho")
    else:  # 'BIOR'
        fre_noisy_patches = utils.bior_2d_forward(noisy_patches)
        fre_basic_patches = utils.bior_2d_forward(basic_patches)

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            weight, nSx_r, group_3d_table = __loop_fn(
                threshold_count,
                i_r,
                j_r,
                fre_noisy_patches,
                ri_rj_N__ni_nj,
                fre_basic_patches,
                group_3d_table,
                sigma,
                useSD,
                acc_pointer,
            )
            # nSx_r = threshold_count[i_r, j_r]
            # group_3D_img = utils.build_3D_group(
            #     fre_noisy_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r
            # )
            # group_3D_est = utils.build_3D_group(
            #     fre_basic_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r
            # )
            # group_3D, weight = wiener_filtering_hadamard(
            #     group_3D_img, group_3D_est, sigma, not useSD
            # )
            # group_3D = group_3D.permute((2, 0, 1))
            #
            # group_3d_table[acc_pointer : acc_pointer + nSx_r] = group_3D
            # acc_pointer += nSx_r
            #
            # if useSD:
            #     weight = utils.sd_weighting(group_3D)
            acc_pointer += nSx_r
            weight_table[i_r, j_r] = weight

    if tau_2D == "DCT":
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
    # denominator = np.pad(denominator, nWien, 'constant', constant_values=1.)
    denominator = pad(denominator, (n_wien, n_wien, n_wien, n_wien), "constant", 1.0)
    # acc_pointer = 0
    # for i_r in row_ind:
    #     for j_r in column_ind:
    #         nSx_r = threshold_count[i_r, j_r]
    #         N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
    #         group_3D = group_3d_table[acc_pointer : acc_pointer + nSx_r]
    #         acc_pointer += nSx_r
    #         weight = weight_table[i_r, j_r]
    #         for n in range(nSx_r):
    #             ni, nj = N_ni_nj[n]
    #             patch = group_3D[n]
    #             # todo: possible to make this not in a loop i think. check shapes / values
    #             numerator[ni : ni + k_wien, nj: nj + k_wien] += (
    #                 patch * kaiser_window * weight
    #             )
    #             denominator[ni : ni + k_wien, nj: nj + k_wien] += kaiser_window * weight
    #
    # img_denoised = numerator / denominator
    img_basic = __agg_loop(
        row_ind=row_ind,
        column_ind=column_ind,
        threshold_count=threshold_count,
        ri_rj_N__ni_nj=ri_rj_N__ni_nj,
        group_table=group_3d_table,
        weight_table=weight_table,
        numerator=numerator,
        sliding_len=k_wien,
        kaiser_window=kaiser_window,
        denominator=denominator,
    )
    return img_basic
