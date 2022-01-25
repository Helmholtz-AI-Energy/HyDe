# import numpy as np
from typing import Tuple, Union

import torch
from torch.nn.functional import pad

from .. import dct
from ..utils import symmetric_pad
from . import utils
from .filtering import ht_filtering_hadamard, wiener_filtering_hadamard

__all__ = ["bm3d"]


def bm3d(
    noisy_im: torch.Tensor,
    sigma: float = 25.0,
    bndry_sz_h: int = 16,  # n_H -> 16 (from BM3D_py)
    patch_size_h: int = 8,  # k_H -> N1 = 8
    num_patches_h: int = 16,  # N_H -> N2 = 16
    patch_buff_h: int = 3,  # p_H -> Nstep = 3
    tau_match_h: Union[int, float] = 3000,  # tauMatch_H -> 3000
    use_sd_h: bool = False,  # useSD_H -> False (from BM3D_py)
    trans_2d_h: str = "BIOR",  # tau_2D_H -> "BIOR"
    hard_thres_h: float = 2.7,  # lambda3D_H -> lambda_thr3D = 2.7
    bndry_sz_w: int = 16,  # n_W -> ??? 16 (from BM3D_py)
    patch_size_w: int = 8,  # k_W -> N1_wiener = 8
    num_patches_w: int = 32,  # N_W -> N2_wiener = 32
    patch_buff_w: int = 3,  # p_W -> Nstep_wiener = 3
    tau_match_w: Union[int, float] = 400,  # -> tau_match_wiener = 400
    use_sd_w: bool = True,  # -> ??? (from BM3D_py)
    trans_2d_w: str = "DCT",  # tau_2D_W  -> "DCT"
):
    """
    todo: check the following things:
        lambda_thr2D        = 0;   %% threshold parameter for the coarse initial denoising used in the d-distance measure

    BM3D denoising. This is composed of two steps:

        1. estimate the image noise using hard thresholding during the collaborative filtering
        2. Using both the original image and the result of the first step, use Wiener filtering
            to determine the remaining noise

    For more information see [1].

    This implementation is based heavily on https://github.com/Ryanshuai/BM3D_py/

    Parameters
    ----------
    noisy_im : torch.Tensor
        noisy image
    sigma : float
        expected noise (standard deviation)

    The following parameters exist for both hard thresholding (..._h) and wiener filtering (..._w).
    The doc strings are the same for each entry but they can have different effects in the
    sub-functions.

    bndry_sz_h : int
        size of the boundary around img_noisy
    patch_size_h : int
        length of side of patch, patches are (patch_size_h x patch_size_h)
    num_patches_h : int
        maximum similar patches wanted for each patch
    patch_buff_h : int
        number of pixels between patches (size of buffer between patches)
    tau_match_h : int, float
        threshold determine whether two patches are similar
    use_sd_h : bool
        if true, use weight based on the standard variation of the 3D group for the first
        (resp. second) step, otherwise use the number of non-zero coefficients after Hard
        Thresholding.
    trans_2d_h : str
        Transformation to apply to reduce/find/remove the noice in the image
        Can either be "BIOR" or "DCT".
        Other methods are not implemented and may cause shape issues in later functions
    hard_thres_h : float
        Threshold for Hard Thresholding
    bndry_sz_w : int
        size of the boundary around img_noisy
    patch_size_w : int
        length of side of patch, patches are (patch_size_w x patch_size_w)
    num_patches_w : int
        maximum similar patches wanted for each patch
    patch_buff_w : int
        number of pixels between patches (size of buffer between patches)
    tau_match_w : int, float
        threshold determine whether two patches are similar
    use_sd_w : bool
        if true, use weight based on the standard variation of the 3D group for the first
        (resp. second) step, otherwise use the number of non-zero coefficients after Hard
        Thresholding.
    trans_2d_w : str
        Transformation to apply to reduce/find/remove the noice in the image
        Can either be "BIOR" or "DCT".
        Other methods are not implemented and may cause shape issues in later functions

    Returns
    -------
    img_basic: torch.Tensor
        the output of the 1st denoising step (hard thresholding)
    img_denoised: torch.Tensor
        the output of the 2nd denoising step (wiener filtering). The second step uses the output
        of the first as an imput

    Notes
    -----
    [1] Lebrun, M. (2012). An Analysis and Implementation of the BM3D Image Denoising Method. Image Processing On Line, 2, 175–213.
    """
    # todo: raise statements for transforms
    patch_size_h = 8 if (trans_2d_h == "BIOR" or sigma < 40.0) else patch_size_h
    patch_size_w = 8 if (trans_2d_w == "BIOR" or sigma < 40.0) else patch_size_w

    # todo: is this be needed?
    # tau_match_h = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
    # tau_match_w = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches

    noisy_im_p = symmetric_pad(noisy_im, bndry_sz_h)
    img_basic = _bm3d_1st_step_ht(
        sigma=sigma,
        img_noisy=noisy_im_p,
        img_bndry_sz=bndry_sz_h,
        patch_sz=patch_size_h,
        npatches=num_patches_h,
        patch_buff=patch_buff_h,
        lambda_hard_3d=hard_thres_h,
        tau_match=tau_match_h,
        use_sd=use_sd_h,
        transform=trans_2d_h,
    )
    img_basic = img_basic[bndry_sz_h:-bndry_sz_h, bndry_sz_h:-bndry_sz_h]

    assert not torch.any(torch.isnan(img_basic))
    img_basic_p = symmetric_pad(img_basic, bndry_sz_w)
    noisy_im_p = symmetric_pad(noisy_im, bndry_sz_w)
    img_denoised = _bm3d_2nd_step_hadamard(
        sigma=sigma,
        img_noisy=noisy_im_p,
        img_basic=img_basic_p,
        img_bndry_sz=bndry_sz_w,
        patch_sz=patch_size_w,
        npatches=num_patches_w,
        patch_buff=patch_buff_w,
        tau_match=tau_match_w,
        use_sd=use_sd_w,
        transform=trans_2d_w,
    )
    img_denoised = img_denoised[bndry_sz_w:-bndry_sz_w, bndry_sz_w:-bndry_sz_w]

    return img_basic, img_denoised


def _bm3d_1st_step_ht(
    sigma: float,
    img_noisy: torch.Tensor,
    img_bndry_sz: int,
    patch_sz: int,
    npatches: int,
    patch_buff: int,
    lambda_hard_3d: float,
    tau_match: Union[int, float],
    use_sd: bool,
    transform: str,
):
    """
    1st step of BM3D denoising. This step uses hard thersholding during the collaborative filtering

    Parameters
    ----------
    sigma: float
        the expected noise of an image
    img_noisy: torch.Tensor
        the noisy image
    img_bndry_sz: int
        size of the boundary around img_noisy
    patch_sz: int
        length of side of patch, patches are (patch_size_h x patch_size_h)
    npatches: int
        maximum similar patches wanted for each patch
    patch_buff: int
        number of pixels between patches (size of buffer between patches)
    lambda_hard_3d: float
        Threshold for Hard Thresholding
    tau_match: int, float
        threshold determine whether two patches are similar
    use_sd: bool
        if true, use weight based on the standard variation of the 3D group for the first
        (resp. second) step, otherwise use the number of non-zero coefficients after Hard
        Thresholding.
    transform: str
        Transformation to apply to reduce/find/remove the noice in the image
        Can either be "BIOR" or "DCT".
        Other methods are not implemented and may cause shape issues in later functions

    Returns
    -------
    denoised_ht : torch.Tensor
        the denoised image
    """
    height, width = img_noisy.shape[0], img_noisy.shape[1]
    dev = img_noisy.device
    dtp = img_noisy.dtype
    # create a tensor of indices max_size, N, step -> range(nHard, height - kHard + 1, pHard)
    row_ind = utils.indices_initialize(height - patch_sz + 1, img_bndry_sz, patch_buff, device=dev)
    col_ind = utils.indices_initialize(width - patch_sz + 1, img_bndry_sz, patch_buff, device=dev)

    kaiser_window = utils.get_kaiser_window(patch_sz, dev=dev)
    ri_rj_n__ni_nj, threshold_count = utils.precompute_block_matching(
        img_noisy,
        patch_size=patch_sz,
        npatches=npatches,
        boundary_sz=img_bndry_sz,
        tau_match=tau_match,
    )
    # ri_rj_n__ni_nj -> The top N most similar patches to the referred patch
    # threshold_count -> (according to tau_match) how many patches are similar to the referred one
    group_len = int(torch.sum(threshold_count))
    group_3d_table = torch.zeros((group_len, patch_sz, patch_sz), dtype=dtp, device=dev)
    weight_table = torch.zeros((height, width), dtype=dtp, device=dev)
    all_patches = utils.image2patches(img_noisy, patch_sz, patch_sz)

    if transform == "DCT":
        fre_all_patches = dct.dct_2d(all_patches)
    elif transform == "BIOR":
        fre_all_patches = utils.bior_2d_forward(all_patches)
    else:  # anything different
        raise ValueError(f"the given transform ({transform}) is not in ['BIOR', 'DCT']")

    acc_pointer = 0

    for i_r in row_ind:
        for j_r in col_ind:
            nsx_r = threshold_count[i_r, j_r].item()
            group_3d = utils.build_3d_group(fre_all_patches, ri_rj_n__ni_nj[i_r, j_r], nsx_r)
            group_3d, weight = ht_filtering_hadamard(group_3d, sigma, lambda_hard_3d, not use_sd)
            group_3d = group_3d.permute((2, 0, 1))
            group_3d_table[acc_pointer : acc_pointer + nsx_r] = group_3d
            acc_pointer += nsx_r

            if use_sd:
                weight = utils.sd_weighting(group_3d)

            weight_table[i_r, j_r] = weight

    if transform == "DCT":
        group_3d_table = dct.idct_2d(group_3d_table)
    else:  # 'BIOR'
        group_3d_table = utils.bior_2d_reverse(group_3d_table)

    # aggregation part
    numerator = torch.zeros_like(img_noisy)
    denominator = torch.zeros(
        (img_noisy.shape[0] - 2 * img_bndry_sz, img_noisy.shape[1] - 2 * img_bndry_sz),
        dtype=img_noisy.dtype,
        device=img_noisy.device,
    )
    denominator = pad(
        denominator,
        (img_bndry_sz, img_bndry_sz, img_bndry_sz, img_bndry_sz),
        "constant",
        1.0,
    )
    # print(group_3d_table.device, weight_table.device)
    img_basic = __agg_loop(
        row_ind=row_ind,
        col_ind=col_ind,
        threshold_count=threshold_count,
        ri_rj_n__ni_nj=ri_rj_n__ni_nj,
        group_table=group_3d_table,
        weight_table=weight_table,
        numerator=numerator,
        patch_sz=patch_sz,
        kaiser_window=kaiser_window,
        denominator=denominator,
    )
    print("finished 1st step")
    return img_basic


@torch.jit.script
def __agg_loop(
    row_ind: torch.Tensor,
    col_ind: torch.Tensor,
    threshold_count: torch.Tensor,
    ri_rj_n__ni_nj: torch.Tensor,
    group_table: torch.Tensor,
    weight_table: torch.Tensor,
    numerator: torch.Tensor,
    patch_sz: int,
    kaiser_window: torch.Tensor,
    denominator: torch.Tensor,
):
    # this function abstracts the aggregation of the numerator and denominator
    # i dont understand why all of this is done, i got it from the source and it works...
    acc_pointer = 0
    for i_r in row_ind:
        for j_r in col_ind:
            nsx_r = threshold_count[i_r, j_r]
            n_ni_nj = ri_rj_n__ni_nj[i_r, j_r]
            group = group_table[acc_pointer : acc_pointer + nsx_r]
            acc_pointer += nsx_r
            weight = weight_table[i_r, j_r] * kaiser_window
            # aggregate the indices for setting
            # grp_sh = group[0].shape[0]
            # dim0 = torch.empty((group[0].shape[0] * nsx_r), dtype=torch.long)
            # dim1 = torch.empty((group[0].shape[0] * nsx_r), dtype=torch.long)
            for n in range(int(nsx_r.item())):
                ni = n_ni_nj[n][0]
                nj = n_ni_nj[n][1]
                patch = group[n]
                # dim0[n*grp_sh: (n+1)*grp_sh] = torch.arange(ni, ni + patch_sz, device=row_ind.device)
                # dim1[n*grp_sh: (n+1)*grp_sh] = torch.arange(nj, nj + patch_sz, device=row_ind.device)
                # print(numerator[ni: ni + patch_sz, nj: nj + patch_sz].shape, patch.shape, weight.shape)
                numerator[ni : ni + patch_sz, nj : nj + patch_sz] += patch * weight
                denominator[ni : ni + patch_sz, nj : nj + patch_sz] += weight

            # # ni = n_ni_nj[:nsx_r, 0]
            # # nj = n_ni_nj[:nsx_r, 1]
            # patchs = group[:nsx_r]
            # print(dim0)
            # print('h\n', numerator[dim0, dim1].shape, patchs.shape, weight.shape)
            # hld = patchs * weight
            # numerator[dim0, dim1] += hld.flatten()
            # denominator[dim0, dim1] += weight.expand((nsx_r * grp_sh * grp_sh))

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
    # calculate the weights for the 2nd bm3d step
    nsx_r = threshold_count[i_r, j_r]
    group_3d_img = utils.build_3d_group(fre_noisy_patches, ri_rj_n__ni_nj[i_r, j_r], nsx_r)
    group_3d_est = utils.build_3d_group(fre_basic_patches, ri_rj_n__ni_nj[i_r, j_r], nsx_r)
    group_3d, weight = wiener_filtering_hadamard(group_3d_img, group_3d_est, sigma, not use_sd)
    group_3d = group_3d.permute((2, 0, 1))

    group_3d_table[acc_pointer : acc_pointer + nsx_r] = group_3d

    if use_sd:
        weight = utils.sd_weighting(group_3d)

    return weight, nsx_r, group_3d_table


def _bm3d_2nd_step_hadamard(
    sigma,
    img_noisy,
    img_basic,
    img_bndry_sz,
    patch_sz,
    npatches,
    patch_buff,
    tau_match,
    use_sd,
    transform,
):
    """
    The second step of BM3D denoising. This uses both the original noisy image and the results of
    the first step. This uses Wiener filtering instead of hard thresholding.

    Parameters
    ----------
    sigma: float
        the expected noise of an image
    img_noisy: torch.Tensor
        the noisy image
    img_basic: torch.Tensor
        the result of the first step of m3d
    img_bndry_sz: int
        size of the boundary around img_noisy
    patch_sz: int
        length of side of patch, patches are (patch_size_h x patch_size_h)
    npatches: int
        maximum similar patches wanted for each patch
    patch_buff: int
        number of pixels between patches (size of buffer between patches)
    tau_match: int, float
        threshold determine whether two patches are similar
    use_sd: bool
        if true, use weight based on the standard variation of the 3D group for the first
        (resp. second) step, otherwise use the number of non-zero coefficients after Hard
        Thresholding.
    transform: str
        Transformation to apply to reduce/find/remove the noice in the image
        Can either be "BIOR" or "DCT".
        Other methods are not implemented and may cause shape issues in later functions

    Returns
    -------
    denoised_image : torch.Tensor
        the denoised image using Wiener filtering
    """
    height, width = img_noisy.shape[0], img_noisy.shape[1]
    dev = img_noisy.device
    dtp = img_noisy.dtype

    row_ind = utils.indices_initialize(height - patch_sz + 1, img_bndry_sz, patch_buff, device=dev)
    col_ind = utils.indices_initialize(width - patch_sz + 1, img_bndry_sz, patch_buff, device=dev)

    kaiser_window = utils.get_kaiser_window(patch_sz, dev=dev)
    ri_rj_n__ni_nj, threshold_count = utils.precompute_block_matching(
        img_basic,
        patch_size=patch_sz,
        npatches=npatches,
        boundary_sz=img_bndry_sz,
        tau_match=tau_match,
    )
    group_len = int(torch.sum(threshold_count))
    group_3d_table = torch.zeros((group_len, patch_sz, patch_sz), dtype=dtp, device=dev)
    weight_table = torch.zeros((height, width), dtype=dtp, device=dev)

    noisy_patches = utils.image2patches(img_noisy, patch_sz, patch_sz)
    basic_patches = utils.image2patches(img_basic, patch_sz, patch_sz)
    if transform == "DCT":
        fre_noisy_patches = dct.dct_2d(noisy_patches, "ortho")
        fre_basic_patches = dct.dct_2d(basic_patches, "ortho")
    else:  # 'BIOR'
        fre_noisy_patches = utils.bior_2d_forward(noisy_patches)
        fre_basic_patches = utils.bior_2d_forward(basic_patches)

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in col_ind:
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

    if transform == "DCT":
        group_3d_table = dct.idct_2d(group_3d_table, "ortho")
    else:  # 'BIOR'
        group_3d_table = utils.bior_2d_reverse(group_3d_table)

    # aggregation part
    numerator = torch.zeros_like(img_noisy)
    denominator = torch.zeros(
        (img_noisy.shape[0] - 2 * img_bndry_sz, img_noisy.shape[1] - 2 * img_bndry_sz),
        dtype=dtp,
        device=dev,
    )
    denominator = pad(
        denominator,
        (img_bndry_sz, img_bndry_sz, img_bndry_sz, img_bndry_sz),
        "constant",
        1.0,
    )

    img_basic = __agg_loop(
        row_ind=row_ind,
        col_ind=col_ind,
        threshold_count=threshold_count,
        ri_rj_n__ni_nj=ri_rj_n__ni_nj,
        group_table=group_3d_table,
        weight_table=weight_table,
        numerator=numerator,
        patch_sz=patch_sz,
        kaiser_window=kaiser_window,
        denominator=denominator,
    )
    return img_basic
