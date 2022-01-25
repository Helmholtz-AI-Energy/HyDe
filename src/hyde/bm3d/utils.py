import math
from typing import Union

import pytorch_wavelets as twave
import torch

# from scipy.fftpack import dct, idct
from torch.nn.functional import pad

from .. import dwt3d
from ..dct import dct, idct

# todo: either make my own dct package from the github or use this one:
#       https://github.com/zh217/torch-dct


__all__ = [
    "add_gaussian_noise",
    "bior_2d_forward",
    "bior_2d_reverse",
    "build_3d_group",
    "closest_power_of_2",
    "get_kaiser_window",
    "hadamard",
    "image2patches",
    "indices_initialize",
    "precompute_block_matching",
    "sd_weighting",
]


def add_gaussian_noise(im: torch.Tensor, sigma: float, seed: int = None):
    """
    Add gaussian noise to an image (`im`). The added noise will have a standard deviation of `sigma`

    Parameters
    ----------
    im : torch.Tensor
        the base image
    sigma : float
        the standard deviation of the gaussian noise to be added to the image
    seed : int, optional
        if given, this defines the seed for the RNG

    Returns
    -------
    image + noise
        image will be cast to uint8
    """
    if seed is not None:
        torch.random.manual_seed(seed)
    im = im + (sigma * torch.randn(*im.shape)).astype(torch.int)
    im = torch.clip(im, 0.0, 255.0)
    im = im.to(torch.uint8)
    return im


def bior_2d_forward(img: torch.Tensor, wavelet_name: str = "bior1.1"):
    """
    Forward discrete wavelet transform. This is intended for the bior family of wavelets.
    There is also a non-standard joining of the high and low level coefficients done after the
    decomposition. This is from the source code and the result is used in the BM3D steps.

    Parameters
    ----------
    img : torch.Tensor
        image that will be transformed
    wavelet_name
        the name of the wavelet to use for the transformation

    Returns
    -------
    decomposed image composed into a single torch.Tensor

    Notes
    -----
    this should be used with the `bio_2d_reverse` function, for which the input is the output of
    this function.
    """
    assert img.shape[-1] == img.shape[-2]
    assert type(wavelet_name) is str
    iter_max = int(math.log2(img.shape[-1]))
    dtp = img.dtype

    # in the source code, this uses 1.5 but ive found that 1.1 behaves better for the test
    # images. cause is unclear
    bior_fwd = dwt3d.DWTForward(J=iter_max, wave=wavelet_name, padding_method="periodization")
    coeffs = bior_fwd(img.to(torch.float32))

    wave_im = torch.zeros_like(img)

    N = 1
    low_levels = coeffs[1]
    low_levels.reverse()
    wave_im[..., :N, :N] = coeffs[0].to(dtp)
    for i in range(0, iter_max):
        # this inverts the off-diag elements and puts them into the opposite corners
        wave_im[..., N : 2 * N, N : 2 * N] = low_levels[i][:, :, 2].to(dtp)
        wave_im[..., 0:N, N : 2 * N] = -low_levels[i][:, :, 1].to(dtp)
        wave_im[..., N : 2 * N, 0:N] = -low_levels[i][:, :, 0].to(dtp)
        N *= 2

    return wave_im


def bior_2d_reverse(bior_img: torch.Tensor, wavelet_name: str = "bior1.1"):
    """
    Inverse/reverse discrete wavelet transform. This is intended for the bior family of wavelets.
    There is also a non-standard decomposition of the high and low level coefficients done before
    the inverse decomposition. This is from the source code and the result is used in the BM3D
    steps.

    Parameters
    ----------
    bior_img : torch.Tensor
        output of the
    wavelet_name
        the name of the wavelet to use for the transformation

    Returns
    -------
    reconstructed input

    Notes
    -----
    this should be used with the `bio_2d_reverse` function, for which the input is the output of
    this function.
    """
    assert bior_img.shape[-1] == bior_img.shape[-2]
    iter_max = int(math.log2(bior_img.shape[-1]))

    bior_inv = dwt3d.DWTInverse(wave=wavelet_name, padding_method="periodization")

    N = 1
    rec_coeffs = [bior_img[..., 0:1, 0:1].unsqueeze(1), []]
    for i in range(iter_max):
        LL = bior_img[..., N : 2 * N, N : 2 * N].unsqueeze(1)  # sz: N x 3 x Y x Y
        HL = -bior_img[..., 0:N, N : 2 * N].unsqueeze(1)
        LH = -bior_img[..., N : 2 * N, 0:N].unsqueeze(1)
        # todo: check values, this unsqueeze is definitely iffy
        t = torch.cat((LH, HL, LL), dim=1).unsqueeze(1)
        # t = (LH, HL, LL)
        rec_coeffs[1].append(t)
        N *= 2
    rec_coeffs[1].reverse()
    rec_im = bior_inv(rec_coeffs).squeeze(1)

    return rec_im


# @torch.jit.script
def build_3d_group(
    fre_all_patches: torch.Tensor, sim_patch_pos: torch.Tensor, nsx_r: int
) -> torch.Tensor:
    """
    Stack frequency patches into a 3D block

    Parameters
    ----------
    fre_all_patches: torch.Tensor
        input tensor which holds all patches
    sim_patch_pos: torch.Tensor
        the position of the N most similar patches. N is a parameter of the BM3D function.
        This Tensor is calculated by the `precompute_block_matching` function
    nsx_r: int
        number of similar patches according to a threshold calculation

    Returns
    -------
    the 3D block
    """
    _, _, k, k_ = fre_all_patches.shape
    assert k == k_
    group_3D = fre_all_patches[sim_patch_pos[:nsx_r, 0], sim_patch_pos[:nsx_r, 1]].permute(
        (1, 2, 0)
    )
    return group_3D  # shape=(k, k, nSx_r)


def closest_power_of_2(m: torch.Tensor, max_: int):
    """
    Determine the closest power of 2 for all elements in a Tensor which are less than max_

    Parameters
    ----------
    m: torch.Tensor
        input tensor
    max_: int
        the maximum

    Returns
    -------
    tensor with the values with the closest value of 2 for each value
    """
    m = torch.where(max_ < m, max_, m)
    while max_ > 1:
        m = torch.where((max_ // 2 < m) * (m < max_), max_ // 2, m)
        max_ = max_ // 2
    return m


# TODO: add RGB support: need to have the estimate sigma function for the multiple dims
#       if the image is greyscale, then it is just the float value

# @torch.jit.script
def __get_add_patch_matrix(
    h: int,
    w: int,
    boundary_sz: int,
    patch_sz: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
):
    # helper function to abstract the creation of the adding row and column matrices?
    row_add = torch.eye(h - 2 * boundary_sz, device=device, dtype=dtype)
    # row_add = np.pad(row_add, nHW, 'constant')
    row_add = pad(row_add, (boundary_sz, boundary_sz, boundary_sz, boundary_sz), "constant")
    row_add_mat = row_add.clone()
    for k in range(1, patch_sz):
        row_add_mat += __translation_2d_mat(row_add, right=k, down=0)

    column_add = torch.eye(w - 2 * boundary_sz, device=device, dtype=dtype)
    # column_add = np.pad(column_add, nHW, 'constant')
    column_add = pad(column_add, (boundary_sz, boundary_sz, boundary_sz, boundary_sz), "constant")
    column_add_mat = column_add.clone()
    for k in range(1, patch_sz):
        column_add_mat += __translation_2d_mat(column_add, right=0, down=k)

    return row_add_mat, column_add_mat


def get_kaiser_window(pts: int, dev: Union[torch.device, str]):
    """
    Get a 2D Kaiser window. This will generate a 1D, pts-point Kaiser window first, then it will
    perform an outer product to make it 2D.

    Parameters
    ----------
    pts : int
        the number of points for the Kaiser windo

    Returns
    -------
    window : torch.Tensor
        2D Kaiser window

    """
    # k = np.kaiser(kHW, 2)
    k = torch.kaiser_window(pts, False, beta=2, device=dev)
    k_2d = torch.outer(k, k)  # k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


def hadamard(n: int, dtype: torch.dtype = torch.float, dev: torch.device = torch.device("cpu")):
    """
    Lifted directly from scipy and adapted for torch

    Construct a Hadamard matrix.
    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction. `n` must be a power of 2.

    Parameters
    ----------
    n : int
        The order of the matrix. `n` must be a power of 2.
    dtype : torch.dtype
        The data type of the array to be constructed.

    Returns
    -------
    H : (n, n) torch.Tensor
        The Hadamard matrix.
    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be " "a power of 2")

    H = torch.tensor([[1]], dtype=dtype, device=dev)

    # Sylvester's construction
    for i in range(0, lg2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))

    return H


def image2patches(im, patch_h, patch_w):
    """
    Cut an image into patches

    Parameters
    ----------
    im : torch.Tensor
        the base image
    patch_h : int
        the patch height
    patch_w : int
        the patch width

    Returns
    -------
    patches : torch.Tensor
        4D tensor containing the image patches
    """
    im_h, im_w = im.shape[0], im.shape[1]
    im_h_idx = torch.arange(im_h - patch_h + 1)
    im_h_idx = im_h_idx.reshape((im_h_idx.shape[0], 1, 1, 1))
    im_w_idx = torch.arange(im_w - patch_w + 1)
    im_w_idx = im_w_idx.reshape((1, im_w_idx.shape[0], 1, 1))
    # [np.newaxis, :, np.newaxis, np.newaxis]
    patch_h_idx = torch.arange(patch_h)
    patch_h_idx = patch_h_idx.reshape((1, 1, patch_h_idx.shape[0], 1))
    # [np.newaxis, np.newaxis, :, np.newaxis]
    patch_w_idx = torch.arange(patch_w)
    patch_w_idx = patch_w_idx.reshape((1, 1, 1, patch_w_idx.shape[0]))
    # [np.newaxis, np.newaxis, np.newaxis, :]
    h_idx = im_h_idx + patch_h_idx
    w_idx = im_w_idx + patch_w_idx

    return im[h_idx, w_idx]  # .astype(np.float64)


def indices_initialize(max_size, N, step, device):
    """
    create a range of indices with a step as given. If the last element (max_size - N - 1)
    if not included in the list, it is cat'ed to the end.

    Parameters
    ----------
    max_size
    N
    step
    device

    Returns
    -------

    """
    ind = torch.arange(N, max_size - N, step, device=device)
    if ind[-1] < max_size - N - 1:
        ind = torch.cat((ind, torch.tensor([max_size - N - 1], device=device)), dim=0)
    return ind


# @torch.jit.script
def precompute_block_matching(
    img: torch.Tensor, patch_size: int, npatches: int, boundary_sz: int, tau_match: float
):
    """
    precompute similar blocks using threshold distances

    Parameters
    ----------
    img : torch.Tensor
        the base image
    patch_size : int
        length of side of patch, patches are (patch_size_h x patch_size_h)
    npatches : int
        maximum similar patches wanted for each patch
    boundary_sz : int
        size of the boundary around img_noisy
    tau_match : float
        threshold determine whether two patches are similar

    Returns
    -------
    ri_rj_N__ni_nj : torch.Tensor
        The top N most similar patches to the referred patch
    threshold_count : torch.Tensor
        how many patches are similar to the referred one according to tau_match
    """
    # img = img.astype(np.float64)
    height, width = img.shape
    Ns = 2 * boundary_sz + 1
    dev = img.device
    threshold = tau_match * patch_size * patch_size
    sum_table = (
        torch.ones((Ns, Ns, height, width), dtype=img.dtype, device=dev) * 2 * threshold
    )  # di, dj, ph, pw

    # ---------------------------------------------------------------------------------
    row_add = torch.eye(height - 2 * boundary_sz, dtype=img.dtype, device=dev)
    # row_add = np.pad(row_add, nHW, 'constant')
    row_add = pad(row_add, (boundary_sz, boundary_sz, boundary_sz, boundary_sz), "constant")
    row_add_mat = row_add.clone()
    for k in range(1, patch_size):
        row_add_mat += torch.roll(row_add, k, dims=1)
        # mat = torch.roll(mat, 0, dims=0)
        # row_add_mat += __translation_2d_mat(row_add, right=k, down=0)

    column_add = torch.eye(width - 2 * boundary_sz, dtype=img.dtype, device=dev)
    # column_add = np.pad(column_add, nHW, 'constant')
    column_add = pad(column_add, (boundary_sz, boundary_sz, boundary_sz, boundary_sz), "constant")
    column_add_mat = column_add.clone()
    for k in range(1, patch_size):
        column_add_mat += torch.roll(column_add, k, dims=0)
        # column_add_mat += __translation_2d_mat(column_add, right=0, down=k)
    # ---------------------------------------------------------------------------------

    # row_add_mat, column_add_mat = __get_add_patch_matrix(
    #     height, width, boundary_sz, patch_size, dtype=img.dtype, device=dev
    # )

    # diff_margin = \
    #   np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), nHW, 'constant', constant_values=0.)
    hold = torch.ones(
        (height - 2 * boundary_sz, width - 2 * boundary_sz), dtype=img.dtype, device=dev
    )
    diff_margin = pad(hold, (boundary_sz, boundary_sz, boundary_sz, boundary_sz), "constant", 0.0)

    sum_margin = (1 - diff_margin) * 2 * threshold

    for di in range(-boundary_sz, boundary_sz + 1):
        for dj in range(-boundary_sz, boundary_sz + 1):
            _block_matching_helper(
                img,
                dj,
                di,
                diff_margin,
                row_add_mat,
                column_add_mat,
                sum_table,
                boundary_sz,
                sum_margin,
            )
            # sum_table (2n+1, 2n+1, height, width) (from repo)

    sum_table = sum_table.reshape((Ns * Ns, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose(1, 0)  # ph_pw__di_dj

    # argsort = np.argpartition(sum_table_T, range(NHW))[:, :NHW]
    # the above argpartition is essentially a torch.topk.indices with largest=False
    bottomk = torch.topk(sum_table_T, npatches, largest=False).indices
    bottomk[:, 0] = (Ns * Ns - 1) // 2
    # bottomk[:, 0] = torch.div(Ns * Ns - 1, 2, rounding_mode="floor")
    bottomk_di = torch.div(bottomk, Ns, rounding_mode="floor") - boundary_sz
    bottomk_dj = bottomk % Ns - boundary_sz
    hold = torch.arange(height, device=dev).unsqueeze(-1).unsqueeze(-1)
    near_pi = bottomk_di.reshape((height, width, -1)) + hold
    hold = torch.arange(width, device=dev).unsqueeze(0).unsqueeze(-1)
    near_pj = bottomk_dj.reshape((height, width, -1)) + hold
    ri_rj_N__ni_nj = torch.cat((near_pi.unsqueeze(-1), near_pj.unsqueeze(-1)), dim=-1)

    sum_filter = torch.where(sum_table_T < threshold, 1, 0)
    # threshold_count = torch.sum(sum_filter, axis=1)
    threshold_count = sum_filter.sum(1)
    threshold_count = closest_power_of_2(threshold_count, max_=npatches)
    threshold_count = threshold_count.reshape((height, width))

    return ri_rj_N__ni_nj, threshold_count


@torch.jit.script
def _block_matching_helper(
    img: torch.Tensor,
    dj: int,
    di: int,
    diff_margin: torch.Tensor,
    row_add_mat: torch.Tensor,
    column_add_mat: torch.Tensor,
    sum_table: torch.Tensor,
    boundary_sz: int,
    sum_margin: torch.Tensor,
):
    t_img = torch.roll(img, -dj, dims=1)
    t_img = torch.roll(t_img, -di, dims=0)
    # t_img = __translation_2d_mat(img, right=-dj, down=-di)
    diff_table_2 = (img - t_img) * (img - t_img) * diff_margin

    sum_diff_2 = row_add_mat @ diff_table_2 @ column_add_mat
    # todo: check maximum behavior v numpy
    sum_table[di + boundary_sz, dj + boundary_sz] = torch.maximum(sum_diff_2, sum_margin)


# @torch.jit.script
def sd_weighting(group_3D: torch.Tensor) -> torch.Tensor:
    """
    Calculate the standard deviation weighting

    Parameters
    ----------
    group_3D : torch.Tensor
        the 3D group of patches to weight with the standard deviation

    Returns
    -------
    weights : torch.Tensor
        the sd weights
    """
    N = group_3D.numel()

    mean = torch.sum(group_3D)
    std = torch.sum(group_3D * group_3D)

    res = (std - mean * mean / N) / (N - 1)
    zed = torch.tensor(0.0, device=group_3D.device, dtype=group_3D.dtype)
    weight = 1.0 / torch.sqrt(res) if res > zed else zed
    return weight


def __translation_2d_mat(mat, right, down):
    # utility function to roll a tensor to the right then down
    mat = torch.roll(mat, right, dims=1)
    mat = torch.roll(mat, down, dims=0)
    return mat
