import math

import pytorch_wavelets as twave
import pywt
import torch

# from scipy.fftpack import dct, idct
from torch.nn.functional import pad

from ..dct import dct, idct

# todo: either make my own dct package from the github or use this one:
#       https://github.com/zh217/torch-dct


__all__ = [
    "add_gaussian_noise",
    "bior_2d_forward",
    "bior_2d_reverse",
    "build_3d_group",
    "closest_power_of_2",
    "compute_psnr",
    "get_add_patch_matrix",
    "get_coef",
    "get_kaiser_window",
    "hadamard",
    "image2patches",
    "image2patches_naive",
    "indices_initialize",
    "precompute_block_matching",
    "sd_weighting",
    "translation_2d_mat",
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
    bior_fwd = twave.DWTForward(J=iter_max, wave=wavelet_name, mode="periodization")
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

    bior_inv = twave.DWTInverse(wave=wavelet_name, mode="periodization")

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


@torch.jit.script
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
    group_3D = torch.zeros((nsx_r, k, k), device=fre_all_patches.device)
    for n in range(nsx_r):
        ni = sim_patch_pos[n][0]
        nj = sim_patch_pos[n][1]
        group_3D[n] = fre_all_patches[ni, nj]
    group_3D = group_3D.permute((1, 2, 0))
    return group_3D  # shape=(k, k, nSx_r)


def closest_power_of_2(m, max_):
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
        max_ //= 2
    return m


def compute_psnr(img1, img2):
    """
    Compute the peak signal to noise ratio between two images

    Parameters
    ----------
    img1 : torch.Tensor
    img2 : torch.Tensor

    Returns
    -------
    float
    """
    # img1 = img1.astype(np.float64) / 255.
    # img2 = img2.astype(np.float64) / 255.
    img1 = img1.to(torch.float32) / 255.0
    img2 = img2.to(torch.float32) / 255.0
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(1.0 / mse)


# TODO: add RGB support: need to have the estimate sigma function for the multiple dims
#       if the image is greyscale, then it is just the float value


def get_add_patch_matrix(h, w, nHW, kHW, device="cpu", dtype=torch.float):
    """


    Parameters
    ----------
    h
    w
    nHW
    kHW
    device
    dtype

    Returns
    -------

    """
    row_add = torch.eye(h - 2 * nHW, device=device, dtype=dtype)
    # row_add = np.pad(row_add, nHW, 'constant')
    row_add = pad(row_add, (nHW, nHW, nHW, nHW), "constant")
    row_add_mat = row_add.clone()
    for k in range(1, kHW):
        row_add_mat += translation_2d_mat(row_add, right=k, down=0)

    column_add = torch.eye(w - 2 * nHW, device=device, dtype=dtype)
    # column_add = np.pad(column_add, nHW, 'constant')
    column_add = pad(column_add, (nHW, nHW, nHW, nHW), "constant")
    column_add_mat = column_add.clone()
    for k in range(1, kHW):
        column_add_mat += translation_2d_mat(column_add, right=0, down=k)

    return row_add_mat, column_add_mat


def get_coef(kHW):
    # todo: optimize
    coef_norm = torch.zeros(kHW * kHW)
    coef_norm_inv = torch.zeros(kHW * kHW)
    coef = 0.5 / ((float)(kHW))
    for i in range(kHW):
        for j in range(kHW):
            if i == 0 and j == 0:
                coef_norm[i * kHW + j] = 0.5 * coef
                coef_norm_inv[i * kHW + j] = 2.0
            elif i * j == 0:
                coef_norm[i * kHW + j] = 0.7071067811865475 * coef
                coef_norm_inv[i * kHW + j] = 1.414213562373095
            else:
                coef_norm[i * kHW + j] = 1.0 * coef
                coef_norm_inv[i * kHW + j] = 1.0

    return coef_norm, coef_norm_inv


def get_kaiser_window(kHW):
    # k = np.kaiser(kHW, 2)
    k = torch.kaiser_window(kHW, False, beta=2)
    k_2d = torch.outer(k, k)  # k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


def hadamard(n: int, dtype: torch.dtype = torch.float):
    """
    Lifted directly from scipy and adapted for torch

    Construct an Hadamard matrix.
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

    H = torch.tensor([[1]], dtype=dtype)

    # Sylvester's construction
    for i in range(0, lg2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))

    return H


def image2patches(im, patch_h, patch_w):
    """
    :cut the image into patches
    :param im:
    :param patch_h:
    :param patch_w:
    :return:
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
    # print('h', im.shape)
    return im[h_idx, w_idx]  # .astype(np.float64)


def image2patches_naive(im, patch_h, patch_w):
    """
    :cut the image into patches
    :param im:
    :param patch_h:
    :param patch_w:
    :return:
    """
    im_h, im_w = im.shape[0], im.shape[1]
    patch_table = torch.zeros(
        (im_h - patch_h + 1, im_w - patch_w + 1, patch_h, patch_w)
    )  # og: dtype=np.float64
    for i in range(im_h - patch_h + 1):
        for j in range(im_w - patch_w + 1):
            patch_table[i][j] = im[i : i + patch_h, j : j + patch_w]

    return patch_table


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


def precompute_block_matching(img, kHW, NHW, nHW, tauMatch):
    """
    precompute block matching

    :search for similar patches
    :param img: input image
    :param kHW: length of side of patch
    :param NHW: how many patches are stacked
    :param nHW: length of side of search area
    :param tauMatch: threshold determine whether two patches are similar
    :return ri_rj_N__ni_nj: The top N most similar patches to the referred patch
    :return threshold_count: according to tauMatch how many patches are similar to the referred one
    """
    # img = img.astype(np.float64)
    height, width = img.shape
    Ns = 2 * nHW + 1
    threshold = tauMatch * kHW * kHW
    sum_table = (
        torch.ones((Ns, Ns, height, width), dtype=img.dtype) * 2 * threshold
    )  # di, dj, ph, pw
    row_add_mat, column_add_mat = get_add_patch_matrix(
        height, width, nHW, kHW, dtype=img.dtype, device=img.device
    )

    # diff_margin = \
    #   np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), nHW, 'constant', constant_values=0.)
    hold = torch.ones((height - 2 * nHW, width - 2 * nHW), dtype=img.dtype)
    diff_margin = pad(hold, (nHW, nHW, nHW, nHW), "constant", 0.0)

    sum_margin = (1 - diff_margin) * 2 * threshold

    for di in range(-nHW, nHW + 1):
        for dj in range(-nHW, nHW + 1):
            t_img = translation_2d_mat(img, right=-dj, down=-di)
            diff_table_2 = (img - t_img) * (img - t_img) * diff_margin

            sum_diff_2 = row_add_mat @ diff_table_2 @ column_add_mat
            # todo: check maximum behavior v numpy
            sum_table[di + nHW, dj + nHW] = torch.maximum(sum_diff_2, sum_margin)
            # sum_table (2n+1, 2n+1, height, width) (from repo)

    sum_table = sum_table.reshape((Ns * Ns, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose(1, 0)  # ph_pw__di_dj

    # argsort = np.argpartition(sum_table_T, range(NHW))[:, :NHW]
    # the above argpartition is essentially a torch.topk.indices with largest=False
    bottomk = torch.topk(sum_table_T, NHW, largest=False).indices
    bottomk[:, 0] = (Ns * Ns - 1) // 2
    bottomk_di = bottomk // Ns - nHW
    bottomk_dj = bottomk % Ns - nHW
    hold = torch.arange(height).unsqueeze(-1).unsqueeze(-1)
    near_pi = bottomk_di.reshape((height, width, -1)) + hold
    hold = torch.arange(width).unsqueeze(0).unsqueeze(-1)
    near_pj = bottomk_dj.reshape((height, width, -1)) + hold
    ri_rj_N__ni_nj = torch.cat((near_pi.unsqueeze(-1), near_pj.unsqueeze(-1)), dim=-1)

    sum_filter = torch.where(sum_table_T < threshold, 1, 0)
    threshold_count = torch.sum(sum_filter, axis=1)
    threshold_count = closest_power_of_2(threshold_count, max_=NHW)
    threshold_count = threshold_count.reshape((height, width))

    return ri_rj_N__ni_nj, threshold_count


@torch.jit.script
def sd_weighting(group_3D: torch.Tensor) -> torch.Tensor:
    N = group_3D.numel()

    mean = torch.sum(group_3D)
    std = torch.sum(group_3D * group_3D)

    res = (std - mean * mean / N) / (N - 1)
    zed = torch.tensor(0.0, device=group_3D.device, dtype=group_3D.dtype)
    weight = 1.0 / torch.sqrt(res) if res > zed else zed
    return weight


def translation_2d_mat(mat, right, down):
    mat = torch.roll(mat, right, dims=1)
    mat = torch.roll(mat, down, dims=0)
    return mat
