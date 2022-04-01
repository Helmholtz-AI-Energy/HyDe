from typing import Iterable, Optional, Tuple, Union

import kornia.metrics as km
import numpy as np
import torch
from torch.nn.functional import max_pool2d, pad

from .logging import get_logger

logger = get_logger()

__all__ = [
    "adaptive_median_filtering",
    "atleast_3d",
    "custom_pca_image",
    "diff",
    "diff_dim0_replace_last_row",
    "estimate_hyperspectral_noise",
    "hysime",
    "normalize",
    "normalize_w_consts",
    "peak_snr",
    "rescale_wo_shift",
    "sam",
    "scale_wo_shift",
    "snr",
    "soft_threshold",
    "sure_thresh",
    "sure_soft_modified_lr2",
    "symmetric_pad",
    "vertical_difference",
    "vertical_difference_transpose",
    "undo_normalize",
]


def adaptive_median_filtering(image: torch.Tensor, max_size: int) -> torch.Tensor:
    """
    Perform adaptive median filtering on `image`. The median filter starts at size 3-by-3 and iterates up
    to size `max_size` x `max_size`.

    This will use pooling from torch to parallelize this function. However, at time of writing (25.01.2022)
    there does not exist a memory efficient way to do median pooling. This results in a larger memory
    footprint than may be expected.

    Parameters
    ----------
    image: torch.Tensor
        image to do adaptive median filtering on
    max_size: int
        MUST BE ODD AND > 3
        the kernel size to use in the filters

    Returns
    -------
    adaptive_median: torch.Tensor

    Original Lisense
    ----------------
    Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
    Digital Image Processing Using MATLAB, Prentice-Hall, 2004
    $Revision: 1.5 $  $Date: 2003/11/21 14:19:05 $

    """
    # NOTE: image must be 2D here

    # max_size must be an odd, positive integer greater than 1.
    if max_size <= 1 or max_size % 2 == 0:
        raise ValueError(f"max_size must be an odd integer > 1, currently: {max_size}")
    # % Initial setup.
    f = torch.zeros_like(image)

    already_procd = torch.zeros_like(image).to(dtype=torch.bool)
    # % Begin filtering.
    for k in range(3, max_size + 1, 2):
        imagep = symmetric_pad(image, k).unsqueeze(0).unsqueeze(0)
        # ordfilt2 -> can use pooling from torch
        kernel = k, k
        stride = 1, 1
        sl = (slice(k, k + image.shape[0]), slice(k, k + image.shape[1]))
        zmin = -max_pool2d(imagep, kernel, stride=stride)
        zmin = zmin.squeeze()[sl]

        zmax = max_pool2d(imagep, kernel, stride=stride)
        zmax = zmax.squeeze()[sl]

        zmed = imagep.unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1])
        zmed = zmed.contiguous().view(zmed.size()[:4] + (-1,)).median(dim=-1)[0].squeeze()
        zmed = zmed[sl]

        process_using_level_b = (zmed > zmin) & (zmax > zmed) & ~already_procd
        zB = (image > zmin) & (zmax > image)

        output_zxy = process_using_level_b & zB
        output_zmed = process_using_level_b & ~zB
        f[output_zxy] = image[output_zxy]
        f[output_zmed] = zmed[output_zmed]

        already_procd = already_procd | process_using_level_b
        if torch.all(already_procd):
            break

    return zmed


def atleast_3d(image):
    """
    to ensure the image has at least 3 dimensions
    if the input image already has at least 3 dimensions, just return the image
    otherwise, extend the dimensionality of the image to 3 dimensions

    Parameters
    ----------
    image : torch.Tensor
        input image

    Return
    ------
    image : torch.Tensor
        input image
    """
    dim = list(image.shape)

    if len(dim) >= 3:
        return image
    else:
        dim.append(1)
        return image.view(dim)


def custom_pca_image(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This will do a standard SVD on a 2D projections of `img`. `U @ S`
    is reshaped into a 3D tensor to match the shape im img at the start.
    MUST BE IN (ROWS, COLS, CHANNELS)

    Parameters
    ----------
    img: torch.Tensor
        MUST BE IN (ROWS, COLS, CHANNELS)

    Returns
    -------
    v_pca: torch.Tensor
        the vh from torch.linalg.svd(img, (nr * nc, p), full_matrices=False)
    pc: torch.Tensor
        U @ s reshaped into the number of rows and columns of the input img
    """
    nr, nc, p = img.shape
    # y_w -> h x w X c
    im1 = torch.reshape(img, (nr * nc, p))
    u, s, v_pca = torch.linalg.svd(im1, full_matrices=False)
    # need to modify u and s
    pc = torch.matmul(u, torch.diag(s))
    pc = pc.reshape((nc, nr, p))
    return v_pca, pc


def diff(x: torch.Tensor, n: int = 1, dim=0) -> torch.Tensor:
    """
    Find the differences in x. This will return a torch.Tensor of the same shape as `x`
    with the requisite number of zeros added to the end (`n`).

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    n: int, optional
        the number of rows between each row for which to calculate the diff
    dim: int, optional
        the dimension to find the diff on

    Returns
    -------
    diffs : torch.Tensor
        the differences. This result is the same size as `x`.
    """
    y = torch.zeros_like(x)
    ret = x
    for _ in range(n):
        ret = torch.diff(ret, dim=dim)
    y[: ret.shape[0]] = ret
    return y


def diff_dim0_replace_last_row(x: torch.Tensor) -> torch.Tensor:
    """
    Find the single row differences in x and then put the second to last row as the last row in
    the result

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    diff_0_last_row: torch.Tensor
        the single row differences with the second to last row and the last row
    """
    u0 = (-1.0 * x[0]).unsqueeze(0)
    u1 = (-1.0 * torch.diff(x, dim=0))[:-1]
    u2 = (x[-2]).unsqueeze(0)
    ret = torch.cat([u0, u1, u2], dim=0)
    return ret


def prep_out_bregman(image: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    sets the values of `out` to be what symmetric padding image with a single row/col would be

    Parameters
    ----------
    image: torch.Tensor
        original image
    out: torch.Tensor
        output tensor for denoise TV Bregman

    Returns
    -------
    torch.Tensor:
        out tensor set with values from image
    """
    out_rows, out_cols = out.shape[:2]
    rows, cols = out_rows - 2, out_cols - 2

    out[1 : out_rows - 1, 1 : out_cols - 1] = image

    out[0, 1 : out_cols - 1] = image[1, :]
    out[1 : out_rows - 1, 0] = image[:, 1]
    out[out_rows - 1, 1 : out_cols - 1] = image[rows - 1, :]
    out[1 : out_rows - 1, out_cols - 1] = image[:, cols - 1]
    return out


def estimate_hyperspectral_noise(
    data: torch.Tensor,
    noise_type: str = "additive",
    calculation_dtype: torch.dtype = torch.float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Infer the noise in a hyperspectral dataset. Assumes that the
    reflectance at a given band is well modelled by a linear regression
    on the remaining bands

    Parameters
    ----------
    data: torch.Tensor
        an LxN matrix with the hyperspectral data where L is the number of bands
        and N is the number of pixels
    noise_type: str, optional
        The type of noise to estimate.
        Options: ["additive", "poisson"]
        Default: "additive"
    calculation_dtype:
        Option to change the datatype for the noise calculation
        Default: torch.float

    Returns
    -------
    w: torch.Tensor
        the noise estimates for every pixel (LxN)
    r_w: torch.Tensor
        the noise correlation matrix (LxL)
    """
    # data must be a torch tensor
    if noise_type == "poisson":
        sqdat = torch.sqrt(data * (data > 0))  # prevent negative values
        u, r_u = _est_additive_noise(sqdat, calculation_dtype)  # noise estimates
        x = (sqdat - u) ^ 2  # signal estimates
        w = torch.sqrt(x) * u * 2
        r_w = (w @ torch.conj(w)) / data.shape[1]
    elif noise_type == "additive":
        w, r_w = _est_additive_noise(data, calculation_dtype)
    else:
        raise ValueError(
            f"noise_type must be one of ['additive', 'poisson'], currently {noise_type}"
        )
    return w, r_w


@torch.jit.script
def _est_additive_noise(
    subdata: torch.Tensor, calculation_dtype: torch.dtype = torch.float
) -> Tuple[torch.Tensor, torch.Tensor]:
    # estimate the additive noise in the given data with a certain precision
    eps = 1e-6
    dim0data, dim1data = subdata.shape
    dtp = subdata.dtype
    subdata = subdata.to(dtype=calculation_dtype)
    w = torch.zeros(subdata.shape, dtype=calculation_dtype, device=subdata.device)
    ddp = subdata @ torch.conj(subdata).T
    hld = (ddp + eps) @ torch.eye(int(dim0data), dtype=calculation_dtype, device=subdata.device)
    ddpi = torch.inverse(hld)
    for i in range(dim0data):
        xx = ddpi - (torch.outer(ddpi[:, i], ddpi[i, :]) / ddpi[i, i])
        # XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
        ddpa = ddp[:, i]
        # RRa = RR(:,i);
        ddpa[i] = 0.0
        # RRa(i)=0; % this remove the effects of XX(:,i)
        beta = xx @ ddpa
        # beta = XX * RRa;
        beta[i] = 0
        # beta(i)=0; % this remove the effects of XX(i,:)
        w[i, :] = subdata[i, :] - (beta @ subdata)
    # ret = torch.diag(torch.diag(ddp / dim1data))
    # Rw=diag(diag(w*w'/N));
    # print("here", w.shape)
    hold2 = torch.matmul(w, w.T) / float(subdata.shape[1])
    ret = torch.diag(torch.diagonal(hold2))
    w = w.to(dtype=dtp)
    ret = ret.to(dtype=dtp)
    return w, ret


def hysime(input, noise, noise_corr):
    """
    HySime: Hyperspectral signal subspace estimation, adapted from [1].

    Parameters
    ----------
    input : torch.Tensor
    noise : torch.Tensor
    noise_corr : torch.Tensor

    Returns
    -------
    signal_subspace_dim : torch.Tensor
    eigs_span_subspace : torch.Tensor
        matrix which columns are the eigenvalues that span the signal subspace

    References
    ----------
    [1] S. Devika and S. M. K. Chaitanya, "Signal estimation of hyperspectral data using HYSIME algorithm," 2016
        International Conference on Research Advances in Integrated Navigation Systems (RAINS), 2016, pp. 1-3,
        doi: 10.1109/RAINS.2016.7764372.
    """
    # [L N] = size(y);
    l, n = input.shape
    # [Ln Nn] = size(n);
    ln, nn = noise.shape
    # [d1 d2] = size(Rn);
    d1, d2 = noise_corr.shape
    if ln != l or nn != n:
        raise RuntimeError("incompatible sizes for noise and input")
    if d1 != d2 or d1 != l:
        print("Bad noise correlation matrix")
        noise_corr = noise @ noise.T / float(n)

    x = input - noise
    # signal correlation matrix estimates
    ry = (input @ input.T) / float(n)
    rx = (x @ x.T) / float(n)
    # eigen values of Rx in decreasing order, equation(15)
    u, s, vh = torch.linalg.svd(rx, full_matrices=False)
    # dx = diag(D);
    # Rn = Rn + sum(diag(Rx)) / L / 10 ^ 5 * eye(L);
    noise_corr += rx.sum() / l / 100000 * torch.eye(l, device=input.device, dtype=input.dtype)
    # Py = diag(E'*Ry*E); %equation (23)
    py = torch.diag(u.T @ ry @ u)
    # Pn = diag(E'*Rn*E); %equation (24)
    pn = torch.diag(u.T @ noise_corr @ u)
    # cost_F = -Py + 2 * Pn; % equation(22)
    cost_f = -py + 2 * pn
    # print(cost_f)
    # kf = sum(cost_F < 0);
    sig_subspace_dim = (cost_f < 0).sum()
    # [dummy, ind_asc] = sort(cost_F, 'ascend');
    _, indices = torch.sort(cost_f)
    # Ek = E(:, ind_asc(1: kf));
    eigs_span_subspace = u[:, indices[:sig_subspace_dim]]
    return sig_subspace_dim, eigs_span_subspace


def normalize(
    image: Union[torch.Tensor, np.ndarray], by_band=False, band_dim: int = -1
) -> Tuple[Union[torch.Tensor, np.ndarray], dict]:
    """
    Normalize an input between 0 and 1. If `by_band` is True, the normalization will
    be done for each band of the image (assumes [h, w, band] shape).
    Normalization constants are returned in a dictionary.
    These normalization constants can be used with the :func:`undo_normalize` function
    like this: `hyde.utils.undo_normalize(normalized_image, **constants, by_band=by_band)`

    Parameters
    ----------
    image: torch.Tensor
    by_band: bool, optional
        if True, normalize each band individually.
    band_dim: int, optional
        if by_band is true, then this defines which band to iterate over

    Returns
    -------
    normalized_input: torch.Tensor
    constants: dict
        normalization constants
        keys: mins, maxs
    """
    out = torch.zeros_like(image)
    if by_band:
        mins, maxs = [], []
        for b in range(image.shape[band_dim]):
            sl = [
                slice(None),
            ] * image.ndim
            sl[band_dim] = b
            # sl = [slice(None), slice(None), b]
            sl = tuple(sl)
            max_y = image[sl].max()  # [0]
            maxs.append(max_y)
            min_y = image[sl].min()  # [0]
            mins.append(min_y)
            out[sl] = (image[sl] - min_y) / (max_y - min_y)
        min_y = tuple(mins)
        max_y = tuple(maxs)
    else:
        # normalize the entire image, not based on the band
        sl = [
            slice(None),
        ] * image.ndim
        sl = tuple(sl)
        max_y = image[sl].max()
        min_y = image[sl].min()

        out[sl] = (image[sl] - min_y) / (max_y - min_y)
    return out, {"mins": min_y, "maxs": max_y}


def normalize_w_consts(image, consts, band_dim=-1):
    """
    Normalize a signal with some given consts

    Parameters
    ----------
    image
    consts: dict
        must contain the keys 'mins' and maxs' to work properly.
        These keys should be the correct shape to normalize given the band_dim and
        the by_band flag.
    band_dim: int

    Returns
    -------

    """
    out = torch.zeros_like(image)
    mins, maxs = consts["mins"], consts["maxs"]
    for b in range(image.shape[band_dim]):
        sl = [
            slice(None),
        ] * image.ndim
        sl[band_dim] = b
        # sl = [slice(None), slice(None), b]
        sl = tuple(sl)
        out[sl] = (image[sl] - mins[b]) / (maxs[b] - mins[b])
    return out


def peak_snr(
    img1: torch.Tensor, img2: torch.Tensor, bandwise: bool = False, band_dim: int = -1
) -> torch.Tensor:
    """
    Compute the peak signal-to-noise ratio between two images

    Parameters
    ----------
    img1 : torch.Tensor
    img2 : torch.Tensor
    bandwise: bool
        If true, do the calculation for each band
        default: False (calculation done for the full image)
    band_dim: int
        the dimension which has the image bands.
        default: -1 (i.e. last dim -> [h, w, bands])

    Returns
    -------
    snr : float
        peak signal-to-noise ration
    """
    if not bandwise:
        return km.psnr(img1, img2, img2.max())
    else:
        out = torch.zeros(img1.shape[band_dim], device=img1.device)
        sl = [
            slice(None),
        ] * img1.ndim
        for b in range(img1.shape[band_dim]):
            sl[band_dim] = b
            out[b] = km.psnr(img1[sl], img2[sl], img2[sl].max())
        return out


def rescale_wo_shift(
    x1: torch.Tensor,
    factor: Union[int, float],
    mins: torch.Tensor,
    maxs: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Undo the scaling from :func:`scale_wo_shift`.

    Parameters
    ----------
    x1: torch.Tensor
    factor: int, float
        scaling factor
    mins: torch.Tensor
        minimum values returned by :func:`scale_wo_shift`.
    maxs: torch.Tensor
        maximum values returned by :func:`scale_wo_shift`.
    eps: float
        floating point precision to use here

    Returns
    -------
    scaled: torch.Tensor
        scaled tensor
    consts: dict
        dictionary with the min and max values to be used in un-scaling
    """
    nx, ny, nz = x1.shape
    x = x1.reshape((nx * ny, nz))

    idx = torch.nonzero(maxs - mins < eps)
    if len(idx) > 0:
        y1 = x
    else:
        y1 = x / factor * (maxs - mins).repeat(nx * ny, 1)
    y1 = y1.reshape((nx, ny, nz))
    return y1


def sam(noise: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Measure spectral similarity using spectral angle mapper. Result is in radians.

    Parameters
    ----------
    noise: torch.Tensor
    reference: torch.Tensor

    Returns
    -------
    SAM: torch.Tensor
        spectral similarity in radians. If inputs are matrices, matrices are returned.
        i.e. a MxN is returned if a MxNxC is given
    """
    if len(noise.shape) == 1:
        # case 1: vectors -> easy
        numer = torch.dot(noise, reference)
        denom = torch.linalg.norm(noise) * torch.linalg.norm(reference)
    else:
        # case 2: matrices -> return a MxN if MxNxC is given
        numer = torch.sum(noise * reference, dim=-1)
        denom = torch.linalg.norm(noise, dim=-1) * torch.linalg.norm(reference, dim=-1)
    eps = torch.finfo(denom.dtype).eps
    return torch.arccos(numer / (denom + eps))


def scale_wo_shift(
    x1: torch.Tensor, factor: Union[int, float], eps: float = 1e-7
) -> Tuple[torch.Tensor, dict]:
    """
    Scale a torch tensor with this formula:
        y = x / (max(x, dim=0) - min(x, dim=0)) * factor
    before this formula is applied x is reshaped to be 2D (maintaining the last dimension).
    The reverse transform is :func:`rescale_wo_shift`.

    Parameters
    ----------
    x1: torch.Tensor
    factor: int, float
        scaling factor
    eps: float
        floating point precision to use here

    Returns
    -------
    scaled: torch.Tensor
        scaled tensor
    consts: dict
        dictionary with the min and max values to be used in un-scaling
    """
    nx, ny, nz = x1.shape
    x = x1.reshape((nx * ny, nz))
    minx = x.min(0)[0]
    maxx = x.max(0)[0]
    idx = torch.nonzero(maxx - minx < eps)
    if len(idx) > 0:
        y1 = x
    else:
        y1 = x / (maxx - minx).repeat(nx * ny, 1) * factor
    y1 = y1.reshape((nx, ny, nz))

    return y1, {"mins": minx, "maxs": maxx}


def snr(noisy: torch.Tensor, ref_signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the general signal-to-noise ratio and the peak signal-to-noise ratios.
    Both values returned are in dB.

    Parameters
    ----------
    noisy: torch.Tensor
        The noisy image, i.e. the measured signal
    ref_signal: torch.Tensor
        the reference signal, i.e. the signal WITHOUT noise

    Returns
    -------
    snr: torch.Tensor
        general signal-to-noise ratio in dB
    psnr: torch.Tensor
        peak signal-to-noise ratio in dB
    """
    err = torch.sum(torch.pow(noisy - ref_signal, 2)) / noisy.numel()
    snr = 10 * torch.log10(torch.mean(torch.pow(ref_signal, 2)) / err)

    mse = ((noisy - ref_signal) ** 2).mean()
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        psnr = 0
    else:
        max_pixel = ref_signal.max()
        psnr = 20 * torch.log10(max_pixel / mse.sqrt())
    return snr, psnr


def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Calculate the soft threshold of an input

    Parameters
    ----------
    x: torch.Tensor
        input data
    threshold: int, float, torch.Tensor
        threshold to test against

    Returns
    -------
    torch.Tensor with the soft threshold result
    """
    y = x.abs() - threshold
    # y = torch.where(y > 0, y, torch.tensor(0.0, dtype=x.dtype, device=x.device))
    y[y <= 0] = 0

    y = y / (y + threshold) * x
    return y


def sure_thresh(signal: torch.Tensor) -> torch.Tensor:
    """
    This function find the threshold value adapted to the given signal using a modified rule
    based on SURE for a soft threshold estimator.

    Parameters
    ----------
    signal : torch.Tensor
        the input for which to find the threshold

    Returns
    -------
    threshold
    """
    # using principle of Stein's Unbiased Risk Estimate.)
    if signal.ndim:
        signal = signal.unsqueeze(1)

    dev = signal.device
    n, m = signal.shape
    sx = torch.sort(torch.abs(signal), dim=0)[0].T
    sx2 = sx ** 2
    hold = (n - 2 * torch.arange(1, n + 1, device=dev)).T
    n1 = hold.repeat(1, m)
    hold = torch.arange(n - 1, -1, -1, device=dev)
    n2 = hold.T.repeat(1, m)
    if sx2.shape[1] >= 1:
        cs1 = torch.cumsum(sx2, dim=0)
    else:
        cs1 = sx2
    risks = (n1 + cs1 + n2 * sx2) / n
    _, best_min = torch.min(risks, 1)
    thr = sx[0, best_min].item()
    return thr


def sure_soft_modified_lr2(
    x: torch.Tensor, tuning_interval: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    This will apply a soft threshold and calculate Stein's unbiased risk estimator. The current
    recommendation is to use `sure_thresh` instead.

    Parameters
    ----------
    x : torch.Tensor
        input signal for which to calculate the threshold for
    tuning_interval : torch.Tensor, optional
        the search interval for selecting the optimum tuning parameter

    Returns
    -------
    sure_threshold : torch.Tensor
        the value of the SURE with soft thresholding
    h_opt : torch.Tensor
        The optimum tuning parameter
    tuning_interval : torch.Tensor
        Search interval for selecting the optimum tuning parameter
    min_sure : torch.Tensor
        Min value of SURE
    """
    N = x.shape[0]
    if tuning_interval is None:
        n = 15
        t_max = torch.sqrt(torch.log(torch.tensor(n, device=x.device)))
        tuning_interval = torch.linspace(0, t_max.item(), n, device=x.device)

    n = len(tuning_interval)
    x = x.clone()
    x = x.repeat(n, 1).T
    t = tuning_interval.repeat(N, 1)
    abv_zero = (x.abs() - t) > 0

    x_t = x ** 2 - t ** 2
    # x_t = torch.where(x_t > 0, x_t, torch.tensor(0.0, dtype=x.dtype, device=x.device))
    x_t[x_t <= 0] = 0

    sure1 = torch.sum(2 * abv_zero - x_t, dim=0)
    min_sure, min_idx = torch.min(sure1, dim=0)
    h_opt = tuning_interval[min_idx]
    return sure1, h_opt, tuning_interval, min_sure


def symmetric_pad(tens: torch.Tensor, n: Union[int, Iterable]) -> torch.Tensor:
    """
    Replacement for symmetric padding in torch (not in function space)
    padding goes from last dim backwards, with the same notation as used in torch.
    Parameters
    ----------
    tens : torch.Tensor
        the tensor to pad
        must be 2D!
    n : int, list
        the amount to pad to the 2D tensor
    Returns
    -------
    padded: torch.Tensor
        2D tensor with symmetric padding
    """
    # img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    if isinstance(n, int):
        n = [n] * tens.ndim * 2
    # todo: implement this with one-sided padding (only on top/bottom and only on left/right
    padded = pad(tens, n, "constant", 0.0)
    if tens.ndim > 2:
        # get edge of left side
        if n[-6] != 0:
            og_edge_flp = padded[:, :, n[-6] + 1 : n[-6] * 2 + 1].flip(dims=[2])
            padded[:, :, : n[-6]] = og_edge_flp
        # right side
        if n[-5] != 0:
            og_edge_flp = padded[:, :, -n[-5] * 2 - 1 : -n[-5] - 1].flip(dims=[2])
            padded[:, :, -n[-5] :] = og_edge_flp

    if tens.ndim > 1:
        # get edge of left side
        if n[-4] != 0:
            og_edge_flp = padded[:, n[-4] + 1 : n[-4] * 2 + 1].flip(dims=[1])
            padded[:, : n[-4]] = og_edge_flp
        # right side
        if n[-3] != 0:
            og_edge_flp = padded[:, -n[-3] * 2 - 1 : -n[-3] - 1].flip(dims=[1])
            padded[:, -n[-3] :] = og_edge_flp

    if n[-2] != 0:
        og_edge_flp = padded[n[-2] + 1 : n[-2] * 2 + 1].flip(dims=[0])
        padded[: n[-2]] = og_edge_flp
        # top
    if n[-1] != 0:
        og_edge_flp = padded[-n[-1] * 2 - 1 : -n[-1] - 1].flip(dims=[0])
        padded[-n[-1] :] = og_edge_flp
        # bottom
    return padded


def vertical_difference(x: torch.Tensor, n: int = 1):
    """
    Find the row differences in x. This will return a torch.Tensor of the same shape as `x`
    with the requisite number of zeros added to the end (`n`).

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    n: int
        the number of rows between each row for which to calculate the diff

    Returns
    -------
    torch.Tensor with the column differences. This result is the same size as `x`.
    """
    y = torch.zeros_like(x)
    ret = x
    for _ in range(n):
        ret = torch.diff(ret, dim=0)
    y[: ret.shape[0]] = ret
    return y


def vertical_difference_transpose(x: torch.Tensor):
    """
    Find the column differences in x for a vector, with extra flavor.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    diff: torch.Tensor
    """
    # TODO: examples and characterize output of this function, also more specific up top
    u0 = (-1.0 * x[0]).unsqueeze(0)
    u1 = (-1.0 * torch.diff(x, dim=0))[:-1]
    u2 = (x[-2]).unsqueeze(0)
    ret = torch.cat([u0, u1, u2], dim=0)
    return ret


def undo_normalize(
    image: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor, by_band=False, band_dim: int = -1
) -> torch.Tensor:
    """
    Undo the normalization to the original scale defined by the mins/maxs paraeters.
    See the :func:`normalize` function for more details

    Parameters
    ----------
    image: torch.Tensor
    mins: torch.Tensor
    maxs: torch.Tensor
    by_band: bool, optional
    band_dim: int, optional
        if by_band is true, then this defines which band to iterate over

    Returns
    -------
    rescaled_tensor: torch.Tensor
    """
    if by_band:
        out = torch.zeros_like(image)
        for b in range(image.shape[band_dim]):
            sl = [
                slice(None),
            ] * image.ndim
            sl[band_dim] = b
            out[sl] = image[sl] * (maxs[b] - mins[b]) + mins[b]
    else:
        # normalize the entire image, not based on the band
        out = image * (maxs - mins) + mins
    return out
