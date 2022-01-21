from typing import Iterable, Optional, Tuple, Union

import torch
from torch.nn.functional import pad

__all__ = [
    "add_noise_db",
    "atleast_3d",
    "custom_pca_image",
    "denoise_tv_bregman",
    "diff",
    "diff_dim0_replace_last_row",
    "estimate_hyperspectral_noise",
    "normalize",
    "un_normalize",
    "snr",
    "soft_threshold",
    "sure_thresh",
    "sure_soft_modified_lr2",
    "symmetric_pad",
]


def add_noise_db(image, snr_db):
    noise_to_add = 10 ** (snr_db / 20)
    # snr = 10 * torch.log10(torch.mean(image ** 2) / torch.mean(noise_tensor ** 2))
    # sig = 10 * torch.log10(torch.mean(image ** 2))
    noise = torch.zeros_like(image).normal_(std=noise_to_add)
    print(f"Added Noise [dB]: {10 * torch.log10(torch.mean(torch.pow(noise, 2)))}")
    return noise + image


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


def denoise_tv_bregman(
    image: torch.Tensor,
    weight: float,
    max_iter: int = 100,
    eps: float = 1e-3,
    isotropic: bool = True,
) -> torch.Tensor:
    """
    Perform total-variation denoising using split-Bregman optimization.

    Parameters
    ----------
    image : torch.Tensor
        Input data to be denoised.
    weight : float
        Denoising weight. The smaller the 'weight', the more denoising (at
        the expense of less similarity to the 'input').
    max_iter : int, optional
        Maximal number of iterations used for the optimization.
    eps : float, optional
        The threshold of distance between denoised image in iterations
        The algorithm stops when image distance is smaller than eps
    isotropic : bool, optional
        Switch between isotropic and anisotropic TV denoising.

    Returns
    -------
    denoised_image : torch.Tensor

    Sources
    -------
    - https://github.com/shakes76/PatternFlow/blob/master/algorithms/denoise/denoise_tv_bregman/denoise_tv_bregman.py
    - skimage.restoration.denoise_tv_bregman

    """
    image = atleast_3d(image)

    rows, cols, dims = image.shape
    rows2 = rows + 2
    cols2 = cols + 2

    shape_extend = (rows2, cols2, dims)
    # out is firstly created as zeros-like tensor with size as shape_extend
    out = torch.zeros(shape_extend, dtype=image.dtype, device=image.device)

    dx = out.clone().detach()
    dy = out.clone().detach()
    bx = out.clone().detach()
    by = out.clone().detach()

    # out = prep_out_bregman(image, out)
    out_rows, out_cols = out.shape[:2]
    rows, cols = out_rows - 2, out_cols - 2

    out[1 : out_rows - 1, 1 : out_cols - 1] = image

    out[0, 1 : out_cols - 1] = image[1, :]
    out[1 : out_rows - 1, 0] = image[:, 1]
    out[out_rows - 1, 1 : out_cols - 1] = image[rows - 1, :]
    out[1 : out_rows - 1, out_cols - 1] = image[:, cols - 1]

    weight = torch.tensor(weight, device=image.device)
    lam = 2 * weight
    rmse = float("inf")
    norm = weight + 4 * lam

    # iterative optimization method
    uprev_c = out[1:-1, 1:-1, :].clone()
    _denoise_inner_loop(
        uprev_c, max_iter, eps, out, weight, lam, image, norm, rmse, bx, by, dx, dy, isotropic
    )

    # return the denoised image excluding the extended area
    return out[1:-1, 1:-1]


@torch.jit.script
def _denoise_inner_loop(
    uprev_c: torch.Tensor,
    max_iter: int,
    eps: float,
    out: torch.Tensor,
    weight: torch.Tensor,
    lam: torch.Tensor,
    image: torch.Tensor,
    norm: torch.Tensor,
    rmse: float,
    bx: torch.Tensor,
    by: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    isotropic: bool,
):
    i = 0
    rows = image.shape[0]
    cols = image.shape[1]
    while i < max_iter and rmse > eps:
        rmse = torch.tensor(0.0, device=image.device)
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                uprev = uprev_c[r - 1, c - 1]

                # forward derivatives
                ux = out[r, c + 1] - uprev
                uy = out[r + 1, c] - uprev

                # Gauss-Seidel method
                unew = (
                    lam
                    * (
                        out[r + 1, c]
                        + out[r - 1, c]
                        + out[r, c + 1]
                        + out[r, c - 1]
                        + dx[r, c - 1]
                        - dx[r, c]
                        + dy[r - 1, c]
                        - dy[r, c]
                        - bx[r, c - 1]
                        + bx[r, c]
                        - by[r - 1, c]
                        + by[r, c]
                    )
                    + weight * image[r - 1, c - 1]
                ) / norm
                out[r, c] = unew

                # update root mean square error
                tx = unew - uprev
                rmse += (tx * tx).sum()

                bxx = bx[r, c]
                byy = by[r, c]

                # d_subproblem after reference [4]
                dxx = dx[r, c] * 0
                dyy = dy[r, c] * 0
                if isotropic:
                    tx = ux + bxx
                    ty = uy + byy
                    s = torch.sqrt(tx * tx + ty * ty)
                    dxx = s * lam * tx / (s * lam + 1)
                    dyy = s * lam * ty / (s * lam + 1)
                else:
                    s = ux + bxx
                    whgt = torch.nonzero(s > 1 / lam)
                    if len(whgt) > 0:
                        dxx[whgt] = s[whgt] - 1 / lam
                    whls = torch.nonzero(s < -1 / lam)
                    if len(whls) > 0:
                        dxx[whls] = s[whls] + 1 / lam

                    s = uy + byy
                    whgt = torch.nonzero(s > 1 / lam)
                    if len(whgt) > 0:
                        dyy[whgt] = s[whgt] - 1 / lam
                    whls = torch.nonzero(s < -1 / lam)
                    if len(whls) > 0:
                        dyy[whls] = s[whls] + 1 / lam

                dx[r, c] = dxx
                dy[r, c] = dyy

                bx[r, c] += ux - dxx
                by[r, c] += uy - dyy
        rmse = torch.sqrt(rmse / float(image.numel()))
        print(i, rmse)
        i += 1


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
        # torch.diff does the *last* axis but matlab diff
        #       does it on the *first* non-1 dimension
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
    noise_type: torch.Tensor = "additive",
    calculation_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
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
    subdata = subdata.to(dtype=calculation_dtype)
    w = torch.zeros(subdata.shape, dtype=calculation_dtype, device=subdata.device)
    ddp = subdata @ torch.conj(subdata).T
    hld = (ddp + eps) @ torch.eye(int(dim0data), dtype=calculation_dtype, device=subdata.device)
    ddpi = torch.eye(
        hld.shape[0], hld.shape[1], dtype=calculation_dtype, device=subdata.device
    ) @ torch.inverse(hld)
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
        w[i, :] = subdata[i, :] - (beta.T @ subdata)
    # ret = torch.diag(torch.diag(ddp / dim1data))
    # Rw=diag(diag(w*w'/N));
    hold2 = torch.matmul(w, w.conj().t())
    ret = torch.diag(torch.diagonal(hold2))
    w = w.to(dtype=torch.float)
    ret = ret.to(dtype=torch.float)
    return w, ret


def custom_pca_image(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MUST BE IN (ROWS, COLS, CHANNELS)

    Parameters
    ----------
    img

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


def normalize(image: torch.Tensor, by_band=False) -> Tuple[torch.Tensor, dict]:
    if by_band:
        out = torch.zeros_like(image)
        for b in range(image.shape[-1]):
            # print(image[:, :, b].max())
            max_y = image[:, :, b].max()  # [0]
            min_y = image[:, :, b].min()  # [0]
            out[:, :, b] = (image[:, :, b] - min_y) / (max_y - min_y)
    else:
        # normalize the entire image, not based on the band
        max_y = image.max()
        min_y = image.min()
        out = (image - min_y) / (max_y - min_y)
    return out, {"mins": min_y, "maxs": max_y}


def un_normalize(image: torch.Tensor, mins, maxs, by_band=False) -> torch.Tensor:
    if by_band:
        out = torch.zeros_like(image)
        for b in range(image.shape[-1]):
            out[:, :, b] = image * (maxs[b] - mins[b]) + mins[b]
    else:
        # normalize the entire image, not based on the band
        out = image * (maxs - mins) + mins
    return out


def snr(noisy, ref_signal):
    err = torch.sum(torch.pow(noisy - ref_signal, 2)) / noisy.numel()
    snr = 10 * torch.log10(torch.mean(torch.pow(ref_signal, 2)) / err)

    mse = ((noisy - ref_signal) ** 2).mean()
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        psnr = 0
    else:
        max_pixel = 255
        psnr = 20 * torch.log10(max_pixel / mse.sqrt())
    return snr, psnr


def soft_threshold(x: torch.Tensor, threshold: Union[int, float, torch.Tensor]) -> torch.Tensor:
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
    hld = torch.abs(x) - threshold
    y = torch.where(hld > 0, hld, torch.tensor(0.0, dtype=x.dtype))
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
    # based on MATLAB's: thselect function with  `rigsure` option (adaptive threshold selection
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
    This function was adapted from a MATLAB directory, however it is currently unused.

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
        tuning_interval = torch.linspace(0, t_max.item(), n)

    n = len(tuning_interval)
    x = x.clone()
    x = x.repeat(n, 1).T
    t = tuning_interval.repeat(N, 1)
    abv_zero = (x.abs() - t) > 0

    x_t = x ** 2 - t ** 2
    # MATLAB: x_t=max(x_t,0) -> this replaces the things below 0 with 0
    x_t = torch.where(x_t > 0, x_t, torch.tensor(0.0, dtype=x.dtype, device=x.device))

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
    og_sz = tens.shape
    # elif isinstance(n)
    # todo: implement this with one-sided padding (only on top/bottom and only on left/right
    padded = pad(tens, n, "constant", 0.0)
    # print(padded[..., 0])
    if tens.ndim > 2:
        cycle = list(range(og_sz[2])) + list(range(og_sz[2] - 1, -1, -1))
        for i in range(n[-6]):  # dim 2, low inds
            # (reflect the cols first -> taste)
            ind = cycle[(i + 1) % len(cycle)] - og_sz[1]
            padded[:, :, i] = padded[:, :, ind]
            # padded[:, :, 0 + i] = padded[:, :, 2 * n[-6] - i - 1]
            # n == 4
            # 0 -> 7, 1 -> 6, 2 -> 5, 3 -> 4, break

        cycle = list(reversed(range(-og_sz[2], 0))) + list(range(-og_sz[2], 0))
        for i in range(n[-5]):  # dim 2, high inds
            sind = og_sz[2] - padded.shape[2] + i
            gind = og_sz[2] - padded.shape[2] + cycle[i % len(cycle)]
            padded[:, :, sind] = padded[:, :, gind]
            # padded[:, :, -1 - (0 + i)] = padded[:, :, -2 * n[-5] + i]
            # n == 4
            # -1 -> -8, -2 -> -7, -3 -> -6, -4 -> -5, break

    if tens.ndim > 1:
        cycle = list(range(og_sz[1])) + list(range(og_sz[1] - 1, -1, -1))
        for i in range(n[-4]):  # dim 1, left side (low inds)
            ind = cycle[(i + 1) % len(cycle)] - og_sz[1]
            # (reflect the cols first -> taste)
            padded[:, 0 + i] = padded[:, ind]
            # 0 -> 7, 1 -> 6, 2 -> 5, 3 -> 4, break

        cycle = list(reversed(range(-og_sz[1], 0))) + list(range(-og_sz[1], 0))
        for i in range(n[-3]):  # dim 1, right side (end / high inds))
            sind = og_sz[1] - padded.shape[1] + i
            gind = og_sz[1] - padded.shape[1] + cycle[i % len(cycle)]
            padded[:, sind] = padded[:, gind]
            # -1 -> -8, -2 -> -7, -3 -> -6, -4 -> -5, break

    cycle = list(range(og_sz[0])) + list(range(og_sz[0] - 1, -1, -1))
    for i in range(n[-2]):  # dim 0, top, (low inds)
        ind = cycle[(i + 1) % len(cycle)] - og_sz[0]
        padded[i] = padded[ind]

    cycle = list(reversed(range(-og_sz[0], 0))) + list(range(-og_sz[0], 0))
    for i in range(n[-1]):  # dim 0, bottom, (low inds)
        sind = og_sz[0] - padded.shape[0] + i
        gind = og_sz[0] - padded.shape[0] + cycle[i % len(cycle)]
        padded[sind] = padded[gind]

    return padded
