from typing import Optional, Tuple

import torch
from torch.nn.functional import pad
import time

__all__ = [
    "custom_pca_image",
    "denoise_tv_bregman",
    "diff",
    "diff_dim0_replace_last_row",
    "estimate_hyperspectral_noise",
    "soft_threshold",
    "sure_thresh",
    "sure_soft_modified_lr2",
    "symmetric_pad",
]


def denoise_tv_bregman(
    image: torch.Tensor,
    weight: float = 5.0,
    max_iter: int = 100,
    eps: float = 1e-3,
    isotropic: bool = True,
    channel_axis: Optional[int] = None,
) -> torch.Tensor:
    """
    Lifted from `scikit learn <https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman>`_
    and adapted for torch.

    Perform total-variation denoising using split-Bregman optimization.
    Total-variation denoising (also know as total-variation regularization)
    tries to find an image with less total-variation under the constraint
    of being similar to the input image, which is controlled by the
    regularization parameter ([1]_, [2]_, [3]_, [4]_).
    Parameters
    ----------
    image : torch.Tensor
        Input data to be denoised (converted using img_as_float`).
    weight : float
        Denoising weight. The smaller the `weight`, the more denoising (at
        the expense of less similarity to the `input`). The regularization
        parameter `lambda` is chosen as `2 * weight`.
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when::
            SUM((u(n) - u(n-1))**2) < eps
    max_iter : int, optional
        Maximal number of iterations used for the optimization.
    isotropic : boolean, optional
        Switch between isotropic and anisotropic TV denoising.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    torch.Tensor
        Denoised image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    .. [2] Tom Goldstein and Stanley Osher, "The Split Bregman Method For L1
           Regularized Problems",
           ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
    .. [3] Pascal Getreuer, "Rudin–Osher–Fatemi Total Variation Denoising
           using Split Bregman" in Image Processing On Line on 2012–05–19,
           https://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf
    .. [4] https://web.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf
    """
    rows = image.shape[0]
    cols = image.shape[1]
    try:
        dims = image.shape[2]
    except IndexError:
        dims = 1

    weight = torch.tensor(weight, dtype=image.dtype, device=image.device)

    shape_ext = (rows + 2, cols + 2, dims)
    out = torch.zeros(shape_ext, dtype=image.dtype, device=image.device)

    if channel_axis is not None:
        channel_out = torch.zeros(shape_ext[:2] + (1,), dtype=image.dtype, device=image.device)
        for c in range(image.shape[-1]):
            # the algorithm below expects 3 dimensions to always be present.
            # slicing the array in this fashion preserves the channel dimension
            # for us
            channel_in = image[..., c : c + 1]
            if not channel_in.is_contiguous():
                channel_in = channel_in.contiguous()
            _denoise_tv_bregman_work(
                channel_in, weight, max_iter, eps, isotropic, channel_out
            )
            out[..., c] = channel_out[..., 0]
    else:
        if not image.is_contiguous():
            image = image.contiguous()
        _denoise_tv_bregman_work(image, weight, max_iter, eps, isotropic, out)

    return out[1:-1, 1:-1].squeeze()


def _denoise_tv_bregman_work(
    img: torch.Tensor,
    weight: float,
    max_iter: int,
    eps: float,
    isotropic: bool,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Lifted from scikit learn's cython implementation and adapted for torch

    See :func:`denoise_tv_bregman <hyde.utils.denoise_tv_bregman>` for more information.
    This function does the actual work of the function.

    Parameters
    ----------
    img : torch.Tensor
        Input data to be denoised (converted using img_as_float`).
    weight : float
        Denoising weight. The smaller the `weight`, the more denoising (at
        the expense of less similarity to the `input`). The regularization
        parameter `lambda` is chosen as `2 * weight`.
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when::
            SUM((u(n) - u(n-1))**2) < eps
    max_iter : int, optional
        Maximal number of iterations used for the optimization.
    isotropic : boolean, optional
        Switch between isotropic and anisotropic TV denoising.

    Returns
    -------
    torch.Tensor
        Denoised image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    .. [2] Tom Goldstein and Stanley Osher, "The Split Bregman Method For L1
           Regularized Problems",
           ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
    .. [3] Pascal Getreuer, "Rudin–Osher–Fatemi Total Variation Denoising
           using Split Bregman" in Image Processing On Line on 2012–05–19,
           https://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf
    .. [4] https://web.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf
    """
    try:
        rows, cols, dims = img.shape
    except ValueError:
        rows, cols = img.shape
        dims = 1
        img = img.unsqueeze(-1)
    total = rows * cols * dims

    try:
        rmse = torch.finfo(img.dtype).max
    except TypeError:
        # this will fail if the image is ints, but it must be floats anyway, casting
        img = img.to(torch.float)  # TODO: float32 or 64?
        rmse = torch.finfo(img.dtype).max

    if out is None:
        out = torch.zeros_like(img)

    dx = out.clone()
    dy = out.clone()
    bx = out.clone()
    by = out.clone()
    # float defs
    # ux, uy, uprev, unew, bxx, byy, dxx, dyy, s, tx, ty
    i = 0
    lam = 2 * weight
    norm = weight + 4 * lam
    # more defs: out_rows, out_cols (ssize_t)

    # skipping the nogil from cython
    out_rows, out_cols = out.shape[:2]
    print(out.shape, img.shape)
    out[1 : out_rows - 1, 1 : out_cols - 1] = img

    # reflect image
    out[0, 1 : out_cols - 1] = img[1, :]
    out[1 : out_rows - 1, 0] = img[:, 1]
    out[out_rows - 1, 1 : out_cols - 1] = img[rows - 1, :]
    out[1 : out_rows - 1, out_cols - 1] = img[:, cols - 1]

    while i < max_iter and rmse > eps:

        rmse = 0

        _split_bregmann_innerloop(
            out, rows, cols, dims, lam, dx, dy, bx, by, isotropic, rmse, img, weight, norm
        )

        rmse = torch.sqrt(torch.tensor(rmse / total, device=img.device))
        i += 1
    return out


def _split_bregmann_innerloop(
    out: torch.Tensor,
    rows: int,
    cols: int,
    dims: int,
    lam: float,
    dx: torch.Tensor,
    dy: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
    isotropic: bool,
    rmse: float,
    img: torch.Tensor,
    weight: float,
    norm: float,
) -> None:

    row_range = torch.arange(1, rows + 1)
    col_range = torch.arange(1, cols + 1)
    k_range = torch.arange(dims)

    for r in row_range:
        # rm1_out = out[r - 1]
        # rp1_out = out[r + 1]
        # r_out = out[r]
        for c in col_range:
            for k in k_range:
                uprev = out[r, c, k]

                # forward derivatives
                ux = out[r, c + 1, k] - uprev
                uy = out[r + 1, c, k] - uprev

                # Gauss-Seidel method
                # t0 = time.perf_counter()
                unew = (
                    lam
                    * (
                        out[r + 1, c, k] #.item()
                        + out[r - 1, c, k]#.item()
                        + out[r, c + 1, k]#.item()
                        + out[r, c - 1, k]#.item()
                        + dx[r, c - 1, k]#.item()
                        - dx[r, c, k]#.item()
                        + dy[r - 1, c, k]#.item()
                        - dy[r, c, k]#.item()
                        - bx[r, c - 1, k]#.item()
                        + bx[r, c, k]#.item()
                        - by[r - 1, c, k]#.item()
                        + by[r, c, k]#.item()
                    )
                    + weight * img[r - 1, c - 1, k]
                ) / norm
                # print(r, c, k, "time for unew", time.perf_counter() - t0)
                out[r, c, k] = unew

                # update root mean square error
                tx = unew - uprev
                rmse += tx * tx

                bxx = bx[r, c, k]
                byy = by[r, c, k]

                # dxx = torch.tensor([], device=img.device)
                # dyy = torch.tensor([], device=img.device)

                # d_subproblem after reference [4]
                if isotropic:
                    tx = ux + bxx
                    ty = uy + byy
                    # s = torch.sqrt(torch.tensor(tx * tx + ty * ty, device=img.device))
                    s = (tx * tx + ty * ty) ** 0.5
                    dxx = s * lam * tx / (s * lam + 1)
                    dyy = s * lam * ty / (s * lam + 1)
                else:
                    s = ux + bxx
                    if s > 1 / lam:
                        dxx = s - 1 / lam
                    elif s < -1 / lam:
                        dxx = s + 1 / lam
                    else:
                        dxx = 0

                    s = uy + byy
                    if s > 1 / lam:
                        dyy = s - 1 / lam
                    elif s < -1 / lam:
                        dyy = s + 1 / lam
                    else:
                        dyy = 0

                dx[r, c, k] = dxx
                dy[r, c, k] = dyy

                bx[r, c, k] += ux - dxx
                by[r, c, k] += uy - dyy


def diff(x: torch.Tensor, n: int = 1, dim=0):
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


def diff_dim0_replace_last_row(x: torch.Tensor):
    """
    Find the single row differences in x and then put the second to last row as the last row in
    the result

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    diff_0_last_row
        the single row differences with the second to last row and the last row
    """
    u0 = (-1.0 * x[0]).unsqueeze(0)
    u1 = (-1.0 * torch.diff(x, dim=0))[:-1]
    u2 = (x[-2]).unsqueeze(0)
    ret = torch.cat([u0, u1, u2], dim=0)
    return ret


def estimate_hyperspectral_noise(
    data,
    noise_type="additive",
    calculation_dtype: torch.dtype = torch.float,
):
    """
    Infer teh noise in a hyperspectral dataset. Assumes that the
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


def custom_pca_image(img):
    """
    MUST BE IN (ROWS, COLS, CHANNELS)

    Parameters
    ----------
    img

    Returns
    -------

    """
    nr, nc, p = img.shape
    # y_w -> h x w X c
    im1 = torch.reshape(img, (nr * nc, p))
    u, s, v_pca = torch.linalg.svd(im1, full_matrices=False)
    # need to modify u and s
    pc = torch.matmul(u, torch.diag(s))
    pc = pc.reshape((nc, nr, p))
    return v_pca, pc


def soft_threshold(x: torch.Tensor, threshold):
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


def sure_thresh(signal: torch.Tensor):
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


def sure_soft_modified_lr2(x: torch.Tensor, tuning_interval=None):
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


def symmetric_pad(tens, n):
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
    padded
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
