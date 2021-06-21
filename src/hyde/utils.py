from typing import Tuple

import torch
from torch.nn.functional import pad

__all__ = [
    "diff",
    "diff_dim0_replace_last_row",
    "estimate_hyperspectral_noise",
    "soft_threshold",
    "sure_thresh",
    "sure_soft_modified_lr2",
    "symmetric_pad",
]


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


# def pad(mat, mode, l, loc, s3, loc_z):
#     """
#     extension('2d', 'sym', Y, [2^L-mod(nr1,2^L), 0], 'r',  0,  'r')
#     extension(TYPE,  MODE, X,                     L, LOC, s3, LOCz)
#     Parameters
#     ----------
#     mat
#     mode
#     l
#     loc
#     s3
#     loc_z
#
#     Returns
#     -------
#
#     """
#
#     # def pad(x, pad_widths, mode):
#     """Extend a 1D signal using a given boundary mode.
#
#     This function operates like :func:`numpy.pad` but supports all signal
#     extension modes that can be used by PyWavelets discrete wavelet transforms.
#
#     Parameters
#     ----------
#     x : ndarray
#         The array to pad
#     pad_widths : {sequence, array_like, int}
#         Number of values padded to the edges of each axis.
#         ``((before_1, after_1), … (before_N, after_N))`` unique pad widths for
#         each axis. ``((before, after),)`` yields same before and after pad for
#         each axis. ``(pad,)`` or int is a shortcut for
#         ``before = after = pad width`` for all axes.
#     mode : str, optional
#         Signal extension mode, see :ref:`Modes <ref-modes>`.
#
#     Returns
#     -------
#     pad : ndarray
#         Padded array of rank equal to array with shape increased according to
#         ``pad_widths``.
#
#     Notes
#     -----
#     The performance of padding in dimensions > 1 may be substantially slower
#     for modes ``'smooth'`` and ``'antisymmetric'`` as these modes are not
#     supported efficiently by the underlying :func:`numpy.pad` function.
#
#     Note that the behavior of the ``'constant'`` mode here follows the
#     PyWavelets convention which is different from NumPy (it is equivalent to
#     ``mode='edge'`` in :func:`numpy.pad`).
#     """
#     # process pad_widths exactly as in numpy.pad
#     pad_widths = np.array(pad_widths)
#     pad_widths = np.round(pad_widths).astype(np.intp, copy=False)
#     if pad_widths.min() < 0:
#         raise ValueError("pad_widths must be > 0")
#     pad_widths = np.broadcast_to(pad_widths, (x.ndim, 2)).tolist()
#
#     if mode in ['symmetric', 'reflect']:
#         xp = np.pad(x, pad_widths, mode=mode)
#     elif mode in ['periodic', 'periodization']:
#         if mode == 'periodization':
#             # Promote odd-sized dimensions to even length by duplicating the
#             # last value.
#             edge_pad_widths = [(0, x.shape[ax] % 2)
#                                for ax in range(x.ndim)]
#             x = np.pad(x, edge_pad_widths, mode='edge')
#         xp = np.pad(x, pad_widths, mode='wrap')
#     elif mode == 'zero':
#         xp = np.pad(x, pad_widths, mode='constant', constant_values=0)
#     elif mode == 'constant':
#         xp = np.pad(x, pad_widths, mode='edge')
#     elif mode == 'smooth':
#         def pad_smooth(vector, pad_width, iaxis, kwargs):
#             # smooth extension to left
#             left = vector[pad_width[0]]
#             slope_left = (left - vector[pad_width[0] + 1])
#             vector[:pad_width[0]] = \
#                 left + np.arange(pad_width[0], 0, -1) * slope_left
#
#             # smooth extension to right
#             right = vector[-pad_width[1] - 1]
#             slope_right = (right - vector[-pad_width[1] - 2])
#             vector[-pad_width[1]:] = \
#                 right + np.arange(1, pad_width[1] + 1) * slope_right
#             return vector
#
#         xp = np.pad(x, pad_widths, pad_smooth)
#     elif mode == 'antisymmetric':
#         def pad_antisymmetric(vector, pad_width, iaxis, kwargs):
#             # smooth extension to left
#             # implement by flipping portions symmetric padding
#             npad_l, npad_r = pad_width
#             vsize_nonpad = vector.size - npad_l - npad_r
#             # Note: must modify vector in-place
#             vector[:] = np.pad(
#                 vector[pad_width[0]:-pad_width[-1]],
#                 pad_width, mode='symmetric'
#             )
#             vp = vector
#             r_edge = npad_l + vsize_nonpad - 1
#             l_edge = npad_l
#             # width of each reflected segment
#             seg_width = vsize_nonpad
#             # flip reflected segments on the right of the original signal
#             n = 1
#             while r_edge <= vp.size:
#                 segment_slice = slice(
#                     r_edge + 1,
#                     min(r_edge + 1 + seg_width, vp.size)
#                     )
#                 if n % 2:
#                     vp[segment_slice] *= -1
#                 r_edge += seg_width
#                 n += 1
#
#             # flip reflected segments on the left of the original signal
#             n = 1
#             while l_edge >= 0:
#                 segment_slice = slice(max(0, l_edge - seg_width), l_edge)
#                 if n % 2:
#                     vp[segment_slice] *= -1
#                 l_edge -= seg_width
#                 n += 1
#             return vector
#
#         xp = np.pad(x, pad_widths, pad_antisymmetric)
#     elif mode == 'antireflect':
#         xp = np.pad(x, pad_widths, mode='reflect', reflect_type='odd')
#     else:
#         raise ValueError(
#             ("unsupported mode: {}. The supported modes are {}").format(
#                 mode, Modes.modes
#             )
#         )
#     return xp
#     pass


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
    """
    x=repmat(x,1,n);        ??
    t=repmat(t1,N,1);       ??

    abs_x=abs(x);
    s = abs_x - t;
    s = (s > 0);

    x_t=x.^2-t.^2;

    x_t=max(x_t,0);
    sure1=sum(2*s-x_t);

    [Min_sure,idx]=min(sure1);
    h_opt=t1(idx);
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
    # x_t, _ = torch.max(x_t, dim=0)
    # MATLAB: x_t=max(x_t,0) -> this replaces the things below 0 with 0
    x_t = torch.where(x_t > 0, x_t, torch.tensor(0.0, dtype=x.dtype))

    sure1 = torch.sum(2 * abv_zero - x_t, dim=0)
    min_sure, min_idx = torch.min(sure1, dim=0)
    # min_idx = torch.argmin(sure1, dim=0)
    h_opt = tuning_interval[min_idx]
    return sure1, h_opt, tuning_interval, min_sure


def symmetric_pad(tens, n):
    """
    Replacement for symmetric padding in torch (not in function space)

    padding goes from last dim backwards

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
