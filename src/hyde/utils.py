from typing import Tuple

import torch

__all__ = [
    "estimate_hyperspectral_noise",
    "soft_threshold",
    "sure_soft_modified_lr2",
    "vertical_difference",
    "vertical_difference_transpose",
]


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
    x_t, _ = torch.max(x_t, dim=0)
    sure1 = torch.sum(2 * abv_zero - x_t, dim=0)
    min_sure = torch.min(sure1)
    min_idx = torch.argmin(sure1)
    h_opt = tuning_interval[min_idx]
    return sure1, h_opt, tuning_interval, min_sure


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
    y = torch.where(hld > 0, hld, torch.tensor(0.0))
    y = y / (y + threshold) * x
    return y


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
        # torch.diff does the *last* axis but matlab diff
        #       does it on the *first* non-1 dimension
        ret = torch.diff(ret, dim=0)
    y[: ret.shape[0]] = ret
    return y


def vertical_difference_transpose(x: torch.Tensor):
    """
    Find the row differences in x for a vector, with extra flavor.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    TODO: examples and characterize output of this function, also more specific up top
    """
    u0 = (-1.0 * x[0]).unsqueeze(0)
    u1 = (-1.0 * torch.diff(x, dim=0))[:-1]
    u2 = (x[-2]).unsqueeze(0)
    ret = torch.cat([u0, u1, u2], dim=0)
    return ret
