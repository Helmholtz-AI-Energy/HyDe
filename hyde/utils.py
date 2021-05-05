import torch
import numpy as np
import pywt
import collections
import scipy as sp
import math
from typing import Tuple

__all__ = [
    "daubcqf",
    "estimate_hyperspectral_noise",
    "soft_threshold",
    "sure_soft_modified_lr2",
    "vectorize",
    "vertical_difference",
    "vertical_difference_transpose",
]


def daubcqf(N, TYPE='min'):
    """
    Computes the Daubechies' scaling and wavelet filters (normalized to sqrt(2)).

    Adapted from: https://pythonhosted.org/pyrwt/_modules/rwt/wavelets.html#daubcqf for use
    with torch....

    Parameters
    ----------
    N : int
        Length of filter (must be even)
    TYPE : ['min', 'max', 'mid'], optional (default='min')
        Distinguishes the minimum phase, maximum phase and mid-phase solutions
        ('min', 'max', or 'mid').

    Returns
    -------
    h_0 : array-like, shape = [N]
        Daubechies' scaling filter
    h_1 : array-like, shape = [N]
        Daubechies' wavelet filter

    Examples
    --------
    >>> from rwt.wavelets import daubcqf
    >>> h_0, h_1 = daubcqf(N=4, TYPE='min')
    >>> print h_0, h_1
    [[0.4830, 0.8365, 0.2241, -0.1294]] [[0.1294, 0.2241, -0.8365, 0.4830]]
    """

    assert N % 2 == 0, 'No Daubechies filter exists for odd length'

    K = int(N/2)
    a = 1
    p = 1
    q = 1

    h_0 = np.array([1.0, 1.0])
    for j in range(1, K):
        a = -a * 0.25 * (j + K - 1)/j
        h_0 = np.hstack((0, h_0)) + np.hstack((h_0, 0))
        p = np.hstack((0, -p)) + np.hstack((p, 0))
        p = np.hstack((0, -p)) + np.hstack((p, 0))
        q = np.hstack((0, q, 0)) + a * p

    q = np.sort(np.roots(q))

    qt = q[:K-1]

    if TYPE == 'mid':
        if K % 2 == 1:
            ind = np.hstack((np.arange(0, N-2, 4), np.arange(1, N-2, 4)))
        else:
            ind = np.hstack(
                (
                    1,
                    np.arange(3, K-1, 4),
                    np.arange(4, K-1, 4),
                    np.arange(N-3, K, -4),
                    np.arange(N-4, K, -4)
                )
            )
        qt = q[ind]

    h_0 = np.convolve(h_0, np.real(np.poly(qt)))
    # todo: devices
    h_0 = torch.tensor(h_0)

    h_0 = math.sqrt(2.) * h_0 / torch.sum(h_0)

    h_0 = torch.reshape(h_0, (1, -1))

    if TYPE == 'max':
        h_0 = torch.fliplr(h_0)

    assert torch.abs(torch.sum(h_0 ** 2))-1 < 1e-4, 'Numerically unstable for this value of "N".'

    h_1 = torch.rot90(h_0, 2).clone()
    h_1[0, :N:2] = -h_1[0, :N:2]

    return h_0, h_1


def estimate_hyperspectral_noise(
        data,
        noise_type="additive",
        calculation_dtype: torch.dtype=torch.float,
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
    # todo: verbose options
    # data must be a torch tensor

    if noise_type == "poisson":
        sqdat = torch.sqrt(data * (data > 0))  # todo: this feels wrong...
        # sqy = sqrt(y.*(y>0));          % prevent negative values
        u, r_u = _est_additive_noise(sqdat, calculation_dtype)
        # [u Ru] = estAdditiveNoise(sqy,verb); % noise estimates
        x = (sqdat - u) ^ 2
        # x = (sqy - u).^2;            % signal estimates
        w = torch.sqrt(x) * u * 2
        # w = sqrt(x).*u*2;
        r_w = (w @ torch.conj(w)) / data.shape[1]
        # Rw = w*w'/N;
    elif noise_type == "additive":
        w, r_w = _est_additive_noise(data, calculation_dtype)
        # [w Rw] = estAdditiveNoise(y,verb); % noise estimates
    else:
        raise ValueError(f"noise_type must be one of ['additive', 'poisson'], currently {noise_type}")
    return w, r_w


@torch.jit.script
def _est_additive_noise(subdata: torch.Tensor, calculation_dtype: torch.dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-6
    # print(subdata[:5, :5])
    dim0data, dim1data = subdata.shape
    subdata = subdata.to(dtype=calculation_dtype)
    w = torch.zeros(subdata.shape, dtype=calculation_dtype, device=subdata.device)
    ddp = subdata @ torch.conj(subdata).T
    hld = (ddp + eps) @ torch.eye(int(dim0data), dtype=calculation_dtype, device=subdata.device)
    ddpi = torch.eye(hld.shape[0], hld.shape[1], dtype=calculation_dtype, device=subdata.device) @ torch.inverse(hld)
    # ddp = ddp.to(dtype=torch.float)
    # subdata = subdata.to(dtype=torch.float)
    for i in range(dim0data):
        xx = ddpi - (torch.outer(ddpi[:, i], ddpi[i, :]) / ddpi[i, i])
        # if i == 1:
        #     print(xx[:10, 0])
        # xx = xx.to(torch.float)
        # XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
        ddpa = ddp[:, i]
        # RRa = RR(:,i);
        ddpa[i] = 0.
        # RRa(i)=0; % this remove the effects of XX(:,i)
        beta = (xx @ ddpa)
        # print(beta.shape)
        # beta = XX * RRa;
        beta[i] = 0
        # beta(i)=0; % this remove the effects of XX(i,:)
        # print(torch.conj(beta).T.shape, subdata.shape)
        w[i, :] = subdata[i, :] - (beta.T @ subdata)
        # if i == 4:
        #     print(beta.T[:10])
        # w(i,:) = r(i,:) - beta'*r; % note that beta(i)=0 => beta(i)*r(i,:)=0
    # ret = torch.diag(torch.diag(ddp / dim1data))
    # Rw=diag(diag(w*w'/N));
    hold2 = torch.matmul(w, w.conj().t())
    # print(w[:10, 0])
    ret = torch.diag(torch.diagonal(hold2))
    w = w.to(dtype=torch.float)
    ret = ret.to(dtype=torch.float)
    return w, ret


def sure_thresh(x: torch.Tensor):
    if x.ndim:
        x = x.unsqueeze(1)
    # print(x.shape)
    n, m = x.shape
    #         sx = sort(abs(x),1);
    #         sx2 = sx.^2;
    sx = torch.sort(torch.abs(x), dim=0)[0].T  #.to(torch.float64)
    sx2 = sx ** 2
    # print(sx.shape)
    # print('h', sx[-10:])
    #         N1 = repmat((n-2*(1:n))',1,m);
    hold = (n-2*torch.arange(1, n + 1)).T
    n1 = hold.repeat(1, m)#.to(torch.float64)
    # print(torch.where(n1 < 0))
    hold = torch.arange(n-1, -1, -1)
    # print(torch.where(hold < 0))
    #         N2 = repmat((n-1:-1:0)',1,m);
    n2 = hold.T.repeat(1, m)#.to(torch.float64)
    # print('n2', n2)
    #         CS1 = cumsum(sx2,1);
    if sx2.shape[1] >= 1:
        cs1 = torch.cumsum(sx2, dim=0)  #.to(torch.float64)
    else:
        cs1 = sx2
    # print('sx2', sx2.shape)
    # print(n1[0, -10:])
    #         risks = (N1+CS1+N2.*sx2)./n;
    # print(n1.shape, cs1.shape, n2.shape, sx2.shape)
    risks = (n1 + cs1 + n2 * sx2) / n
    # print('r', risks.shape)
    #         [~,best] = min(risks,[],1, 'linear');
    _, best_min = torch.min(risks, 1)
    # print('b', best_min)
    #         % thr will be row vector
    #         thr = sx(best);
    thr = sx[0, best_min].item()
    # print(best_min, thr)
    return thr

    # n = x.shape[0]
    #
    # # a = mtlb_sort(abs(X)).^2
    # a = torch.sort(torch.abs(x))[0] ** 2
    #
    # c = torch.linspace(n - 1, 0, n)
    # s = torch.cumsum(a, dim=0) + c * a
    # risk = (n - (2 * torch.arange(n)) + s) / n
    # # risk = (n-(2*(1:n))+(cumsum(a,'m')+c(:).*a))/n;
    # ibest = torch.argmin(risk)
    # THR = torch.sqrt(a[ibest])
    # return THR


def sure_soft_modified_lr2(x: torch.Tensor, t1=None):
    """
    SUREsoft -- apply soft_threshold threshold + compute Stein's unbiased risk estimator
    %  Usage
    %    [sure1,h_opt,t1,Min_sure] = SUREsoft(x,t1,stdev);
    %    [sure1,h_opt,t1,Min_sure] = SUREsoft(x,t1);
    %    [sure1,h_opt,t1,Min_sure] = SUREsoft(x);
    %  Inputs
    %    x      input coefficients
    %    t1 : Search interval for selecting the optimum tuning parameter
    %    stdev  noise standard deviation (default is 1)
    %  Outputs
    %    sure1    the value of the SURE, using soft_threshold thresholding
    %    h_opt : The optimum tuning parameter
    %    t1 : Search interval for selecting the optimum tuning parameter
    %    Min_sure : Min value of SURE

    Parameters
    ----------
    x
    t1
    stdev

    Returns
    -------

    """
    N = x.shape[0]
    if t1 is None:
        n = 15
        t_max = torch.sqrt(torch.log(torch.tensor(n, device=x.device)))
        t1 = torch.linspace(0, t_max.item(), n)

    n = len(t1)
    x = x.clone()
    x = x.repeat(n, 1).T
    t = t1.repeat(N, 1)
    abv_zero = (x.abs() - t) > 0

    x_t = x ** 2 - t ** 2
    x_t, _ = torch.max(x_t, dim=0)
    sure1 = torch.sum(2 * abv_zero - x_t, dim=0)
    min_sure = torch.min(sure1)
    min_idx = torch.argmin(sure1)
    h_opt = t1[min_idx]
    return sure1, h_opt, t1, min_sure


def soft_threshold(x: torch.Tensor, threshold):
    # y = max(abs(x) - T, 0) -> this will replace the value below with 0
    hld = torch.abs(x) - threshold
    y = torch.where(hld > 0, hld, torch.tensor(0.))
    # y, _ = torch.max(torch.abs(x) - threshold, dim=0)
    y = y / (y + threshold) * x
    return y


def vectorize(x):
    return torch.reshape(x, (1, x.numel()))


def vertical_difference(x: torch.Tensor, n: int = 1):
    """
    Find the vertical differences in x

    Parameters
    ----------
    x
    n

    Returns
    -------

    """
    # y=zeros(size(x));
    y = torch.zeros_like(x)
    ret = x
    for _ in range(n):
        # todo: torch.diff does the *last* axis but matlab diff does it on the *first* non-1 dimension
        ret = torch.diff(ret, dim=0)
    # y(1:end-1,:)=diff(x,n);
    y[:ret.shape[0]] = ret
    return y


def vertical_difference_transpose(x: torch.Tensor):
    """
    Find the vertical differences in x

    Parameters
    ----------
    x
    n

    Returns
    -------

    """
    # u0 = -z(1,:);
    # u1 = -diff(z);
    # u2 = z(end-1,:);
    # y = [u0; u1(1:(end-1),:); u2];
    # print(x.shape)
    u0 = (-1. * x[0]).unsqueeze(0)
    # todo: torch.diff does the *last* axis but matlab diff does it on the *first* non-1 dimension
    u1 = (-1. * torch.diff(x, dim=0))[:-1]
    u2 = (x[-2]).unsqueeze(0)
    # print(u0.shape, u1.shape, u2.shape)
    ret = torch.cat([u0, u1, u2], dim=0)
    # print(u2)
    return ret
