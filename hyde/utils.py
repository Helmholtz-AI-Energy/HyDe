import torch
import numpy as np
import pywt
import collections
import scipy as sp
import math

__all__ = [
    "daubcqf",
    "estimate_hyperspectral_noise",
    "soft",
    "sure_soft_modified_lr2",
    "vectorize",
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


def estimate_hyperspectral_noise(data, noise_type="additive", verbose=False):
    # todo: verbose options
    # data must be a torch tensor

    if noise_type == "poisson":
        sqdat = torch.sqrt(data * (data > 0))  # todo: this feels wrong...
        # sqy = sqrt(y.*(y>0));          % prevent negative values
        u, r_u = _est_additive_noise(sqdat)
        # [u Ru] = estAdditiveNoise(sqy,verb); % noise estimates
        x = (sqdat - u) ^ 2
        # x = (sqy - u).^2;            % signal estimates
        w = torch.sqrt(x) * u * 2
        # w = sqrt(x).*u*2;
        r_w = (w @ torch.conj(w)) / data.shape[1]
        # Rw = w*w'/N;
    else:
        w, r_w = _est_additive_noise(data)
        # [w Rw] = estAdditiveNoise(y,verb); % noise estimates
    return w, r_w


def _est_additive_noise(subdata):
    eps = 1e-6
    dim0data, dim1data = subdata.shape
    w = torch.zeros(subdata.shape, device=subdata.device)
    ddp = subdata @ torch.conj(subdata).T
    hld = (ddp + eps) @ torch.eye(int(dim0data))
    ddpi = torch.eye(*tuple(hld.shape)) @ torch.inverse(hld)
    for i in range(dim0data):
        xx = ddpi - torch.outer(ddpi[:, i], ddpi[i, :]) / ddpi[i, i]
        # XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
        ddpa = ddp[:, i]
        # RRa = RR(:,i);
        ddpa[i] = 0
        # RRa(i)=0; % this remove the effects of XX(:,i)
        beta = xx @ ddpa
        # beta = XX * RRa;
        beta[i] = 0
        # beta(i)=0; % this remove the effects of XX(i,:)
        w[i, :] = subdata[i, :] - (torch.conj(beta).T @ subdata)
        # w(i,:) = r(i,:) - beta'*r; % note that beta(i)=0 => beta(i)*r(i,:)=0
    # ret = torch.diag(torch.diag(ddp / dim1data))
    # Rw=diag(diag(w*w'/N));
    hold2 = torch.matmul(w, w.conj().t())
    ret = torch.diag(torch.diagonal(hold2))
    return w, ret


def soft(x: torch.Tensor, threshold):
    y, _ = torch.max(torch.abs(x) - threshold, dim=0)
    y = y / (y + threshold) * x
    return y


def vectorize(x):
    return torch.reshape(x, (1, x.numel()))


def sure_soft_modified_lr2(x: torch.Tensor, t1=None, stdev=None):
    """
    SUREsoft -- apply soft threshold + compute Stein's unbiased risk estimator
    %  Usage
    %    [sure1,h_opt,t1,Min_sure] = SUREsoft(x,t1,stdev);
    %    [sure1,h_opt,t1,Min_sure] = SUREsoft(x,t1);
    %    [sure1,h_opt,t1,Min_sure] = SUREsoft(x);
    %  Inputs
    %    x      input coefficients
    %    t1 : Search interval for selecting the optimum tuning parameter
    %    stdev  noise standard deviation (default is 1)
    %  Outputs
    %    sure1    the value of the SURE, using soft thresholding
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
    if stdev is None and t1 is None:
        n = 15
        t_max = torch.sqrt(torch.log(torch.tensor(n, device=x.device)))
        t1 = torch.linspace(0, t_max.item(), n)
    if stdev is None:
        stdev = 1

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
