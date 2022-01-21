import math
from typing import Iterable, Optional, Tuple, Union

import torch
from torch.nn.functional import pad

__all__ = [
    "add_noise_std",
    "add_noise_db",
    "atleast_3d",
    "custom_pca_image",
    "diff",
    "diff_dim0_replace_last_row",
    "estimate_hyperspectral_noise",
    "hysime",
    "normalize",
    "peak_snr",
    "snr",
    "soft_threshold",
    "sure_thresh",
    "sure_soft_modified_lr2",
    "symmetric_pad",
    "vertical_difference",
    "vertical_difference_transpose",
    "undo_normalize",
]


def add_noise_std(image, sigma, noise_type, iid=True):
    # switch which_case
    if isinstance(sigma, int):
        sigma = [0.10, 0.08, 0.06, 0.04, 0.02][sigma]
    if noise_type == "additive" and iid:
        noise = sigma * torch.randn(image.shape)
        img_noisy = image + noise
        # case 'case1'
        #     %--------------------- Case 1 --------------------------------------
        #
        #     % zero-mean Gaussian noise is added to all the bands of the Washington DC Mall
        #     % and Pavia city center data.
        #     % The noise standard deviation values are 0.02, 0.04, 0.06, 0.08, and 0.10, respectively.
        #     noise_type='additive';
        #     iid = 1; %It is true that noise is i.i.d.
        #         %generate noisy image
        #         noise = sigma.*randn(size(img_clean));
        #         img_noisy=img_clean+noise;
    elif noise_type == "additive" and not iid:
        sigma = torch.rand(image.shape[-1]) * 0.1
        noise = torch.randn(image.shape)
        for band in range(image.shape[-1]):
            noise[:, :, band] *= sigma[band]
        img_noisy = image + noise

        # case 'case2'
        #     %---------------------  Case 2 ---------------------
        #
        #     % Different variance zero-mean Gaussian noise is added to
        #     % each band of the two HSI datasets.
        #     % The std values are randomly selected from 0 to 0.1.
        #     noise_type='additive';
        #     iid = 0; %noise is not i.i.d.
        #     rand('seed',0);
        #     sigma = rand(1,band)*0.1;
        #     randn('seed',0);
        #     noise= randn(size(img_clean));
        #     for cb=1:band
        #         noise(:,:,cb) = sigma(cb)*noise(:,:,cb);
        #
        #     end
        #     img_noisy=img_clean+noise;
    elif noise_type == "poisson":
        img_wN = image
        snr_db = 15
        snr_set = torch.exp(
            snr_db * torch.log(torch.tensor(10, device=image.device, dtype=image.dtype)) / 10
        )
        # case 'case3'
        #     %  ---------------------  Case 3: Poisson Noise ---------------------
        #     noise_type='poisson';
        #      iid = NaN; % noise_type is set to 'poisson',
        #     img_wN = img_clean;
        #
        #     snr_db = 15;
        #     snr_set = exp(snr_db*log(10)/10);

        rc = image.shape[0] * image.shape[1]
        bands = image.shape[-1]
        img_wn_noisy = torch.zeros((bands, rc), dtype=image.dtype, device=image.device)
        for i in range(bands):
            img_wntmp = img_wN[:, :, i].unsqueeze(0)
            # reshape(img_wN(:,:,i),[1,N]);
            # img_wNtmp = max(img_wNtmp,0)
            # todo: check to make sure that the max above just takes the stuff above 0
            img_wntmp[img_wntmp <= 0] = 0
            # factor = snr_set/( sum(img_wNtmp.^2)/sum(img_wNtmp) );
            factor = snr_set / ((img_wntmp ** 2).sum() / img_wntmp.sum())
            # img_wN_scale(i,1:N) = factor*img_wNtmp;
            # img_wN_scale[i] = factor * img_wNtmp
            # % Generates Poisson random samples
            # img_wN_noisy(i,1:N) = poissrnd(factor*img_wNtmp);
            img_wn_noisy[i] = torch.poisson(factor * img_wntmp)
        #         img_noisy = reshape(img_wN_noisy', [row, column band]);
        img_noisy = img_wn_noisy.T.reshape(image.shape)
    else:
        raise ValueError(f"noise type must be one of [poissson, additive], currently: {noise_type}")
    return img_noisy


def add_noise_db(signal: torch.Tensor, noise_pow: Union[int, float]) -> torch.Tensor:
    """
    Add Gaussian white noise to a torch.Tensor. The *power* of the noise is controlled
    by the `noise_pow` parameter. This value is in dB

    Parameters
    ----------
    signal: torch.Tensor
    noise_pow: int, float

    Returns
    -------
    noisy_signal: torch.Tensor
    """
    noise_to_add = 10 ** (noise_pow / 20)
    # snr = 10 * torch.log10(torch.mean(image ** 2) / torch.mean(noise_tensor ** 2))
    # sig = 10 * torch.log10(torch.mean(image ** 2))
    noise = torch.zeros_like(signal).normal_(std=noise_to_add)
    print(f"Added Noise [dB]: {10 * torch.log10(torch.mean(torch.pow(noise, 2)))}")
    return noise + signal


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
    # if Ln~=L | Nn~=N, % n is an empty matrix or with different size
    #   error('empty noise matrix or its size does not agree with size of y\n'),
    # end
    # ## above is just raise statements
    # if (d1~=d2 | d1~=L)
    #     fprintf('Bad noise correlation matrix\n'),
    #     Rn = n * n'/N;
    # end
    x = input - noise
    # Ry = y * y'/N;   % sample correlation matrix
    ry = input @ input.T / float(n)
    rx = x @ x.T / float(n)  # signal correlation matrix estimates
    # Ry = np.dot(y, y.T) / N
    # Rx = np.dot(x, x.T) / N
    # E, dx, V = np.linalg.svd(Rx)
    u, s, vh = torch.linalg.svd(rx)
    # [E, D] = svd(Rx); % eigen values of Rx in decreasing order, equation(15)
    # dx = diag(D);
    #
    # if verbose, fprintf(1, 'Estimating the number of endmembers\n');end
    # Rn = Rn + sum(diag(Rx)) / L / 10 ^ 5 * eye(L);
    noise_corr += (
        rx.diag().sum() / float(l) / 1.0e5 * torch.eye(l, device=input.device, dtype=input.dtype)
    )
    #
    # Py = diag(E'*Ry*E); %equation (23)
    py = torch.diag(u.T @ ry @ u)
    # Pn = diag(E'*Rn*E); %equation (24)
    pn = torch.diag(u.T @ noise_corr @ u)
    # cost_F = -Py + 2 * Pn; % equation(22)
    cost_f = -py + 2 * pn
    # kf = sum(cost_F < 0);
    sig_subspace_dim = (cost_f < 0).sum()
    # [dummy, ind_asc] = sort(cost_F, 'ascend');
    _, indices = torch.sort(cost_f)
    # Ek = E(:, ind_asc(1: kf));
    eigs_span_subspace = u[:, indices[:sig_subspace_dim]]
    return sig_subspace_dim, eigs_span_subspace


def peak_snr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute the peak signal to noise ratio between two images

    Parameters
    ----------
    img1 : torch.Tensor
    img2 : torch.Tensor

    Returns
    -------
    snr : float
        peak signal to noise ration
    """
    img1 = img1.to(torch.float32) / 255.0
    img2 = img2.to(dtype=torch.float32, device=img1.device) / 255.0
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0.0
    return 10 * math.log10(1.0 / mse)


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

    Returns
    -------
    normalized_input: torch.Tensor
    constants: dict
        normalization constants
        keys: mins, maxs
    """
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
    tens : torch.Tensor
        the tensor to pad
        must be 2D!
    n : int, list
        the amount to pad to the 2D tensor

    Returns
    -------

    """
    # TODO: examples and characterize output of this function, also more specific up top
    u0 = (-1.0 * x[0]).unsqueeze(0)
    u1 = (-1.0 * torch.diff(x, dim=0))[:-1]
    u2 = (x[-2]).unsqueeze(0)
    ret = torch.cat([u0, u1, u2], dim=0)
    return ret


def undo_normalize(
    image: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor, by_band=False
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

    Returns
    -------
    rescaled_tensor: torch.Tensor
    """
    if by_band:
        out = torch.zeros_like(image)
        for b in range(image.shape[-1]):
            out[:, :, b] = image * (maxs[b] - mins[b]) + mins[b]
    else:
        # normalize the entire image, not based on the band
        out = image * (maxs - mins) + mins
    return out
