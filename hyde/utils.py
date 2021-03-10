import torch
import numpy as np
import pywt
import collections


__all__ = ["daubcqf", "estimate_hyperspectral_noise", "image_pca", "vectorize", "sure_soft_modified_lr2"]


def daubcqf(filter_len, filter_type='min', reverse_img_sorting=False):
    """
    %    [h_0,h_1] = daubcqf(N,TYPE);
    %
    %    Function computes the Daubechies' scaling and wavelet filters
    %    (normalized to sqrt(2)).
    %
    %    Input:
    %       N    : Length of filter (must be even)
    %       TYPE : Optional parameter that distinguishes the minimum phase,
    %              maximum phase and mid-phase solutions ('min', 'max', or
    %              'mid'). If no argument is specified, the minimum phase
    %              solution is used.
    %
    %    Output:
    %       h_0 : Minimal phase Daubechies' scaling filter
    %       h_1 : Minimal phase Daubechies' wavelet filter
    %
    %    Example:
    %       N = 4;
    %       TYPE = 'min';
    %       [h_0,h_1] = daubcqf(N,TYPE)
    %       h_0 = 0.4830 0.8365 0.2241 -0.1294
    %       h_1 = 0.1294 0.2241 -0.8365 0.4830
    %
    %    Reference: "Orthonormal Bases of Compactly Supported Wavelets",
    %                CPAM, Oct.89
    Parameters
    ----------
    filter_len
    filter_type

    Returns
    -------

    """
    if filter_len % 2 == 1:
        raise ValueError(f"`filter_len` must be even, currently {filter_len}")

    k, a, p, q = filter_len // 2, 1, [1], [1]
    h_0 = [1, 1]
    # just do it with lists for now
    for j in range(1, k):
        a = -a * 0.25 * (j + k - 1) / j
        h_0 = [q + w for q, w in zip([0] + h_0, h_0 + [0])]
        p = [q + w for q, w in zip([0] + [-1 * f for f in p], p + [0])]
        p = [q + w for q, w in zip([0] + [-1 * f for f in p], p + [0])]
        q = [0] + q + [0]
        q = [i + a * f for i, f in zip(q, p)]

    # need roots (easiest in numpy)
    q = np.array(q)
    unsorted = np.roots(q)
    sort_index = np.argsort(np.abs(unsorted))
    qsorted = unsorted[sort_index]
    if reverse_img_sorting:
        # get real, then swap the first real number with the one after
        inds = list(range(qsorted.size))
        swapped_inds = inds.copy()
        reals, uniq_inds = np.unique(qsorted.real, return_index=True)
        for u in reversed(uniq_inds):
            del inds[u]
        flipped_next = False
        for r in inds:
            if r in uniq_inds:
                if not flipped_next:
                    hold = swapped_inds[r + 1]
                    swapped_inds[r + 1] = swapped_inds[r]
                    swapped_inds[r] = hold
                    flipped_next = True
                else:
                    flipped_next = False

        qsorted = qsorted[swapped_inds]
    qt = qsorted[:k-1]

    if filter_type == 'mid':
        if k % 2 == 1:
            pass
            sl1 = qsorted[:filter_len - 2:4]  # 4 might need to be 3 here
            sl2 = qsorted[1:filter_len - 2:4]
            qt = np.concatenate((sl1, sl2))
            # qt = q([1:4: N - 2    2: 4:N - 2])
        else:
            sl1 = qsorted[0]  # 4 might need to be 3 here
            sl2 = qsorted[3:k - 1:4]
            sl3 = qsorted[4:k - 1: 4]
            sl4 = qsorted[filter_len - 4:k-1:-4]
            # qt = q([1     4:4:K-1     5:4:K-1     N-3:-4:K    N-4:-4:K]);
            sl5 = qsorted[filter_len - 5:k-2:-4]
            if sl1.size == 1:
                sl1 = [sl1]
            if sl2.size == 1:
                sl2 = [sl1]
            if sl3.size == 1:
                sl3 = [sl1]
            if sl4.size == 1:
                sl4 = [sl1]
            if sl5.size == 1:
                sl5 = [sl1]
            qt = np.concatenate((sl1, sl2, sl3, sl4, sl5))

    h_0 = np.polymul(h_0, np.poly(qt).real)
    h_0 = np.sqrt(2) * (h_0 / np.sum(h_0))  # normalize to root 2
    
    if filter_type == "max":
        h_0 = np.flip(h_0)
    if np.abs(np.sum(h_0 ** 2)) - 1. > 1e-4:
        raise ValueError(f"Numerically unstable for this number of filers ({filter_len})")
    h_1 = np.flip(h_0)
    h_1[:filter_len:2] = -1 * h_1[:filter_len:2]
    return torch.tensor(h_0), torch.tensor(h_1)


def estimate_hyperspectral_noise(data, noise_type="additive", verbose=False):
    # todo: verbose options
    # data must be a torch tensor

    if noise_type == "poisson":
        sqdat = torch.sqrt(data * (data > 0))  # todo: this feels wrong...
        # sqy = sqrt(y.*(y>0));          % prevent negative values
        u, r_u = _est_additive_noise(sqdat, verbose)
        # [u Ru] = estAdditiveNoise(sqy,verb); % noise estimates
        x = (sqdat - u) ^ 2
        # x = (sqy - u).^2;            % signal estimates
        w = torch.sqrt(x) * u * 2
        # w = sqrt(x).*u*2;
        r_w = (w @ torch.conj(w)) / data.shape[1]
        # Rw = w*w'/N;
    else:
        w, r_w = _est_additive_noise(data, verbose)
        # [w Rw] = estAdditiveNoise(y,verb); % noise estimates
    return w, r_w


def _est_additive_noise(subdata, verbose=False):
    eps = 1e-6
    dim0data, dim1data = subdata.shape
    w = torch.zeros(subdata.shape, device=subdata.device)
    ddp = subdata @ torch.conj(subdata)
    hld = ddp + eps * torch.eye(dim0data).shape
    ddpi = torch.eye(hld.shape) / hld
    for i in range(dim0data):
        xx = ddpi - (ddpi[:, i] @ ddpi[i, :]) / ddpi[i, i]
        # XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
        ddpa = ddp[:, i]
        # RRa = RR(:,i);
        ddpa[i] = 0
        # RRa(i)=0; % this remove the effects of XX(:,i)
        beta = xx @ ddpa
        # beta = XX * RRa;
        beta[i] = 0
        # beta(i)=0; % this remove the effects of XX(i,:)
        w[i, :] = subdata[i, :] - (torch.conj(beta) @ subdata)
        # w(i,:) = r(i,:) - beta'*r; % note that beta(i)=0 => beta(i)*r(i,:)=0
    ret = torch.diag(torch.diag(ddp / dim1data))
    # Rw=diag(diag(w*w'/N));
    return w, ret


def image_pca(image):
    nr, nc, p = image.shape
    im1 = torch.reshape(image, (nr * nc, p))
    # todo: test if full_matrices is correct, in matlab it is 'econ' option for svd
    u, s, v = torch.linalg.svd(im1, full_matrices=False)
    pc1 = u @ s
    return torch.reshape(pc1, (nr, nc, p))


def soft(x: torch.Tensor, threshold):
    y, _ = torch.max(torch.abs(x) - threshold, dim=0)  # keepdim=True?
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
    N = x.len()
    if stdev is None and t1 is None:
        n = 15
        t_max = torch.sqrt(torch.log(torch.tensor(n)))
        t1 = torch.linspace(0, t_max.item(), n)
    if stdev is None:
        stdev = 1

    n = len(t1)
    x = x.clone()
    x = x.repeat(1, n)
    t = t1.repeat(N, 1)
    # abs_x =     # create an above 0 mask
    abv_zero = (x.abs() - t) > 0  # todo: check this. not sure about the notation in matlab: s = (s > 0)

    x_t = x ** 2 - t ** 2
    x_t, _ = torch.max(x_t, dim=0)
    sure1 = torch.sum(2 * abv_zero - x_t)
    min_sure = torch.min(sure1)
    min_idx = torch.argmin(sure1)
    h_opt = t1[min_idx]
    return sure1, h_opt, t1, min_sure


def pad_and_reflect(pad_size, side, reflect_side, ):
    # pad with numpy's reflect mode on the sides
    # no extension in the 3rd dim
    # i think we can use the torch.nn.ReflectionPad2d(...)
    pass


def modified_wextend(s3, loc, locz):
    # called extend in the code
    pass

# for i=1:n3
#     Y(:,:,i) = wextend(TYPE,MODE,X(:,:,i),L,LOC);
#       -> np.pad(mode='symmetric'
# end
#
# if s3>n3
#     error('Symetric extension in the z-direction is impossible, the extension size is larger than data size')
# end
# if s3~=0 && n3~=0
#     switch LOCz
#         case {'lr', 'rl'}
#             y1=flipdim(Y(:,:,1:floor(s3/2)),3);
#             y2=flipdim(Y(:,:,end-ceil(s3/2)+1:end),3);
#             y3 = cat(3,y1,Y);
#             Y = cat(3,y3,y2);
#             if mod(s3,3)~=0
#                warning(' The extension is not symetric, one component more in the right ')
#             end
#         case {'l'}
#             y1=flipdim(Y(:,:,1:s3),3);
#             Y = cat(3,y1,Y);
#         case {'r'}
#             y2=flipdim(Y(:,:,end-s3+1:end),3);
#             Y = cat(3,Y,y2);
#         otherwise
#             warning('The data is not expended in the z diretion')
#     end
# end


if __name__ == "__main__":
    daubcqf(filter_len=24, filter_type='mid')
