import torch
from . import dwt3d, utils

__all__ = ["WSRRR"]


class WSRRR(torch.nn.Module):
    """
    WSRRR - Wavelet-based Sparse Reduced-Rank Regression

    This method is explained in detail in [1].

    The method is based on minimizing a sparse regularization problem subject to an orthogonality constraint. A cyclic descent-type algorithm is derived for solving the minimization problem. For selecting the tuning parameters, we propose a method based on Stein's unbiased risk estimation. It is shown that the hyperspectral image can be restored using a few sparse components. The method is evaluated using signal-to-noise ratio and spectral angle distance for a simulated noisy data set and by classification accuracies for a real data set.

    References
    ----------
    [1] B. Rasti, J. R. Sveinsson and M. O. Ulfarsson, "Wavelet-Based Sparse Reduced-Rank
        Regression for Hyperspectral Image Restoration," in IEEE Transactions on Geoscience and
        Remote Sensing, vol. 52, no. 10, pp. 6688-6698, Oct. 2014, doi: 10.1109/TGRS.2014.2301415.
    """

    def __init__(self, decomp_level=3, wavelet_level=5, padding_method="symmetric"):
        super(WSRRR, self).__init__()
        self.decomp_level = decomp_level  # L
        self.wavelet_name = "db" + str(wavelet_level)
        self.device = "cpu"

        self.padding_method = padding_method

        self.dwt_forward = dwt3d.DWTForwardOverwrite(
            decomp_level,
            self.wavelet_name,
            self.padding_method,
            device=self.device,
        )
        self.dwt_inverse = dwt3d.DWTInverse(
            wave=self.wavelet_name, padding_method=self.padding_method, device=self.device
        )

    def forward(self, x: torch.Tensor, rank: int):
        """
        Denoise an image `x` using the HyRes algorithm.

        Parameters
        ----------
        x: torch.Tensor
            input image
        rank: int
            the band rank to cut to. bands less than or equal to this band will be used for
            denoising. others will be excluded in the threshold calculations

        Returns
        -------
        denoised_image : torch.Tensor
        """
        if x.device != self.device:
            self.device = x.device
            self.dwt_forward = dwt3d.DWTForwardOverwrite(
                self.decomp_level,
                self.wavelet_name,
                self.padding_method,
                device=self.device,
            )
            self.dwt_inverse = dwt3d.DWTInverse(
                wave=self.wavelet_name, padding_method=self.padding_method, device=self.device
            )

        # L=3; -> level of decomp
        og_rows, og_cols, og_channels = x.shape

        # % Noise Variance Estimation
        _, v_dwt_low, v_dwt_highs = self.dwt_forward.forward(x.permute((2, 0, 1)).unsqueeze(0))
        # out shape is N x C x H x W
        v_dwt_2d, filter_starts = dwt3d.construct_2d_from_filters(low=v_dwt_low, highs=v_dwt_highs)

        # testing the reshape and inverse stuff
        eps = 1e-30
        # this gets the median of the FIRST high level filters
        omega = torch.median(torch.abs(v_dwt_2d[filter_starts[-1] ** 2 :]), dim=0)[0] / 0.6745 + eps
        # % Covariance matrix
        # Omega_1=permute(sigma(:).^2,[3,2,1]);
        # Omega=repmat(Omega_1,[nx1,ny1,1]);
        omega = omega.reshape((1, 1, omega.numel())).repeat(og_rows, og_cols, 1)

        inp = torch.pow(omega, -1) * x

        # % D^T*Y is fixed through the derivation it is better to be calculated out
        # % of the loop
        # [WY_tilda,s1,s2]=twoDWTon3Ddata(Omega.^-.5.*Y,L,qmf,'FWT_PO_1D_2D_3D_fast');
        _, wy_tilda_low, wy_tilda_highs = self.dwt_forward(inp.permute((2, 0, 1)).unsqueeze(0))

        wy_tilda_2d, wy_filter_starts = dwt3d.construct_2d_from_filters(
            low=wy_tilda_low, highs=wy_tilda_highs
        )

        # [V,PC]=PCA_image(Omega.^-.5.*Y);
        v, pc = utils.custom_pca_image(inp)
        v = v[:, : rank + 1]
        thresh = torch.zeros((rank + 1, self.decomp_level + 2), dtype=x.dtype, device=x.device)
        wx = torch.zeros((wy_tilda_2d.shape[0], v.shape[1]), dtype=x.dtype, device=x.device)

        stop = filter_starts[0] ** 2
        # anything which cuts at `stop` gets the low filters from the DWT decomposition
        for cc in range(200):
            # W=WY_tilda*V;
            w = wy_tilda_2d @ v
            for i in range(rank + 1):
                index = (i % (rank + 1), i // (rank + 1))
                if cc == 0:
                    _, thresh[index], _, _ = utils.sure_soft_modified_lr2(w[:stop, i])
                if thresh[index] == 0:
                    wx[:stop, i] = w[:stop, i]
                else:
                    wx[:stop, i] = utils.soft_threshold(w[:stop, i], thresh[i, 0])

                # the next loop applies thresholding to the high parts of the filters
                for j in range(self.decomp_level):
                    st = wy_filter_starts[j] ** 2
                    try:
                        sp = wy_filter_starts[j + 1] ** 2
                    except IndexError:
                        sp = None
                    idx = slice(st, sp)

                    if cc == 0:
                        _, thresh[i, j], _, _ = utils.sure_soft_modified_lr2(w[idx, i])
                    if thresh[i, j] == 0:
                        wx[idx, i] = w[idx, i]
                    else:
                        wx[idx, i] = utils.soft_threshold(w[idx, i], thresh[i, j])
            # M=WX'*WY_tilda;
            m = wx.T @ wy_tilda_2d
            # [C,S2,G] = svd(M,'econ');
            c, s2, g = torch.linalg.svd(m, full_matrices=False)  # float64??
            # V=(C*G')';
            v = torch.conj(c @ g).T

        dwt_inv_in = wx @ v.T

        dwt_inv_low, dwt_inv_highs = dwt3d.construct_filters_from_2d(
            matrix=dwt_inv_in, filter_starts=filter_starts, decomp_level=self.decomp_level
        )

        d_w_vt = self.dwt_inverse((dwt_inv_low, dwt_inv_highs))
        d_w_vt = d_w_vt.squeeze().permute((1, 2, 0))

        xx = omega * d_w_vt
        xx = xx[:og_rows, :og_cols, :og_channels]

        # # PCs=ItwoDWTon3Ddata(WX,s1,s2,lvl,qmf,'IWT_PO_1D_2D_3D_fast');
        wx_low, wx_highs = dwt3d.construct_filters_from_2d(wx, filter_starts, self.decomp_level)
        pcs = self.dwt_inverse((wx_low, wx_highs))
        # PCs=PCs(1:nr,1:nc,:);
        pcs = pcs.squeeze().permute((1, 2, 0))[:og_rows, :og_cols]

        return xx, pcs


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import scipy.io as sio

    input = sio.loadmat("/path/git/Codes_4_HyMiNoR/HyRes/Indian.mat")
    imp = input["Indian"].reshape(input["Indian"].shape, order="C")

    t0 = time.perf_counter()
    input_tens = torch.tensor(imp, dtype=torch.float32)
    hyres = WSRRR()
    output = hyres(input_tens, 5)
    print(time.perf_counter() - t0)

    s = torch.sum(input_tens ** 2.0)
    d = torch.sum((input_tens - output) ** 2.0)
    snr = 10 * torch.log10(s / d)
    print(snr)

    imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="gray")  # , vmin=50., vmax=120.)
    plt.show()
