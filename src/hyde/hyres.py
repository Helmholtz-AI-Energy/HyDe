import torch

from . import dwt3d, utils

__all__ = ["HyRes"]


class HyRes(torch.nn.Module):
    """
    HyRes -- Automatic Hyperspectral Restoration Using low-rank and sparse modeling.

    The model used is :math:`Y=D_2XV'+N` and penalized least squares with :math:`\ell_1` penalty.
    The formula to restore the signal is:

    .. math::

        argmax(0.5 * ||Y-D_2XV'||_F^2+\lambda||X||_1)

    This method relies on Daubechies wavelets for wavelet decomposition

    Parameters
    ----------
    decomp_level : int, optional
        the level of the wavelet decomposition to do
        default: 5
    wavelet_level : int, optional
        the integer value indicating which Daubechies wavelet to use. i.e. 5 -> db5
        default: 5
    padding_method : str, optional
        the method used to pad the image during the DWT transform.
        options: [zero, symmetric, periodization, reflect, periodic]
        default: "symmetric"

    Notes
    -----
    Algorithmic questions should be forwarded to the original authors. This is purely an
    implementation of the algorithm detailed in [1].

    References
    ----------
    [1] B. Rasti, M. O. Ulfarsson and P. Ghamisi, "Automatic Hyperspectral Image Restoration
    Using Sparse and Low-Rank Modeling," in IEEE Geoscience and Remote Sensing Letters, vol. 14,
    no. 12, pp. 2335-2339, Dec. 2017, doi: 10.1109/LGRS.2017.2764059.
    """

    def __init__(self, decomp_level=5, wavelet_level=5, padding_method="symmetric"):
        super(HyRes, self).__init__()
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

    def forward(self, x: torch.Tensor):
        """
        Denoise an image `x` using the HyRes algorithm.

        Parameters
        ----------
        x: torch.Tensor
            input image

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
        # need to have the dims be (num images (1), C_in, H_in, W_in) for twave ops
        # current order: rows, columns, bands (H, W, C) -> permute tuple (2, 0, 1)
        og_rows, og_cols, og_channels = x.shape
        two_d_shape = (og_rows * og_cols, og_channels)
        # current shape: h x w X c
        # current shape: w x h X c -> unclear why this needs to be this way...
        w, _ = utils.estimate_hyperspectral_noise(
            x.reshape(two_d_shape).T,
            calculation_dtype=torch.float64,
        )

        if og_rows % (2 ** self.decomp_level) != 0:
            x = utils.symmetric_pad(
                x, [0, 0, 0, 0, 0, 2 ** self.decomp_level - (og_rows % 2 ** self.decomp_level)]
            )
        if og_cols % (2 ** self.decomp_level) != 0:
            x = utils.symmetric_pad(
                x, [0, 0, 0, 2 ** self.decomp_level - (og_cols % 2 ** self.decomp_level), 0, 0]
            )

        padded_shape = tuple(x.shape)

        p_rows, p_cols, p_ch = x.shape
        eps = 1e-30
        omega1 = (torch.sqrt(torch.var(w, dim=1)) + eps) ** 2
        # todo: sigma is the same as MATLAB
        omega1 = omega1.reshape((1, 1, omega1.numel())).repeat(p_rows, p_cols, 1)
        y_w = torch.pow(omega1, -0.5) * x
        # -------- custom PCA_Image stuff ----------------------
        nr, nc, p = y_w.shape
        # y_w -> h x w X c
        im1 = torch.reshape(y_w, (nr * nc, p))
        u, s, v_pca = torch.linalg.svd(im1, full_matrices=False)
        # need to modify u and s
        pc = torch.matmul(u, torch.diag(s))
        pc = pc.reshape((nc, nr, p))
        # -------------------------------------------------------
        # next is twoDWTon3Ddata -> requires permute + unsqueeze
        pc = pc.to(torch.float)  # no-op if already float

        # pc -> h x w x c
        v_dwt_full, v_dwt_lows, v_dwt_highs = self.dwt_forward.forward(
            pc.permute((2, 0, 1)).unsqueeze(0)
        )
        # need to put it back into the order of all the other stuff reshape into 2D
        # v_dwt_lows -> n x c x h x w ---> need: h x w x c
        # permute back is 1, 2, 0
        v_dwt_permed = v_dwt_full.squeeze().permute((1, 2, 0))
        v_dwt_permed = v_dwt_permed.reshape(
            (v_dwt_permed.shape[0] * v_dwt_permed.shape[1], og_channels)
        )

        norm_v_dwt = torch.linalg.norm(v_dwt_permed) ** 2
        n_xyz = v_dwt_permed.numel()
        sure, min_sure = None, []
        rank = None
        for rank in range(og_channels):
            sure_i, thresh_i, t1, min_sure_i = utils.sure_soft_modified_lr2(v_dwt_permed[:, rank])
            if thresh_i != 0:
                v_dwt_permed[:, rank] = utils.soft_threshold(v_dwt_permed[:, rank], thresh_i)

            if sure is None:
                sure = sure_i.unsqueeze(1)
            else:
                sure = torch.cat([sure, sure_i.unsqueeze(1)], dim=1)

            sure[:, rank] = torch.sum(sure, dim=1) + norm_v_dwt - n_xyz
            min_sure_h = torch.min(sure[:, rank])
            min_sure.append(min_sure_h)
            if rank > 1 and min_sure[rank] > min_sure[rank - 1]:
                break

        inv_lows = v_dwt_lows[:, :rank]
        inv_highs = [asdf[:, :rank] for asdf in v_dwt_highs]
        y_est_sure_model_y = self.dwt_inverse((inv_lows, inv_highs))
        # y_est_sure_model_y -> n x c x h x w  -> perm back: squeeze -> 1, 2, 0 (h, w, c)
        y_est_sure_model_y = y_est_sure_model_y.squeeze().permute((1, 2, 0))
        y_est_sure_model_y = y_est_sure_model_y.reshape((padded_shape[0] * padded_shape[1], rank))
        if y_est_sure_model_y.dtype != x.dtype:
            y_est_sure_model_y = y_est_sure_model_y.to(x.dtype)

        # ------ inverse PCA stuff -----------------------
        y_restored = (omega1 ** 0.5) * torch.matmul(y_est_sure_model_y, v_pca[:rank, :]).reshape(
            padded_shape
        )
        return y_restored[:og_rows, :og_cols, :og_channels]


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import scipy.io as sio

    input = sio.loadmat("/path/git/Codes_4_HyMiNoR/HyRes/Indian.mat")
    imp = input["Indian"].reshape(input["Indian"].shape, order="C")

    t0 = time.perf_counter()
    input_tens = torch.tensor(imp, dtype=torch.float32)
    hyres = HyRes()
    output = hyres(input_tens)
    print(time.perf_counter() - t0)

    s = torch.sum(input_tens ** 2.0)
    d = torch.sum((input_tens - output) ** 2.0)
    snr = 10 * torch.log10(s / d)
    print(snr)

    imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="gray")  # , vmin=50., vmax=120.)
    plt.show()
