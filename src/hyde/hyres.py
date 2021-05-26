import matplotlib.pyplot as plt
import torch

from . import dwt3d, utils

__all__ = ["HyRes"]


class HyRes(torch.nn.Module):
    """
    HyRes -- Automatic Hyperspectral Restoration Using low-rank and sparse modeling.

    The model used is :math:`Y=D_2XV'+N` and penalized least squares with :math:`\ell_1` penalty.
    The formula to restore the signal is:
    ..math::
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

    Notes
    -----
    Algorithmic questions should be forwarded to the original authors. This is purely an
    implementation of the algorithm detailed in [1].

    References
    ----------
    [1] B. Rasti, M. O. Ulfarsson and P. Ghamisi, "Automatic Hyperspectral Image Restoration Using
        Sparse and Low-Rank Modeling," in IEEE Geoscience and Remote Sensing Letters, vol. 14,
        no. 12, pp. 2335-2339, Dec. 2017, doi: 10.1109/LGRS.2017.2764059.
    """

    def __init__(self, decomp_level=5, wavelet_level=5):
        super(HyRes, self).__init__()
        self.decomp_level = decomp_level  # L
        self.wavelet_name = "db" + str(wavelet_level)
        self.device = "cpu"

        self.mode = "symmetric"

        self.dwt_forward = dwt3d.DWTForwardOverwrite(
            decomp_level,
            self.wavelet_name,
            self.mode,
            device=self.device,
        )
        self.dwt_inverse = dwt3d.DWTInverse(
            wave=self.wavelet_name, mode=self.mode, device=self.device
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
                self.mode,
                device=self.device,
            )
            self.dwt_inverse = dwt3d.DWTInverse(
                wave=self.wavelet_name, mode=self.mode, device=self.device
            )
        # need to have the dims be (num images (1), C_in, H_in, W_in) for twave ops
        # current order: rows, columns, bands (H, W, C) -> permute tuple (2, 0, 1)
        og_rows, og_cols, og_channels = x.shape
        two_d_shape = (og_rows * og_cols, og_channels)
        # current shape: h x w X c
        x = x.permute((1, 0, 2))
        # current shape: w x h X c -> unclear why this needs to be this way...
        w, _ = utils.estimate_hyperspectral_noise(
            x.reshape(two_d_shape).T,
            calculation_dtype=torch.float64,
        )
        x = x.permute((1, 0, 2))
        # x -> h x w X c

        p_rows, p_cols, p_ch = x.shape
        eps = 1e-30
        omega1 = (torch.sqrt(torch.var(w, dim=1).T) + eps) ** 2
        omega1 = omega1.reshape((1, 1, omega1.numel())).repeat(p_rows, p_cols, 1)
        y_w = torch.pow(omega1, -0.5) * x
        # -------- custom PCA_Image stuff ----------------------
        nr, nc, p = y_w.shape
        y_w = y_w.permute((1, 0, 2))  # needed to make the arrays equal to each other (vs matlab)
        # y_w -> h x w X c
        im1 = torch.reshape(y_w, (nr * nc, p))
        u, s, v_pca = torch.linalg.svd(im1, full_matrices=False)
        v_pca = torch.conj(v_pca.T)
        # need to modify u and s
        pc = torch.matmul(u, torch.diag(s))
        pc = pc.reshape((nc, nr, p)).permute((1, 0, 2))
        # -------------------------------------------------------
        # next is twoDWTon3Ddata -> requires permute + unsqueeze
        if pc.dtype != torch.float:
            pc = pc.to(torch.float)

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

        test_sure = []
        rank = None
        for rank in range(og_channels):
            sure2test = utils.sure_thresh(v_dwt_permed[:, rank])
            test_sure.append(sure2test)
            if rank > 1 and test_sure[rank] <= test_sure[rank - 1]:
                break

        inv_lows = v_dwt_lows[:, :rank]
        inv_highs = [asdf[:, :rank] for asdf in v_dwt_highs]
        y_est_sure_model_y = self.dwt_inverse((inv_lows, inv_highs))
        # y_est_sure_model_y -> n x c x h x w  -> perm back: squeeze -> 1, 2, 0 (h, w, c)
        y_est_sure_model_y = y_est_sure_model_y.squeeze().permute((1, 2, 0))
        dwt_inv_shape = y_est_sure_model_y.shape
        if dwt_inv_shape[0] > og_rows:
            # dif = og_rows - dwt_inv_shape[0]
            y_est_sure_model_y = y_est_sure_model_y[:og_rows, :og_cols]
        y_est_sure_model_y = y_est_sure_model_y.reshape((og_rows * og_cols, rank))
        if y_est_sure_model_y.dtype != x.dtype:
            y_est_sure_model_y = y_est_sure_model_y.to(x.dtype)

        # ------ inverse PCA stuff -----------------------
        # reshape to 2D (rows*cols, channels)
        y_restored = (omega1 ** 0.5) * torch.matmul(y_est_sure_model_y, v_pca[:, :rank].T).reshape(
            (og_rows, og_cols, og_channels)
        )
        return y_restored


if __name__ == "__main__":
    import time

    import scipy.io as sio

    # input = sio.loadmat("/home/daniel/git/Codes_4_HyMiNoR/HyRes/img_noisy_npdB21.mat")
    # imp = input["img_noisy_npdB21"].reshape(input["img_noisy_npdB21"].shape, order="C")
    input = sio.loadmat("/home/daniel/git/Codes_4_HyMiNoR/HyRes/Indian.mat")
    imp = input["Indian"].reshape(input["Indian"].shape, order="C")

    # print(imp.flags)
    t0 = time.perf_counter()
    input_tens = torch.tensor(imp, dtype=torch.float32)
    hyres = HyRes()
    output = hyres(input_tens)
    print(time.perf_counter() - t0)
    mdic = {"a": output.numpy(), "label": "img_noisy_npdB21_hyde_denoised"}
    # sio.savemat("/home/daniel/git/Codes_4_HyMiNoR/HyRes/indian_hyde_denoised.mat", mdic)

    comp = sio.loadmat("/home/daniel/git/HyRes/hyres_final.mat")
    # comp = sio.loadmat("/home/daniel/git/Codes_4_HyMiNoR/HyRes/img_noisy_npdB21_denoised.mat")
    comp_tens = torch.tensor(comp["Y_restored"], dtype=torch.float32)

    s = torch.sum(input_tens ** 2.0)
    d = torch.sum((input_tens - output) ** 2.0)
    snr = 10 * torch.log10(s / d)
    print(snr)

    # channel = 0
    compare_btw = output[:, :, :] - comp_tens[:, :, :]

    print(torch.mean(compare_btw), torch.std(compare_btw))
    # print(torch.min(output[..., 0]))
    # import matplotlib.image as mpimg

    # imgplot = plt.imshow(input_tens.numpy()[:, :, 0])
    imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="gray")  # , vmin=50., vmax=120.)
    plt.show()
