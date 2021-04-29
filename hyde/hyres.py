import pytorch_wavelets as twave
import pywt
import torch
import torch.nn as nn
import utils
import dwt3d

import scipy.io as sio


# define this as a class like a transform
class HyRes:

    def __init__(self, decomp_level=5, filter_coef=10):
        self.decomp_level = decomp_level  # L
        self.filter_coef = filter_coef  # NFC
        wavelet = 'db' + str(self.filter_coef // 2)

        self.mode = 'symmetric'

        self.dwt_forward = twave.DWTForward(J=self.decomp_level, wave=wavelet, mode=self.mode)
        self.dwt_inverse = twave.DWTInverse(wave=wavelet, mode=self.mode)

    def forward(self, x: torch.Tensor):
        # todo: need to have the dims be (num images (1), C_in, H_in, W_in) for twave ops
        # todo: current order: rows, columns, bands (H, W, C) -> permute tuple (2, 0, 1)
        perm_tuple = (2, 0, 1)
        og_rows, og_cols, og_channels = x.shape
        two_d_shape = (og_rows * og_cols, og_channels)

        w, _ = utils.estimate_hyperspectral_noise(torch.conj(x.reshape(two_d_shape)).T)
        sigma = torch.sqrt(torch.conj(torch.var(torch.conj(w).T, dim=0)).T)

        p_rows, p_cols, p_ch = x.shape

        eps = 1e-6
        omega1 = (sigma+eps) ** 2
        omega1 = omega1.reshape((1, 1, omega1.numel())).repeat(p_rows, p_cols, 1)
        y_w = torch.pow(omega1, -0.5) * x
        # -------- custom PCA_Image stuff ----------------------
        nr, nc, p = y_w.shape
        im1 = torch.reshape(y_w, (nr * nc, p))
        # todo: test if full_matrices is correct, in matlab it is 'econ' option for svd
        u, s, v_pca = torch.linalg.svd(im1, full_matrices=False)
        # need to modify u and s
        pc = torch.matmul(u[:, :s.numel()], torch.diagflat(s)).reshape((nr, nc, p))
        # -------------------------------------------------------
        # next is twoDWTon3Ddata -> requires permute + unsqueeze
        # TODO: determine if the padding is okay here...this will be different from the matlab ones
        v_dwt_lows, v_dwt_highs = self.dwt_forward.forward(pc.permute(perm_tuple).unsqueeze(0))
        # need to put it back into the order of all the other stuff reshape into 2D
        v_dwt_permed = v_dwt_lows.squeeze().permute((1, 2, 0))
        v_dwt_permed = v_dwt_permed.reshape((v_dwt_permed.shape[0] * v_dwt_permed.shape[1], og_channels))
        norm_y = torch.norm(v_dwt_lows) ** 2
        nelem = v_dwt_permed.numel()

        sure_vals, sure1, min_sure1 = None, None, []
        for c in range(og_channels):
            v_pca_y = v_pca[:, :c + 1]
            sure_lp, _, _, _ = utils.sure_soft_modified_lr2(v_dwt_permed[:, c])
            if sure_vals is None:
                sure_vals = sure_lp.unsqueeze(1)
                sure1 = torch.empty_like(sure_vals)
            else:
                sure_vals = torch.cat((sure_vals, sure_lp.unsqueeze(1)), dim=1)
                sure2 = torch.empty_like(sure_vals)
                sure2[:, : sure1.shape[1]] = sure1
                sure1 = sure2
            sure1[:, c] = torch.sum(sure_vals, dim=1) + norm_y - nelem

            min_sure1_h, _ = sure1[:, c].min(0)
            min_sure1.append(min_sure1_h)

        _, rank_sel_sure = torch.tensor(min_sure1).min(0)

        inv_lows = v_dwt_lows[:, :rank_sel_sure + 1]
        inv_highs = [asdf[:, :rank_sel_sure + 1] for asdf in v_dwt_highs]
        y_est_sure_model_y = self.dwt_inverse((inv_lows, inv_highs)).squeeze().permute((1, 2, 0)).reshape((og_rows*og_cols, rank_sel_sure + 1))

        # ------ inverse PCA stuff -----------------------
        # reshape to 2D (rows*cols, channels)
        y_restored = (omega1 ** 0.5) * torch.matmul(y_est_sure_model_y, v_pca_y[:, :rank_sel_sure + 1].T).reshape((og_rows, og_cols, og_channels))

        return y_restored


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    input = sio.loadmat("/home/daniel/git/HyRes/Indian.mat")
    input_tens = torch.tensor(input["Indian"], dtype=torch.float32)
    test = HyRes()
    output = test.forward(input_tens)
    print(time.perf_counter() - t0)
    import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg

    # imgplot = plt.imshow(input_tens.numpy()[:, :, 0])
    imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="Greys")
    plt.show()
