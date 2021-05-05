import torch
import pytorch_wavelets as twave
import utils
import dwt3d

import scipy.io as sio


class HyRes:

    def __init__(self, decomp_level=5, filter_coef=10):
        self.decomp_level = decomp_level  # L
        self.filter_coef = filter_coef  # NFC
        wavelet = 'db' + str(self.filter_coef // 2)

        self.mode = 'symmetric'  # this has shown the most similar results, more testing required

        # self.dwt_forward = twave.DWTForward(J=self.decomp_level, wave=wavelet, mode=self.mode)
        self.dwt_forward = dwt3d.DWTForwardOverwrite(decomp_level, wavelet, self.mode)
        self.dwt_inverse = twave.DWTInverse(wave=wavelet, mode=self.mode)

    def forward(self, x: torch.Tensor):
        # todo: need to have the dims be (num images (1), C_in, H_in, W_in) for twave ops
        # todo: current order: rows, columns, bands (H, W, C) -> permute tuple (2, 0, 1)
        og_rows, og_cols, og_channels = x.shape
        two_d_shape = (og_rows * og_cols, og_channels)
        # print(x.shape)
        # current shape: h x w X c
        # todo: move the permutations and transpose into the function?
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
        omega1 = (torch.sqrt(torch.var(w, dim=1).T)+eps) ** 2
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
        # TODO: determine if the padding is okay here...this will be different from the matlab ones
        if pc.dtype != torch.float:
            pc = pc.to(torch.float)

        # pc -> h x w x c
        v_dwt_full, v_dwt_lows, v_dwt_highs = self.dwt_forward.forward(pc.permute((2, 0, 1)).unsqueeze(0))
        # need to put it back into the order of all the other stuff reshape into 2D
        # v_dwt_lows -> n x c x h x w ---> need: h x w x c
        # permute back is 1, 2, 0
        v_dwt_permed = v_dwt_full.squeeze().permute((1, 2, 0))
        v_dwt_permed = v_dwt_permed.reshape((v_dwt_permed.shape[0] * v_dwt_permed.shape[1], og_channels))

        test_sure = []
        rank = None
        for rank in range(og_channels):
            sure2test = utils.sure_thresh(v_dwt_permed[:, rank])
            test_sure.append(sure2test)
            if rank > 1 and test_sure[rank] <= test_sure[rank-1]:
                break

        # print(rank)
        inv_lows = v_dwt_lows[:, :rank]
        inv_highs = [asdf[:, :rank] for asdf in v_dwt_highs]
        y_est_sure_model_y = self.dwt_inverse((inv_lows, inv_highs))
        # y_est_sure_model_y -> n x c x h x w  -> perm back: squeeze -> 1, 2, 0 (h, w, c)
        y_est_sure_model_y = y_est_sure_model_y.squeeze().permute((1, 2, 0))
        dwt_inv_shape = y_est_sure_model_y.shape
        if dwt_inv_shape[0] > og_rows:
            # dif = og_rows - dwt_inv_shape[0]
            y_est_sure_model_y = y_est_sure_model_y[:og_rows, :og_cols]
        # todo: error in here about it not relating properly, sizes are mismatched!
        y_est_sure_model_y = y_est_sure_model_y.reshape((og_rows * og_cols, rank))
        if y_est_sure_model_y.dtype != x.dtype:
            y_est_sure_model_y = y_est_sure_model_y.to(x.dtype)

        # ------ inverse PCA stuff -----------------------
        # reshape to 2D (rows*cols, channels)
        y_restored = (omega1 ** 0.5) * torch.matmul(y_est_sure_model_y, v_pca[:, :rank].T).reshape((og_rows, og_cols, og_channels))
        return y_restored


class HyMiNoR:

    def __init__(self):
        self.hyres = HyRes()

    def forward(self, x: torch.Tensor, lam: int = 10):
        # H_M -> x ; lambda -> lam
        mu1, mu2, its = 0.5, 0.5, 50
        print(x.shape)
        hyres_result = self.hyres.forward(x)
        # H=HyRes(H_M);
        m, n, d = hyres_result.shape
        mn = m * n
        # [m,n,d]=size(H);
        # mn=m*n;
        hyres_reshaped_t = torch.conj(hyres_result.reshape((mn, d))).T
        # Y=reshape(H,mn,d)';
        # todo: torch.float
        l1 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # L1=zeros(d,mn);
        l2 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # L2=L1;
        v1 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # V1=L1;
        v2 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # V2=L1;
        xx = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # X=L1;
        #
        eye_d = torch.eye(d, device=x.device, dtype=x.dtype)
        # print(eye_d.shape)
        hold = utils.vertical_difference_transpose(
            utils.vertical_difference(eye_d)
        )
        u, s, v = torch.linalg.svd(hold.to(torch.float64), full_matrices=False)
        u = u.to(torch.float32)
        s = s.to(torch.float32)
        v = torch.conj(v.T).to(torch.float32)
        # print((u @ torch.diag(s) @ v).shape)
        # todo: the linalg torch solver gives slightly different values
        # u, s, v = torch.svd(hold)
        # [U,S,V] = svd(Dvt(Dv(speye(d))),'econ');
        # print('v', v.shape, s.shape, v.shape)
        # print(1. / torch.diag(mu1 + mu2 * torch.diag(s)))
        # print('u', u.shape)
        # hold = torch.diag(1. / (mu1 + mu2 * s))
        # hold = torch.diag(1. / (mu1 + mu2 * s)) @ torch.conj(u).T
        # # hold2 = hold * torch.conj(u).T
        # print(hold[0])
        # magic = v @ hold #torch.diag(1. / (mu1 + mu2 * s)) @ torch.conj(u).T
        # todo: find bug in matmul stuffs
        #   middle arg is same, so is v and u:  :D
        # hold1 =
        magic = torch.chain_matmul(
            v,
            torch.diag(1. / (mu1 + mu2 * s)),
            u.T,
        ).to(torch.float32)
        # magic += -1*magic[0, -1]
        # Majic=V*diag(1./(mu1+mu2*diag(S)))*U';
        # print('m', magic[:5, -5:])
        # for i=1:iter
        # print(hyres_reshaped_t[0, :10])
        for _ in range(its):
            # subminimization problems
            hold1 = -mu1 * (v1 - hyres_reshaped_t - l1)
            hold2 = mu2 * utils.vertical_difference_transpose((v2 - l2))
            xx = magic @ (hold1 + hold2)
            # X=Majic*(-mu1*(V1-Y-L1)+mu2*Dvt(V2-L2));
            # % V-Step
            v1 = utils.soft_threshold(hyres_reshaped_t - xx + l1, 1. / mu1)
            # V1=soft_threshold(Y-X+L1,1/mu1);
            dv_xx = utils.vertical_difference(xx)
            v2 = utils.soft_threshold(dv_xx + l2, lam / mu2)
            # V2=soft_threshold(Dv(X)+L2,lambda/mu2);
            # %Updating L1 and L2
            l1 = l1 + hyres_reshaped_t - xx - v1
            # L1=L1+Y-X-V1;
            l2 = l2 + dv_xx - v2
            # L2=L2+Dv(X)-V2;
        # H_DN=reshape(X',m,n,d);
        ret = torch.reshape(torch.conj(xx).T, (m, n, d))
        return ret


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    input = sio.loadmat("/home/daniel/git/Codes_4_HyMiNoR/noisy_J_M_8_09.mat")
    input_tens = torch.tensor(input["noisy_J_M_8_09"], dtype=torch.float32)
    test = HyMiNoR()
    output = test.forward(input_tens)
    print(time.perf_counter() - t0)

    comp = sio.loadmat("/home/daniel/git/Codes_4_HyMiNoR/noisy_J_M_8_09_og_result.mat")
    # comp = sio.loadmat("/home/daniel/git/Codes_4_HyMiNoR/HyRes/img_noisy_npdB21_denoised.mat")
    # print(comp.keys())
    comp_tens = torch.tensor(comp["Y_restored"], dtype=torch.float32)

    import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg

    s = torch.sum(input_tens ** 2.)
    # print(s)
    d = torch.sum((input_tens - output) ** 2.)
    # print(d)
    snr = 10 * torch.log10(s / d)
    print(snr)

    diff = output - comp_tens
    print(diff.mean(), diff.std())

    # # imgplot = plt.imshow(input_tens.numpy()[:, :, 0])
    # imgplot = plt.imshow(diff.numpy()[:, :, 9], cmap="gray")
    print(output.max(), output.min())
    imgplot = plt.imshow(output.numpy()[:, :, 39], cmap="gray", vmin=-0.05, vmax=1.0)

    plt.show()
