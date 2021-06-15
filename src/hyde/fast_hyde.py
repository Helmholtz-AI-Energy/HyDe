import torch

from . import bm3d, dwt3d, utils

__all__ = ["FastHyDe"]


class FastHyDe(torch.nn.Module):
    """
    TODO: fix the license thing
    """

    def __init__(
        self,
    ):
        super(FastHyDe, self).__init__()

    def forward(self, img: torch.Tensor, noise_type, iid, k_subspace):
        # [rows, cols, B] = size(img_ori);
        rows, cols, b = img.shape
        # N=rows*cols;
        n = rows * cols
        # ----------------------------- Data transformation ---------------------------------
        # Observed data with additive Gaussian non-iid noise or Poissonian noise are transformed
        # in order to  to have additive Gaussian i.i.d. noise before the denoisers are applied.
        #
        # switch noise_type
        if noise_type == "additive":
            #     case 'additive'
            if iid == 0:  # additive Gaussian non-iid noise, applying eq. ?
                # Y = reshape(img_ori, N, B)';
                y = torch.reshape(img, (n, b)).T
                # [w Rw] = estNoise(Y,noise_type);
                w, r_w = utils.estimate_hyperspectral_noise(
                    y,
                    noise_type,
                    calculation_dtype=img.dtype,
                )
                # Rw_ori = Rw;
                r_w_og = r_w.clone()
                # Y = sqrt(inv(Rw_ori))*Y;
                y = torch.sqrt(torch.inverse(r_w_og)) @ y
                # img_ori = reshape(Y', rows, cols, B);
                img = torch.reshape(torch.conj(y.T), (rows, cols, b))
        elif noise_type == "poisson":
            # case 'poisson'
            # applying the Anscombe transform, which converts Poissonion noise into
            # approximately additive  noise.
            # img_ori = 2*sqrt(abs(img_ori+3/8));
            img = 2 * torch.sqrt(torch.abs(img + 3 / 8))

        # Y = reshape(img_ori, N, B)';
        y = torch.reshape(img, (n, b)).T

        # subspace estimation using HySime or SVD
        # [w Rw] = estNoise(Y,'additive');
        w, r_w = utils.estimate_hyperspectral_noise(y)
        # [~, E]=hysime(Y,w,Rw);
        _, e = utils.hysime(data=y, noise_mat=w, noise_corr_mat=r_w)
        # %[E,~,~]= svd(Y,'econ');
        #
        # E=E(:,1:k_subspace);
        e = e[:, : k_subspace + 1]  # todo: + 1???
        #
        # eigen_Y = E'*Y;
        eigen_y = e.T @ y
        #
        # %% --------------------------Eigen-image denoising ------------------------------------

        # eigen_Y_bm3d=[];
        # eigen_y_bm3d = torch.empty((k_subspace, n))
        # for i=1:k_subspace
        for i in range(k_subspace):
            # produce eigen-image
            #     eigen_im = eigen_Y(i,:);
            eigen_im = eigen_y[i]
            #     min_x = min(eigen_im);
            min_x = torch.min(eigen_im)  # dim?
            # max_x = max(eigen_im);
            max_x = torch.max(eigen_im)  # dim?
            # eigen_im = eigen_im - min_x;
            eigen_im -= min_x
            # scale = max_x-min_x;
            scale = max_x - min_x
            #
            # %scale to [0,1]
            # eigen_im = reshape(eigen_im, rows, cols)/scale;
            eigen_im = torch.reshape(eigen_im, (rows, cols)) / scale
            #
            # %estimate noise from Rw
            # sigma = sqrt(E(:,i)'*Rw*E(:,i))/scale;
            # sigma = torch.sqrt(e[:, i].T @ r_w @ e[:, i]) / scale
            #
            # [~, filt_eigen_im] = BM3D(1,eigen_im, sigma*255);
            # _, filt_eigen_im = bm3d.bm3d() ???
            #
            # eigen_Y_bm3d(i,:) = reshape(filt_eigen_im*scale + min_x, 1,N);
            pass

        #
        #
        # end
        #
        # % reconstruct data using denoising engin images
        # Y_reconst = E*eigen_Y_bm3d;
        #
        # %% ----------------- Re-transform ------------------------------
        #
        #
        # switch noise_type
        #     case 'additive'
        #         if iid==0
        #             Y_reconst = sqrt(Rw_ori)*Y_reconst;
        #         end
        #
        #     case 'poisson'
        #         Y_reconst =(Y_reconst/2).^2-3/8;
        # end
        #
        # image_fasthyde=[];
        #
        # image_fasthyde = reshape(Y_reconst',rows,cols,B);
        pass


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import scipy.io as sio

    input = sio.loadmat("/path/git/Codes_4_HyMiNoR/HyRes/Indian.mat")
    imp = input["Indian"].reshape(input["Indian"].shape, order="C")

    t0 = time.perf_counter()
    input_tens = torch.tensor(imp, dtype=torch.float32)
    fast_hyde = FastHyDe()
    output = fast_hyde(input_tens)
    # print(time.perf_counter() - t0)
    #
    # s = torch.sum(input_tens ** 2.0)
    # d = torch.sum((input_tens - output) ** 2.0)
    # snr = 10 * torch.log10(s / d)
    # print(snr)
    #
    # imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="gray")  # , vmin=50., vmax=120.)
    # plt.show()
