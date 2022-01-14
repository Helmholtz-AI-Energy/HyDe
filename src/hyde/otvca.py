import torch

from . import dwt3d, utils

from skimage.restoration import denoise_tv_bregman

# import split_bregman


__all__ = ["OTVCA"]


class OTVCA(torch.nn.Module):
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
        super(OTVCA, self).__init__()
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

    def forward(self, x: torch.Tensor, features: int, num_itt: int = 200, lam: float = 0.01):
        """
        Denoise an image `x` using the HyRes algorithm.

        Parameters
        ----------
        x: torch.Tensor
            input image
        features: int
            Number of features to extract, this could be selected equal to the
            number of classes of interests in the scene
        num_itt: int
            Number of iterations; 200 iterations are default value
        lam: float
            Tuning parameter; Default is 0.01 for normalized HSI

        Returns
        -------
        denoised_image : torch.Tensor
            Hyperspectral features extracted (3D matrix)
        """
        # key: Y -> x (image), r_max -> features, tol -> num_itt
        # [nr1, nc1, p1] = size(Y);
        nr1, nc1, p1 = x.shape
        # RY = reshape(Y, nr1 * nc1, p1);
        x_2d = x.reshape((nr1 * nc1, p1))
        # value
        # recustion
        # m = min(Y(:));
        x_min = x.min()
        # M = max(Y(:));
        x_max = x.max()
        # NRY = (RY - m) / (M - m);
        normalized_y = (x_2d - x_min) / (x_max - x_min)
        # [~, ~, V1] = svd(NRY, 'econ');
        # matlab returns the non-transposed V, torch DOES transpose
        _, _, vh = torch.linalg.svd(normalized_y, full_matrices=False)
        v1 = vh.T
        # V = V1(:, 1: r_max);
        v = v1[:, :features]
        # FE = zeros(nr1, nc1, r_max);
        fe = torch.zeros((nr1, nc1, features), dtype=x.dtype, device=x.device)
        # for fi=1:tol
        #     C1 = NRY * V(:, 1: r_max);
        #     PC = reshape(C1, nr1, nc1, r_max);
        #     for j = 1:r_max
        #         FE(:,:, j)=splitBregmanROF(PC(1: nr1, 1: nc1, j), 1 / lambda , .1);
        #     end
        #     fprintf('%d\t', fi)
        #     RFE = reshape(FE, nr1 * nc1, r_max);
        #     M = NRY'*RFE;
        #     [C, ~, G] = svd(M, 'econ');
        #     V = (C * G');
        for fi in range(num_itt):
            c1 = normalized_y @ v[:, :features]
            pc = c1.reshape((nr1, nc1, features))
            print(fe.shape)
            # for j in range(features):
            #     print(j)
            fe[:, :, :] = torch.tensor(denoise_tv_bregman(image=pc[:nr1, :nc1, :].numpy(),
                weight=1/lam, eps=0.1))

            # fe[:, :, :] = utils.denoise_tv_bregman(
            #     image=pc[:nr1, :nc1, :], weight=1 / lam, eps=0.1
            # )
            fe_reshape = fe.reshape((nr1 * nc1, features))
            m = normalized_y.T @ fe_reshape
            c, _, gh = torch.linalg.svd(m, full_matrices=False)
            print(fi, m.shape, c.shape, gh.shape)
            v = c @ gh
            break
        # end
        # Yr=reshape(RFE * V',nr1,nc1,p1);
        return (fe_reshape @ v.T).reshape((nr1, nc1, p1))


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import scipy.io as sio

    # input = sio.loadmat("/path/git/Codes_4_HyMiNoR/HyRes/Indian.mat")
    # imp = input["Indian"].reshape(input["Indian"].shape, order="C")
    #
    # t0 = time.perf_counter()
    # input_tens = torch.tensor(imp, dtype=torch.float32)
    # hyres = WSRRR()
    # output = hyres(input_tens, 5)
    # print(time.perf_counter() - t0)
    #
    # s = torch.sum(input_tens ** 2.0)
    # d = torch.sum((input_tens - output) ** 2.0)
    # snr = 10 * torch.log10(s / d)
    # print(snr)
    #
    # imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="gray")  # , vmin=50., vmax=120.)
    # plt.show()
