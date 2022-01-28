import torch

from hyde.lowlevel import utils
from hyde.spatial_domain import fast_hyde

__all__ = ["L1HyMixDe"]


class L1HyMixDe(torch.nn.Module):
    """
    Implementation of the L1HyMixDe method [1] for improving on the FastHyDe method.
    This method will show less speed-up than others because it relies upon the BM3D
    software package which only utilizes CPUs.

    References
    ----------
    [1] L. Zhuang and M. K. Ng, "Hyperspectral Mixed Noise Removal By $\ell _1$-Norm-Based Subspace Representation,"
        in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 1143-1157,
        2020, doi: 10.1109/JSTARS.2020.2979801.

    """

    def __init__(self):
        super(L1HyMixDe, self).__init__()

    def forward(
        self,
        img: torch.Tensor,
        k_subspace: int = 10,
        p: float = 0.05,
        max_iter: int = 10,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Run the L1HyMixDe method on an image. The loop will break if there is no more progress made even if
        `max_iter` is not reached.

        Parameters
        ----------
        img: torch.Tensor
            image to de-noise
        k_subspace: int
            The number of signal subspaces to scan.
            default: 10
        p: float
            the percentage of elements corrupted by impulse noise and stripes
            default: 0.05
        max_iter: int
            number of iterations to run the denoising method.
        normalize: bool
            if true, normalize the data (by band) before the algorithm.
            default: True.

        Returns
        -------
        denoised_image: torch.Tensor
        """
        if normalize:
            img, consts = utils.normalize(img, by_band=True)
        row, col, band = img.shape
        n = row * col
        y_og = img.reshape((n, band)).T
        # %% -------------Subspace Learning Against Mixed Noise---------------------
        # %An adaptive median filter is applied to noisy image to remove the bulk of
        # %impulse noise and stripes
        img_median = torch.zeros_like(img)
        for ib in range(band):
            img_median[:, :, ib] = utils.adaptive_median_filtering(img[:, :, ib], 21)
        y_median = img_median.reshape((n, band)).T
        # %detect pixel indexes of impulse noise and stripes
        # img_dif =  abs(img-img_median) ;
        img_dif = torch.abs(img - img_median)
        mask_outlier = img_dif > p

        img_remove_outlier = img.clone()
        img_remove_outlier[mask_outlier] = img_median[mask_outlier]

        y_remove_outlier = img_remove_outlier.reshape((n, band)).T
        x, r_w = utils.estimate_hyperspectral_noise(
            y_remove_outlier, "additive", calculation_dtype=torch.float64
        )
        # %data whitening so that noise variances of each band are same
        hold = torch.linalg.inv(torch.sqrt(r_w))
        y_og = hold @ y_og
        y_median = hold @ y_median
        y_remove_outlier = hold @ y_remove_outlier
        #
        # %Subspace learning from the coarse image without stripes and impulse noise
        e, s, _ = torch.linalg.svd(y_remove_outlier @ y_remove_outlier.T / n)
        e = e[:, :k_subspace]
        # %% --------------------------L1HyMixDe-------------------------------------
        # %Initialization
        z = e.T @ y_median
        img_dif = img = img_median
        v = img_dif.reshape((n, band)).T
        d = torch.zeros((band, n), dtype=img.dtype, device=img.device)
        # Noise covariance matrix Rw_fasthyde is identity matrix because the image has been whitened.
        rw_fasthyde = torch.eye(band, dtype=img.dtype, device="cpu")
        zold = None
        crits = []
        for it in range(max_iter):
            # %% Updating Z: Z_{k+1} = argmin_Z lambda*phi(Z) + mu/2 || Y-EZ-V_k-D_k||_F^2
            # %Equivlance: Z_{k+1} = argmin_Z lambda/mu*phi(Z) +  1/2 || Y-EZ-V_k-D_k||_F^2
            y_aux = y_og - v + d
            # %FastHyDe
            # Z = FastHyDe_fixEreturnZ(img_aux, E, Rw_fasthyde);
            eigen_y = e.T @ y_aux
            # end of FastHyDe_fixEreturnZ ---------------
            # %% --------------------------Eigen-image denoising ------------------------------------
            z = fast_hyde.fast_hyde_eigen_image_denoising(
                img, k_subspace, rw_fasthyde, e, eigen_y, n
            )
            z = torch.tensor(z, dtype=e.dtype, device=e.device)
            # %% Updating V: V_{k+1} = argmin_V ||V||_1 + mu/2 || Y-EZ_{k+1}-V-D_k||_F^2
            # V = sign(V_aux).*max(abs(V_aux)-par,0);
            yez = y_og - e @ z
            v_aux = yez + d
            par = 1
            hold = torch.abs(v_aux) - par
            # filter out negaive values matlap -> max(matrix, 0)
            hold[hold < 0] = 0
            v = torch.sign(v_aux) * hold
            # %% Updating D: D_{k+1} = D_k - (Y-EZ_{k+1}-V_{k+1})
            d += yez - v
            if it > 0:
                crits.append(torch.norm(z - zold) / torch.norm(zold))
                # print(it, crits[-1], sum(crits[-5:]) / len(crits[-5:]) - crits[-1])
            if it > 1 and (
                crits[-1] < 0.001 or sum(crits[-5:]) / len(crits[-5:]) - crits[-1] < 1e-4
            ):
                break
            zold = z.clone()

        y_denoised = e @ z
        y_denoised = torch.sqrt(r_w) @ y_denoised
        image_fasthyde = y_denoised.T.reshape((row, col, band))
        if normalize:
            image_fasthyde = utils.undo_normalize(image_fasthyde, **consts, by_band=True)
        return image_fasthyde
