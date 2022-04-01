import bm3d
import numpy as np
import torch

from hyde.lowlevel import utils

__all__ = ["FastHyDe"]


class FastHyDe(torch.nn.Module):
    """
    The FastHyDe method is detailed in [1] and [2]. It is designed to use BM3D and eigenvalue decompositions
    to remove Gaussian and Poissonian noise.

    For more details about how to run the algorithm, see the :func:`forward` function.

    References
    ----------
    [1] L. Zhuang and J. M. Bioucas-Dias,
        "Fast hyperspectral image denoising based on low rank and sparse
        representations, in 2016 IEEE International Geoscience and Remote
        Sensing Symposium (IGARSS 2016), 2016.
    [2] L. Zhuang and J. M. Bioucas-Dias,
        "Fast hyperspectral image denoising and inpainting based on low rank
        and sparse representations, Submitted to IEEE Journal of Selected
        Topics in Applied Earth Observations and Remote Sensing, 2017.
        URL: http://www.lx.it.pt/~bioucas/files/submitted_ieee_jstars_2017.pdf


    Original Copyright
    ------------------
    Copyright (July, 2017):
                Lina Zhuang (lina.zhuang@lx.it.pt)
                &
                José Bioucas-Dias (bioucas@lx.it.pt)

    FastHyDe is distributed under the terms of
    the GNU General Public License 2.0.

    Permission to use, copy, modify, and distribute this software for
    any purpose without fee is hereby granted, provided that this entire
    notice is included in all copies of any software which is or includes
    a copy or modification of this software and in all copies of the
    supporting documentation for such software.
    This software is being provided "as is", without any express or
    implied warranty.  In particular, the authors do not make any
    representation or warranty of any kind concerning the merchantability
    of this software or its fitness for any particular purpose."
    """

    def __init__(
        self,
    ):
        super(FastHyDe, self).__init__()

    def forward(
        self,
        img: torch.Tensor,
        noise_type: str = "additive",
        iid: bool = True,
        k_subspace: int = 10,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Run the FastHyDe denoising algorithm.

        Parameters
        ----------
        img: torch.Tensor
            hyperspectral dataset. The format is assumed to be [rows, cols, bands].
        noise_type: str
            What type of noise is expected in the image
            must be one of [additive, poisson], default: additive
        iid: bool
            If gaussian (additive noise) is i.i.d. or not. This is not used for poissonian noise.
            default: True
        k_subspace: int
            The number of signal subspaces to scan.
            default: 10
        normalize: bool
            if true, normalize the data (by band) before the algorithm.
            default: True.

        Returns
        -------
        denoised image: torch.Tensor
        """
        # [rows, cols, B] = size(img_ori);
        rows, cols, b = img.shape
        # N=rows*cols;
        n = rows * cols
        # ----------------------------- Data transformation ---------------------------------
        # Observed data with additive Gaussian non-iid noise or Poissonian noise are transformed
        # in order to  to have additive Gaussian i.i.d. noise before the denoisers are applied.
        if normalize:
            img, consts = utils.normalize(img, by_band=True)

        # switch noise_type
        if noise_type == "additive":
            #     case 'additive'
            if iid == 0:  # additive Gaussian non-iid noise, applying eq. ?
                y = torch.reshape(img, (n, b)).T
                w, r_w = utils.estimate_hyperspectral_noise(
                    y,
                    noise_type,
                    calculation_dtype=img.dtype,
                )
                r_w_og = r_w.clone()
                # Y = sqrt(inv(Rw_ori))*Y;
                y = torch.sqrt(torch.inverse(r_w_og)) @ y
                img = torch.reshape(torch.conj(y.T), (rows, cols, b))
        elif noise_type == "poisson":
            # case 'poisson'
            # applying the Anscombe transform, which converts Poissonion noise into
            # approximately additive noise.
            # img_ori = 2*sqrt(abs(img_ori+3/8));
            img = 2 * torch.sqrt(torch.abs(img + 3 / 8))

        # Y = reshape(img_ori, N, B)';
        y = torch.reshape(img, (n, b)).T

        # subspace estimation using HySime or SVD
        w, r_w = utils.estimate_hyperspectral_noise(
            y, noise_type=noise_type, calculation_dtype=torch.float64
        )

        # [~, E]=hysime(Y,w,Rw);
        _, e = utils.hysime(input=y, noise=w, noise_corr=r_w)
        e = e[:, :k_subspace]
        # eigen_Y = E'*Y;
        eigen_y = e.T @ y
        # %% --------------------------Eigen-image denoising ------------------------------------
        eigen_y_bm3d = fast_hyde_eigen_image_denoising(img, k_subspace, r_w, e, eigen_y, n)
        eigen_y_bm3d = torch.tensor(eigen_y_bm3d, dtype=img.dtype, device=img.device)
        # % reconstruct data using denoising engin images
        y_reconst = e @ eigen_y_bm3d[: e.shape[1]]
        # %% ----------------- Re-transform ------------------------------
        if noise_type == "additive":
            if not iid:
                y_reconst = torch.sqrt(r_w_og @ y_reconst)
        elif noise_type == "poisson":
            y_reconst = (y_reconst / 2) ** 2 - 3 / 8
        image_fasthyde = y_reconst.T.reshape((rows, cols, b))
        if normalize:
            image_fasthyde = utils.undo_normalize(image_fasthyde, **consts, by_band=True)
        return image_fasthyde


def fast_hyde_eigen_image_denoising(img, k_subspace, r_w, e, eigen_y, n) -> np.ndarray:
    # %% --------------------------Eigen-image denoising ------------------------------------
    # send slices of the image to the GPU if that is the case,
    rows, cols, b = img.shape
    np_dtype = np.float32 if img.dtype is torch.float32 else np.float64
    eigen_y_bm3d = np.empty((k_subspace, n), dtype=np_dtype)
    ecpu = e.to(device="cpu", non_blocking=True)
    r_w = r_w.to(device="cpu", non_blocking=True)

    nxt_eigen = eigen_y[0].cpu()
    mx = min(k_subspace, eigen_y.shape[0])
    for i in range(mx):
        lp_eigen = nxt_eigen.numpy()
        if i < mx - 1:
            nxt_eigen = eigen_y[i + 1].to(device="cpu", non_blocking=True)
        # produce eigen-image
        eigen_im = lp_eigen
        min_x = np.min(eigen_im)
        max_x = np.max(eigen_im)
        eigen_im -= min_x
        scale = max_x - min_x
        # normalize eigen_im
        eigen_im = np.reshape(eigen_im, (rows, cols)) / scale
        if i == 0:
            ecpu = ecpu.numpy()
            r_w = r_w.numpy()
        sigma = np.sqrt(ecpu[:, i].T @ r_w @ ecpu[:, i]) / scale

        filt_eigen_im = bm3d.bm3d(eigen_im, sigma)

        eigen_y_bm3d[i, :] = (filt_eigen_im * scale + min_x).reshape(eigen_y_bm3d[i, :].shape)

    return eigen_y_bm3d
