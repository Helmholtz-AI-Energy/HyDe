import bm3d
import numpy as np
import torch

from hyde.lowlevel import utils

__all__ = ["FastHyIe"]


class FastHyIe(torch.nn.Module):
    """
    The FastHyIe method is detailed in [1] and [2]. It is designed to use BM3D and eigenvalue decompositions
    to remove Gaussian and Poissonian noise as well as fill in missing data (in-painting).

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
        super(FastHyIe, self).__init__()

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        noise_type="additive",
        iid=True,
        k_subspace=10,
        normalize=True,
    ):
        """
        Run the FastHyDe denoising algorithm.

        Parameters
        ----------
        img: torch.Tensor
            hyperspectral dataset. The format is assumed to be [rows, cols, bands].
        mask: torch.Tensor
            mask matrix of the same size as img.
            values of 1 in the mask indicate that there is viable data  for that element
            values of 0 in the mask indicate that there is no viable data for that element
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
        rows, cols, b = img.shape
        n = rows * cols
        m2d = mask.reshape((n, b)).T
        m_pix = torch.ones((rows, cols), dtype=torch.bool, device=img.device)
        m_pix[torch.nonzero(mask.sum(dim=-1) < b, as_tuple=True)] = 0

        # normalize and get the location of NOT 0 values in the image
        if normalize:
            img, consts = utils.normalize(img, by_band=True)
        y = torch.reshape(img, (n, b)).T
        m_pix = m_pix.flatten()
        loc_tmp = torch.nonzero(m_pix == 1, as_tuple=True)[0]

        # ----------------------------- Data transformation ---------------------------------
        # Observed data with additive Gaussian non-iid noise or Poissonian noise are transformed
        # in order to  to have additive Gaussian i.i.d. noise before the denoisers are applied.

        # switch noise_type
        if noise_type == "additive":
            #     case 'additive'
            if iid == 0:  # additive Gaussian non-iid noise, applying eq. ?
                # Y = reshape(img_ori, N, B)';
                y = torch.reshape(img, (n, b)).T
                # [w Rw] = estNoise(Y,noise_type);
                w, r_w = utils.estimate_hyperspectral_noise(
                    y[:, loc_tmp],
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
            # approximately additive noise.
            img = 2 * torch.sqrt(torch.abs(img + 3 / 8))

        # subspace estimation using HySime or SVD
        w, r_w = utils.estimate_hyperspectral_noise(
            y[:, loc_tmp], noise_type=noise_type, calculation_dtype=torch.float64
        )
        _, e = utils.hysime(input=y[:, loc_tmp], noise=w, noise_corr=r_w)
        e = e[:, :k_subspace]

        k_subspace = e.shape[1] if e.shape[1] < k_subspace else k_subspace

        eigen_y = torch.zeros((e.shape[1], y.shape[1]), dtype=y.dtype, device=y.device)
        eigen_y[:, loc_tmp] = e.T @ y[:, loc_tmp]
        # % For incompletely observed pixels:
        for icorrupt in torch.nonzero(m_pix == 0, as_tuple=True)[0]:
            mi = torch.diag(m2d[:, icorrupt])
            non_strp_band = torch.nonzero(m2d[:, icorrupt] == 1, as_tuple=True)[0]
            mi = mi[non_strp_band]
            yi = y[non_strp_band, icorrupt]

            eigen_y[:, icorrupt] = torch.inverse(e.T @ mi.T @ mi @ e) @ e.T @ mi.T @ yi

        # %% --------------------------Eigen-image denoising ------------------------------------
        # send slices of the image to the GPU if that is the case,
        np_dtype = np.float32 if img.dtype is torch.float32 else np.float64
        eigen_y_bm3d = np.empty((k_subspace, n), dtype=np_dtype)
        ecpu = e.to(device="cpu", non_blocking=True)
        r_w = r_w.to(device="cpu", non_blocking=True)

        nxt_eigen = eigen_y[0].cpu()
        for i in range(k_subspace):
            lp_eigen = nxt_eigen.numpy()
            if i < k_subspace - 1:
                nxt_eigen = eigen_y[i + 1].to(device="cpu", non_blocking=True)
            # produce eigen-image
            #     eigen_im = eigen_Y(i,:);
            eigen_im = lp_eigen
            #     min_x = min(eigen_im);
            min_x = np.min(eigen_im)  # dim?
            # max_x = max(eigen_im);
            max_x = np.max(eigen_im)  # dim?
            # eigen_im = eigen_im - min_x;
            eigen_im -= min_x
            # scale = max_x-min_x;
            scale = max_x - min_x
            # %scale to [0,1]
            # eigen_im = reshape(eigen_im, rows, cols)/scale;
            eigen_im = np.reshape(eigen_im, (rows, cols)) / scale
            # %estimate noise from Rw
            # sigma = sqrt(E(:,i)'*Rw*E(:,i))/scale;
            if i == 0:
                ecpu = ecpu.numpy()
                r_w = r_w.numpy()
            sigma = np.sqrt(ecpu[:, i].T @ r_w @ ecpu[:, i]) / scale
            # [~, filt_eigen_im] = BM3D(1,eigen_im, sigma*255);
            filt_eigen_im = bm3d.bm3d(eigen_im, sigma)
            # eigen_Y_bm3d(i,:) = reshape(filt_eigen_im*scale + min_x, 1,N);
            eigen_y_bm3d[i, :] = (filt_eigen_im * scale + min_x).reshape(eigen_y_bm3d[i, :].shape)

        eigen_y_bm3d = torch.tensor(eigen_y_bm3d, dtype=img.dtype, device=img.device)
        # % reconstruct data using denoising engin images
        y_reconst = e @ eigen_y_bm3d
        # Y_reconst = E*eigen_Y_bm3d;
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
