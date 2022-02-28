from typing import Tuple

import torch
from skimage.restoration import denoise_tv_bregman

from hyde.lowlevel import utils

__all__ = ["OTVCA"]


class OTVCA(torch.nn.Module):
    """
    OTVCA - orthogonal total variation component analysis

    This method is explained in detail in [1].

    Abstract (excerpt from [1])
    -------------------
        In this paper, a novel feature extraction method, called orthogonal total variation component analysis (OTVCA),
        is proposed for remotely sensed hyperspectral data. The features are extracted by minimizing a total variation
        (TV) penalized optimization problem. The TV penalty promotes piecewise smoothness of the extracted features
        which is useful for classification. A cyclic descent algorithm called OTVCA-CD is proposed for solving the
        minimization problem.

    References
    ----------
    [1] B. Rasti, M. O. Ulfarsson and J. R. Sveinsson, "Hyperspectral Feature Extraction Using Total Variation
        Component Analysis," in IEEE Transactions on Geoscience and Remote Sensing, vol. 54, no. 12, pp. 6976-6985,
        Dec. 2016, doi: 10.1109/TGRS.2016.2593463.
    """

    def __init__(self):
        super(OTVCA, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        features: int,
        num_itt: int = 10,
        lam: float = 0.01,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise an image `x` using the HyRes algorithm.

        Parameters
        ----------
        x: torch.Tensor
            input image
        features: int
            Number of features to extract, this could be selected equal to the
            number of classes of interests in the scene
        num_itt: int, optional
            Number of iterations; 200 iterations are default value
        lam: float, optional
            Tuning parameter; Default is 0.01 for normalized HSI
            the value passed to the denoising algorithm is 1/lam
        normalize: bool, optional
            flag indicating if the input should be normalized

        Returns
        -------
        denoised_image : torch.Tensor
            Hyperspectral features extracted (3D matrix)
        fe : torch.Tensor
            Extracted features
        """
        nr1, nc1, p1 = x.shape

        if normalize:
            x, consts = utils.normalize(x, by_band=True)

        x_2d = x.reshape((nr1 * nc1, p1))
        _, _, vh = torch.linalg.svd(x_2d, full_matrices=False)
        v1 = vh.T
        v = v1[:, :features]
        fe = torch.zeros((nr1, nc1, features), dtype=x.dtype, device=x.device)

        for fi in range(num_itt):
            c1 = x_2d @ v[:, :features]
            pc = c1.reshape((nr1, nc1, features))

            fe = denoise_tv_bregman(image=pc.cpu().numpy(), weight=1 / lam, eps=0.1, isotropic=True)
            fe = torch.tensor(fe, dtype=x.dtype, device=x.device)

            fe_reshape = fe.reshape((nr1 * nc1, features))
            m = x_2d.T @ fe_reshape
            c, _, gh = torch.linalg.svd(m, full_matrices=False)
            v = c @ gh

        denoised_image = (fe_reshape @ v.T).reshape((nr1, nc1, p1))
        if normalize:
            denoised_image = utils.undo_normalize(denoised_image, **consts, by_band=True)
        return denoised_image, fe
