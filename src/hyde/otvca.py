from typing import Union

import torch

from . import dwt3d, utils

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
        self, x: torch.Tensor, features: int, num_itt: int = 200, lam: float = 0.05
    ) -> Union[torch.Tensor, torch.Tensor]:
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
        fe : torch.Tensor
            Extracted features
        """
        # key: Y -> x (image), r_max -> features, tol -> num_itt
        nr1, nc1, p1 = x.shape
        x_2d = x.reshape((nr1 * nc1, p1))
        x_min = x.min()
        x_max = x.max()
        # NRY = (RY - m) / (M - m);
        normalized_y = (x_2d - x_min) / (x_max - x_min)
        _, _, vh = torch.linalg.svd(normalized_y, full_matrices=False)
        v1 = vh.T
        v = v1[:, :features]
        fe = torch.zeros((nr1, nc1, features), dtype=x.dtype, device=x.device)

        for fi in range(num_itt):
            c1 = normalized_y @ v[:, :features]
            pc = c1.reshape((nr1, nc1, features))

            fe = utils.denoise_tv_bregman(image=pc, weight=1 / lam)

            fe_reshape = fe.reshape((nr1 * nc1, features))
            m = normalized_y.T @ fe_reshape
            c, _, gh = torch.linalg.svd(m, full_matrices=False)
            v = c @ gh

        yr = (fe_reshape @ v.T).reshape((nr1, nc1, p1))
        return yr, fe
