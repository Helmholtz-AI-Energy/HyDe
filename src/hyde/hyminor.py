import torch

from . import utils
from .hyres import HyRes

__all__ = ["HyMiNoR"]


class HyMiNoR:
    """
    HyMiNoT -- Hyperspectral Mixed Gaussian and Sparse Noise Reduction

    This is a two step mixed nose removal technique for hypersperctral images which was presented
    in [1] The two steps are:

        1. The Gaussian noise is removed using :func:`HyRes <hyde.hyres.HyRes>`
        2. The sparse noised is solved as: :math:`\min\limits_{X} ||H-X||_1+ \lambda ||X*D'||_1` where :math:`D` is the difference matrix.

    The data used should be normalized to range from 0 to 1.

    Parameters
    ----------
    decomp_level : int, optional
        the level of the wavelet decomposition to do
        default: 5
    wavelet_level : int, optional
        the integer value indicating which Daubechies wavelet to use. i.e. 5 -> db5
        default: 5
    padding_method : str, optional
        the method used to pad the image during the DWT transform.
        options: [zero, symmetric, periodization, reflect, periodic]
        default: "symmetric"

    Notes
    -----
    Algorithmic questions should be forwarded to the original authors. This is purely an
    implementation of the algorithm detailed in [1].

    References
    ----------
    [1] B. Rasti, P. Ghamisi and J. A. Benediktsson, "Hyperspectral Mixed Gaussian and Sparse Noise Reduction," in IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 3, pp. 474-478, March 2020, doi: 10.1109/LGRS.2019.2924344.
    """

    def __init__(self, decomp_level=5, wavelet_level=5, padding_method="symmetric"):
        self.hyres = HyRes(
            decomp_level=decomp_level,
            wavelet_level=wavelet_level,
            padding_method=padding_method,
        )

    def forward(self, x: torch.Tensor, lam: float = 10.0, iterations: int = 50):
        """
        Do the HyMiNoR decomposition.

        Parameters
        ----------
        x : torch.Tensor
            the image/array to be de-noised
        lam : float, optional
            the tuning parameter
            default: 10.
        iterations : int, optional
            the number of iterations to do
            default: 50

        Returns
        -------
        denoised image : torch.Tensor
        """
        base_dtype = x.dtype
        # H_M -> x ; lambda -> lam
        mu1, mu2, its = 0.5, 0.5, iterations
        # H=HyRes(H_M);
        hyres_result = self.hyres.forward(x)

        m, n, d = hyres_result.shape
        # [m,n,d]=size(H);
        mn = m * n
        # Y=reshape(H,mn,d)';
        hyres_reshaped_t = torch.conj(hyres_result.reshape((mn, d))).T
        l1 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        l2 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        v1 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        v2 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        xx = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # X=L1;
        eye_d = torch.eye(d, device=x.device, dtype=x.dtype)
        hold = utils.diff_dim0_replace_last_row(utils.diff(eye_d))
        u, s, v = torch.linalg.svd(hold.to(torch.float64), full_matrices=False)
        v = torch.conj(v.T)

        magic = torch.chain_matmul(
            v,
            torch.diag(1.0 / (mu1 + mu2 * s)),
            torch.conj(u.T),
        ).to(base_dtype)

        for i in range(its):
            # subminimization problems
            hold1 = -mu1 * (v1 - hyres_reshaped_t - l1)
            hold2 = mu2 * utils.diff_dim0_replace_last_row((v2 - l2))
            # X=Majic*(-mu1*(V1-Y-L1)+mu2*Dvt(V2-L2));
            xx = magic @ (hold1 + hold2)

            v1 = utils.soft_threshold(hyres_reshaped_t - xx + l1, 1.0 / mu1)
            dv_xx = utils.diff(xx)
            v2 = utils.soft_threshold(dv_xx + l2, float(lam) / mu2)
            # Updating L1 and L2
            l1 += hyres_reshaped_t - xx - v1
            l2 += dv_xx - v2

        ret = torch.reshape(xx.T, (m, n, d))
        return ret


if __name__ == "__main__":
    import time

    import scipy.io as sio

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

    s = torch.sum(input_tens ** 2.0)
    # print(s)
    d = torch.sum((input_tens - output) ** 2.0)
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
