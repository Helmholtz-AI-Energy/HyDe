import warnings
from typing import Tuple, Union

import torch

from hyde.lowlevel import dwt3d, utils

__all__ = ["FORPDN_SURE"]


class FORPDN_SURE(torch.nn.Module):
    """
    HyRes -- Automatic Hyperspectral Restoration Using low-rank and sparse modeling.

    The model used is :math:`Y=D_2XV'+N` and penalized least squares with :math:`\ell_1` penalty.
    The formula to restore the signal is:

    .. math::

        argmax(0.5 * ||Y-D_2XV'||_F^2+\lambda||X||_1)

    This method relies on Daubechies wavelets for wavelet decomposition

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
    [1] B. Rasti, M. O. Ulfarsson and P. Ghamisi, "Automatic Hyperspectral Image Restoration
    Using Sparse and Low-Rank Modeling," in IEEE Geoscience and Remote Sensing Letters, vol. 14,
    no. 12, pp. 2335-2339, Dec. 2017, doi: 10.1109/LGRS.2017.2764059.
    """

    def __init__(
        self,
        padding_method="symmetric",
    ):
        super(FORPDN_SURE, self).__init__()
        self.decomp_level = None  # L
        self.wavelet_name = None  # "db" + str(wavelet_level)
        self.padding_method = padding_method

        # both of these are defined during the forward function
        self.dwt_forward = None
        self.dwt_inverse = None

    def forward(
        self,
        img: torch.Tensor,
        s_int_st: torch.Tensor = torch.tensor([0, 0, 0, 0]),
        s_int_step: torch.Tensor = torch.tensor([0.001, 0.01, 0.1, 1]),
        s_int_end: torch.Tensor = torch.tensor([0.01, 0.1, 1, 10]),
        domain: str = "wavelet",
        scale: bool = True,
        scale_const: Union[int, float] = 255,
        wavelet_level: int = 6,
    ) -> torch.Tensor:
        """
        Call the FORPDN_SURE method.
        NOTE: the decomposition level for wavelet-based denoising is 1 minus the length of the s_int_* Tensors.

        Parameters
        ----------
        img: torch.Tensor
            signal image to denoise
        s_int_st: torch.Tensor
            Starting points of the SURE intervals
            must be the same size as s_int_step, s_int_end
        s_int_step: torch.Tensor
            Step size of the SURE intervals
            must be the same size as s_int_st, s_int_end
        s_int_end: torch.Tensor
            Ending points of the SURE intervals
            must be the same size as s_int_st, s_int_step
        domain: str, optional
            domain on which to do denoising
            options: [signal, wavelet]
            default: wavelet
        scale: bool, optional
            Toggle determining if `img` should be scaled
            default: True
        scale_const: int, float, optional
            Constant used to scale `img`
            default: 255
        wavelet_level: int, optional
            wavelet level to be used during wavelet-based denoising (only used when domain == wavelet)
            default: 6

        Returns
        -------
        denoised_image: torch.Tensor
        noise_std: torch.Tensor
            noise standard deviation for each band as a vector
        """
        self.wavelet_name = "db" + str(wavelet_level)
        rows, cols, bands = img.shape
        if scale:
            img, consts = utils.scale_wo_shift(img, factor=scale_const)

        # put all of the sure interval tensors on the same sevice
        s_int_st = s_int_st.to(device=img.device, non_blocking=True)
        s_int_step = s_int_step.to(device=img.device, non_blocking=True)
        s_int_end = s_int_end.to(device=img.device, non_blocking=True)

        # % Decomposition level
        # decomp_lvl=length(s_int_st)-1;
        # from matlab docs: lengths -> "For arrays with more dimensions, the length is max(size(X))"
        self.decomp_level = max(tuple(s_int_st.shape)) - 1  # TODO: minus 1 needed

        # set up the DWT transforms to be used later
        if domain == "wavelet":
            self.dwt_forward = dwt3d.DWTForward(
                self.decomp_level,
                self.wavelet_name,
                self.padding_method,
                device=img.device,
            )
            self.dwt_inverse = dwt3d.DWTInverse(
                wave=self.wavelet_name, padding_method=self.padding_method, device=img.device
            )

        # Symmetric extension
        if rows % (2 ** self.decomp_level) != 0:
            img = utils.symmetric_pad(
                img, [0, 0, 0, 0, 0, 2 ** self.decomp_level - (rows % 2 ** self.decomp_level)]
            )
        if cols % (2 ** self.decomp_level) != 0:
            img = utils.symmetric_pad(
                img, [0, 0, 0, 2 ** self.decomp_level - (cols % 2 ** self.decomp_level), 0, 0]
            )

        rowsp, colsp, bandsp = img.shape
        yp_m = torch.reshape(img.permute((1, 0, 2)), (rowsp * colsp, bandsp))
        xp2 = yp_m.clone()
        # Noise estimation
        w, r_w = utils.estimate_hyperspectral_noise(
            yp_m.T,
            calculation_dtype=torch.float64,
        )
        sigma = torch.sqrt(torch.var(w.T, dim=0).T)

        # get the eigenvalues to be used in the domains later
        c12 = torch.diag(sigma)
        u, s, vh = torch.linalg.svd(c12 @ utils.diff_dim0_replace_last_row(utils.diff(c12)))
        c1 = torch.pow(c12, 2)
        c12u = c12 @ u
        c12v = vh @ c12  # vh is already trandposed (matlab doesnt do that)

        if s_int_step[0] == 0:
            t = torch.zeros(3, device=img.device, dtype=torch.long)
        else:
            t = torch.arange(s_int_st[0], s_int_end[0] + s_int_step[0], s_int_step[0])

        diag_s = s
        if domain == "signal":
            xp, nsig, opt_params, sure1 = self._forpdns(
                img, t, rowsp, colsp, bandsp, sigma, diag_s, c1, c12u, c12v, yp_m
            )
        elif domain == "wavelet":
            xp, nsig, opt_params, sure1 = self._forpdnt(
                img, t, sigma, c1, c12u, c12v, diag_s, s_int_st, s_int_end, s_int_step, xp2
            )
        else:
            raise ValueError(
                f"'domain' must be one of [signal, wavelet]. case matters. current value: {domain}"
            )

        image_forpdn = xp[:rows, :cols, :bands]
        if scale:
            image_forpdn = utils.rescale_wo_shift(image_forpdn, **consts, factor=scale_const)
        return image_forpdn

    @staticmethod
    def _forpdns(img, t: torch.Tensor, rowsp, colsp, bandsp, sigma, diag_s, c1, c12u, c12v, yp_m):
        # signal based decomposition
        x1 = img.reshape((rowsp * colsp, bandsp))
        x1 = x1.repeat(torch.pow(sigma, 2), (rowsp * colsp, 1)) * x1
        sure = torch.zeros((t.shape[0], 2), dtype=img.dtype, device=img.device)
        for i, lam in enumerate(t):
            sig_lam_eye = torch.diag(1.0 / (1.0 / (lam * diag_s) + 1.0))
            majc = c1 - c12u @ (sig_lam_eye @ c12v)
            xp_m = (majc @ x1.T).T
            norm1 = torch.linalg.norm(yp_m - xp_m, ord=2)
            sure[i, 0] = norm1 / (rowsp * colsp) + 2 * torch.trace(majc)
            if i > 1 and sure[i, 0] > sure[i - 1, 0]:
                break
        opt_params = t[torch.argmin(sure)]
        xp = xp_m.reshape((rowsp, colsp, bandsp))
        return xp, sigma, opt_params, sure

    def _forpdnt(self, img, t, sigma, c1, c12u, c12v, diag_s, s_int_st, s_int_end, s_int_step, xp2):
        # wavelet based decomposition
        opt_params = torch.zeros((self.decomp_level + 1, 1), dtype=img.dtype, device=img.device)
        # do wavelet transform
        img_dwt_lows, img_dwt_highs = self.dwt_forward.forward(img.permute((2, 0, 1)).unsqueeze(0))
        ax1_dwt_full, ax1_filter_starts = dwt3d.construct_2d_from_filters(
            low=img_dwt_lows, highs=img_dwt_highs
        )

        ax = torch.pow(sigma.T, -2) * ax1_dwt_full
        sure = torch.zeros((t.shape[0], 2), dtype=img.dtype, device=img.device)
        # xp2 is the output after the transforms
        xp2 = torch.zeros_like(ax1_dwt_full)
        c_break = 0
        st = 0
        sp = ax1_filter_starts[0][0] * ax1_filter_starts[0][1]
        # ax1_filter_starts[0] ** 2
        idx = slice(st, sp)
        # operate on the `low` filters from dwt
        for i, lam in enumerate(t):
            sig_lam_eye = torch.diag(1.0 / (1.0 / (lam * diag_s) + 1.0))
            majc = c1 - c12u @ (sig_lam_eye @ c12v)
            # this will set the output based on the low filters from the DWT transform
            #   and the eigenvectors/values from earlier
            xp0 = (majc @ ax[idx].T).T
            norm1 = torch.linalg.norm(ax1_dwt_full[idx] - xp0, ord=2) ** 2
            sure[i, 0] = norm1 / float(sp - st) + 2 * torch.trace(majc)
            if i > 1 and sure[i, 0] > sure[i - 1, 0]:
                c_break = 1
                break
            xp2[idx] = xp0.clone()

        if c_break == 0:
            warnings.warn(
                "The optimum regularization parameter has not found for the 1st interval. "
                "consider increasing the 1st interval.",
                UserWarning,
            )
        i3 = torch.argmin(sure)
        opt_params[0] = t[i3]
        sure1 = [
            sure,
        ]
        if opt_params[0] == t[0]:
            warnings.warn(
                "SURE selects zero for LL: To get better performance decrease s_int_step for the 1st interval."
                "Probably you can decrease s_int_end as well.",
                UserWarning,
            )

        for j in range(self.decomp_level):
            # need to iterate over the levels.
            # the number high filters from DWT are equal to the decomp level
            c_break = 0
            try:
                t = torch.arange(
                    s_int_st[j + 1],
                    s_int_end[j + 1],
                    s_int_step[j + 1],
                    device=img.device,
                    dtype=img.dtype,
                )
                tst = s_int_step[j + 1]
            except IndexError:  # this is the last loop case
                t = s_int_step[-1]
                tst = s_int_step[-1]

            if tst == 0:
                t = torch.tensor(
                    [
                        0,
                    ],
                    device=img.device,
                    dtype=img.dtype,
                )

            sure *= 0
            # ax1_filter_starts has the filter sizes, since they are square, we know where they start in the 2D matrix.
            st = ax1_filter_starts[j][0] * ax1_filter_starts[j][1]
            # ax1_filter_starts[j] ** 2
            try:
                sp = ax1_filter_starts[j + 1][0] * ax1_filter_starts[j + 1][1]
                # ax1_filter_starts[j + 1] ** 2
            except IndexError:
                sp = ax1_dwt_full.shape[0]
            idx = slice(st, sp)
            for i, lam in enumerate(t):
                sig_lam_eye = torch.diag(1.0 / (1.0 / (lam * diag_s) + 1.0))
                majc = c1 - c12u @ (sig_lam_eye @ c12v)
                # this will set the output based on the high filters from the DWT transform
                #   and the eigenvectors/values from earlier
                xp0 = (majc @ ax[idx].T).T
                norm1 = torch.linalg.norm(ax1_dwt_full[idx] - xp0, ord=2) ** 2
                sure[i, 0] = norm1 / ((sp - st) + 2 * torch.trace(majc))
                if i > 1 and sure[i, 0] > sure[i - 1, 0]:
                    c_break = i
                    break
                xp2[idx] = xp0
            if c_break == 1:
                warnings.warn(
                    f"The optimum regularization parameter has not found for the {j+1}'th interval, you might want "
                    f"to consider to increase this interval.",
                    UserWarning,
                )
            i2 = torch.argmin(sure)  # todo: dim??
            opt_params[j + 1] = t[i2]
            sure1.append(sure)
            if opt_params[j + 1] == t[0]:
                warnings.warn(
                    f"SURE selects zero, please decrease s_int_step for the {j + 1}-th interval, Probably you can decrease"
                    f" s_int_end as well.",
                    UserWarning,
                )
        lows, highs = dwt3d.construct_filters_from_2d(
            matrix=xp2, filter_starts=ax1_filter_starts, decomp_level=self.decomp_level
        )
        xp = self.dwt_inverse((lows, highs))
        xp = xp.squeeze().permute((1, 2, 0))
        return xp, sigma, opt_params, sure1
