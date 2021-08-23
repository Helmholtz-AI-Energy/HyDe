import pytorch_wavelets.dwt.lowlevel as lowlevel
import pywt
import torch

__all__ = [
    "DWTForwardOverwrite",
    "DWTInverse",
    "construct_filters_from_2d",
    "construct_2d_from_filters",
]


class DWTForwardOverwrite(torch.nn.Module):
    """
    Mirrors the setup of the matlab function `FWT2_PO_fast`.
    This will compute a normal (2D) DWT transform and will layer the found filters in to a 2D
    matrix. After each decomposition level, the found filers will be put into a 2D matrix
    starting at (0, 0). The filters take this form, where the top left point is (0, 0):

    .. code-block:: python

        -------------------
        |        |        |
        | cA(LL) | cH(LH) |
        |        |        |
        -------------------
        |        |        |
        | cV(HL) | cD(HH) |
        |        |        |
        -------------------

    Parameters
    ----------
    decomp_level : int, optional
        how many levels to decompose a given image
        default: 1 (a single level)
    wave: str, pywt.Wavelet, optional
        which wavelet to use in the wavelet decomposition
        default: db1
    padding_method: str, optional
        the padding method to use during the decomposition
        options: [zero, symmetric, periodization, constant, reflect, periodic]
        default: zero
    device: str, torch.Device, optional
        the torch device to do the calculations on
    """

    def __init__(self, decomp_level=1, wave="db1", padding_method="zero", device="cpu"):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row, device=device)
        self.register_buffer("h0_col", filts[0])
        self.register_buffer("h1_col", filts[1])
        self.register_buffer("h0_row", filts[2])
        self.register_buffer("h1_row", filts[3])
        # todo: register the buffer for the output
        self.decomp_level = decomp_level
        self.padding_method = padding_method

    def forward(self, x):
        """
        Forward pass of the DWT.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns
        -------
        overwritten_results : torch.Tensor
            the 2D torch Tensor which has all of the results in it
        yl : torch.Tensor
            the lowpass coefficients. yl has shape :math:`(N, C_{in}, H_{in}', W_{in}')`.
        yh : torch.Tensor
            the bandpass coefficients. yh is a list of length `self.decomp_level` with the first
            entry being the finest scale coefficients. it has shape :math:`list(N, C_{in}, 3,
            H_{in}'', W_{in}'')`. The new dimension in yh iterates over the LH, HL and HH
            coefficients.

        Notes
        -----
        :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly downsampled shapes of
        the DWT pyramid.
        """
        yh = []
        ll = x
        padding_method = lowlevel.mode_to_int(self.padding_method)
        full = None

        if x.dtype != self.h0_col.dtype:
            self.h0_col = self.h0_col.to(x.dtype)
            self.h1_col = self.h1_col.to(x.dtype)
            self.h0_row = self.h0_row.to(x.dtype)
            self.h1_row = self.h1_row.to(x.dtype)

        # prev_ll_d0 = None
        # Do a multilevel transform
        for lvl in range(self.decomp_level):
            # Do a level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, padding_method
            )

            s = ll.shape[-2]
            if full is None:  # first iteration
                full_shape = list(ll.shape)
                full_shape[-1] *= 2
                full_shape[-2] *= 2
                full = torch.zeros(full_shape, device=ll.device, dtype=ll.dtype)
            full[:, :, :s, :s] = ll
            full[:, :, :s, s : s * 2] = high[:, :, 0]
            full[:, :, s : s * 2, :s] = high[:, :, 1]
            full[:, :, s : s * 2, s : s * 2] = high[:, :, 2]

            yh.append(high)

        return full, ll, yh


class DWTInverse(torch.nn.Module):
    """
    Performs a 2d DWT Inverse reconstruction of an array

    Parameters
    ----------
    wave : str, pywt.Wavelet, tuple(np.ndarray)
        Which wavelet to use.
        Options: [
            string to pass to pywt.Wavelet constructor,
            pywt.Wavelet class,
            tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row),
            ]
        default: "db1"
    padding_method : str
        the padding scheme to use. Options: 'zero', 'symmetric', 'reflect' or 'periodization'
        default: "zero"
    device : str, torch.Device
        the device to use for the calculation
    """

    def __init__(self, wave="db1", padding_method="zero", device="cpu"):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row, device=device)
        self.register_buffer("g0_col", filts[0])
        self.register_buffer("g1_col", filts[1])
        self.register_buffer("g0_row", filts[2])
        self.register_buffer("g1_row", filts[3])
        self.padding_method = padding_method

    def forward(self, coeffs=None):
        """
        Do the 2D DWT inverse reconstruction for a set of coefficients

        Parameters
        ----------
        coeffs: tuple, torch.Tensor
            tuple of lowpass and bandpass coefficients, where yl is a lowpass tensor of shape
            :math:`(N, C_{in}, H_{in}', W_{in}')` and yh is a list of bandpass tensors of shape
            :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match the format returned
            by DWTForward.
            If this input is a torch tensor, then this will assume that `coeffs` is the overwritten
            results returned by :func:`DWTForwardOverwrite <hyde.dwt3d.DWTForwardOverwrite>`

        Returns
        -------
        torch.Tensor
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Notes
        -----
        - :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly downsampled shapes
        of the DWT pyramid.
        - Can have None for any of the highpass scales and will treat the values as zeros (not
        in an efficient way though).
        """
        yl, yh = coeffs

        ll = yl
        padding_method = lowlevel.mode_to_int(self.padding_method)

        if ll.dtype != self.g0_col.dtype:
            self.g0_col = self.g0_col.to(ll.dtype)
            self.g1_col = self.g1_col.to(ll.dtype)
            self.g0_row = self.g0_row.to(ll.dtype)
            self.g1_row = self.g1_row.to(ll.dtype)

        # Do a multilevel inverse transform
        for h in yh[::-1]:  # this is the reversed list
            if h is None:
                h = torch.zeros(
                    ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device
                )

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = lowlevel.SFB2D.apply(
                ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, padding_method
            )
        return ll


def construct_2d_from_filters(low, highs):
    """
    construct a 2D matrix from the filters out of DWT3D

    this loop builds a list to be cat'ed together
    once done, it will be in the form of:

        .. code::

            |--------------|
            |     low      |
            | ------------ |
            | last filters |
            | ------------ |
            | next last h  |
            | ------------ |
            |     ...      |

    this is the same form that the matlab code has it in

    Parameters
    ----------
    low
    highs

    Returns
    -------

    """
    hold = low.squeeze().permute((1, 2, 0))
    ret_2d = [
        hold.reshape((hold.shape[0] * hold.shape[1], hold.shape[2])),
    ]
    filter_starts = []

    for i in reversed(range(len(highs))):
        filter_starts.append(highs[i][:, :, 0].shape[-1])
        for j in range(3):
            hold = highs[i][:, :, j].squeeze().permute((1, 2, 0))
            ret_2d.append(hold.reshape((hold.shape[0] * hold.shape[1], hold.shape[2])))
    ret_2d = torch.cat(ret_2d, dim=0)
    return ret_2d, filter_starts


def construct_filters_from_2d(matrix, filter_starts, decomp_level):
    """
    construct the filters in the proper shape for the DWT inverse forward step

    Parameters
    ----------
    matrix
    filter_starts
    decomp_level

    Returns
    -------

    """
    exp = filter_starts[0]
    low = matrix[: exp ** 2].reshape((exp, exp, matrix.shape[-1]))
    low = low.permute(2, 0, 1).unsqueeze(0)
    highs = []
    last_end = exp ** 2
    for lvl in range(decomp_level):
        exp = filter_starts[lvl]
        lp_list = [None, None, None]
        for i in range(1, 4):
            next_end = last_end + exp ** 2
            lp_list[i - 1] = (
                matrix[last_end:next_end]
                .reshape((exp, exp, matrix.shape[-1]))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(2)
            )
            last_end = next_end
        highs.append(torch.cat(lp_list, dim=2))

    highs.reverse()
    return low, highs
