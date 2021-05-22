import pytorch_wavelets.dwt.lowlevel as lowlevel
import pywt
import torch

__all__ = ["DWTForwardOverwrite"]


class DWTForwardOverwrite(torch.nn.Module):
    """
    mirrors the setup of function wc = FWT2_PO_fast(x,L,qmf)
    % FWT2_PO_fast -- 2-d MRA wavelet transform (periodized, orthogonal)
    %  Usage
    %    wc = FWT2_PO_fast(x,L,qmf)
    %  Inputs
    %    x     2-d object (nx by ny by nz array, nx,ny,nz dyadic)
    %    L     coarse level
    %    qmf   quadrature mirror filter
    %  Outputs
    %    wc    2-d wavelet transform
    %
    %  Description
    %    A three-dimensional Wavelet Transform is computed for the
    %    array x.  To reconstruct, use IWT2_PO_fast.
    %
    %  See Also
    %    IWT2_PO_fast, MakeONFilter
    %
    % This is a fast implementation for FWT2_PO from Wavelab
    % (c) 2013 Behnood Rasti
    % behnood.rasti@gmail.com

    """

    def __init__(self, decomp_level=1, wave="db1", mode="zero", device="cpu"):
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
        self.mode = mode

    def forward(self, x):
        """Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)
        ret = None

        # Do a multilevel transform
        for _ in range(self.decomp_level):
            # Do a level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode
            )

            s = ll.shape[-2]
            if ret is None:
                ret_shape = list(ll.shape)
                ret_shape[-1] *= 2
                ret_shape[-2] *= 2
                ret = torch.zeros(ret_shape, device=ll.device, dtype=ll.dtype)
            ret[:, :, :s, :s] = ll
            ret[:, :, :s, s : s * 2] = high[:, :, 0]
            ret[:, :, s : s * 2, :s] = high[:, :, 1]
            ret[:, :, s : s * 2, s : s * 2] = high[:, :, 2]

            yh.append(high)

        return ret, ll, yh


class DWTInverse(torch.nn.Module):
    """Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave="db1", mode="zero", device="cpu"):
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
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
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
                ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode
            )
        return ll
