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

    def __init__(self, decomp_level=1, wave="db1", mode="zero"):
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
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
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
