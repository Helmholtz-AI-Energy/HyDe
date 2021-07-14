import torch
import numpy as np

from . import dwt3d, utils

__all__ = ["WSRRR"]


class WSRRR(torch.nn.Module):
    """

    """

    def __init__(self, decomp_level=3, wavelet_level=5, padding_method="symmetric"):
        super(WSRRR, self).__init__()
        self.decomp_level = decomp_level  # L
        self.wavelet_name = "db" + str(wavelet_level)
        self.device = "cpu"

        self.padding_method = padding_method

        self.dwt_forward = dwt3d.DWTForwardOverwrite(
            decomp_level,
            self.wavelet_name,
            self.padding_method,
            device=self.device,
        )
        self.dwt_inverse = dwt3d.DWTInverse(
            wave=self.wavelet_name, padding_method=self.padding_method, device=self.device
        )

    def forward(self, x: torch.Tensor, r):
        """
        Denoise an image `x` using the HyRes algorithm.

        Parameters
        ----------
        x: torch.Tensor
            input image

        Returns
        -------
        denoised_image : torch.Tensor
        """
        if x.device != self.device:
            self.device = x.device
            self.dwt_forward = dwt3d.DWTForwardOverwrite(
                self.decomp_level,
                self.wavelet_name,
                self.padding_method,
                device=self.device,
            )
            self.dwt_inverse = dwt3d.DWTInverse(
                wave=self.wavelet_name, padding_method=self.padding_method, device=self.device
            )

        # L=3; -> level of decomp
        # [nr,nc,p]=size(Y);
        og_rows, og_cols, og_channels = x.shape
        two_d_shape = (og_rows * og_cols, og_channels)

        # if mod(nr,2^L)~=0
        #     Y=extension('2d','sym',Y,[2^L-mod(nr,2^L),0],'r',0,'r');
        # end
        # if mod(nc,2^L)~=0
        #     Y=extension('2d','sym',Y,[0,2^L-mod(nc,2^L)],'r',0,'r');
        # end

        if og_rows % (2 ** self.decomp_level) != 0:
            x = utils.symmetric_pad(
                x, [0, 0, 0, 0, 0, 2 ** self.decomp_level - (og_rows % 2 ** self.decomp_level)]
            )
        if og_cols % (2 ** self.decomp_level) != 0:
            x = utils.symmetric_pad(
                x, [0, 0, 0, 2 ** self.decomp_level - (og_cols % 2 ** self.decomp_level), 0, 0]
            )

        # [nx1,ny1,nz1]=size(Y);
        padded_shape = tuple(x.shape)

        # NFC=10;%number of filter coefficients       -> this is wavelet level * 2
        # qmf = daubcqf(NFC,'min');%wavelet filter
        # this is handled in the init


        """
        2021.14.07
        
        the issue in this now seems to be related to that the indices used during parts of this 
        are not what they should be. since the DWT transform being used is not the same, 
        the padding is also not the same. therefor, I need to figure out what these indices are 
        doing before i can correct the bug.
        """























        # % Noise Variance Estimation
        # [WY,s1,s2]=twoDWTon3Ddata(Y,L,qmf,'FWT_PO_1D_2D_3D_fast');
        # s1/s2 are not needed
        v_dwt_full, v_dwt_lows, v_dwt_highs = self.dwt_forward.forward(
            x.permute((2, 0, 1)).unsqueeze(0)
        )  # out shape is N x C x H x W
        # todo: unsure if this will do exactly what i want, need to test
        # print(v_dwt_full.shape, v_dwt_lows.shape, [i.shape for i in v_dwt_highs])
        v_dwt_permed = v_dwt_full.squeeze().permute((1, 2, 0))  # H x W x C

        nx1, ny1, nz1 = v_dwt_permed.shape  # padded_shape

        v_dwt_permed = v_dwt_permed.reshape(
            (v_dwt_permed.shape[0] * v_dwt_permed.shape[1], og_channels)
        )

        """
        okay so the dwt_forward does things differently than the matlab version of it. unless you
        want to rewrite all of the matlab one (nope) then you need to make this work without it.
        it worked for hyres/hyminor so it should work for this.
        
        there are some fishy things that are happening, but this should work at some point. its 
        likely that the issue in the the for loop below
        """

        start = int(3 * (nx1 / 2.) * (ny1 / 2.))  # todo: minus 1 for this and next line?
        stop = int(4 * (nx1 / 2.) * (ny1 / 2.))
        # idx_sigma = 3*(nx1/2)*(ny1/2)+1:4*(nx1/2)*(ny1/2);
        # print(start, stop)
        idx_sigma = torch.tensor(list(range(start, stop)), device=x.device)
        # todo: check for correctness !!!!

        eps = 1e-30
        # sigma = (median(abs(WY(idx_sigma,:)))/0.6745)'+eps;
        # print(v_dwt_permed[idx_sigma].shape)
        # print(v_dwt_permed[:6, :6])
        # todo: what does idx_sigma do?
        # sigma = torch.median(torch.abs(v_dwt_permed[idx_sigma]), dim=0)[0] / 0.6745
        # sigma = (sigma.T + eps).unsqueeze(1)  # todo: is this needed?
        sigma = (torch.sqrt(torch.var(v_dwt_permed[idx_sigma], dim=0)) + eps) ** 2

        # % Covariance matrix
        omega = sigma ** 2
        print(omega.shape, np.cov(v_dwt_permed.numpy()).shape)
        # Omega_1=permute(sigma(:).^2,[3,2,1]);
        # Omega=repmat(Omega_1,[nx1,ny1,1]);
        omega = omega.reshape((1, 1, omega.numel())).repeat(og_rows, og_cols, 1)

        # % D^T*Y is fixed through the derivation it is better to be calculated out
        # % of the loop
        # [WY_tilda,s1,s2]=twoDWTon3Ddata(Omega.^-.5.*Y,L,qmf,'FWT_PO_1D_2D_3D_fast');
        # print(omega.shape, x.shape)
        inp = torch.sqrt(omega) * x
        wy_tilda_full, wy_tilda_lows, wy_tilda_highs = self.dwt_forward(
            inp.permute((2, 0, 1)).unsqueeze(0)
        )
        wy_tilda_permed = wy_tilda_full.squeeze().permute((1, 2, 0))
        wy_tilda_permed = wy_tilda_permed.reshape(
            (wy_tilda_permed.shape[0] * wy_tilda_permed.shape[1], og_channels)
        )
        #
        # [V,PC]=PCA_image(Omega.^-.5.*Y);
        v, pc = utils.custom_pca_image(inp)
        # %%%%%%%%%%%%%%%%%%%%%%%%%
        # V=V(:,1:r);
        v = v[:, :r]  # +1 or not?
        # thresh=zeros(r,L+1);
        thresh = torch.zeros((r, self.decomp_level + 1), dtype=x.dtype, device=x.device)
        # WX=zeros(size(WY_tilda*V));
        wx = torch.zeros((wy_tilda_permed.shape[0], v.shape[1]), dtype=x.dtype, device=x.device)
        # for cc=1:200 % while epsilon>=n || k<3
        lvl = self.decomp_level
        nx1 = wy_tilda_lows.shape[0]
        ny1 = wy_tilda_lows.shape[1]
        stop = int((nx1 / 2. ** lvl) * (ny1 / 2. ** lvl))
        # print(stop)
        for cc in range(3):  # should be 200
            # W=WY_tilda*V;
            w = wy_tilda_permed @ v  # todo: might need to slice here
            # for i=1:r
            for i in range(r):
                index = (i % r, i // r)
                if cc == 0:  # todo: move outside both loops?
                    _, thresh[index], _, _ = utils.sure_soft_modified_lr2(w[:stop, i])
                    print(cc, i, thresh[index])
                # if thresh(i)==0 %
                # in matlab this is the global column index for some reason.... need to convert it
                # print(thresh[index])
                if thresh[index] == 0:
                    # WX(1:(nx1/2^L)*(ny1/2^L),i)= W(1:(nx1/2^L)*(ny1/2^L),i);
                    wx[:stop, i] = w[:stop, i]
                # else
                else:
                    # WX(1:(nx1/2^L)*(ny1/2^L),i)= soft(W(1:(nx1/2^L)*(ny1/2^L),i),thresh(i,1));
                    wx[:stop, i] = utils.soft_threshold(w[:stop, i], thresh[i, 0])
                # for j=1:L
                for j in range(self.decomp_level):
                    st = int((nx1 / 2. ** (lvl - j + 1)) * (ny1 / 2. ** (lvl - j + 1.)) + 1.)
                    sp = int(4 * (nx1 / 2. ** (lvl - j + 1)) * (ny1 / 2. ** (lvl - j + 1)))
                    idx = list(range(st, sp))
                    # idx=(nx1/2^(lvl-j+1))*(ny1/2^(lvl-j+1))+1:4*(nx1/2^(lvl-j+1))*(ny1/2^(lvl-j+1));
                    # if cc==1
                    if cc == 0:
                        _, thresh[i, j + 1], _, _ = utils.sure_soft_modified_lr2(w[idx, i])
                        # [sure(i,:),thresh(i,j+1),t1,Min_sure(i)] = SUREsoft_modified_LR2(W(idx,i));
                    # if thresh(i,j+1)==0 % debuging for happening NAN
                    if thresh[i, j + 1] == 0:
                        # WX(idx,i)=W(idx,i);
                        wx[idx, i] = w[idx, i]
                    else:
                        # WX(idx,i) = soft(W(idx,i),thresh(i,j+1));
                        wx[idx, i] = utils.soft_threshold(w[idx, i], thresh[i, j + 1])
            # M=WX'*WY_tilda;
            m = wx.T @ wy_tilda_permed
            # [C,S2,G] = svd(M,'econ');
            c, s2, g = torch.linalg.svd(m, full_matrices=False)  # float64??
            # V=(C*G')';
            v = torch.conj(c @ g).T

        # [D_W_VT]=ItwoDWTon3Ddata(WX*V',s1,s2,lvl,qmf,'IWT_PO_1D_2D_3D_fast');
        # need to make the other stuff in the proper way
        # print(torch.nonzero(wx[:, 0] == 0.00).shape, wx.shape)
        dwt_inv_in = (wx @ v.T)

        dim01 = int(dwt_inv_in.shape[0] ** 0.5)
        colors = int(dwt_inv_in.shape[1])
        dwt_inv_in = dwt_inv_in.reshape((dim01, dim01, colors))

        # something be fucked with the shapes here
        first_highs = [s[:, :colors] for s in v_dwt_highs]
        # print([f.shape for f in first_highs], dwt_inv_in.shape)
        # print("shapes", dwt_inv_in.permute((2, 0, 1)).unsqueeze(0).shape)
        print(dwt_inv_in[:10, :10, 0])
        d_w_vt = self.dwt_inverse(
            (dwt_inv_in[:23, :23].permute((2, 0, 1)).unsqueeze(0), first_highs)
            #, coeff_dims=coeff_dims[::-1]
        )

        # todo: order of the two wy/v dwt stuff
        d_w_vt = d_w_vt.squeeze().permute((1, 2, 0))
        # X =Omega.^.5.*D_W_VT;
        xx = torch.sqrt(omega) * d_w_vt
        # PCs=ItwoDWTon3Ddata(WX,s1,s2,lvl,qmf,'IWT_PO_1D_2D_3D_fast');
        wx_dwt = wx.reshape((dim01, dim01, int(wx.shape[1])))
        inv_highs = [asdf[:, :int(wx.shape[1])] for asdf in wy_tilda_highs]
        print(wx_dwt.shape)
        pcs = self.dwt_inverse((wx_dwt[:23, :23].permute((2, 0, 1)).unsqueeze(0), inv_highs))
        #, coeff_dims=coeff_dims[::-1])
        # X=X(1:nr,1:nc,1:p);
        xx = xx[:og_rows, :og_cols, :og_channels]
        # PCs=PCs(1:nr,1:nc,:);
        pcs = pcs.squeeze().permute((1, 2, 0))[:og_rows, :og_cols]
        return xx, pcs

        # ================================ fin ==========================================

        # need to have the dims be (num images (1), C_in, H_in, W_in) for twave ops
        # current order: rows, columns, bands (H, W, C) -> permute tuple (2, 0, 1)

        # current shape: h x w X c
        # current shape: w x h X c -> unclear why this needs to be this way...

        # return y_restored[:og_rows, :og_cols, :og_channels]


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import scipy.io as sio

    input = sio.loadmat("/path/git/Codes_4_HyMiNoR/HyRes/Indian.mat")
    imp = input["Indian"].reshape(input["Indian"].shape, order="C")

    t0 = time.perf_counter()
    input_tens = torch.tensor(imp, dtype=torch.float32)
    hyres = WSRRR()
    output = hyres(input_tens, 5)
    print(time.perf_counter() - t0)

    s = torch.sum(input_tens ** 2.0)
    d = torch.sum((input_tens - output) ** 2.0)
    snr = 10 * torch.log10(s / d)
    print(snr)

    imgplot = plt.imshow(output.numpy()[:, :, 0], cmap="gray")  # , vmin=50., vmax=120.)
    plt.show()
