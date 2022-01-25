import torch

from . import fast_hyde, utils

__all__ = ["L1HyMixDe"]


class L1HyMixDe(torch.nn.Module):
    def __init__(self):
        super(L1HyMixDe, self).__init__()

    def forward(self, img: torch.Tensor, k_subspace, p):
        # [row, col, band] = size(img);
        row, col, band = img.shape
        # N=row*col;
        n = row * col
        # Y_ori = reshape(img, N, band)';
        y_og = img.reshape((n, band)).T
        #
        # %% -------------Subspace Learning Against Mixed Noise---------------------
        # %An adaptive median filter is applied to noisy image to remove the bulk of
        # %impulse noise and stripes
        # for ib=1:band
        #     img_median(:,:,ib) = adpmedian(img(:,:,ib), 21);
        # end
        img_median = torch.zeros_like(img)
        for ib in range(band):
            img_median[:, :, ib] = utils.adaptive_median_filtering(img[:, :, ib], 21)
        # Y_median = reshape(img_median, N, band)';
        y_median = img_median.reshape((n, band)).T
        # %detect pixel indexes of impulse noise and stripes
        # img_dif =  abs(img-img_median) ;
        img_dif = torch.abs(img - img_median)
        #  mask_outlier =(img_dif>p);
        mask_outlier = img_dif > p
        #
        #  img_remove_outlier = img;
        # img_remove_outlier(mask_outlier) = img_median(mask_outlier);
        img_remove_outlier = img.clone()
        img_remove_outlier[mask_outlier] = img_median[mask_outlier]

        # Y_remove_outlier = reshape(img_remove_outlier, N, band)';
        y_remove_outlier = img_remove_outlier.reshape((n, band)).T
        #
        # [w Rw] = estNoise(Y_remove_outlier,'additive');
        x, r_w = utils.estimate_hyperspectral_noise(
            y_remove_outlier, "additive", calculation_dtype=torch.float64
        )
        # Rw_ori = Rw;
        # %data whitening so that noise variances of each band are same
        # Y_ori = inv(sqrt(Rw))*Y_ori;
        # img = reshape(Y_ori', row, col, band);
        hold = torch.linalg.inv(torch.sqrt(r_w))
        y_og = hold @ y_og
        # img = y_og.T.reshape((row, col, band))
        # Y_median = inv(sqrt(Rw))*Y_median;
        # img_median = reshape(Y_median', row, col, band);
        # Y_remove_outlier= inv(sqrt(Rw))*Y_remove_outlier;
        y_median = hold @ y_median
        # img = y_median.T.reshape((row, col, band))
        y_remove_outlier = hold @ y_remove_outlier
        #
        # %Subspace learning from the coarse image without stripes and impulse noise
        # [E,S,~] = svd(Y_remove_outlier*Y_remove_outlier'/N);
        # E = E(:,1:k_subspace);
        e, s, _ = torch.linalg.svd(y_remove_outlier @ y_remove_outlier.T / n)
        e = e[:, :k_subspace]
        # %% --------------------------L1HyMixDe-------------------------------------
        # %Initialization
        # Z = E'*Y_median;
        # img_dif =  img-img_median ;
        # V = reshape(img_dif, N, band)';
        # D = zeros(band,N);
        # mu = 1;
        z = e.T @ y_median
        img_dif = img = img_median
        v = img_dif.reshape((n, band)).T
        d = torch.zeros((band, n), dtype=img.dtype, device=img.device)
        # mu = 1
        zold = None
        for it in range(40):  # range limit??
            # %% Updating Z: Z_{k+1} = argmin_Z lambda*phi(Z) + mu/2 || Y-EZ-V_k-D_k||_F^2
            # %Equivlance: Z_{k+1} = argmin_Z lambda/mu*phi(Z) +  1/2 || Y-EZ-V_k-D_k||_F^2
            # Y_aux = Y_ori-V+D;
            # img_aux = reshape( Y_aux', row, col, band);
            y_aux = y_og - v + d
            # img_aux = y_aux.T.reshape((row, col, band))
            # %FastHyDe
            rw_fasthyde = torch.eye(band, dtype=img.dtype, device=img.device)
            # Rw_fasthyde =  eye(band); %Noise covariance matrix Rw_fasthyde is identity matrix because
            # %the image has been whitened.
            # Z = FastHyDe_fixEreturnZ(img_aux, E, Rw_fasthyde);
            # see below ----------
            # [Lines, Columns, B] = size(img_aux);
            # Y = reshape(img_ori, N, B)'; -> y_aux
            # k_subspace =  size(E,2);
            # eigen_Y = E'*Y;
            eigen_y = e.T @ y_aux
            # end of FastHyDe_fixEreturnZ ---------------
            # %% --------------------------Eigen-image denoising ------------------------------------
            z = fast_hyde.fast_hyde_eigen_image_denoising(
                img, k_subspace, rw_fasthyde, e, eigen_y, n
            )
            # %% Updating V: V_{k+1} = argmin_V ||V||_1 + mu/2 || Y-EZ_{k+1}-V-D_k||_F^2
            #
            # V_aux = Y_ori-E*Z+D;
            # par =   1;
            # V = sign(V_aux).*max(abs(V_aux)-par,0);
            yez = y_og - e @ z
            v_aux = yez + d
            par = 1
            hold = torch.abs(v_aux) - par
            # filter out negaive values matlap -> max(matrix, 0)
            hold[hold < 0] = 0
            v = torch.sign(v_aux) * hold
            # %% Updating D: D_{k+1} = D_k - (Y-EZ_{k+1}-V_{k+1})
            # D =  D+  ( Y_ori-E*Z-V );
            d += yez - v
            # if ite>1
            #     criterion(ite) =  norm(Z-Z_old,'fro')/norm(Z_old,'fro');
            #     if criterion(ite)<0.001
            #         break;
            #     end
            # end
            if it > 0:
                criterion = torch.norm(z - zold) / torch.norm(zold)
                if criterion < 0.001:
                    break
            zold = z.clone()
        # end
        #
        # figure; plot(criterion,'-o');
        # Y_denoised = E*Z;
        # Y_denoised = sqrt(Rw_ori)*Y_denoised;
        # img_denoised = reshape(Y_denoised', row, col, band);
        y_denoised = e @ z
        y_denoised = torch.sqrt(r_w) @ y_denoised
        return y_denoised.T.reshape((row, col, band))
