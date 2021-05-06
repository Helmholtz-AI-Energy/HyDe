import scipy.io as sio
import torch
import utils
from hyres import HyRes


class HyMiNoR:
    def __init__(self):
        self.hyres = HyRes()

    def forward(self, x: torch.Tensor, lam: int = 10):
        # H_M -> x ; lambda -> lam
        mu1, mu2, its = 0.5, 0.5, 50
        print(x.shape)
        hyres_result = self.hyres.forward(x)
        # H=HyRes(H_M);
        m, n, d = hyres_result.shape
        mn = m * n
        # [m,n,d]=size(H);
        hyres_reshaped_t = torch.conj(hyres_result.reshape((mn, d))).T
        # Y=reshape(H,mn,d)';
        l1 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        l2 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        v1 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        v2 = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        xx = torch.zeros((d, mn), device=x.device, dtype=x.dtype)
        # X=L1;
        eye_d = torch.eye(d, device=x.device, dtype=x.dtype)
        hold = utils.vertical_difference_transpose(utils.vertical_difference(eye_d))
        u, s, v = torch.linalg.svd(hold.to(torch.float64), full_matrices=False)
        u = u.to(torch.float32)
        s = s.to(torch.float32)
        v = torch.conj(v.T).to(torch.float32)

        magic = torch.chain_matmul(
            v,
            torch.diag(1.0 / (mu1 + mu2 * s)),
            u.T,
        ).to(torch.float32)
        for _ in range(its):
            # subminimization problems
            hold1 = -mu1 * (v1 - hyres_reshaped_t - l1)
            hold2 = mu2 * utils.vertical_difference_transpose((v2 - l2))
            xx = magic @ (hold1 + hold2)
            # X=Majic*(-mu1*(V1-Y-L1)+mu2*Dvt(V2-L2));
            # % V-Step
            v1 = utils.soft_threshold(hyres_reshaped_t - xx + l1, 1.0 / mu1)
            # V1=soft_threshold(Y-X+L1,1/mu1);
            dv_xx = utils.vertical_difference(xx)
            v2 = utils.soft_threshold(dv_xx + l2, lam / mu2)
            # V2=soft_threshold(Dv(X)+L2,lambda/mu2);
            # Updating L1 and L2
            l1 = l1 + hyres_reshaped_t - xx - v1
            # L1=L1+Y-X-V1;
            l2 = l2 + dv_xx - v2
            # L2=L2+Dv(X)-V2;
        # H_DN=reshape(X',m,n,d);
        ret = torch.reshape(torch.conj(xx).T, (m, n, d))
        return ret


if __name__ == "__main__":
    import time

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
