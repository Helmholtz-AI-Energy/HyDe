import argparse
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

import hyde
from hyde import logging

# from torchvision.transforms import Compose
from hyde.nn.datasets.transforms import (  # AddNoiseDeadline,; AddNoiseImpulse,; AddNoiseNonIIDdB,; AddNoiseStripe,; RandChoice,
    AddGaussianNoise,
)

logger = logging.get_logger()

import time

import pandas as pd

# function to generate the noisy images


def generate_noisy_images(base_image, save_loc, noise_type="gaussian"):
    base_image = Path(base_image)
    save_loc = Path(save_loc)
    if noise_type == "gaussian":
        noise_levels = (20, 30, 40)
        transform = AddGaussianNoise
        kwargs = {"scale_factor": 1.0}
    else:
        raise NotImplementedError("implement mixed/complex noise cases")

    input = sio.loadmat(base_image)
    imp_clean = input["houston"].reshape(input["houston"].shape, order="C").astype(np.float32)
    # print(imp_clean.max())
    # print(imp_clean.reshape((imp_clean.shape[0]*imp_clean.shape[1], imp_clean.shape[2])).max(0))
    imp_torch = torch.from_numpy(imp_clean)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(imp_clean.dtype)

    for nl in noise_levels:
        sv_folder = save_loc / str(nl)
        sv_folder.mkdir(exist_ok=True)
        tfm = transform(nl, **kwargs)
        for i in range(15):
            print(nl, i)
            lp_img = tfm(imp_torch)
            mdic = {"houston": lp_img.numpy(), "label": nl}
            sio.savemat(sv_folder / f"{i}.mat", mdic)


max_houston = 65517.0

gaussian_noise_removers_args = {
    "FORPDN_SURE": {
        "s_int_st": torch.tensor([0, 0, 0, 0]),
        "s_int_step": torch.tensor([0.001, 0.01, 0.1, 1]),
        "s_int_end": torch.tensor([0.01, 0.1, 1, 10]),
        "domain": "wavelet",
        "scale": True,
        "scale_const": max_houston,
        "wavelet_level": 6,
    },
    "HyMiNoR": {"lam": 10.0, "iterations": 50},
    "HyRes": {},
    "OTVCA": {"features": 6, "num_itt": 10, "lam": 0.01},
    "WSRRR": {"rank": 10},
    "FastHyDe": {"noise_type": "additive", "iid": True, "k_subspace": 10, "normalize": True},
    "L1HyMixDe": {
        "k_subspace": 10,
        "p": 0.05,
        "max_iter": 10,
        "normalize": True,
    },
    "nn": {"band_dim": -1, "normalize": True, "permute": True},
}

nn_noise_removers = {
    "qrnn3d": "pretrained-models/qrnn3d/hyde-qrnn3d-gauss-l2.pth",
    "qrnn2d": "pretrained-models/qrnn2d/qrnn2d_gauss-l2.pth",
    "memnet": "pretrained-models/memnet/memnet-2d_gauss-l2.pth",
    "memnet3d": "pretrained-models/memnet3d/memnet3d_gauss-l2.pth",
    "denet": "pretrained-models/denet/gauss_denet-2d-l2.pth",
    "denet3d": "pretrained-models/denet3d/denet3d_gauss_3d-l2.pth",
    "2d-models": ["DeNet", "MemNet"],
}

# out_df -> cols = [method, 20dB, 30dB, 40dB]


def benchmark(file_loc, method, device, output, original):
    # TODO: get the method, noise_level, and device from command line args
    # method_results = {20: [], 30: [], 40: []}
    if method not in nn_noise_removers:
        nn = False
        method_call = getattr(hyde, method)()
    else:
        method_call = hyde.NNInference(arch=method, pretrained_file=nn_noise_removers[method])
        nn = True
    # TODO: load and update the pandas dict with the results
    # out = Path(output)
    og = sio.loadmat(original)["houston"]
    og = og.reshape(og.shape, order="F")
    # print(og)
    original_im = torch.from_numpy(og.astype(np.float32)).to(device=device)
    out_df = pd.DataFrame(columns=["noise", "method", "device", "psnr", "sam", "time"])
    print(out_df)
    # print(original_im.mean(-1))
    for noise in [20, 30, 40]:
        # todo: see if the file exists
        working_dir = Path(file_loc) / str(noise)
        psnrs, sads, times = [], [], []
        for c, fil in enumerate(working_dir.iterdir()):  # data loading and method for each file
            # 1. load data + convert to torch
            dat_i = sio.loadmat(fil)
            dat_i = dat_i["houston"]
            dat_i = torch.from_numpy(dat_i).to(device=device, dtype=torch.float)
            # 2. start timer
            t0 = time.perf_counter()
            if nn:
                kwargs = gaussian_noise_removers_args["nn"]
                is2d = method in nn_noise_removers["2d-models"]
            else:
                kwargs = gaussian_noise_removers_args[method]
                is2d = False
            if len(kwargs) > 0:
                dat_i = dat_i if not is2d else dat_i[:, :, :31]
                res = method_call(dat_i, **kwargs)
            else:
                res = method_call(dat_i)
            t1 = time.perf_counter() - t0
            psnr = hyde.peak_snr(res, original_im)
            # print((original_im).mean(-1))
            sam = hyde.sam(res, original_im).mean()
            times.append(t1)
            psnrs.append(psnr.item())
            sads.append(sam.item())
            print(f"file: {fil} time: {t1}, psnr: {psnr}, sam: {sam}")
            # break

        times = np.array(times)
        psnrs = np.array(psnrs)
        sads = np.array(sads)

        tsorted = times.argsort()
        good_idxs = tsorted[1:-1]

        times = times[good_idxs]
        psnrs = psnrs[good_idxs]
        sads = sads[good_idxs]

        # ret =

        pd_dict = {
            "noise": noise,
            "method": method,
            "device": device,
            "psnr": psnrs.mean(),
            "sam": sads.mean(),
            "time": times.mean(),
        }

        # save the results
        ret_df = pd.Series(pd_dict)
        out_df = out_df.append(ret_df, ignore_index=True)
        # print(ret_df)

    print(ret_df)
    noise_out = output / "python-benchmarks.csv"
    if not noise_out.exists():
        ret_df.to_csv(noise_out)

    else:
        # load the existing DF and append to the bottom of it
        existing = pd.read_csv(noise_out)
        new = existing.append(ret_df, ignore_index=True)
        new.to_csv(noise_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyDe Benchmarking")
    cla = hyde.nn.parsers.benchmark_parser(parser)
    logger.info(cla)
    # generate_noisy_images(base_image="/mnt/ssd/hyde/houston.mat", save_loc="/mnt/ssd/hyde/")
    benchmark(
        file_loc=cla.data_dir,
        method=cla.method,
        device=cla.device,
        output=cla.output_dir,
        original=cla.original_image,
    )
    # "/mnt/ssd/hyde/", method="HyRes", device="cuda", output=None, original="/mnt/ssd/hyde/houston.mat")