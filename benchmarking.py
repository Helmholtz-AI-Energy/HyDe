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
    GaussianSNRLevel,
)

logger = logging.get_logger()

import gc
import time

import pandas as pd

# function to generate the noisy images


def generate_noisy_images(base_image, save_loc, noise_type="gaussian"):
    base_image = Path(base_image)
    save_loc = Path(save_loc)
    if noise_type == "gaussian":
        noise_levels = (20, 30, 40)
        transform = GaussianSNRLevel
        kwargs = {}
        # kwargs = {"scale_factor": 255.0 / 65517.0}  # houston dataset is int16
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

    for nl in noise_levels:
        sv_folder = save_loc / str(nl)
        sv_folder.mkdir(exist_ok=True)
        tfm = transform(nl, **kwargs)
        print(sv_folder)
        for i in range(15):
            print(nl, i)
            lp_img = tfm(imp_torch)
            mdic = {"houston": lp_img.numpy(), "label": nl}
            sio.savemat(sv_folder / f"{i}.mat", mdic)


gaussian_noise_removers_args = {
    "WSRRR": {"rank": 10},
    "HyMiNoR": {"lam": 10.0, "iterations": 50},
    "FORPDN_SURE": {
        "s_int_st": torch.tensor([0, 0, 0, 0]),
        "s_int_step": torch.tensor([0.001, 0.01, 0.1, 1]),
        "s_int_end": torch.tensor([0.01, 0.1, 1, 10]),
        "domain": "wavelet",
        "scale": True,
        "scale_const": 255,
        "wavelet_level": 6,
    },
    "HyRes": {},
    "OTVCA": {"features": 6, "num_itt": 10, "lam": 0.01},
    "FastHyDe": {"noise_type": "additive", "iid": True, "k_subspace": 10, "normalize": True},
    "L1HyMixDe": {
        "k_subspace": 8,
        "p": 0.1,
        "max_iter": 10,  # 40
        "normalize": True,
    },
    "nn": {"band_dim": -1, "normalize": True, "permute": True},
}

nn_noise_removers = {
    "qrnn3d": "pretrained-models/qrnn3d/new-noise-gaussian-bs4x4-l2.pth",  # hyde-bs16-blindinc-gaussian-qrnn3d-l2.pth",
    "qrnn2d": "pretrained-models/qrnn2d/new-noise-gaussian-bs4x4-l2.pth",  # hyde-bs16-blindinc-gaussian-qrnn2d-l2.pth",
    "memnet": "pretrained-models/memnet/new-noise-gaussian-bs4x4-l2.pth",  # hyde-bs16-blindinc-gaussian-memnet-l2.pth",
    "memnet3d": "pretrained-models/memnet3d/new-noise-gaussian-bs4x4-l2.pth",  # hyde-bs16-blindinc-gaussian-memnet3d-l2",
    "denet": "pretrained-models/denet/new-noise-gaussian-bs4x4-l2.pth",  # hyde-bs16-blindinc-gaussian-denet-l2.pth",
    "denet3d": "pretrained-models/denet3d/new-noise-gaussian-bs4x4-l2.pth",  # hyde-bs16-blindinc-gaussian-denet3d-l2.pth",
    "memnet_hyres": "pretrained-models/memnet_hyres/new-noise-gaussian-bs4x4-l2.pth",
    # =======
    #     "qrnn3d": "pretrained-models/qrnn3d/hyde-bs16-blindinc-gaussian-qrnn3d-l2.pth",
    #     "qrnn2d": "pretrained-models/qrnn2d/hyde-bs16-blindinc-gaussian-qrnn2d-l2.pth",
    #     "memnet": "pretrained-models/memnet/hyde-bs16-blindinc-gaussian-memnet-l2.pth",
    #     "memnet3d": "pretrained-models/memnet3d/hyde-bs16-blindinc-gaussian-memnet3d-l2.pth",
    #     "denet": "pretrained-models/denet/hyde-bs16-blindinc-gaussian-denet-l2.pth",
    #     "denet3d": "pretrained-models/denet3d/hyde-bs16-blindinc-gaussian-denet3d-l2.pth",
    "is2d": ["denet", "memnet", "memnet_hyres"],
}

# out_df -> cols = [method, 20dB, 30dB, 40dB]


def benchmark(file_loc, method, device, output, original):
    # TODO: get the method, noise_level, and device from command line args
    # method_results = {20: [], 30: [], 40: []}
    if method not in nn_noise_removers:
        nn = False
        method_call = getattr(hyde, method)()
    else:
        # is2d = method in nn_noise_removers["2d-models"]
        is2d = method in nn_noise_removers["is2d"]
        method_call = hyde.NNInference(
            arch=method,
            pretrained_file=nn_noise_removers[method],
            band_window=10,
            window_shape=512,
        )
        # is2d = method in nn_noise_removers["is2d"]

        # if method in ["qrnn3d", "qrnn2d", "memnet3d", "denet3d"]:
        #     method_call = hyde.NNInference(
        #         arch=method, pretrained_file=nn_noise_removers[method], band_window=5,
        #     )
        # else:
        #     method_call = hyde.NNInference(arch=method, pretrained_file=nn_noise_removers[method])
        nn = True
    # TODO: load and update the pandas dict with the results
    output = Path(output)
    og = sio.loadmat(original)["houston"]
    og = og.reshape(og.shape)

    original_im = torch.from_numpy(og.astype(np.float32)).to(device=device)
    out_df = None

    for noise in [20, 30, 40]:
        # todo: see if the file exists
        working_dir = Path(file_loc) / str(noise)
        psnrs, sads, times, mems, snrs = [], [], [], [], []
        for c, fil in enumerate(working_dir.iterdir()):  # data loading and method for each file
            torch.cuda.reset_peak_memory_stats()
            print(c, fil)
            # 1. load data + convert to torch
            dat_i = sio.loadmat(fil)
            dat_i = dat_i["houston"]
            dat_i = torch.from_numpy(dat_i).to(device=device, dtype=torch.float).contiguous()
            # 2. start timer
            t0 = time.perf_counter()

            if nn:
                kwargs = gaussian_noise_removers_args["nn"]
                if is2d:
                    dat_i = dat_i.squeeze(1)
            else:
                kwargs = gaussian_noise_removers_args[method]

            if len(kwargs) > 0:
                # dat_i = dat_i if not is2d else dat_i[:, :, :31]#.contiguous()
                res = method_call(dat_i, **kwargs)
            else:
                # dat_i = dat_i if not is2d else dat_i[:, :, :31]#.contiguous()
                res = method_call(dat_i)

            t1 = time.perf_counter() - t0
            mems.append(torch.cuda.max_memory_allocated())
            if isinstance(res, tuple):
                res = res[0]

            psnr = hyde.peak_snr(res, original_im)
            snr, _ = hyde.snr(res, original_im)
            sam = hyde.sam(res, original_im).mean()
            times.append(t1)
            psnrs.append(psnr.item())
            sads.append(sam.item())
            snrs.append(snr.item())

            print(f"file: {fil} time: {t1}, psnr: {psnr}, sam: {sam}, snr: {snr}")

        times = np.array(times)
        psnrs = np.array(psnrs)
        sads = np.array(sads)
        snrs = np.array(snrs)
        mem = np.array(mems)

        tsorted = times.argsort()
        good_idxs = tsorted  # [1:-1]

        times = times[good_idxs]
        psnrs = psnrs[good_idxs]
        snrs = snrs[good_idxs]
        mem = mem[good_idxs]

        pd_dict = {
            "noise": noise,
            "method": method,
            "device": device,
            "psnr": psnrs.mean(),
            "snr": snrs.mean(),
            "sam": sads.mean(),
            "time": times.mean(),
            "memory": mem.mean(),
        }
        # save the results
        ret_df = pd.DataFrame(pd_dict, index=[0], columns=list(pd_dict.keys()))
        if out_df is None:
            out_df = pd.DataFrame(pd_dict, index=[0], columns=list(pd_dict.keys()))
        else:
            out_df = pd.concat([out_df, ret_df], ignore_index=True, axis=0)
        # print(out_df)

    # print(ret_df)
    noise_out = output / "python-benchmarks-new2.csv"
    if not noise_out.exists():
        out_df.to_csv(noise_out, index=False)
        print(out_df)
    else:
        # load the existing DF and append to the bottom of it
        existing = pd.read_csv(noise_out)
        new = pd.concat([existing, out_df], ignore_index=True, axis=0)
        new.to_csv(noise_out, index=False)
        print(new)


def load_n_calc_snr(original, directory, method="fasthyde"):

    for noise in [20, 30, 40]:
        # working_dir = Path(file_loc) / str(noise)
        psnrs, sads, snrs = [], [], []
        # for c, fil in enumerate(working_dir.iterdir()):  # data loading and method for each file
        for c in range(15):
            torch.cuda.reset_peak_memory_stats()
            # 1. load data + convert to torch
            mat = Path(directory) / f"{method}-denoised-nonorm-{noise}-{c}.mat"
            res = sio.loadmat(mat)["restored"]

            psnr = hyde.peak_snr(res, original)
            snr, _ = hyde.snr(res, original)
            sam = hyde.sam(res, original).mean()
            psnrs.append(psnr.item())
            sads.append(sam.item())
            snrs.append(snr.item())

        psnrs = np.array(psnrs)
        sads = np.array(sads)
        snrs = np.array(snrs)

        pd_dict = {
            "noise": noise,
            "method": method,
            "psnr": psnrs.mean(),
            "snr": snrs.mean(),
            "sam": sads.mean(),
        }
        # save the results
        ret_df = pd.DataFrame(pd_dict, index=[0], columns=list(pd_dict.keys()))
        print(noise)
        print(ret_df)


if __name__ == "__main__":
    # import os
    #
    # print(os.sched_getaffinity(0))
    # torch.set_num_threads(24)
    # print(torch.__config__.parallel_info())
    #
    # parser = argparse.ArgumentParser(description="HyDe Benchmarking")
    # cla = hyde.nn.parsers.benchmark_parser(parser)
    # print(cla)
    # logger.info(cla)
    #
    # pd.set_option("display.max_rows", 500)
    # pd.set_option("display.max_columns", 500)
    # pd.set_option("display.width", 1000)
    # # generate_noisy_images(base_image="/mnt/ssd/hyde/houston.mat", save_loc="/mnt/ssd/hyde/")
    #
    # # generate_noisy_images(base_image=cla.original_image, save_loc=cla.data_dir)
    # benchmark(
    #     file_loc=cla.data_dir,
    #     method=cla.method,
    #     device=cla.device,
    #     output=cla.output_dir,
    #     original=cla.original_image,
    # )

    # # for method in gaussian_noise_removers_args:
    # for method in nn_noise_removers:
    #     print(method)
    #     benchmark(
    #         "/mnt/ssd/hyde/",
    #         method=method,
    #         device="cuda",
    #         output="/mnt/ssd/hyde/",
    #         original="/mnt/ssd/hyde/houston.mat",
    #     )

    methods = {
        # "hyres": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/hyminor",
        # "hyminor": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/hyminor",
        # "wsrrr": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/hyminor",
        # "otvca": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/otvca",
        # "fosrpdn": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/forpdn",
        "fasthyde": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/fasthyde/Demo_FastHyDe_FastHyIn/",
        "l1hymixde": "/hkfs/work/workspace/scratch/qv2382-hyde2/matlab-stuff/lyhymixde/L1HyMixDe/",
    }
    og = sio.loadmat("/hkfs/work/workspace/scratch/qv2382-hyde2/benchmark-data/houston.mat")[
        "houston"
    ]
    og = og.reshape(og.shape)

    original_im = torch.from_numpy(og.astype(np.float32))

    for method in methods:
        load_n_calc_snr(original=None, directory=methods[method], method=method)
#     # Convert matlab results
#
#     times = [    1.8486,    1.8469,    1.8500,    1.8481,    1.8501,    1.8473,    1.8482,
#     1.8479,    1.8484,    1.8465,    1.8511,    1.8500   , 1.8487,    1.8475,
#     1.8498
# ]8,   30.8920,   30.8682,   30.8568,   30.8483,
#
#     psnrs = [   30.8542,   30.8584,   30.850   30.8527,   30.8592,   30.8532   ,30.8574,   30.8446   ,30.8535,   30.8554,
#    30.8670
# ]
#     sads = [    0.2276    ,0.2274,    0.2273,    0.2271,    0.2283,    0.2274,    0.2277,
#     0.2279,    0.2284    ,0.2283  ,  0.2278,    0.2271  ,  0.2284,    0.2275,
#     0.2275
# ]
#
#     times = np.array(times) * 10**3
#     psnrs = np.array(psnrs)
#     sads = np.array(sads)
#     # mem = np.array(mems)
#
#     tsorted = times.argsort()
#     good_idxs = tsorted  # [1:-1]
#
#     times = times[good_idxs]
#     psnrs = psnrs[good_idxs]
#     sads = sads[good_idxs]
#     print("time\tpsnr\tsad")
#     print(f"{times.mean()}\t{psnrs.mean()}\t{sads.mean()}")
