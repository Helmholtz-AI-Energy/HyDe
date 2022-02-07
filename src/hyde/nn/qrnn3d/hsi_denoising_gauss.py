import argparse
import os
import time
from functools import partial
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from kornia.losses import SSIMLoss
from torch._six import string_classes
from torch.utils.data import DataLoader
from utils import train_argparse

from hyde.lowlevel import logging
from hyde.nn import comm
from hyde.nn.general_nn_utils import (
    AddGaussianNoise,
    AddGaussianNoiseBlind,
    MultipleWeightedLosses,
)
from hyde.nn.qrnn3d import dataset_utils as ds_utils
from hyde.nn.qrnn3d import helper, models, training_utils

logger = logging.get_logger()


def main():
    """Training settings"""
    parser = argparse.ArgumentParser(
        description="QRNN3D Hyperspectral Image Denoising (Gaussian Noise)"
    )
    # cla == command line arguments
    cla = train_argparse(parser)
    logger.info(cla)

    # self.prefix = cla.prefix
    # parser.add_argument("--prefix", "-p", type=str, default="denoise", help="prefix")
    prefix = cla.arch

    # Engine.setup(self):
    basedir = join("checkpoints", cla.arch)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if cla.no_cuda:
        cuda = False
    else:
        cuda = torch.cuda.is_available()
    logger.debug("Cuda Acess: %d" % cuda)

    torch.manual_seed(cla.seed)
    np.random.seed(cla.seed)
    if cuda:
        torch.cuda.manual_seed(cla.seed)

    """Model"""
    logger.info(f"=> creating model: {cla.arch}")
    net = models.__dict__[cla.arch]()
    # initialize parameters
    # init params will set the model params with a random distribution
    # helper.init_params(net, init_type=cla.init)  # disable for default initialization
    distributed = False
    bandwise = net.bandwise
    # world_size = 1
    if torch.cuda.device_count() > 1 and cuda:
        # print("cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
        logger.info("Spawning torch groups for DDP")
        group = comm.init(method=cla.comm_method)

        loc_rank = dist.get_rank() % torch.cuda.device_count()
        # world_size = dist.get_world_size()
        device = torch.device("cuda", loc_rank)
        logger.info(f"Default GPU: {device}")
        net = models.__dict__[cla.arch]()
        net = net.to(device)

        net = nn.SyncBatchNorm.convert_sync_batchnorm(net, group)

        net = nn.parallel.DistributedDataParallel(net, device_ids=[device.index])
        # net = nn.SyncBatchNorm.convert_sync_batchnorm(net, group)

        logger.info("Finished conversion to SyncBatchNorm")

    if cla.loss == "l2":
        criterion = nn.MSELoss()
    elif cla.loss == "l1":
        criterion = nn.L1Loss()
    elif cla.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    elif cla.loss == "ssim":
        criterion = SSIMLoss(window_size=11, max_val=1)
    elif cla.loss == "l2_ssim":
        criterion = MultipleWeightedLosses(
            [nn.MSELoss(), SSIMLoss(window_size=11, max_val=1.0)], weight=[1, 2.5e-3]
        )

    logger.info(criterion)

    if cuda:
        net.cuda()
        criterion = criterion.cuda()

    writer = None
    if cla.tensorboard:
        writer = helper.get_summary_writer(os.path.join(cla.basedir, "logs"), cla.cla.prefix)

    """Optimization Setup"""
    optimizer = optim.Adam(net.parameters(), lr=cla.lr, weight_decay=cla.wd, amsgrad=False)

    # """Resume previous model"""
    if cla.resume:
        # Load checkpoint.
        torch.load(cla.resume_path, not cla.no_ropt)
    else:
        logger.info("==> Building model..")
        logger.debug(net)

    # net = net.to(torch.float16)

    cudnn.benchmark = True

    # train_transform_1 = AddGaussianNoise(34)
    common_transform_2 = transforms.RandomCrop((256, 256))
    # train_transform_2 = AddGaussianNoiseBlind(max_sigma_db=40, min_sigma_db=20)
    # harder_train_transform = AddGaussianNoiseBlind(max_sigma_db=40, min_sigma_db=20)

    # set_icvl_64_31_TL_1 = ds_utils.ICVLDataset(
    #     cla.datadir,
    #     transform=AddGaussianNoise(40), # train_transform_2,
    # )
    # worker_init_fn is in dataset -> just getting the seed
    # print(cla.batch_size * world_size)
    # icvl_64_31_TL_1 = DataLoader(
    #     set_icvl_64_31_TL_1,
    #     batch_size=cla.batch_size,
    #     shuffle=True,
    #     num_workers=cla.workers,
    #     pin_memory=torch.cuda.is_available(),
    #     persistent_workers=False,
    # )

    set_icvl_64_31_TL_2 = ds_utils.ICVLDataset(
        cla.datadir,
        common_transforms=common_transform_2,
        easy_transform=AddGaussianNoise(15),
        medium_transform=AddGaussianNoise(30),
        last_transform=AddGaussianNoise(40), #AddGaussianNoiseBlind(max_sigma_db=40, min_sigma_db=10),
    )
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(set_icvl_64_31_TL_2)
    else:
        train_sampler = None

    # worker_init_fn is in dataset -> just getting the seed
    icvl_64_31_TL_2 = DataLoader(
        set_icvl_64_31_TL_2,
        batch_size=cla.batch_size,
        shuffle=True,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
        # worker_init_fn=None,
        persistent_workers=False,
        sampler=train_sampler,
    )

    # -------------- validation loader -------------------------

    """Test-Dev"""
    basefolder = cla.val_datadir

    val_dataset = ds_utils.ICVLDataset(
        basefolder,
        easy_transform=AddGaussianNoiseBlind(
            max_sigma_db=40, min_sigma_db=10
        ),  # blind gaussain noise
        common_transforms=transforms.RandomCrop((256, 256)),
        val=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cla.batch_size,
        shuffle=False,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
    )

    helper.adjust_learning_rate(optimizer, cla.lr)
    # epoch_per_save = cla.save_freq
    max_epochs = 100
    best_val_loss, best_val_psnr = 100000, 0
    epochs_wo_best = 0

    for epoch in range(max_epochs):
        # torch.manual_seed(epoch)
        # torch.cuda.manual_seed(epoch)
        # np.random.seed(epoch)
        # TODO: change the transform to something harder at some point in the training?
        # if epoch == 10:
        #    icvl_64_31_TL_2.transform = harder_train_transform
        if epoch <= 30:
            set_icvl_64_31_TL_2.easy_transform = AddGaussianNoise(1 + epoch // 2)
            logger.info(f"New noise level: {1 + epoch // 2} dB")

        #if epoch == 5:
        #    set_icvl_64_31_TL_2.stage = 1
        if epoch == 30:
            set_icvl_64_31_TL_2.stage = 1

        if epoch == 50:
            set_icvl_64_31_TL_2.stage = 2


        if epoch == 30:
            helper.adjust_learning_rate(optimizer, cla.lr * 0.1)

        ttime = time.perf_counter()
        training_utils.train(
            icvl_64_31_TL_2, net, cla, epoch, optimizer, criterion, bandwise, writer=writer
        )
        ttime = time.perf_counter() - ttime

        vtime = time.perf_counter()
        psnr, ls = training_utils.validate(
            val_loader, "validate", net, cla, epoch, criterion, bandwise, writer=writer
        )
        vtime = time.perf_counter() - vtime

        expected_time_remaining = time.strftime(
            "%H:%M:%S", time.gmtime((ttime + vtime) * (max_epochs - epoch))
        )
        logger.info(f"Expected time remaing: {expected_time_remaining}")

        if epoch == 0:
            logger.info(f"Max mem alocated: {torch.cuda.max_memory_allocated(device=None)}")

        epochs_wo_best += 1

        helper.display_learning_rate(optimizer)
        # if (epoch % epoch_per_save == 0 and epoch > 0) or epoch == max_epochs - 1:
        if psnr > best_val_psnr:
            best_val_psnr = psnr
            epochs_wo_best = 0

        if ls < best_val_loss:
            best_val_loss = ls
            epochs_wo_best = 0

        if epochs_wo_best == 0 or epoch % 10 == 0:
            # best_val_psnr < psnr or best_val_psnr > ls:
            logger.info("Saving current network...")
            model_latest_path = os.path.join(cla.save_dir, prefix, "model_latest_gradual_noise_warmup.pth")
            training_utils.save_checkpoint(
                cla, epoch, net, optimizer, model_out_path=model_latest_path
            )

        # if epochs_wo_best == 5:
        #    logger.info(f"Breaking loop, not improving for 5 epochs, current epoch: {epoch}")
        # break


if __name__ == "__main__":
    main()
