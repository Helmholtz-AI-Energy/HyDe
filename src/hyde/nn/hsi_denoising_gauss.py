import argparse
import os
import random
import time
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from kornia.losses import (
    DiceLoss,
    FocalLoss,
    PSNRLoss,
    SSIMLoss,
    TotalVariation,
    TverskyLoss,
)
from torch.utils.data import DataLoader

from hyde.lowlevel import logging
from hyde.nn import (
    MultipleWeightedLosses,
    SAMLoss,
    comm,
    helper,
    models,
    training_utils,
)
from hyde.nn.datasets import dataset_utils as ds_utils
from hyde.nn.datasets.transforms import GaussianBlindSNRLevel, GaussianSNRLevel
from hyde.nn.parsers import qrnn_parser

logger = logging.get_logger()


def main():
    """Training settings"""
    parser = argparse.ArgumentParser(description="Hyperspectral Image Denoising (Gaussian Noise)")
    # cla == command line arguments
    cla = qrnn_parser(parser)
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
    logger.info("Cuda Acess: %d" % cuda)

    torch.manual_seed(cla.seed)
    np.random.seed(cla.seed)

    """Model"""
    logger.info(f"=> creating model: {cla.arch}")
    net = models.__dict__[cla.arch]()
    # logger.info(net)
    # initialize parameters
    # init params will set the model params with a random distribution
    # helper.init_params(net, init_type=cla.init)  # disable for default initialization
    distributed = False
    # world_size = 1
    if torch.cuda.device_count() > 1 and cuda:
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
        logger.info("Finished conversion to SyncBatchNorm")
        cla.rank = comm.get_rank()
        distributed = True

        comm.set_logger_to_rank0(logger, cla.rank)

        torch.backends.cudnn.benchmark = True

    elif cuda:
        torch.cuda.manual_seed(cla.seed)
        device = torch.device("cuda", 0)
        loc_rank = 0
        cla.rank = 0
    else:
        device = torch.device("cpu")
        loc_rank = None
        cla.rank = 0

    cla.device = device

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
    elif cla.loss == "psnr":
        criterion = PSNRLoss(1)
    elif cla.loss == "tv":
        criterion = TotalVariation()
    elif cla.loss == "dice":
        criterion = DiceLoss()
    elif cla.loss == "l2_psnr":
        criterion = MultipleWeightedLosses(
            [nn.MSELoss(), PSNRLoss(1)],
            weight=[1, 1],  # SSIMLoss(window_size=11, max_val=1.0)], weight=[1, 2.5e-3]
        )
    elif cla.loss == "sam":
        criterion = SAMLoss()

    # elif cla.loss == "tversky":
    #     criterion = TverskyLoss
    # elif cla.loss == "focal":
    #     criterion = FocalLoss

    else:
        raise ValueError(
            f"Loss function must be one of: [l2, l1, smooth_l1, ssim, l2_ssim], currently: {cla.loss}"
        )

    logger.info(criterion)

    if cuda:
        net.cuda(loc_rank)
        criterion = criterion.cuda(loc_rank)

    writer = None
    if cla.tensorboard:
        writer = helper.get_summary_writer(os.path.join(cla.basedir, "logs"), cla.cla.prefix)

    """Optimization Setup"""

    optimizer = optim.Adam(net.parameters(), lr=cla.lr, weight_decay=cla.wd, amsgrad=False)
    # optimizer = optim.SGD(net.parameters(), cla.lr, momentum=0.9, weight_decay=1e-4)

    start_epoch = 0
    # """Resume previous model"""
    if cla.resume_path is not None:
        # Load checkpoint.
        logger.info(f"Resuming training from {cla.resume_path}")
        checkpoint = torch.load(cla.resume_path)
        if not cla.no_resume_opt:
            logger.info("Loading optimizer from checkpoint file")
            optimizer.load_state_dict(checkpoint["optimizer"])

        try:
            net.load_state_dict(checkpoint["net"])
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["net"].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
        if cla.resume_training:
            start_epoch = checkpoint["epoch"]

    else:
        logger.info("==> Building model..")
        helper.init_network(net, cla.nn_init_mode)
        logger.debug(net)

    cudnn.benchmark = True

    torch.manual_seed(cla.rank)
    torch.cuda.manual_seed(cla.rank)
    np.random.seed(cla.rank)
    random.seed(cla.rank)

    crop_size = (256, 256)
    band_norm = True

    train_icvl = ds_utils.ICVLDataset(
        cla.datadir,
        common_transforms=None,
        transform=GaussianSNRLevel(50),
        crop_size=crop_size,
        band_norm=band_norm,
    )
    # if distributed:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_icvl)
    # else:
    train_sampler = None

    # worker_init_fn is in dataset -> just getting the seed
    train_loader = DataLoader(
        train_icvl,
        batch_size=cla.batch_size,
        shuffle=True if train_sampler is None else False,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        sampler=train_sampler,
    )

    # -------------- validation loader -------------------------
    basefolder = cla.val_datadir

    val_dataset = ds_utils.ICVLDataset(
        basefolder,
        transform=GaussianSNRLevel(20),
        val=True,
        crop_size=crop_size,
        band_norm=band_norm,
    )
    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=cla.batch_size,
        shuffle=True if val_sampler is None else False,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
        sampler=val_sampler,
    )

    max_epochs = 50
    best_val_loss, best_val_psnr = 100000, 0
    epochs_wo_best = 0
    base_lr = 1e-3
    cla.lr = base_lr
    helper.adjust_learning_rate(optimizer, base_lr)

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"\t\t--------- Start epoch {epoch} of {max_epochs - 1} ---------\t")

        train_icvl.transform = GaussianBlindSNRLevel(max_sigma_db=30, min_sigma_db=20)
        logger.info("SNR level: 30-20 blind")

        if epoch == 40:  # 120:
            helper.adjust_learning_rate(optimizer, cla.lr * 0.1)
        if epoch == 45:  # 140:
            helper.adjust_learning_rate(optimizer, cla.lr * 0.01)

        helper.display_learning_rate(optimizer)
        ttime = time.perf_counter()
        training_utils.train(
            train_loader,
            net,
            cla,
            epoch,
            optimizer,
            criterion,
            writer=writer,
            iterations=250,
        )
        ttime = time.perf_counter() - ttime

        vtime = time.perf_counter()
        psnr, ls = training_utils.validate(
            val_loader, "validate", net, cla, epoch, criterion, writer=writer
        )
        vtime = time.perf_counter() - vtime

        expected_time_remaining = time.strftime(
            "%H:%M:%S", time.gmtime((ttime + vtime) * (max_epochs - epoch))
        )
        logger.info(f"Expected time remaing: {expected_time_remaining}")

        if epoch == 0:
            logger.info(f"Max mem alocated: {torch.cuda.max_memory_allocated(device=None)}")

        epochs_wo_best += 1

        if psnr > best_val_psnr:
            best_val_psnr = psnr
            epochs_wo_best = 0

        if ls < best_val_loss:
            best_val_loss = ls
            epochs_wo_best = 0

        if epochs_wo_best == 0 or (epoch + 1) % 10 == 0:
            # best_val_psnr < psnr or best_val_psnr > ls:
            logger.info("Saving current network...")
            model_latest_path = os.path.join(
                cla.save_dir, prefix, f"new-noise-gaussian-bs4x4-{cla.loss}.pth"
            )
            training_utils.save_checkpoint(
                cla, epoch, net, optimizer, model_out_path=model_latest_path
            )


if __name__ == "__main__":
    main()
