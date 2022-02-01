import argparse
import os
from functools import partial
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

# import torchvision
import torchvision.transforms as transforms
from kornia.losses import SSIMLoss
from torch.utils.data import DataLoader
from utils import train_argparse

from ...lowlevel import logging
from .. import comm
from ..general_nn_utils import (
    AddNoiseDeadline,
    AddNoiseImpulse,
    AddNoiseNonIIDdB,
    AddNoiseStripe,
    MultipleWeightedLosses,
)
from . import dataset_utils as ds_utils
from . import helper, lmdb_dataset, models, training_utils

logger = logging.get_logger()


def main():
    """Training settings"""
    parser = argparse.ArgumentParser(
        description="QRNN3D Hyperspectral Image Denoising (Gaussian Noise)"
    )
    # cla == command line arguments
    cla = train_argparse(parser)
    logger.debug(cla)

    # self.prefix = cla.prefix
    # parser.add_argument("--prefix", "-p", type=str, default="denoise", help="prefix")
    prefix = cla.prefix

    # Engine.setup(self):
    basedir = join("checkpoints", cla.arch)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if torch.cuda.is_available:
        cuda = True
    device = "cuda" if cuda else "cpu"
    # todo: if there are multiple devices, then launch torch distributed
    #       get from MLPerf
    logger.debug("Cuda Acess: %d" % cuda)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(cla.seed)
    if cuda:
        torch.cuda.manual_seed(cla.seed)

    """Model"""
    logger.info(f"=> creating model: {cla.arch}")
    net = models.__dict__[cla.arch]()
    # initialize parameters
    # init params will set the model params with a random distribution
    helper.init_params(net, init_type=cla.init)  # disable for default initialization

    if torch.cuda.device_count() > 1:
        group = comm.init(method="nccl-mpi")
        # net = nn.parallel.DistributedDataParallel(net, device_ids=cla.gpu_ids)
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net, group)

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
            [nn.MSELoss(), SSIMLoss(window_size=11, max_val=1)], weight=[1, 2.5e-3]
        )

    logger.info(criterion)

    if cuda:
        net.to(device)
        criterion = criterion.to(device)

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
        logger.info(net)

    cudnn.benchmark = True

    tr_ds = lmdb_dataset.LMDBDataset(cla.dataroot, repeat=64)

    HSI2Tensor = partial(ds_utils.HsiToTensor, use_2dconv=net.use_2dconv)
    target_transform = [
        HSI2Tensor(),
    ]
    train_transform = [
        AddNoiseNonIIDdB(),
        transforms.RandomApply([AddNoiseImpulse(), AddNoiseStripe(), AddNoiseDeadline()], p=0.25),
        HSI2Tensor(),
    ]
    common_transforms = [
        transforms.RandomCrop((32, 32)),
    ]

    set_icvl_64_31_TL_1 = ds_utils.GeneralImageDataset(
        tr_ds,
        transform=train_transform,
        target_transform=target_transform,
        common_transforms=common_transforms,
    )
    # worker_init_fn is in dataset -> just getting the seed
    icvl_64_31_TL = DataLoader(
        set_icvl_64_31_TL_1,
        batch_size=cla.batch_size,
        shuffle=True,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=None,
        persistent_workers=True,
    )
    # -------------- validation loader -------------------------

    """Test-Dev"""
    basefolder = cla.val_datadir
    mat_names = ["icvl_512_30", "icvl_512_50"]

    if not net.use_2dconv:
        mat_transform = [
            ds_utils.LoadMatHSI(
                input_key="input", gt_key="gt", transform=lambda x: x[:, ...][None]
            ),
        ]
    else:
        mat_transform = [
            ds_utils.LoadMatHSI(input_key="input", gt_key="gt"),
        ]

    mat_datasets = [
        ds_utils.MatDataFromFolder(
            os.path.join(basefolder, name), size=5, common_transform=mat_transform
        )
        for name in mat_names
    ]

    mat_loaders = [
        DataLoader(
            mat_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=cla.workers,
            pin_memory=torch.cuda.is_available(),
        )
        for mat_dataset in mat_datasets
    ]

    base_lr = cla.lr
    helper.adjust_learning_rate(optimizer, cla.lr)
    epoch_per_save = cla.save_freq
    max_epochs = 100
    for epoch in range(max_epochs):
        s = torch.random.seed()
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        if epoch == 85:
            helper.adjust_learning_rate(optimizer, base_lr * 0.1)
        elif epoch == 95:
            helper.adjust_learning_rate(optimizer, base_lr * 0.01)

        training_utils.train(icvl_64_31_TL, net, cla, epoch, optimizer, criterion, writer=writer)
        training_utils.validate(
            mat_loaders[0], "icvl-validate-noniid", net, cla, epoch, criterion, writer=writer
        )
        training_utils.validate(
            mat_loaders[1], "icvl-validate-mixture", net, cla, epoch, criterion, writer=writer
        )

        helper.display_learning_rate(optimizer)
        if (epoch % epoch_per_save == 0 and epoch > 0) or epoch == max_epochs - 1:
            logger.info("Saving current network...")
            model_latest_path = os.path.join(basedir, prefix, "model_latest.pth")
            training_utils.save_checkpoint(
                cla, epoch, net, optimizer, model_out_path=model_latest_path
            )
