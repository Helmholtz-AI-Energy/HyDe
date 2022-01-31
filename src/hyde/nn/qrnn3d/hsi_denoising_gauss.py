import argparse
import os
from functools import partial
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from kornia.losses import SSIMLoss
from torch.utils.data import DataLoader
from utility import dataset as ds_utils
from utility import helper, indexes, lmdb_dataset
from utils import train_argparse

from ...lowlevel import logging
from .. import comm
from ..nn_utils import AddGaussianNoise, AddGaussianNoiseBlind
from . import models

logger = logging.get_logger()


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return "weight={}".format(self.weight)


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
        criterion = MultipleLoss(
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

    tr_ds = lmdb_dataset.LMDBDataset(cla.dataroot, repeat=16)
    dataset2 = lmdb_dataset.LMDBDataset(cla.dataroot, repeat=64)

    HSI2Tensor = partial(ds_utils.HsiToTensor, use_2dconv=net.use_2dconv)
    target_transform = [
        HSI2Tensor(),
    ]
    train_transform_1 = [AddGaussianNoise(34), HSI2Tensor()]
    common_transform_2 = [
        transforms.RandomCrop((32, 32)),
    ]
    train_transform_2 = [AddGaussianNoiseBlind(max_sigma_db=40), HSI2Tensor()]

    set_icvl_64_31_TL_1 = ds_utils.GeneralImageDataset(
        tr_ds,
        transform=train_transform_1,
        target_transform=target_transform,
    )
    # worker_init_fn is in dataset -> just getting the seed
    icvl_64_31_TL_1 = DataLoader(
        set_icvl_64_31_TL_1,
        batch_size=cla.batch_size,
        shuffle=True,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=None,
        persistent_workers=True,
    )

    set_icvl_64_31_TL_2 = ds_utils.GeneralImageDataset(
        dataset2,
        transform=train_transform_2,
        target_transform=target_transform,
        common_transforms=common_transform_2,
    )
    # worker_init_fn is in dataset -> just getting the seed
    icvl_64_31_TL_2 = DataLoader(
        set_icvl_64_31_TL_2,
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
    max_epochs = 50
    for epoch in range(max_epochs):
        s = torch.random.seed()
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        if epoch == 20:
            helper.adjust_learning_rate(optimizer, base_lr * 0.1)
        elif epoch == 30:
            helper.adjust_learning_rate(optimizer, base_lr)
        elif epoch == 35:
            helper.adjust_learning_rate(optimizer, base_lr * 0.1)
        elif epoch == 30:
            helper.adjust_learning_rate(optimizer, base_lr * 0.01)

        if epoch <= 30:
            train(icvl_64_31_TL_1, net, cla, epoch, optimizer, criterion, writer=writer)
            validate(mat_loaders[1], "icvl-validate-50", net, cla, epoch, criterion, writer=writer)
        else:
            train(icvl_64_31_TL_2, net, cla, epoch, optimizer, criterion)
            validate(mat_loaders[0], "icvl-validate-30", net, cla, epoch, criterion, writer=writer)
            validate(mat_loaders[1], "icvl-validate-50", net, cla, epoch, criterion, writer=writer)

        helper.display_learning_rate(optimizer)
        if (epoch % epoch_per_save == 0 and epoch > 0) or epoch == max_epochs - 1:
            logger.info("Saving current network...")
            model_latest_path = os.path.join(basedir, prefix, "model_latest.pth")
            save_checkpoint(cla, epoch, net, optimizer, model_out_path=model_latest_path)


def train(train_loader, network, cla, epoch, optimizer, criterion, writer=None):
    logger.info(f"\nTrain Loop - Epoch: {epoch}")
    network.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        loss_data = 0
        if network.bandwise:
            outs = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                out = network(i)
                outs.append(out)
                loss = criterion(out, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
        total_norm = nn.utils.clip_grad_norm_(network.parameters(), cla.clip)
        optimizer.step()

        train_loss += loss_data
        avg_loss = train_loss / (batch_idx + 1)

        if batch_idx % cla.log_freq == 0:
            logger.info(
                f"Epoch: {epoch} iteration: {batch_idx} Loss: {avg_loss} Norm: {total_norm}"
            )

    if writer is not None:
        writer.add_scalar(join(cla.prefix, "train_loss_epoch"), avg_loss, epoch)


def validate(valid_loader, name, network, cla, epoch, criterion, writer=None):
    network.eval()
    validate_loss = 0
    total_psnr = 0
    logger.info(f"Validation: Epoch: {epoch} dataset name: {name}")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if not torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            loss_data = 0
            if network.bandwise:
                outs = []
                for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                    out = network(i)
                    outs.append(out)
                    loss = criterion(out, t)
                    loss_data += loss.item()
                outputs = torch.cat(outs, dim=1)
            else:
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                loss_data += loss.item()

            psnr = torch.mean(indexes.cal_bwpsnr(outputs, targets))

            validate_loss += loss_data
            avg_loss = validate_loss / (batch_idx + 1)

            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx + 1)

            if batch_idx % cla.log_freq == 0:
                logger.info(f"Loss: {avg_loss} | PSNR: {avg_psnr}")

    logger.info(f"Final: Loss: {avg_loss} | PSNR: {avg_psnr}")

    if writer is not None:
        writer.add_scalar(join(cla.prefix, name, "val_loss_epoch"), avg_loss, epoch)
        writer.add_scalar(join(cla.prefix, name, "val_psnr_epoch"), avg_psnr, epoch)

    return avg_psnr, avg_loss


def save_checkpoint(cla, epoch, network, optimizer, model_out_path=None, **kwargs):
    if not model_out_path:
        model_out_path = join(cla.basedir, cla.prefix, f"model_epoch_{epoch}.pth")

    state = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    state.update(kwargs)

    if not os.path.isdir(join(cla.basedir, cla.prefix)):
        os.makedirs(join(cla.basedir, cla.prefix))

    torch.save(state, model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
