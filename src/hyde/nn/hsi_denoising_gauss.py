import argparse
import os
import time
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from kornia.losses import SSIMLoss
from torch.utils.data import DataLoader

from hyde.lowlevel import logging
from hyde.nn import MultipleWeightedLosses, comm, helper, models, training_utils
from hyde.nn.datasets import dataset_utils as ds_utils
from hyde.nn.datasets.transforms import AddGaussianNoise, AddGaussianNoiseBlind
from hyde.nn.parsers import qrnn_parser

logger = logging.get_logger()


def main():
    # print('testing 123')
    # return None

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
    # initialize parameters
    # init params will set the model params with a random distribution
    # helper.init_params(net, init_type=cla.init)  # disable for default initialization
    distributed = False
    bandwise = net.bandwise
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

    # """Resume previous model"""
    if cla.resume:
        # Load checkpoint.
        torch.load(cla.resume_path, not cla.no_ropt)
    else:
        logger.info("==> Building model..")
        helper.init_network(net, cla.nn_init_mode)
        logger.debug(net)

    cudnn.benchmark = True

    # AddGaussianNoiseBlind(max_sigma_db=40, min_sigma_db=10),

    crop_size = (512, 512)

    train_icvl = ds_utils.ICVLDataset(
        cla.datadir,
        common_transforms=None,
        transform=AddGaussianNoise(15),
        crop_size=crop_size,
    )
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_icvl)
    else:
        train_sampler = None

    # worker_init_fn is in dataset -> just getting the seed
    train_loader = DataLoader(
        train_icvl,
        batch_size=cla.batch_size,
        shuffle=True if train_sampler is None else False,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        sampler=train_sampler,
    )

    # -------------- validation loader -------------------------
    basefolder = cla.val_datadir

    val_dataset = ds_utils.ICVLDataset(
        basefolder,
        transform=AddGaussianNoiseBlind(max_sigma_db=40, min_sigma_db=10),  # blind gaussain noise
        val=True,
        crop_size=crop_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cla.batch_size,
        shuffle=True,  # shuffle this dataset, we will fix the noise transforms
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
    )

    helper.adjust_learning_rate(optimizer, cla.lr)
    # epoch_per_save = cla.save_freq
    max_epochs = 100
    best_val_loss, best_val_psnr = 100000, 0
    epochs_wo_best = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5)

    for epoch in range(max_epochs):
        logger.info(f"\n\t --------- Start epoch {epoch} ---------\n")
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        np.random.seed(epoch)
        # TODO: change the transform to something harder at some point in the training?
        # if epoch == 10:
        #    icvl_64_31_TL_2.transform = harder_train_transform

        # 5, 10, 20, 30, 40, 50, blind
        # if epoch < 20:
        #     noise = 20
        if epoch < 40:
            noise = 20
        elif epoch < 60:
            noise = 30
        else:
            noise = None

        # if epoch == 30:
        #    # RESET LR???
        #    scheduler._reset()
        #    helper.adjust_learning_rate(optimizer, cla.lr)
        #    helper.adjust_learning_rate(optimizer, cla.lr)

        # if epoch == 20:
        #     helper.adjust_learning_rate(optimizer, cla.lr * 0.1)
        # elif epoch == 30:
        #     helper.adjust_learning_rate(optimizer, cla.lr)
        # elif epoch == 35:
        #     helper.adjust_learning_rate(optimizer, cla.lr * 0.1)
        # elif epoch == 45:
        #     helper.adjust_learning_rate(optimizer, cla.lr * 0.01)
        # elif epoch == 45:
        #     helper.adjust_learning_rate(optimizer, cla.lr * 0.01)

        if noise is not None:
            train_icvl.transform = AddGaussianNoise(noise)
            logger.info(f"Noise level: {noise} dB")
        else:
            train_icvl.transform = AddGaussianNoiseBlind(max_sigma_db=40, min_sigma_db=20)
            logger.info("Noise level: BLIND!")

        # if epoch == 70:
        #     helper.adjust_learning_rate(optimizer, cla.lr * 0.1)
        helper.display_learning_rate(optimizer)
        ttime = time.perf_counter()
        training_utils.train(
            train_loader, net, cla, epoch, optimizer, criterion, bandwise, writer=writer
        )
        ttime = time.perf_counter() - ttime

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
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
            logger.debug(f"Max mem alocated: {torch.cuda.max_memory_allocated(device=None)}")

        scheduler.step(ls)

        epochs_wo_best += 1

        # if (epoch % epoch_per_save == 0 and epoch > 0) or epoch == max_epochs - 1:
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
                cla.save_dir, prefix, f"seeded_20-30db_noise_{cla.loss}.pth"
            )
            training_utils.save_checkpoint(
                cla, epoch, net, optimizer, model_out_path=model_latest_path
            )

        # if epochs_wo_best == 5:
        #    logger.info(f"Breaking loop, not improving for 5 epochs, current epoch: {epoch}")
        # break


if __name__ == "__main__":
    main()
