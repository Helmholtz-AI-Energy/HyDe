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

import torch.distributed as dist

from utils import train_argparse

import hyde.nn.comm as comm

import time

from hyde.lowlevel import logging
from hyde.nn import comm
from hyde.nn.general_nn_utils import (
    AddGaussianNoise,
    AddGaussianNoiseBlind,
    MultipleWeightedLosses,
)
from hyde.nn.qrnn3d import dataset_utils as ds_utils
from hyde.nn.qrnn3d import helper, models, training_utils
from torch._six import string_classes

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
    #helper.init_params(net, init_type=cla.init)  # disable for default initialization
    distributed = False
    bandwise = net.bandwise
    world_size = 1
    if torch.cuda.device_count() > 1 and cuda:
        import torch.distributed as dist
        #print("cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
        logger.info("Spawning torch groups for DDP")
        group = comm.init(method=cla.comm_method)

        loc_rank = dist.get_rank() % torch.cuda.device_count()
        world_size = dist.get_world_size()
        device = torch.device("cuda", loc_rank)
        logger.info(f"Default GPU: {device}")
        net = models.__dict__[cla.arch]()
        net = net.to(device)

        net = nn.SyncBatchNorm.convert_sync_batchnorm(net, group)

        net = nn.parallel.DistributedDataParallel(net, device_ids=[device.index])
        #net = nn.SyncBatchNorm.convert_sync_batchnorm(net, group)

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
            [nn.MSELoss(), SSIMLoss(window_size=11, max_val=1)], weight=[1, 2.5e-3]
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

    net = net.to(torch.bfloat16)

    cudnn.benchmark = True

    # train_transform_1 = AddGaussianNoise(34)
    common_transform_2 = transforms.RandomCrop((256, 256))
    train_transform_2 = AddGaussianNoiseBlind(max_sigma_db=40)

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
        transform=train_transform_2,
        common_transforms=common_transform_2,
    )
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(set_icvl_64_31_TL_2)
    else:
        train_sampler = None

    #def colate_fn(batch):
    #    print('in colate')
    #    elem = batch[0]
    #    elem_type = type(elem)
    #    if isinstance(elem, torch.Tensor):
    #        out = None
    #        if torch.utils.data.get_worker_info() is not None:
    #            # If we're in a background process, concatenate directly into a
    #            # shared memory tensor to avoid an extra copy
    #            numel = sum(x.numel() for x in batch)
    #            storage = elem.storage()._new_shared(numel)
    #            out = elem.new(storage)
    #        return torch.stack(batch, 0, out=out)
    #    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
    #            and elem_type.__name__ != 'string_':
    #        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
    #            # array of string classes and object
    #            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
    #                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
   # 
   #             return default_collate([torch.as_tensor(b) for b in batch])
   #         elif elem.shape == ():  # scalars
   #             return torch.as_tensor(batch)
   #     elif isinstance(elem, float):
   #         return torch.tensor(batch, dtype=torch.float64)
   #     elif isinstance(elem, int):
   #         return torch.tensor(batch)
   #     elif isinstance(elem, string_classes):
   #         return batch
   #     elif isinstance(elem, collections.abc.Mapping):
   #         return {key: default_collate([d[key] for d in batch]) for key in elem}
   #     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
   #         return elem_type(*(default_collate(samples) for samples in zip(*batch)))
   #     elif isinstance(elem, collections.abc.Sequence):
   #         # check to make sure that the elements in batch have consistent size
   #         it = iter(batch)
   #         elem_size = len(next(it))
   #         if not all(len(elem) == elem_size for elem in it):
   #             raise RuntimeError('each element in list of batch should be of equal size')
   #         transposed = zip(*batch)
   #         return [default_collate(samples) for samples in transposed]
   # 
   #     raise TypeError(default_collate_err_msg_format.format(elem_type))

    # worker_init_fn is in dataset -> just getting the seed
    icvl_64_31_TL_2 = DataLoader(
        set_icvl_64_31_TL_2,
        batch_size=cla.batch_size,
        shuffle=True,
        num_workers=cla.workers,
        pin_memory=False, #torch.cuda.is_available(),
        #worker_init_fn=None,
        persistent_workers=False,
        sampler=train_sampler,
        #collate_fn=colate_fn,
    )

    #dist.barrier()
    #logger.info("testing train loader")
    #for d, t in icvl_64_31_TL_2:
    #    print(d.shape, t.shape)
    #logger.info("finished train loader test")

    # -------------- validation loader -------------------------

    """Test-Dev"""
    basefolder = cla.val_datadir

    val_dataset = ds_utils.ICVLDataset(
        basefolder,
        transform=AddGaussianNoiseBlind(max_sigma_db=40),  # blind gaussain noise
        val=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cla.batch_size,
        shuffle=False,
        num_workers=cla.workers,
        pin_memory=torch.cuda.is_available(),
    )

    base_lr = cla.lr
    helper.adjust_learning_rate(optimizer, cla.lr)
    # epoch_per_save = cla.save_freq
    max_epochs = 50
    best_val_loss, best_val_psnr = 100000, 0
    epochs_wo_best = 0
    expected_finish_time = None

    #dist.barrier()

    for epoch in range(max_epochs):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        np.random.seed(epoch)

        #if epoch == 5:
        #    helper.adjust_learning_rate(optimizer, base_lr * 0.1)
        #elif epoch == 10:
        #    helper.adjust_learning_rate(optimizer, base_lr * 0.01)
        
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
            '%H:%M:%S', 
            time.gmtime((ttime + vtime) * (max_epochs - epoch))
        )
        logger.info(f"Expected time remaing: {expected_time_remaining}")

        epochs_wo_best += 1

        helper.display_learning_rate(optimizer)
        # if (epoch % epoch_per_save == 0 and epoch > 0) or epoch == max_epochs - 1:
        if psnr > best_val_psnr:
            best_val_psnr = psnr
            epochs_wo_best = 0

        if ls < best_val_loss:
            best_val_loss = ls
            epochs_wo_best = 0

        if epochs_wo_best == 0:  # best_val_psnr < psnr or best_val_psnr > ls:
            logger.info("Saving current network...")
            model_latest_path = os.path.join(cla.save_dir, prefix, "model_latest.pth")
            training_utils.save_checkpoint(
                cla, epoch, net, optimizer, model_out_path=model_latest_path
            )

        #if epochs_wo_best == 5:
        #    logger.info(f"Breaking loop, not improving for 5 epochs, current epoch: {epoch}")
            #break


if __name__ == "__main__":
    main()
