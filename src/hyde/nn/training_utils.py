import os

import torch
import torch.distributed as dist
from torch.cuda import amp

from ..lowlevel import logging, utils
from .call_nn_inference import inference_windows

__all__ = ["train"]


logger = logging.get_logger()


def _train_loop(train_loader, network, cla, epoch, optimizer, criterion, scaler, iter_num):
    train_loss = 0
    avg_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if not cla.no_cuda and torch.cuda.is_available():
            inputs, targets = inputs.to(cla.device), targets.to(cla.device)
        optimizer.zero_grad()
        loss_data = 0
        with amp.autocast():
            outputs = network(inputs)
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1e6)  # max_norm)
        # loss.backward()
        loss_data += loss.item()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss_data
        avg_loss = train_loss / (batch_idx + 1)

    iter_num += batch_idx
    logger.info(f"Epoch: {epoch} iter_num: {iter_num} Loss: {avg_loss}")
    return avg_loss, iter_num


def train(train_loader, network, cla, epoch, optimizer, criterion, writer=None, iterations=150):
    logger.info(f"Train:\t\tEpoch: {epoch}")
    network.train()

    scaler = amp.GradScaler()
    it = 0
    while True:
        avg_loss, it = _train_loop(
            train_loader=train_loader,
            network=network,
            cla=cla,
            epoch=epoch,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            iter_num=it,
        )
        if it >= iterations:
            break

    if writer is not None:
        writer.add_scalar(os.path.join(cla.prefix, "train_loss_epoch"), avg_loss, epoch)


def validate(valid_loader, name, network, cla, epoch, criterion, writer=None):
    network.eval()
    validate_loss = 0
    total_psnr = 0
    ls, psnrs = [], []
    logger.info(f"Validation:\tEpoch: {epoch} dataset name: {name}")
    is_2d = True if cla.arch in ["memnet", "denet", "memnet_hyres"] else False
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if not cla.no_cuda and torch.cuda.is_available():
                inputs, targets = inputs.to(cla.device), targets.to(cla.device)

            loss_data = 0
            if is_2d:
                # todo: nn_inference style
                inputs = inputs.squeeze(1)
                outputs = inference_windows(
                    network, inputs, window_size=256, band_window=10, buff=(4, 4, 2), frozen=True
                )
                outputs = outputs.squeeze(1)
                targets = targets.squeeze(1)
                loss = criterion(outputs, targets)
                loss_data += loss.item()
            else:
                outputs = network(inputs)
                outputs = outputs.squeeze(1)
                targets = targets.squeeze(1)
                loss = criterion(outputs, targets)
                loss_data += loss.item()

            # data units: [batch, bands, h, w]
            psnr = []
            for d in range(outputs.shape[0]):
                # band dim assumes that the batch is > 1
                psnr.append(utils.peak_snr(outputs[d], targets[d], bandwise=False, band_dim=-3))
            psnrs.extend(psnr)
            psnr = sum(psnr) / float(len(psnr))

            ls.append(loss_data)

            validate_loss += loss_data
            avg_loss = validate_loss / (batch_idx + 1)

            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx + 1)

    if dist.is_initialized():
        # average all the results
        sz = dist.get_world_size()
        red = torch.tensor([avg_psnr, avg_loss], device=cla.device) / float(sz)
        dist.all_reduce(red)
        avg_psnr = red[0].item()
        avg_loss = red[1].item()

    logger.info(f"Final: Loss: {avg_loss} | PSNR: {avg_psnr}")

    if writer is not None:
        writer.add_scalar(os.path.join(cla.prefix, name, "val_loss_epoch"), avg_loss, epoch)
        writer.add_scalar(os.path.join(cla.prefix, name, "val_psnr_epoch"), avg_psnr, epoch)
    avg_psnr = sum(psnrs) / float(len(psnrs))
    avg_loss = sum(ls) / float(len(ls))
    return avg_psnr, avg_loss


def save_checkpoint(cla, epoch, network, optimizer, model_out_path=None, **kwargs):
    if cla.rank != 0:
        return
    if not model_out_path:
        model_out_path = os.path.join(cla.save_dir, cla.arch, f"model_epoch_{epoch}.pth")

    state = {
        "net": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    state.update(kwargs)

    if not os.path.isdir(os.path.join(cla.save_dir, cla.arch)):
        os.makedirs(os.path.join(cla.save_dir, cla.arch))

    torch.save(state, model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
