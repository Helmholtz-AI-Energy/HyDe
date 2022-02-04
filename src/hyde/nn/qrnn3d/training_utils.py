import os

import torch
import torch.nn as nn

from ...lowlevel import logging, utils

__all__ = ["train"]


logger = logging.get_logger()


def train(train_loader, network, cla, epoch, optimizer, criterion, bandwise, writer=None):
    logger.info(f"\nTrain Loop - Epoch: {epoch}")
    network.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if not cla.no_cuda and torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        loss_data = 0
        # print(bandwise, inputs.shape)
        if bandwise:
            outs = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                out = network(i)
                outs.append(out)
                loss = criterion(out, t)
                loss.backward()
                loss_data += loss.item()
        else:
            # print('before input')
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            # print("before backward")
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
        writer.add_scalar(os.path.join(cla.prefix, "train_loss_epoch"), avg_loss, epoch)


def validate(valid_loader, name, network, cla, epoch, criterion, bandwise, writer=None):
    network.eval()
    validate_loss = 0
    total_psnr = 0
    logger.info(f"Validation: Epoch: {epoch} dataset name: {name}")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if not cla.no_cuda and torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            loss_data = 0
            if bandwise:
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

            # data units: [batch, 1, bands, h, w]
            psnr = utils.peak_snr(outputs, targets, bandwise=True, band_dim=1)
            try:
                psnr = torch.mean(psnr)
            except (RuntimeError, TypeError):
                pass

            validate_loss += loss_data
            avg_loss = validate_loss / (batch_idx + 1)

            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx + 1)

            # if batch_idx % cla.log_freq == 0:
            logger.info(f"Loss: {avg_loss} | PSNR: {avg_psnr}")
            break

    logger.info(f"Final: Loss: {avg_loss} | PSNR: {avg_psnr}")

    if writer is not None:
        writer.add_scalar(os.path.join(cla.prefix, name, "val_loss_epoch"), avg_loss, epoch)
        writer.add_scalar(os.path.join(cla.prefix, name, "val_psnr_epoch"), avg_psnr, epoch)

    return avg_psnr, avg_loss


def save_checkpoint(cla, epoch, network, optimizer, model_out_path=None, **kwargs):
    if not model_out_path:
        model_out_path = os.path.join(cla.save_dir, cla.prefix, f"model_epoch_{epoch}.pth")

    state = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    state.update(kwargs)

    if not os.path.isdir(os.path.join(cla.save_dir, cla.prefix)):
        os.makedirs(os.path.join(cla.save_dir, cla.prefix))

    torch.save(state, model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
