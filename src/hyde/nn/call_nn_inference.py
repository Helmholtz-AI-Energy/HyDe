import os

import torch
import torch.nn as nn

from ..lowlevel import logging, utils
from . import models

logger = logging.get_logger()


class NNInference(nn.Module):
    def __init__(self, arch: str, pretrained_file, frozen=True):
        # get model
        super().__init__()
        network = models.__dict__[arch]()
        # load the pretrained weights
        logger.info(f"Loading model from: {pretrained_file}")
        assert os.path.isfile(pretrained_file), "Error: no checkpoint directory found!"
        checkpoint = torch.load(pretrained_file)

        try:
            network.load_state_dict(checkpoint["net"])
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["net"].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            network.load_state_dict(new_state_dict)

        self.network = network
        if "qrnn" in arch:
            self.network.clamp = False
        logger.debug(self.network)
        self.frozen = frozen
        self.network.eval()

    def forward(
        self, image: torch.Tensor, band_dim: int, normalize: bool = False, permute: bool = False
    ):
        """
        Image is not the right shape. needs to have shape [1, bands, height (rows), width (cols)]
        or [H x W x B] with permute=True

        Parameters
        ----------
        image
        band_dim
        normalize
        permute

        Returns
        -------

        """
        if permute and band_dim == -1 and image.ndim == 3:
            image = image.permute((2, 0, 1)).unsqueeze(0)
            band_dim = -3

        if image.ndim == 4:
            image = image.unsqueeze(0)

        if normalize:
            image, consts = utils.normalize(image, by_band=True, band_dim=band_dim)

        self.network = self.network.to(image.device)
        if self.frozen:
            with torch.no_grad():
                ret = self.network(image)
        else:
            ret = self.network(image)

        if permute:
            ret = ret.squeeze().permute((1, 2, 0))
            band_dim = -1

        if normalize:
            ret = utils.undo_normalize(ret, **consts, by_band=True, band_dim=band_dim)

        return ret
