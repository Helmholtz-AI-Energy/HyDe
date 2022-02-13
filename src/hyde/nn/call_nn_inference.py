import os

import torch
import torch.nn as nn

from ..lowlevel import logging
from . import models

logger = logging.get_logger()


class QRNNInference(nn.Module):
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
            self.network.clamp = True
        logger.debug(self.network)
        self.frozen = frozen
        self.network.eval()

    def forward(self, image: torch.Tensor, band_dim: int):
        if image.ndim != 4 or band_dim != 1:
            raise RuntimeError(
                "Image is not the right shape. needs to have shape [1, bands, height (rows), width (cols)]"
            )
        if image.ndim == 4:
            image = image.unsqueeze(0)

        self.network = self.network.to(image.device)
        if self.frozen:
            with torch.no_grad():
                return self.network(image)
        else:
            return self.network(image)
