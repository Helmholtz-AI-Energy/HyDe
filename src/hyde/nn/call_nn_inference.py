import os

import torch
import torch.nn as nn

from ..lowlevel import logging
from . import qrnn3d

logger = logging.get_logger()


class QRNNInference(nn.Module):
    def __init__(self, arch, pretrained_file, frozen=True):
        # get model
        super().__init__()
        network = qrnn3d.models.__dict__[arch]()
        # load the pretrained weights
        logger.info(f"Loading model from: {pretrained_file}")
        assert os.path.isfile(pretrained_file), "Error: no checkpoint directory found!"
        checkpoint = torch.load(pretrained_file)
        network.load_state_dict(checkpoint["net"])
        self.network = network
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
