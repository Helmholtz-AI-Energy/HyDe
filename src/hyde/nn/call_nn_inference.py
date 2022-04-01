import os

import torch
import torch.nn as nn

from ..lowlevel import logging, utils
from . import models

logger = logging.get_logger()


__all__ = ["NNInference", "inference_windows"]


class NNInference(nn.Module):
    def __init__(self, arch: str, pretrained_file, frozen=True, band_window=10, window_shape=256):
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

        self.band_window = band_window
        self.window_shape = window_shape  # band_window is not None

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
        if self.window_shape is None:
            return self._forward(image, band_dim, normalize, permute)

        self.network = self.network.to(image.device, non_blocking=True)
        if permute and image.ndim == 3:
            image = image.permute((2, 0, 1)).unsqueeze(0)
            band_dim = -3
            image = image.unsqueeze(0)

        if normalize:
            image, consts = utils.normalize(image, by_band=True, band_dim=band_dim)

        buff = (4, 4, 2)
        # todo: make class variable -> buffer

        out = inference_windows(
            network=self.network,
            image=image,
            buff=buff,
            window_size=self.window_shape,
            band_window=self.band_window,
            frozen=self.frozen,
        )

        if permute:
            out = out.squeeze().permute((1, 2, 0))
            band_dim = -1

        if normalize:
            out = utils.undo_normalize(out, **consts, by_band=True, band_dim=band_dim)

        return out

    def _forward(self, image, band_dim, normalize, permute):
        if permute and band_dim == -1 and image.ndim == 3:
            image = image.permute((2, 0, 1)).unsqueeze(0)
            band_dim = -3

        if image.ndim == 4:
            image = image.unsqueeze(0)

        if normalize:
            image, consts = utils.normalize(image, by_band=True, band_dim=band_dim)

        self.network = self.network.to(image.device)
        ret = _call_nn(self.network, image, self.frozen)

        if permute:
            ret = ret.squeeze().permute((1, 2, 0))
            band_dim = -1

        if normalize:
            ret = utils.undo_normalize(ret, **consts, by_band=True, band_dim=band_dim)

        return ret


def inference_windows(network, image, buff, window_size, band_window, frozen):
    # THIS ASSMUES THAT THE IMAGE/BATCH IS SHAPE: [..., BANDS, H, W]
    band_dim = -3
    row_dim = -2
    col_dim = -1
    row_w, col_w = window_size, window_size
    sh_b, sh_r, sh_c = image.shape[-3:]
    buff = list(buff)
    if row_w >= sh_r:
        row_w = sh_r
        buff[0] = 0
    if col_w >= sh_c:
        col_w = sh_c
        buff[1] = 0

    # design this to move over windows (like conv groups)
    # ==== rows ============
    dim_r_starts = [row_w, row_w - buff[0]]
    dim_r_set_starts = [row_w, row_w]
    if dim_r_starts[-1] >= sh_r or row_w >= sh_r:
        # if the second window starts after the image ends
        dim_r_starts, dim_r_set_starts = [0], [0]
    else:
        while dim_r_starts[-1] < sh_r:
            # layer the images over one another to avoid tiling (hopfully)
            dim_r_starts.append(dim_r_starts[-1] + row_w - buff[0])
            dim_r_set_starts.append(dim_r_set_starts[-1] + row_w)
        dim_r_starts, dim_r_set_starts = dim_r_starts[:-1], dim_r_set_starts[:-1]
    # replace the first item with 0 (it currently has the end of the first window)
    dim_r_starts[0], dim_r_set_starts[0] = 0, 0
    # ==== cols ============
    dim_c_starts = [col_w, col_w - buff[1]]
    dim_c_set_starts = [col_w, col_w]
    if dim_c_starts[-1] >= sh_c or col_w >= sh_c:
        # if the second window starts after the image ends
        dim_c_starts, dim_c_set_starts = [0], [0]
    else:
        while dim_c_starts[-1] < sh_c:
            # layer the images over one another to avoid tiling (hopfully)
            dim_c_starts.append(dim_c_starts[-1] + col_w - buff[1])
            dim_c_set_starts.append(dim_c_set_starts[-1] + col_w)
        dim_c_starts, dim_c_set_starts = dim_c_starts[:-1], dim_c_set_starts[:-1]
    # replace the first item with 0 (it currently has the end of the first window)
    dim_c_starts[0], dim_c_set_starts[0] = 0, 0
    # ==== bands ============
    bw = band_window
    dim_b_starts = [bw, bw - buff[2]]
    dim_b_set_starts = [bw, bw]
    if dim_b_starts[-1] >= sh_b or bw >= sh_b:
        # if the second window starts after the image ends
        dim_b_starts, dim_b_set_starts = [0], [0]
    else:
        while dim_b_starts[-1] < sh_b:
            # layer the images over one another to avoid tiling (hopfully)
            dim_b_starts.append(dim_b_starts[-1] + bw - buff[2])
            dim_b_set_starts.append(dim_b_set_starts[-1] + bw)
        dim_b_starts, dim_b_set_starts = dim_b_starts[:-1], dim_b_set_starts[:-1]
    # replace the first item with 0 (it currently has the end of the first window)
    dim_b_starts[0], dim_b_set_starts[0] = 0, 0

    out = torch.zeros_like(image)
    # This will drop the data from the windows
    # dim _r is outside loop -> dim_c is inside loop -> dim_b on last loop
    # KEY:
    # sl -> slice the image to get the window
    # cut_slice -> slice the result after applying the network
    # cut_slice_out -> where to put the result after its cut out
    for beg_r, set_r in zip(dim_r_starts, dim_r_set_starts):
        sl = [slice(None)] * image.ndim
        # create overlap
        end_r = beg_r + row_w
        sl[row_dim] = slice(beg_r, end_r)

        # need to make something that
        cut_slice = [slice(None)] * image.ndim
        cut_slice_out = [slice(None)] * image.ndim
        cut_slice_out[row_dim] = slice(set_r, set_r + row_w)
        if beg_r != 0:
            cut_slice[row_dim] = slice(buff[0], None)
            cut_slice_out[row_dim] = slice(beg_r + buff[0], end_r)
        if end_r > sh_r:
            # if we go past the end on the slice,
            # this will cause issue when the network has a fixed number of bands
            sl[row_dim] = slice(sh_r - row_w, sh_r)
            cut_slice[row_dim] = slice(-1 * (row_w - buff[0]), None)
            cut_slice_out[row_dim] = slice(-1 * (row_w - buff[0]), None)
        cut_slice[col_dim] = slice(None)
        for beg_c, set_c in zip(dim_c_starts, dim_c_set_starts):
            # create overlap
            end_c = beg_c + col_w
            sl[col_dim] = slice(beg_c, end_c)

            cut_slice_out[col_dim] = slice(set_c, set_c + col_w)
            if beg_c != 0:
                cut_slice[col_dim] = slice(buff[1], None)
                cut_slice_out[col_dim] = slice(beg_c + buff[1], end_c)
            if end_c > sh_c:
                # if we go past the end on the slice,
                # this will cause issue when the network has a fixed number of bands
                sl[col_dim] = slice(sh_c - col_w, sh_c)
                cut_slice[col_dim] = slice(-1 * (col_w - buff[1]), None)
                cut_slice_out[col_dim] = slice(-1 * (col_w - buff[1]), None)
            cut_slice[band_dim] = slice(None)
            for beg_b, set_b in zip(dim_b_starts, dim_b_set_starts):
                # create overlap
                end_b = beg_b + bw
                sl[band_dim] = slice(beg_b, end_b)

                cut_slice_out[band_dim] = slice(set_b, set_b + bw)
                if beg_b != 0:
                    cut_slice[band_dim] = slice(buff[2], None)
                    cut_slice_out[band_dim] = slice(beg_b + buff[2], end_b)
                if end_b > sh_b:
                    # if we go past the end on the slice,
                    # this will cause issue when the network has a fixed number of bands
                    sl[band_dim] = slice(sh_b - bw, sh_b)
                    cut_slice[band_dim] = slice(-1 * (bw - buff[2]), None)
                    cut_slice_out[band_dim] = slice(-1 * (bw - buff[2]), None)
                out[cut_slice_out] = _call_nn(network=network, image=image[sl], frozen=frozen)[
                    cut_slice
                ]
    return out


def _call_nn(network, image, frozen):
    if frozen:
        with torch.no_grad():
            return network(image)
    else:
        return network(image)
