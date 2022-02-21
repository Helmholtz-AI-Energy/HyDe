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
    ws = window_size
    sh_b, sh_r, sh_c = image.shape[-3:]
    # design this to move over windows (like conv groups)
    # ==== rows ============
    dim_r_starts = [ws, ws - buff[0]]
    dim_r_set_starts = [ws, ws]
    if dim_r_starts[-1] >= sh_r or ws >= sh_r:
        # if the second window starts after the image ends
        dim_r_starts, dim_r_set_starts = [0], [0]
    else:
        while dim_r_starts[-1] < sh_r:
            # layer the images over one another to avoid tiling (hopfully)
            dim_r_starts.append(dim_r_starts[-1] + ws - buff[0])
            dim_r_set_starts.append(dim_r_set_starts[-1] + ws)
        dim_r_starts, dim_r_set_starts = dim_r_starts[:-1], dim_r_set_starts[:-1]
    # replace the first item with 0 (it currently has the end of the first window)
    dim_r_starts[0], dim_r_set_starts[0] = 0, 0
    # ==== cols ============
    dim_c_starts = [ws, ws - buff[0]]
    dim_c_set_starts = [ws, ws]
    if dim_c_starts[-1] >= sh_c or ws >= sh_c:
        # if the second window starts after the image ends
        dim_c_starts, dim_c_set_starts = [0], [0]
    else:
        while dim_c_starts[-1] < sh_c:
            # layer the images over one another to avoid tiling (hopfully)
            dim_c_starts.append(dim_c_starts[-1] + ws - buff[1])
            dim_c_set_starts.append(dim_c_set_starts[-1] + ws)
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
        end_r = beg_r + ws
        sl[row_dim] = slice(beg_r, end_r)

        # need to make something that
        cut_slice = [slice(None)] * image.ndim
        cut_slice_out = [slice(None)] * image.ndim
        cut_slice_out[row_dim] = slice(set_r, set_r + ws)
        if beg_r != 0:
            cut_slice[row_dim] = slice(buff[0], None)
            cut_slice_out[row_dim] = slice(beg_r + buff[0], end_r)
        if end_r > sh_r:
            # if we go past the end on the slice,
            # this will cause issue when the network has a fixed number of bands
            sl[row_dim] = slice(sh_r - ws, sh_r)
            cut_slice[row_dim] = slice(-1 * (ws - buff[0]), None)
            cut_slice_out[row_dim] = slice(-1 * (ws - buff[0]), None)
        cut_slice[col_dim] = slice(None)
        for beg_c, set_c in zip(dim_c_starts, dim_c_set_starts):
            # create overlap
            end_c = beg_c + ws
            sl[col_dim] = slice(beg_c, end_c)

            cut_slice_out[col_dim] = slice(set_c, set_c + ws)
            if beg_c != 0:
                cut_slice[col_dim] = slice(buff[1], None)
                cut_slice_out[col_dim] = slice(beg_c + buff[1], end_c)
            if end_c > sh_c:
                # if we go past the end on the slice,
                # this will cause issue when the network has a fixed number of bands
                sl[col_dim] = slice(sh_c - ws, sh_c)
                cut_slice[col_dim] = slice(-1 * (ws - buff[1]), None)
                cut_slice_out[col_dim] = slice(-1 * (ws - buff[1]), None)
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


class QRNNInference(nn.Module):
    def __init__(self, arch: str, pretrained_file, frozen=True, band_window=None):
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

        self.network.clamp = False

        logger.debug(self.network)
        self.frozen = frozen
        self.network.eval()

        self.band_window = band_window
        self.windows = True  # band_window is not None

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
        if self.windows is None:
            return self._forward(image, band_dim, normalize, permute)
        else:
            self.band_window = 31
            # design this to move over windows (like conv groups)
            # assume band_dim = -1
            min_dim = 0 if image.shape[0] < image.shape[1] else 1

            if min_dim == 0:
                num_windows = image.shape[1] // image.shape[0]
                rem_window = image.shape[1] % image.shape[0]
            else:
                num_windows = image.shape[0] // image.shape[1]
                rem_window = image.shape[0] % image.shape[1]

            out = torch.zeros_like(image)

            sh0, sh1 = image.shape[:2]
            for w in range(num_windows):
                sl = [
                    slice(None),
                ] * image.ndim
                if min_dim == 0:
                    dim0 = slice(sh0)
                    dim1 = slice(sh0 * w, (w + 1) * sh0)
                else:
                    dim0 = slice(sh1 * w, (w + 1) * sh1)
                    dim1 = slice(sh1)
                sl[0] = dim0  # slice(349)
                sl[1] = dim1  # slice(349)

                iters = image.shape[band_dim] // self.band_window
                rem = image.shape[band_dim] % self.band_window
                for i in range(iters):
                    # print(i)
                    beg = i * self.band_window
                    end = (i + 1) * self.band_window
                    # if i == iters - 1:
                    #     end += rem
                    sl[band_dim] = slice(beg, end)
                    out[sl] = self._forward(image[sl], band_dim, normalize, permute)
                if rem != 0:
                    # remainder bands
                    beg = -self.band_window
                    end = end + self.band_window
                    sl[band_dim] = slice(beg, end)
                    out[sl] = self._forward(image[sl], band_dim, normalize, permute)

            # do last window
            sl = [
                slice(None),
            ] * image.ndim
            set_slice = [
                slice(None),
            ] * image.ndim
            if min_dim == 0:
                dim0 = slice(sh0)
                dim1 = slice(sh1 - sh0, sh1)
                sdim0 = dim0
                sdim1 = slice(-rem_window, sh1)
            else:
                dim0 = slice(sh0 - sh1, sh0)
                dim1 = slice(sh1)
                sdim0 = slice(-rem_window, sh0)
                sdim1 = dim1

            sl[0] = dim0  # slice(349)
            sl[1] = dim1  # slice(349)
            set_slice[0] = sdim0
            set_slice[1] = sdim1

            iters = image.shape[band_dim] // self.band_window
            rem = image.shape[band_dim] % self.band_window
            for i in range(iters):
                # print(i)
                beg = i * self.band_window
                end = (i + 1) * self.band_window
                # if i == iters - 1:
                #     end += rem
                sl[band_dim] = slice(beg, end)
                set_slice[band_dim] = slice(beg, end)
                ret = self._forward(image[sl], band_dim, normalize, permute)
                out[set_slice] = ret[sdim0, sdim1]
            if rem != 0:
                # remainder bands
                beg = -self.band_window
                end = end + self.band_window
                sl[band_dim] = slice(beg, end)
                out[sl] = self._forward(image[sl], band_dim, normalize, permute)

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
