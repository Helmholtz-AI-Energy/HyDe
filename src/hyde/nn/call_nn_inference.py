import os

import torch
import torch.nn as nn

from ..lowlevel import logging, utils
from . import models

logger = logging.get_logger()


class NNInference(nn.Module):
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
        if "qrnn" in arch:
            self.network.clamp = False
        logger.debug(self.network)
        self.frozen = frozen
        self.network.eval()

        self.band_window = band_window
        self.windows = 128  # band_window is not None

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

        row_dim = 0
        col_dim = 1
        band_dim = 2
        sh_r, sh_c, sh_b = image.shape
        buff = (8, 8, 8)
        self.network = self.network.to(image.device, non_blocking=True)
        if permute and image.ndim == 3:
            image = image.permute((2, 0, 1)).unsqueeze(0)
            band_dim = -3
            row_dim = -2
            col_dim = -1
            image = image.unsqueeze(0)

        if normalize:
            image, consts = utils.normalize(image, by_band=True, band_dim=band_dim)

        self.band_window = 10  # image.shape[-1]
        # todo: make class variable -> buffer
        ws = 128

        # design this to move over windows (like conv groups)
        # assume band_dim = -1
        # min_dim = 0 if image.shape[0] < image.shape[1] else 1
        # more windows:

        # dim_r_ws = (sh_r + buff[0]) // ws

        dim_r_starts = [ws, ws - buff[0]]
        if dim_r_starts[-1] >= sh_r or ws >= sh_r:
            # if the second window starts after the image ends
            dim_r_starts = [0]
        else:
            while dim_r_starts[-1] < sh_r:
                # layer the images over one another to avoid tiling (hopfully)
                dim_r_starts.append(dim_r_starts[-1] + ws - buff[0])
            dim_r_starts = dim_r_starts[:-1]
        # replace the first item with 0 (it currently has the end of the first window)
        dim_r_starts[0] = 0

        dim_c_starts = [ws, ws - buff[1]]
        if dim_c_starts[-1] >= sh_c or ws >= sh_c:
            # if the second window starts after the image ends
            dim_c_starts = [0]
        else:
            while dim_c_starts[-1] < sh_c:
                # layer the images over one another to avoid tiling (hopfully)
                dim_c_starts.append(dim_c_starts[-1] + ws - buff[1])
            dim_c_starts = dim_c_starts[:-1]
        # replace the first item with 0 (it currently has the end of the first window)
        dim_c_starts[0] = 0

        bw = self.band_window
        dim_b_starts = [bw, bw - buff[2]]
        if dim_b_starts[-1] >= sh_b or bw >= sh_b:
            # if the second window starts after the image ends
            dim_b_starts = [0]
        else:
            while dim_b_starts[-1] < sh_b:
                # layer the images over one another to avoid tiling (hopfully)
                dim_b_starts.append(dim_b_starts[-1] + bw - buff[2])
            dim_b_starts = dim_b_starts[:-1]
        # replace the first item with 0 (it currently has the end of the first window)
        dim_b_starts[0] = 0

        # dim_r_ws = 1 + ((sh_r - ws + buff[0]) // ws)
        # dim_r_rem = abs(sh_r - (dim_r_ws * ws) - buff[0])  # (sh_r + buff[_r]) % ws
        #
        # dim_c_ws = 1 + ((sh_c - ws + buff[1]) // ws)  # (sh_c + buff[1]) // ws
        # dim_c_rem = abs(sh_c - (dim_c_ws * ws) - buff[1])  # (sh1 + buff[1]) % ws
        #
        # bw = self.band_window
        # dim_b_ws = 1 + ((sh_b - bw + buff[2]) // bw)  # (sh_b + buff[2]) // bw
        # dim_b_rem = abs(sh_b - (dim_b_ws * bw) - buff[2])
        # print(dim_r_ws, dim_r_rem, dim_c_ws, dim_c_rem, dim_b_ws, dim_b_rem)

        out = torch.zeros_like(image)
        # dim _r is outside loop -> dim_c is inside loop -> dim_b on last loop
        for beg_r in dim_r_starts:
            sl = [
                slice(None),
            ] * image.ndim
            # create overlap
            end_r = beg_r + ws
            sl[row_dim] = slice(beg_r, end_r)

            # set_sl = [slice(None)] * image.ndim

            for beg_c in dim_c_starts:
                # create overlap
                end_c = beg_c + ws
                sl[col_dim] = slice(beg_c, end_c)

                for beg_b in dim_b_starts:
                    # create overlap
                    end_b = beg_b + bw
                    sl[band_dim] = slice(beg_b, end_b)
                    # print(sl)

                    out[sl] = self._call_nn(image[sl])

        if permute:
            out = out.squeeze().permute((1, 2, 0))
            band_dim = -1

        if normalize:
            out = utils.undo_normalize(out, **consts, by_band=True, band_dim=band_dim)

        return out

    def _call_nn(self, image):
        if self.frozen:
            with torch.no_grad():
                return self.network(image)
        else:
            return self.network(image)

    def _forward(self, image, band_dim, normalize, permute):
        if permute and band_dim == -1 and image.ndim == 3:
            image = image.permute((2, 0, 1)).unsqueeze(0)
            band_dim = -3

        if image.ndim == 4:
            image = image.unsqueeze(0)

        if normalize:
            image, consts = utils.normalize(image, by_band=True, band_dim=band_dim)

        self.network = self.network.to(image.device)
        ret = self._call_nn(image)

        if permute:
            ret = ret.squeeze().permute((1, 2, 0))
            band_dim = -1

        if normalize:
            ret = utils.undo_normalize(ret, **consts, by_band=True, band_dim=band_dim)

        return ret


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
