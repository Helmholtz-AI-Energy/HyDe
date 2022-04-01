"""MemNet"""
import torch
import torch.nn as nn

from ... import lowlevel, transform_domain


class MemNet(nn.Module):
    def __init__(
        self, in_channels, channels, num_memblock, num_resblock, conv3d=False, hyres=False
    ):
        super(MemNet, self).__init__()
        if conv3d:
            in_channels = 1
        self.feature_extractor = BNReLUConv(in_channels, channels, conv3d=conv3d)
        self.reconstructor = BNReLUConv(channels, in_channels, conv3d=conv3d)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1, conv3d=conv3d) for i in range(num_memblock)]
        )
        self.freeze_bn = True
        self.freeze_bn_affine = True
        self.conv3d = conv3d
        if hyres:
            self.hyres = transform_domain.HyRes()
        else:
            self.hyres = None

    def forward(self, x):
        squeezed = False
        if x.ndim == 5 and not self.conv3d:
            x = x.squeeze(1)
            squeezed = True

        # with torch.no_grad()
        # current shape: [batch, band, h, w]
        if self.hyres is not None:
            for b in range(x.shape[0]):
                i = x[b].squeeze().permute((1, 2, 0))
                ret = self.hyres(i).permute((2, 0, 1))
                if self.conv3d:
                    ret = ret.unsqueeze(0)
                x[b] = ret

        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)

        out = out + residual
        if squeezed:
            out = out.unsqueeze(1)
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock, num_memblock, conv3d=False):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels, conv3d=conv3d) for _ in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv(
            (num_resblock + num_memblock) * channels, channels, 1, 1, 0, conv3d=conv3d
        )

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.gate_unit(torch.cat(xs + ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1, conv3d=False):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p, conv3d=conv3d)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p, conv3d=conv3d)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True, conv3d=False):
        super(BNReLUConv, self).__init__()
        if not conv3d:
            self.add_module("bn", nn.BatchNorm2d(in_channels))
            self.add_module("relu", nn.ReLU(inplace=inplace))
            self.add_module("conv", nn.Conv2d(in_channels, channels, k, s, p, bias=False))
        else:
            self.add_module("bn", nn.BatchNorm3d(in_channels))
            self.add_module("relu", nn.ReLU(inplace=inplace))
            self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=False))
