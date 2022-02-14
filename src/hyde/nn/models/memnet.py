"""MemNet"""
import torch
import torch.nn as nn


class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock, conv3d=False):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels, conv3d=conv3d)
        self.reconstructor = BNReLUConv(channels, in_channels, conv3d=conv3d)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1, conv3d=conv3d) for i in range(num_memblock)]
        )
        self.freeze_bn = True
        self.freeze_bn_affine = True

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(1)
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)

        out = out + residual

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
