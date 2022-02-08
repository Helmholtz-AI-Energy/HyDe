import torch.nn as nn

from . import qrnn3d


class ResQRNN3D(nn.Module):
    def __init__(self, in_channels, channels, n_resblocks):
        super(ResQRNN3D, self).__init__()

        bn = True
        act = "tanh"

        # define head module
        m_head = [qrnn3d.BiQRNNConv3D(in_channels, channels, bn=bn, act=act)]

        # define body module
        m_body = [ResBlock(qrnn3d.QRNNConv3D, channels, bn=bn, act=act) for _ in range(n_resblocks)]

        # define tail module
        m_tail = [qrnn3d.BiQRNNConv3D(channels, in_channels, bn=bn, act="none")]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class ResBlock(nn.Module):
    def __init__(self, block, channels, **kwargs):
        super(ResBlock, self).__init__()
        self.layer1 = block(channels, channels, **kwargs)
        self.layer2 = block(channels, channels, **kwargs)

    def forward(self, x, reverse=False):
        res = self.layer1(x, reverse)
        res = self.layer2(x, not reverse)
        res += x

        return res
