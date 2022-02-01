from functools import partial

import torch
import torch.nn as nn

from . import combinations

"""F pooling"""


class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act="tanh"):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == "tanh":
            return Z.tanh(), F.sigmoid()
        elif self.act == "relu":
            return Z.relu(), F.sigmoid()
        elif self.act == "none":
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []

        if not reverse:
            for time, (z, f) in enumerate(
                zip(Z.split(1, 2), F.split(1, 2))
            ):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate(
                (zip(reversed(Z.split(1, 2)), reversed(F.split(1, 2))))
            ):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        # return concatenated hidden states
        return torch.cat(h_time, dim=2)

    def extra_repr(self):
        return "act={}".format(self.act)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == "tanh":
            return Z.tanh(), F1.sigmoid(), F2.sigmoid()
        elif self.act == "relu":
            return Z.relu(), F1.sigmoid(), F2.sigmoid()
        elif self.act == "none":
            return Z, F1.sigmoid(), F2.sigmoid()
        else:
            raise NotImplementedError

    def forward(self, inputs, fname=None):
        h = None
        Z, F1, F2 = self._conv_step(inputs)
        hsl = []
        hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsl.append(h)

        h = None
        for time, (z, f) in enumerate(
            (zip(reversed(zs), reversed(F2.split(1, 2))))
        ):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)

        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        if fname is not None:
            stats_dict = {"z": Z, "fl": F1, "fr": F2, "hsl": hsl, "hsr": hsr}
            torch.save(stats_dict, fname)
        return hsl + hsr


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act="tanh"):
        super(BiQRNNConv3D, self).__init__(
            in_channels,
            hidden_channels,
            combinations.BasicConv3d(in_channels, hidden_channels * 3, k, s, p, bn=bn),
            act=act,
        )


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(
        self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False, bn=True, act="tanh"
    ):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels,
            hidden_channels,
            combinations.BasicDeConv3d(in_channels, hidden_channels * 3, k, s, p, bias=bias, bn=bn),
            act=act,
        )


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act="tanh"):
        super(QRNNConv3D, self).__init__(
            in_channels,
            hidden_channels,
            combinations.BasicConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn),
            act=act,
        )


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act="tanh"):
        super(QRNNDeConv3D, self).__init__(
            in_channels,
            hidden_channels,
            combinations.BasicDeConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn),
            act=act,
        )


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(
        self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1, 2, 2), bn=True, act="tanh"
    ):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels,
            hidden_channels,
            combinations.BasicUpsampleConv3d(
                in_channels, hidden_channels * 2, k, s, p, upsample, bn=bn
            ),
            act=act,
        )


class QRNNREDC3D(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_half_layer,
        sample_idx,
        feature_extractor=None,  # BiQRNNConv3D
        reconstructor=None,  # BiQRNNDeConv3D
        encoder=None,  # QRNN3DEncoder
        decoder=None,  # QRNN3DDecoder
        is_2d=False,
        has_ad=True,
        bn=True,
        act="tanh",
        plain=False,
    ):
        super(QRNNREDC3D, self).__init__()
        assert sample_idx is None or isinstance(sample_idx, list)

        self.enable_ad = has_ad
        if sample_idx is None:
            sample_idx = []
        if is_2d:
            self.feature_extractor = feature_extractor(
                in_channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act
            )
        else:
            self.feature_extractor = feature_extractor(in_channels, channels, bn=bn, act=act)

        self.encoder = encoder(
            channels,
            num_half_layer,
            sample_idx,
            is_2d=is_2d,
            has_ad=has_ad,
            bn=bn,
            act=act,
            plain=plain,
        )
        self.decoder = decoder(
            channels * (2 ** len(sample_idx)),
            num_half_layer,
            sample_idx,
            is_2d=is_2d,
            has_ad=has_ad,
            bn=bn,
            act=act,
            plain=plain,
        )

        if act == "relu":
            act = "none"

        if is_2d:
            self.reconstructor = reconstructor(
                channels, in_channels, bias=True, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act
            )
        else:
            self.reconstructor = reconstructor(channels, in_channels, bias=True, bn=bn, act=act)

    def forward(self, x):
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        if self.enable_ad:
            out, reverse = self.encoder(out, xs, reverse=False)
            out = self.decoder(out, xs, reverse=(reverse))
        else:
            out = self.encoder(out, xs)
            out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.reconstructor(out)
        out = out + xs.pop()
        return out


class QRNN3DEncoder(nn.Module):
    def __init__(
        self,
        channels,
        num_half_layer,
        sample_idx,
        encoder_layer=None,  # QRNNConv3D
        is_2d=False,
        has_ad=True,
        bn=True,
        act="tanh",
        plain=False,
    ):
        super(QRNN3DEncoder, self).__init__()
        # Encoder
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in range(num_half_layer):
            if i not in sample_idx:
                if is_2d:
                    encoder_layer = encoder_layer(
                        channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act
                    )
                else:
                    encoder_layer = encoder_layer(channels, channels, bn=bn, act=act)
            else:
                if is_2d:
                    encoder_layer = encoder_layer(
                        channels,
                        2 * channels,
                        k=(1, 3, 3),
                        s=(1, 2, 2),
                        p=(0, 1, 1),
                        bn=bn,
                        act=act,
                    )
                else:
                    if not plain:
                        encoder_layer = encoder_layer(
                            channels, 2 * channels, k=3, s=(1, 2, 2), p=1, bn=bn, act=act
                        )
                    else:
                        encoder_layer = encoder_layer(
                            channels, 2 * channels, k=3, s=(1, 1, 1), p=1, bn=bn, act=act
                        )

                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse=False):
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer - 1):
                x = self.layers[i](x)
                xs.append(x)
            x = self.layers[-1](x)

            return x
        else:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer - 1):
                x = self.layers[i](x, reverse=reverse)
                reverse = not reverse
                xs.append(x)
            x = self.layers[-1](x, reverse=reverse)
            reverse = not reverse

            return x, reverse


class QRNN3DDecoder(nn.Module):
    def __init__(
        self,
        channels,
        num_half_layer,
        sample_idx,
        decoder_layer=None,  # QRNNDeConv3D
        up_sample_decoder=None,  # QRNNUpsampleConv3d
        is_2d=False,
        has_ad=True,
        bn=True,
        act="tanh",
        plain=False,
    ):
        super(QRNN3DDecoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                if is_2d:
                    decoder_layer = decoder_layer(
                        channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act
                    )
                else:
                    decoder_layer = decoder_layer(channels, channels, bn=bn, act=act)
            else:
                if is_2d:
                    decoder_layer = up_sample_decoder(
                        channels, channels // 2, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act
                    )
                else:
                    if not plain:
                        decoder_layer = up_sample_decoder(channels, channels // 2, bn=bn, act=act)
                    else:
                        decoder_layer = decoder_layer(channels, channels // 2, bn=bn, act=act)

                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs, reverse=False):
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            x = self.layers[0](x)
            for i in range(1, num_half_layer):
                x = x + xs.pop()
                x = self.layers[i](x)
            return x
        else:
            num_half_layer = len(self.layers)
            x = self.layers[0](x, reverse=reverse)
            reverse = not reverse
            for i in range(1, num_half_layer):
                x = x + xs.pop()
                x = self.layers[i](x, reverse=reverse)
                reverse = not reverse
            return x


QRNN3DEncoder = partial(QRNN3DEncoder, QRNNConv3D=QRNNConv3D)

QRNN3DDecoder = partial(
    QRNN3DDecoder, QRNNDeConv3D=QRNNDeConv3D, QRNNUpsampleConv3d=QRNNUpsampleConv3d
)

QRNNREDC3D = partial(
    QRNNREDC3D,
    BiQRNNConv3D=BiQRNNConv3D,
    BiQRNNDeConv3D=BiQRNNDeConv3D,
    QRNN3DEncoder=QRNN3DEncoder,
    QRNN3DDecoder=QRNN3DDecoder,
)
