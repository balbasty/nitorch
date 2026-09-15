"""Vendored 3D U-Net matching the published anatomix checkpoint's architecture.

This is a minimal, dependency-light reimplementation (only `torch` is
required) of the U-Net used by anatomix (Dey et al.), so that nitorch does
not need to depend on the upstream `anatomix` package (which pulls in MONAI,
TorchIO, and other training-time dependencies unnecessary for pure
frozen-weight inference). See ``specs/001-anatomix-registration-features/
research.md`` (decision 4) for the rationale.
"""

import torch
from torch import nn as tnn


def _norm_layer(norm, channels):
    """Instantiate a normalization layer.

    Parameters
    ----------
    norm : {'batch', 'instance', 'none'}
    channels : int

    Returns
    -------
    torch.nn.Module

    """
    if norm == 'batch':
        return tnn.BatchNorm3d(channels)
    elif norm == 'instance':
        return tnn.InstanceNorm3d(channels)
    elif norm == 'none':
        return tnn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm!r} "
                          f"(expected 'batch', 'instance' or 'none')")


def _pool_layer(pooling):
    """Instantiate a stride-2 pooling layer.

    Parameters
    ----------
    pooling : {'max', 'avg'}

    Returns
    -------
    torch.nn.Module

    """
    if pooling == 'max':
        return tnn.MaxPool3d(kernel_size=2, stride=2)
    elif pooling == 'avg':
        return tnn.AvgPool3d(kernel_size=2, stride=2)
    else:
        raise ValueError(f"Unknown pooling type: {pooling!r} "
                          f"(expected 'max' or 'avg')")


class ConvBlock(tnn.Module):
    """Two (Conv3d -> Norm -> ReLU) layers, preserving spatial shape.

    Convolutions carry no bias (redundant ahead of a normalization layer),
    matching the published anatomix checkpoint.
    """

    def __init__(self, in_channels, out_channels, norm='batch'):
        super().__init__()
        self.conv1 = tnn.Conv3d(in_channels, out_channels, kernel_size=3,
                                padding=1, bias=False)
        self.norm1 = _norm_layer(norm, out_channels)
        self.conv2 = tnn.Conv3d(out_channels, out_channels, kernel_size=3,
                                padding=1, bias=False)
        self.norm2 = _norm_layer(norm, out_channels)
        self.act = tnn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class AnatomixUNet(tnn.Module):
    """Parameterized 3D U-Net matching anatomix's published architecture.

    A single (Conv3d -> Norm -> ReLU) "stem" maps `input_nc` to `ngf`
    channels; encoder/decoder channel counts then double at each of
    `num_downs` levels. Skip connections concatenate encoder features into
    the corresponding decoder level. A final, bias-free 3x3x3 convolution
    (no norm/activation after it) maps the last decoder level to
    `output_nc` raw feature channels.

    Parameters
    ----------
    input_nc : int, default=1
        Number of input (image) channels.
    output_nc : int, default=16
        Number of output feature channels.
    num_downs : int, default=4
        Number of downsampling levels.
    ngf : int, default=16
        Number of channels at the first encoder level; doubles at each
        subsequent level.
    norm : {'batch', 'instance', 'none'}, default='batch'
        Normalization layer type.
    interp : {'nearest', 'trilinear'}, default='nearest'
        Decoder upsampling mode.
    pooling : {'max', 'avg'}, default='max'
        Encoder pooling type.

    """

    def __init__(self, input_nc=1, output_nc=16, num_downs=4, ngf=16,
                 norm='batch', interp='nearest', pooling='max'):
        super().__init__()
        if interp not in ('nearest', 'trilinear'):
            raise ValueError(f"Unknown interp mode: {interp!r} "
                              f"(expected 'nearest' or 'trilinear')")
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_downs = num_downs
        self.ngf = ngf
        self.norm = norm
        self.interp = interp
        self.pooling = pooling

        enc_channels = [ngf * (2 ** i) for i in range(num_downs)]
        bottleneck_channels = ngf * (2 ** num_downs)

        self.stem = tnn.Sequential(
            tnn.Conv3d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, ngf),
            tnn.ReLU(inplace=True),
        )

        self.encoders = tnn.ModuleList()
        self.pools = tnn.ModuleList()
        in_ch = ngf
        for out_ch in enc_channels:
            self.encoders.append(ConvBlock(in_ch, out_ch, norm=norm))
            self.pools.append(_pool_layer(pooling))
            in_ch = out_ch

        self.bottleneck = ConvBlock(in_ch, bottleneck_channels, norm=norm)

        self.upsamples = tnn.ModuleList()
        self.decoders = tnn.ModuleList()
        in_ch = bottleneck_channels
        for skip_ch in reversed(enc_channels):
            self.upsamples.append(
                tnn.Upsample(scale_factor=2, mode=interp,
                             align_corners=None if interp == 'nearest' else False)
            )
            self.decoders.append(ConvBlock(in_ch + skip_ch, skip_ch, norm=norm))
            in_ch = skip_ch

        self.out_conv = tnn.Conv3d(in_ch, output_nc, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        factor = 2 ** self.num_downs
        bad = [s for s in x.shape[2:] if s % factor != 0]
        if bad:
            raise ValueError(
                f"AnatomixUNet with num_downs={self.num_downs} requires "
                f"every spatial dimension to be divisible by {factor} "
                f"(so encoder/decoder feature maps line up at each skip "
                f"connection); got spatial shape {tuple(x.shape[2:])}. Pad "
                f"or crop the input, or use a smaller `num_downs`."
            )

        x = self.stem(x)
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.out_conv(x)
