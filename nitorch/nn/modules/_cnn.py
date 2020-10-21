"""Convolutional neural networks (UNet, VAE, etc.)."""

import torch
from torch import nn as tnn
from ._conv import Conv
from ._base import nitorchmodule
from nitorch.core.pyutils import make_list
from collections import OrderedDict


@nitorchmodule
class Encoder(tnn.ModuleList):
    """Encoder network (for U-nets, VAEs, etc.)"""

    def __init__(self, dim, channels, kernel_size=3, skip=False,
                 activation=tnn.ReLU):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        channels : sequence[int]
            Number of channels in each encoding layer.
            There will be len(channels)-1 layers.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        skip : bool, default=False
            Handle skip connections.
            If True, ``forward`` returns  all output layers in a tuple.
        activation : str or type or callable, default='relu'
            Activation function.

        """
        self.dim = dim
        self.skip = skip
        input_channels = list(channels)[:-1]
        output_channels = list(channels)[1:]
        modules = []
        for (i, o) in zip(input_channels, output_channels):
            modules.append(Conv(dim, i, o, kernel_size,
                                stride=2, activation=activation))
        super().__init__(modules)

    def forward(self, x):
        if self.skip:
            return self._forward_skip(x)
        else:
            return self._forward_simple(x)

    def _forward_simple(self, x):
        for layer in self:
            x = layer(x)
        return x

    def _forward_skip(self, x):
        output = [x]
        for layer in self:
            output.append(layer(output[-1]))
        return output[::-1]


@nitorchmodule
class Decoder(tnn.ModuleList):
    """Decoder network (for U-nets, VAEs, etc.)"""

    def __init__(self, dim, channels, kernel_size=3, skip=False,
                 activation=tnn.ReLU, encoder=None):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        channels : sequence[int]
            Number of channels in each decoding layer.
            There will be len(channels)-1 layers.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        activation : str or type or callable, default='relu'
            Activation function.
        encoder : sequence[int], optional
            Number of channels in each encoding layer.
            In a UNet, this argument is necessary to determine the
            number of input/output channels in each convolutional layer.
        """
        self.dim = dim
        self.skip = bool(encoder)
        input_channels = list(channels)[:-1]
        output_channels = list(channels)[1:]
        if self.skip:
            encoder = list(encoder)
            for i, e in enumerate(encoder[-2::-1], 1):
                if i >= len(input_channels):
                    break
                input_channels[i] += e
        modules = []
        for (i, o) in zip(input_channels, output_channels):
            modules.append(Conv(dim, i, o, kernel_size=kernel_size, stride=2,
                                transposed=True, activation=activation))
        super().__init__(modules)

    def forward(self, x):
        if self.skip:
            return self._forward_skip(x)
        else:
            return self._forward_simple(x)

    def _forward_simple(self, x):
        for layer in self:
            x = layer(x)
        return x

    def _forward_skip(self, x):
        inputs = list(x)
        x = inputs.pop(0)
        if len(self) < len(inputs):
            raise TypeError('Not enough decoding layers. Found {}, '
                            'need {}'.format(len(self.decoder), len(inputs)))
        decoder = list(self.children())[:len(inputs)]
        postproc = list(self.children())[len(inputs):]

        # Decoder
        for layer in decoder:
            y = inputs.pop(0)
            oshape = layer.shape(x, stride=2, output_padding=0)[2:]
            yshape = y.shape[2:]
            pad = [i-o for o, i in zip(oshape, yshape)]
            x = layer(x, stride=2, output_padding=pad)
            x = torch.cat((x, y), dim=1)

        # Post-processing (convolutions without upsampling)
        for layer in postproc:
            pad = [((k-1)*d)//2 for k, d in zip(layer.conv.kernel_size,
                                                layer.conv.dilation)]
            x = layer(x, stride=1, padding=pad)

        return x


@nitorchmodule
class StackedConv(tnn.Sequential):
    """Stacked convolutions, without up/down-sampling."""

    def __init__(self, dim, channels, kernel_size=3, activation=tnn.ReLU):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        channels : sequence[int]
            Number of channels in each convolutional layer.
            There will be len(channels)-1 layers.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        activation : str or type or callable, default='relu'
            Activation function.
        """
        self.dim = dim
        input_channels = channels[:-1]
        output_channels = channels[1:]
        modules = []
        kernel_size = make_list(kernel_size, dim)
        for (i, o) in zip(input_channels, output_channels):
            pad = [(k-1)//2 for k in kernel_size]
            modules.append(Conv(dim, i, o, kernel_size, padding=pad,
                                stride=1, activation=activation))
        super().__init__(*modules)


@nitorchmodule
class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(self, dim, input_channels, output_channels,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.ReLU):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        encoder : sequence[int], default=[16, 32, 32, 32]
            Number of channels in each encoding layer.
        decoder : sequence[int], default=[32, 32, 32, 32, 32, 16, 16]
            Number of channels in each decoding layer.
            If the number of decoding layer is larger than the number of
            encoding layers, stacked convolutions are appended to the
            UNet.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        activation : str or type or callable, default='relu'
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
        """
        self.dim = dim

        encoder = list(encoder or [16, 32, 32, 32])
        decoder = list(decoder or [32, 32, 32, 32, 32, 16, 16])
        if len(decoder) < len(encoder):
            # we need as many upsampling steps as downsampling steps
            decoder = make_list(decoder, len(encoder))
        encoder = [input_channels] + encoder
        decoder = encoder[-1:] + decoder + [output_channels]
        stack = decoder[len(encoder)-1:]
        decoder = decoder[:len(encoder)]

        modules = []
        enc = Encoder(dim, encoder, kernel_size=kernel_size,
                      skip=True, activation=activation)
        modules.append(('encoder', enc))
        dec = Decoder(dim, decoder, kernel_size=kernel_size,
                      skip=True, encoder=encoder, activation=activation)
        modules.append(('decoder', dec))
        if len(stack) > 1:
            stack[0] += encoder[0]
            stk = StackedConv(dim, stack, kernel_size=kernel_size,
                              activation=activation)
            modules.append(('stack', stk))
        super().__init__(OrderedDict(modules))

