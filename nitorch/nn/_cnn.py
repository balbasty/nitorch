# -*- coding: utf-8 -*-
"""Convolutional neural networks."""

import torch
from torch import nn as tnn
from ._conv import Conv
from collections import OrderedDict


class Encoder(tnn.ModuleList):
    """Encoder network (for U-nets, VAEs, etc.)"""

    def __init__(self, dim, channels, kernel_size=3, skip=False,
                 activation=tnn.ReLU):
        """

        Args:
            dim (int): Dimension (1|2|3)
            channels (list): Number of channels in each layer
            kernel_size (int or list[int], optional): Kernel size per
                dimension. Default: 3
            skip (bool, optional): Handle skip connections.
                If yes, ``forward`` returns  all output layers in a tuple.
            activation (type or function, optional): Constructor of an
                activation function. Default: ``torch.nn.ReLU``
        """
        self.dim = dim
        self.skip = skip
        input_channels = channels[:-1]
        output_channels = channels[1:]
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


class Decoder(tnn.ModuleList):
    """Decoder network (for U-nets, VAEs, etc.)"""

    def __init__(self, dim, channels, kernel_size=3, skip=False,
                 activation=tnn.ReLU, encoder=None):
        """

        Args:
            dim (int): Dimension (1|2|3)
            channels (list): Number of channels in each decoding layer
            kernel_size (int or list[int], optional): Kernel size per
                dimension. Default: 3
            skip (bool, optional): Handle skip connections.
                If yes, ``forward`` returns  all output layers in a tuple.
            activation (type or function, optional): Constructor of an
                activation function. Default: ``torch.nn.ReLU``
            encoder (list, optional): If skip == True, number of
                channels in each encoding layer. This argument is
                necessary to determine the number of input/output
                channels in each convolutional layer.
        """
        self.dim = dim
        self.skip = skip
        input_channels = channels[:-1]
        output_channels = channels[1:]
        if skip and encoder is not None:
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


class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(self, dim, input_channels, output_channels,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.ReLU):
        """

        Args:
            dim (int): Dimension (1|2|3)
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            encoder (optional, list[int]): Number of channels in eac
                encoding layer. Default: [16, 32, 32, 32]
            decoder (optional, list[int]): Number of channels in each
                decoding layer. Default: [32, 32, 32, 32, 32, 16, 16]
            kernel_size (int or list[int], optional): Kernel size per
                dimension. Default: 3
            activation (type or function, optional): Constructor of an
                activation function. Default: ``torch.nn.ReLU``
        """
        self.dim = dim

        if encoder is None:
            encoder = [16, 32, 32, 32]
        encoder = [input_channels] + encoder
        if decoder is None:
            decoder = [32, 32, 32, 32, 32, 16, 16]
        decoder = decoder + [output_channels]

        enc = Encoder(dim, encoder, kernel_size=kernel_size,
                      skip=True, activation=activation)
        dec = Decoder(dim, decoder, kernel_size=kernel_size,
                      skip=True, encoder=encoder, activation=activation)
        super().__init__(OrderedDict([('encoder', enc), ('decoder', dec)]))


