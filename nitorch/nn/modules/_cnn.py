"""Convolutional neural networks (UNet, VAE, etc.)."""

import torch
from torch import nn as tnn
from ._base import nitorchmodule, Module
from ._conv import (Conv, ConvZeroCentre)
from ._reduction import reductions, Reduction
from nitorch.core.pyutils import make_list
from collections import OrderedDict
import inspect
import math
from .. import check


@nitorchmodule
class Encoder(tnn.ModuleList):
    """Encoder network (for U-nets, VAEs, etc.)"""

    def __init__(self, dim, channels, kernel_size=3, skip=False,
                 activation=tnn.ReLU, batch_norm=False):
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
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.

        """
        self.dim = dim
        self.skip = skip
        input_channels = list(channels)[:-1]
        output_channels = list(channels)[1:]
        modules = []
        for (i, o) in zip(input_channels, output_channels):
            modules.append(Conv(dim, i, o, kernel_size, stride=2,
                                activation=activation, batch_norm=batch_norm))
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
                 activation=tnn.ReLU, batch_norm=False, encoder=None):
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
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
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
                                transposed=True, activation=activation,
                                batch_norm=batch_norm))
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

    def __init__(self, dim, channels, kernel_size=3,
                 activation=tnn.ReLU, final_activation='same',
                 batch_norm=False):
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
        final_activation : str or type or callable, default='same'
            Final activation function. If 'same', same as `activation`.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        self.dim = dim
        input_channels = channels[:-1]
        output_channels = channels[1:]
        modules = []
        kernel_size = make_list(kernel_size, dim)
        nb_module = len(input_channels)
        for l, (i, o) in enumerate(zip(input_channels, output_channels)):
            pad = [(k-1)//2 for k in kernel_size]
            if l == nb_module-1 and final_activation != 'same':
                activation = final_activation
            modules.append(Conv(dim, i, o, kernel_size, padding=pad,
                                stride=1, activation=activation,
                                batch_norm=batch_norm))
        super().__init__(*modules)


@nitorchmodule
class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(self, dim, input_channels, output_channels,
                 encoder=None, decoder=None, kernel_size=3,
                 activation=tnn.ReLU, final_activation='same',
                 batch_norm=False):
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
        final_activation : str or type or callable, default='same'
            Final activation function. If 'same', same as `activation`.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
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
                      skip=True, activation=activation,
                      batch_norm=batch_norm)
        modules.append(('encoder', enc))
        dec = Decoder(dim, decoder, kernel_size=kernel_size, skip=True,
                      encoder=encoder, activation=activation,
                      batch_norm=batch_norm)
        modules.append(('decoder', dec))
        # there should always be at least one stacked conv
        stack[0] += encoder[0]
        stk = StackedConv(dim, stack, kernel_size=kernel_size,
                          activation=activation,
                          final_activation=final_activation,
                          batch_norm=batch_norm)
        modules.append(('stack', stk))
        super().__init__(OrderedDict(modules))


@nitorchmodule
class CNN(tnn.Sequential):
    """Encoding convolutional network (for classification or regression)."""

    def __init__(self, dim, input_channels, output_channels,
                 encoder=None, stack=None, kernel_size=3, reduction='max',
                 activation='relu', final_activation='same', batch_norm=False):
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
        stack : sequence[int], default=[32, 16, 16]
            Number of channels in each fully-connected layer.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        reduction : str or type or callable, default='max'
            Reduction function, that transitions between the encoding
            layers and fully connected layers.
        activation : str or type or callable, default='relu'
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
        final_activation : str or type or callable, default='same'
            Final activation function. If 'same', same as `activation`.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        self.dim = dim

        encoder = list(encoder or [16, 32, 32, 32])
        stack = list(stack or [32, 16, 16])
        encoder = [input_channels] + encoder
        stack = encoder[-1:] + stack + [output_channels]
        if isinstance(reduction, str):
            reduction = reductions.get(reduction, None)
        reduction = reduction(keepdim=True) if inspect.isclass(reduction) \
                    else reduction if callable(reduction) \
                    else None
        if not isinstance(reduction, Reduction):
            raise TypeError('reduction must be a `Reduction` module.')

        modules = []
        enc = Encoder(dim, encoder, kernel_size=kernel_size,
                      activation=activation, batch_norm=batch_norm)
        modules.append(('encoder', enc))
        modules.append(('reduction', reduction))
        stk = StackedConv(dim, stack, kernel_size=1,
                          activation=activation,
                          final_activation=final_activation)
        modules.append(('stack', stk))
        super().__init__(OrderedDict(modules))


class MRF(Module):
    """MRF network"""

    def __init__(self, dim, num_classes, num_iter=20, num_filters=16, num_extra=0,
                 kernel_size=3, activation=tnn.LeakyReLU(0.1), batch_norm=False,
                 w=0.5):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        num_classes : int
            Number of input classes.
        num_iter : int, default=20
            Number of mean-field iterations.
        num_extra : int, default=0
            Number of extra layers between MRF layer and final layer.
        num_filters : int, default=16
            Number of conv filters in first, MRF layer.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        activation : str or type or callable, default='tnn.LeakyReLU(0.1)'
            Activation function.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        w : float, default=0.5
            Weight between new and old prediction [0, 1].

        """
        super().__init__()

        self.dim = dim
        self.num_iter = num_iter
        if w < 0 or w > 1:
            raise ValueError('Parameter w should be between 0 and 1, got {w}'.format(w))
        self.w = w
        if num_classes == 1:
            final_activation = tnn.Sigmoid
        else:
            final_activation = tnn.Softmax(dim=1)
        # make layers
        layers = []
        p = ((kernel_size - 1) // 2,)*self.dim
        layer = ConvZeroCentre(dim, in_channels=num_classes, out_channels=num_filters,
                               kernel_size=kernel_size, activation=activation,
                               batch_norm=batch_norm, bias=False, padding=p)
        layers.append(('mrf', layer))
        for i in range(num_extra):
            layer = Conv(dim, in_channels=num_filters, out_channels=num_filters,
                         kernel_size=1, activation=activation, batch_norm=batch_norm,
                         bias=False)
            layers.append(('extra', layer))
        layer = Conv(dim, in_channels=num_filters, out_channels=num_classes,
                     kernel_size=1, activation=final_activation, batch_norm=batch_norm,
                     bias=False)
        layers.append(('final', layer))
        # build model
        self.layers = tnn.Sequential(OrderedDict(layers))
        # register loss tag
        self.tags = ['mrf']

    def forward(self, seg, ref=None, *, _loss=None, _metric=None):
        """Forward pass, with mean-field iterations.
        """
        seg = torch.as_tensor(seg)

        # sanity check
        check.dim(self.dim, seg)

        # mrf
        with torch.no_grad():
            if self.train():
                # Training: variable number of iterations
                num_iter = int(torch.LongTensor(1).random_(1, self.num_iter))
            else:
                # Testing: fixed number of iterations
                num_iter = self.num_iter
        for i in range(num_iter):
            oseg = seg.clone()
            for layer in self.layers:
                seg = layer(seg)
            seg = self.w*seg + (1 - self.w)*oseg

        # compute loss and metrics
        if ref is not None:
            # sanity checks
            check.dim(self.dim, ref)
            dims = [0] + list(range(2, self.dim+2))
            check.shape(seg, ref, dims=dims)
            self.compute(_loss, _metric, mrf=[seg, ref])

        return seg
