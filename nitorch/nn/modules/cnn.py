"""Convolutional neural networks (UNet, VAE, etc.)."""

import torch
from torch import nn as tnn
from .base import nitorchmodule, Module
from .conv import Conv
from .pool import Pool
from .reduction import reductions, Reduction
from nitorch.core.py import make_list, flatten
from nitorch.core.linalg import matvec
from nitorch.core.utils import movedim
from collections import OrderedDict
import inspect
import math


def interleaved_cat(tensors, dim=0, groups=1):
    if groups == 1:
        return torch.cat(tensors, dim=dim)
    tensors = [t.chunk(groups, dim) for t in tensors]
    tensors = [chunk for chunks in zip(*tensors) for chunk in chunks]
    return torch.cat(tensors, dim=dim)


class Stitch(Module):
    """Stitch tensors together using a learnt linear combination.

    This operation can be been as a 1D convolution, where weights are
    shared across some channels.

    References
    ----------
    ..[1] "Cross-stitch Networks for Multi-task Learning"
          Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert
          CVPR 2016
    """

    def __init__(self, input_groups=2, output_groups=2):
        """

        Parameters
        ----------
        input_groups : int, default=2
            Number of input groups
        output_groups : int, default=2
            Number of output groups
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(input_groups, output_groups))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        """

        Parameters
        ----------
        input : (B, channels_in, *spatial) tensor
            Input tensor
            `channels_in` must be divisible by `input_groups`.

        Returns
        -------
        output : (B, channel_out, *spatial) tensor
            Output stitched tensor
            `channel_out = (channels_in//input_groups)*output_groups`

        """
        # shapes
        batch = input.shape[0]
        nb_channels = input.shape[1]
        spatial = input.shape[2:]
        input_groups = self.weight.shape[0]
        output_groups = self.weight.shape[1]
        input_split = nb_channels//input_groups
        out_channels = input_split * output_groups

        # linear combination
        input = input.unfold(1, input_split, input_split).transpose(1, -1)
        input = matvec(self.weight.t(), input)
        input = movedim(input, -1, 1).reshape([batch, out_channels, *spatial])

        return input


@nitorchmodule
class Encoder(tnn.ModuleList):
    """Encoder network (for U-nets, VAEs, etc.).

    Notes
    -----
    .. The encoder is made of `n` encoding layer.
    .. Each encoding layer is made of `k >= 0` convolutions followed by one
       strided convolution _or_ pooling.
    .. The activation function that follows the very last convolution/pooling
       can be different from the other ones.
    .. If batch normalization is activated, it is performed before each
       convolution.
    .. The `skip` option can be used so that the output of each convolutional
       block is returned (i.e., the first tensor value after a pooling).

    Examples
    --------
    Encoder(dim, 2, [8, 16])
        (B, 2) -> Conv(stride=2) + ReLU -> (B, 8)
               -> Conv(stride=2) + ReLU -> (B, 16)
    Encoder(dim, 2, [[8, 8], [16, 16], [32, 32]], final_activation=None)
        (B, 2) -> Conv(stride=1) + ReLU -> (B, 8)
               -> Conv(stride=2) + ReLU -> (B, 8)
               -> Conv(stride=1) + ReLU -> (B, 16)
               -> Conv(stride=2) + ReLU -> (B, 16)
               -> Conv(stride=1) + ReLU -> (B, 32)
               -> Conv(stride=2)        -> (B, 32)
    Encoder(dim, 2, [8, 16], pool='max', activation=LeakyReLU(0.2))
        (B, 2) -> Conv(stride=1) + MaxPool(stride=2) + LeakyReLU -> (B, 8)
               -> Conv(stride=1) + MaxPool(stride=2) + LeakyReLU -> (B, 16)
    Encoder(dim, 2, [[8, 8], 32], skip=True)
        (B, 2) [returned]
               -> Conv(stride=1) + ReLU -> (B, 8)
               -> Conv(stride=2) + ReLU -> (B, 8)  [returned]
               -> Conv(stride=2) + ReLU -> (B, 32) [returned]

    """

    def __init__(
            self,
            dim,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            pool=None,
            skip=False,
            activation=tnn.ReLU,
            final_activation='same',
            batch_norm=False,
            groups=None,
            final_group='same',
            stitch=1,
            final_stitch='same'):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        input_channels : int
            Number of input channels.
        output_channels : sequence[int]
            Number of output channels in each encoding layer.
            Elements can be integers or nested sequences.
            If nested sequences, multiple convolutions are performed at the
            same resolution.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        skip : bool, default=False
            Handle skip connections.
            If True, ``forward`` returns  all output layers in a tuple.
        activation : str or type or callable, default='relu'
            Activation function.
        final_activation : str or type or callable, default='same'
            Final activation function. If 'same', same as `activation`.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        stride : int or sequence[int], default=2
            Spatial dimensions are divided by this number after each
            encoding layer.
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, use strided convolutions.
        groups : int or sequence[int], default=`stitch`
            Number of groups per layer. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
        final_group : int, default='same'
            Final number of groups
        stitch : int or sequence[int], default=1
            Number of stitched tasks per layer.
        final_stitch : int, default='same'
            Final stitch operation.

        """
        self.dim = dim
        self.skip = skip
        stitch = make_list(stitch, len(output_channels))
        stitch = [stch or 1 for stch in stitch]
        groups = make_list(groups, len(output_channels))
        groups = [group or stch for group, stch in zip(groups, stitch)]
        output_channels = [make_list(l) for l in output_channels]
        input_channels = [input_channels] + [c[-1] for c in output_channels[:-1]]

        final_activation = (activation if final_activation == 'same'
                            else final_activation)
        final_group = groups[-1] if final_group == 'same' else final_group
        final_stitch = groups[-1] if final_stitch == 'same' else final_stitch
        conv_activation = activation
        conv_last_activation = None if pool else activation
        conv_last_stride = 1 if pool else stride
        pool_activation = activation
        modules = []
        all_shapes = zip(input_channels, output_channels, groups, stitch)
        for d, (i, o, g, s) in enumerate(all_shapes):
            if d == len(output_channels) - 1:
                conv_last_activation = None if pool else final_activation
                pool_activation = final_activation
                s = final_stitch or 1
                g = final_group or s
            if len(o) > 1:
                modules.append(StackedConv(
                    dim, i, o, kernel_size,
                    stride=conv_last_stride,
                    activation=conv_activation,
                    final_activation=conv_last_activation,
                    batch_norm=batch_norm,
                    groups=g))
            else:
                modules.append(Conv(
                    dim, i, o[0], kernel_size,
                    stride=conv_last_stride,
                    activation=conv_last_activation,
                    batch_norm=batch_norm,
                    groups=g))
            if pool:
                modules.append(Pool(dim, kernel_size, stride=stride,
                                    activation=pool_activation))
            if s > 1:
                modules.append(Stitch(stitch, stitch))

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
    """Decoder network (for U-nets, VAEs, etc.)

    Notes
    -----
    .. The decoder is made of `n` decoding layer.
    .. Each encoding layer is made of `k >= 0` convolutions followed by one
       strided transposed convolution.
    .. The activation function that follows the very last convolution/pooling
       can be different from the other ones.
    .. If batch normalization is activated, it is performed before each
       convolution.
    .. The `skip` option can be used so that the the input is a list of
       tensors that are concatenated after each transposed convolution.
       Padding is used to ensure that tensors can be concatenated.


    Examples
    --------
    Decoder(dim, 16, [8, 2])
        (B, 16) -> ConvT(stride=2) + ReLU -> (B, 8)
                -> ConvT(stride=2) + ReLU -> (B, 2)
    Decoder(dim, 32, [[16, 16], [8,8], 2], final_activation=None)
        (B, 32) -> Conv(stride=2) + ReLU -> (B, 16)
                -> Conv(stride=1) + ReLU -> (B, 16)
                -> Conv(stride=2) + ReLU -> (B, 8)
                -> Conv(stride=1) + ReLU -> (B, 8)
                -> Conv(stride=2)        -> (B, 2)
    Decoder(dim, [32, 8, 2], [[8, 8], 2])
        Inputs : [(B, 32) a, (B, 8) b, (B, 2) c]
        (B, 2) a -> Conv(stride=1) + ReLU -> (B, 8)
                 -> Conv(stride=2) + ReLU -> (B, 8)
                 -> Cat(b)                -> (B, 16)
                 -> Conv(stride=2) + ReLU -> (B, 2)
                 -> Cat(c)                -> (B, 4)
    """

    def __init__(
            self,
            dim,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            activation=tnn.ReLU,
            final_activation='same',
            batch_norm=False,
            groups=None,
            final_group='same',
            stitch=1,
            final_stitch='same'):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        input_channels : int or sequence[int]
            Number of input channels in each decoding layer.
            A sequence can be provided to concatenate skip connections
            at each layer.
        output_channels
            Number of channels in each decoding layer.
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
        stitch = make_list(stitch, len(output_channels))
        stitch = [stch or 1 for stch in stitch]
        groups = make_list(groups, len(output_channels))
        groups = [group or stch for group, stch in zip(groups, stitch)]
        output_channels = [make_list(l) for l in output_channels]
        nb_layers = len(output_channels)
        self.skip = len(make_list(input_channels)) > 1
        input_channels = make_list(input_channels, nb_layers, default=0)
        for i in range(1, len(input_channels)):
            input_channels[i] += output_channels[i-1][-1]

        final_activation = (activation if final_activation == 'same'
                            else final_activation)
        final_group = groups[-1] if final_group == 'same' else final_group
        final_stitch = groups[-1] if final_stitch == 'same' else final_stitch

        modules = []
        all_shapes = zip(input_channels, output_channels, groups, stitch)
        for layer, (i, o, g, s) in enumerate(all_shapes):
            ConvClass = Conv if len(o) == 1 else StackedConv
            o = o[0] if len(o) == 1 else o
            if layer == len(input_channels) - 1:
                activation = final_activation
                s = final_stitch or 1
                g = final_group or s
            modules.append(ConvClass(
                dim, i, o,
                kernel_size=kernel_size,
                stride=stride,
                transposed=True,
                activation=activation,
                batch_norm=batch_norm,
                groups=g))
            if s > 1:
                modules.append(Stitch(s, s))
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
        for i, layer in enumerate(decoder):
            y = inputs.pop(0)
            oshape = layer.shape(x, stride=2, output_padding=0)[2:]
            yshape = y.shape[2:]
            pad = [i-o for o, i in zip(oshape, yshape)]
            x = layer(x, stride=2, output_padding=pad)
            groups = decoder[i+1].groups if len(decoder) > i+1 else 1
            x = interleaved_cat((x, y), dim=1, groups=groups)

        # Post-processing (convolutions without upsampling)
        for layer in postproc:
            pad = [((k-1)*d)//2 for k, d in zip(layer.conv.kernel_size,
                                                layer.conv.dilation)]
            x = layer(x, stride=1, padding=pad)

        return x


@nitorchmodule
class StackedConv(tnn.Sequential):
    """Stacked convolutions, without up/down-sampling."""

    def __init__(
            self,
            dim,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            transposed=False,
            activation=tnn.ReLU,
            final_activation='same',
            batch_norm=False,
            groups=None,
            final_group='same',
            stitch=1,
            final_stitch='same'):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        input_channels : int
            Number of input channels.
        output_channels : sequence[int]
            Number of output channels in each convolutional layer.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        stride : int or sequence[int], default=1
            Stride in the final convolution.
        activation : str or type or callable, default='relu'
            Activation function.
        final_activation : str or type or callable, default='same'
            Final activation function. If 'same', same as `activation`.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        self.dim = dim

        stitch = make_list(stitch, len(output_channels))
        stitch = [stch or 1 for stch in stitch]
        groups = make_list(groups, len(output_channels))
        groups = [group or stch for group, stch in zip(groups, stitch)]
        output_channels = list(output_channels)
        input_channels = [input_channels] + output_channels[:-1]

        modules = []
        kernel_size = make_list(kernel_size, dim)
        nb_module = len(input_channels)

        final_stride, stride = (stride, 1)
        final_transposed, transposed = (transposed, False)
        final_group = groups[-1] if final_group == 'same' else final_group
        final_stitch = groups[-1] if final_stitch == 'same' else final_stitch

        all_shapes = zip(input_channels, output_channels, groups, stitch)
        for l, (i, o, g, s) in enumerate(all_shapes):
            pad = [(k-1)//2 for k in kernel_size]
            if l == nb_module-1:
                if final_activation != 'same':
                    activation = final_activation
                stride = final_stride
                transposed = final_transposed
                s = final_stitch or 1
                g = final_group or s
            modules.append(Conv(dim, i, o,
                                kernel_size=kernel_size,
                                padding=pad,
                                stride=stride,
                                activation=activation,
                                batch_norm=batch_norm,
                                transposed=transposed,
                                groups=g))
            if s != 1:
                modules.append(Stitch(s, s))
        super().__init__(*modules)


@nitorchmodule
class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(self, dim, input_channels, output_channels,
                 encoder=None, decoder=None, kernel_size=3, stride=2,
                 activation=tnn.ReLU, final_activation='same', pool=None,
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
        activation : str or type or callable or None, default='relu'
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
        final_activation : str or type or callable or None, default='same'
            Final activation function. If 'same', same as `activation`.
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling to use in the encoder. If None, use strided convolutions.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        self.dim = dim

        encoder = list(encoder or [16, 32, 32, 32])
        decoder = list(decoder or [32, 32, 32, 32, 32, 16, 16])
        if len(decoder) < len(encoder):
            # we need as many upsampling steps as downsampling steps
            decoder = make_list(decoder, len(encoder))
        encoder = [make_list(l) for l in encoder]
        encoder_out = [l[-1] for l in reversed(encoder)] + [input_channels]
        decoder = [make_list(l) for l in decoder]
        decoder[-1].append(output_channels)
        stack = flatten(decoder[len(encoder):])
        decoder = decoder[:len(encoder)]
        last_decoder = decoder[-1][-1] + input_channels

        modules = []
        enc = Encoder(dim,
                      input_channels=input_channels,
                      output_channels=encoder,
                      kernel_size=kernel_size,
                      stride=stride,
                      skip=True,
                      activation=activation,
                      batch_norm=batch_norm,
                      pool=pool)
        modules.append(('encoder', enc))

        dec = Decoder(dim,
                      input_channels=encoder_out,
                      output_channels=decoder,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation,
                      batch_norm=batch_norm)
        modules.append(('decoder', dec))

        stk = StackedConv(dim,
                          input_channels=last_decoder,
                          output_channels=stack,
                          kernel_size=kernel_size,
                          activation=activation,
                          final_activation=final_activation,
                          batch_norm=batch_norm)
        modules.append(('stack', stk))

        super().__init__(OrderedDict(modules))


@nitorchmodule
class CNN(tnn.Sequential):
    """Encoding convolutional network (for classification or regression).

    Notes
    -----
    .. The CNN is made of a convolutional encoder, followed by a
       reduction operation across all spatial dimensions, followed by a
       series of fully-connected layers (i.e., 1D convolutions).
    .. Each encoding layer is made of `k >= 0` convolutions followed by one
       strided convolution _or_ pooling.
    .. The very last activation function can be different from the other ones.
    .. If batch normalization is activated, it is performed before each
       encoding convolution.

    """

    def __init__(self, dim, input_channels, output_channels, encoder=None,
                 stack=None, kernel_size=3, stride=2, pool=None, reduction='max',
                 activation='relu', final_activation='same', batch_norm=False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        encoder : sequence[int or sequence[int]], default=[16, 32, 32, 32]
            Number of output channels in each encoding layer.
            If a nested list, multiple convolutions are performed at each
            resolution.
        stack : sequence[int], default=[32, 16, 16]
            Number of output channels in each fully-connected layer.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        stride : int or sequence[int], default=2
            Stride of the encoder (the dimensions are divided by this
            number after each encoding layer).
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Type of pooling performed in the encoder.
            If None, strided convolutions are used.
        reduction : {'max', 'min', 'median', 'mean', 'sum'} or type or callable, default='max'
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
        last_encoder = make_list(encoder[-1])[-1]
        stack = stack + [output_channels]
        if isinstance(reduction, str):
            reduction = reductions.get(reduction, None)
        reduction = reduction(keepdim=True) if inspect.isclass(reduction) \
                    else reduction if callable(reduction) \
                    else None
        if not isinstance(reduction, Reduction):
            raise TypeError('reduction must be a `Reduction` module.')

        modules = []
        enc = Encoder(dim,
                      input_channels=input_channels,
                      output_channels=encoder,
                      kernel_size=kernel_size,
                      stride=stride,
                      pool=pool,
                      activation=activation,
                      batch_norm=batch_norm)
        modules.append(('encoder', enc))

        modules.append(('reduction', reduction))

        stk = StackedConv(dim,
                          input_channels=last_encoder,
                          output_channels=stack,
                          kernel_size=1,
                          activation=activation,
                          final_activation=final_activation)
        modules.append(('stack', stk))

        super().__init__(OrderedDict(modules))

    def forward(self, input):
        """

        Parameters
        ----------
        input : (batch, input_channels, *spatial) tensor

        Returns
        -------
        output : (batch, output_channels) tensor

        """
        input = super().forward(input)
        input = input.reshape(input.shape[:2])
        return input


@nitorchmodule
class MRF(tnn.Sequential):
    """Categorical MRF network."""

    def __init__(self, dim, num_classes, num_filters=16, num_extra=1,
                 kernel_size=3, activation=tnn.LeakyReLU(0.2),
                 batch_norm=False, final_activation='same', bias=False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
        num_classes : int
            Number of input classes.
        num_extra : int, default=0
            Number of extra layers between MRF layer and final layer.
        num_filters : int, default=16
            Number of conv filters in first, MRF layer.
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
        activation : str or type or callable, default='tnn.LeakyReLU(0.2)'
            Activation function.
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        final_activation : str or type or callable, default='same'
            Final activation function. If 'same', same as `activation`.
        bias : bool, default=False
            Adds learnable bias to the output.
        """
        self.dim = dim

        # make layers
        modules = []
        p = ((kernel_size - 1) // 2,)*self.dim  # for 'same' convolution in first layer
        module = Conv(dim, in_channels=num_classes, out_channels=num_filters,
                      kernel_size=kernel_size, activation=activation,
                      batch_norm=batch_norm, bias=bias, padding=p)
        # zero-centre
        if self.dim == 3:
            module.conv.weight.data[:, :, kernel_size // 2, kernel_size // 2, kernel_size // 2] = 0
        elif self.dim == 2:
            module.conv.weight.data[:, :, kernel_size // 2, kernel_size // 2] = 0
        else:
            module.conv.weight.data[:, :, kernel_size // 2] = 0
        modules.append(('mrf', module))
        for i in range(num_extra):
            module = Conv(dim, in_channels=num_filters, out_channels=num_filters,
                         kernel_size=1, activation=activation, batch_norm=batch_norm,
                         bias=bias)
            modules.append(('extra', module))
        module = Conv(dim, in_channels=num_filters, out_channels=num_classes,
                     kernel_size=1, activation=final_activation, batch_norm=batch_norm,
                     bias=bias)
        modules.append(('final', module))
        # build model
        super().__init__(OrderedDict(modules))
