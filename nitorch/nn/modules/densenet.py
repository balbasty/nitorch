"""Densely-cnnected networks

References
----------
..[1] "Densely Connected Convolutional Networks"
      Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
      CVPR (2017)
      https://arxiv.org/abs/1608.06993
"""

import torch
import torch.nn as tnn
from nitorch.core import py, utils
from ..base import Module, nitorchmodule
from .conv import ConvBlock
from .encode_decode import DownStep, UpStep


@nitorchmodule
class DenseBottleneck(tnn.Sequential):
    """
    A 1d-conv followed by a spatial conv.
    """
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 bottleneck,
                 kernel_size=3,
                 padding_mode='zeros',
                 groups=1,
                 bias=True,
                 dilation=1,
                 activation=tnn.ReLU,
                 batch_norm=True,
                 order='nac'):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int
            Number of channels in the input image.

        out_channels : int
            Number of channels in the output image.

        bottleneck : int, optional
            The number of output channels after each 1d bottleneck

        kernel_size : int or sequence[int], default=3
            Size of the convolution kernel.

        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.

        groups : int, default=1
            Number of blocked connections from input channels to
            output channels. Using this parameter is an alternative to
            the use of 'sequence' input/output channels. In that case,
            the number of input and output channels in each group is
            found by dividing the ``input_channels`` and ``output_channels``
            with ``groups``.

        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.

        dilation : int, default=1

        activation : str or type or callable, default=ReLU
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation

        batch_norm : bool or type or callable, default=True
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).

        order : permutation of 'nca', default='nac'
            Order in which to perform the normalization (n), convolution (c)
            and activation (a).

        """
        opt = dict(
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            batch_norm=batch_norm,
            order=order,
        )
        conv1 = ConvBlock(dim, in_channels, bottleneck, kernel_size=1, **opt)
        conv3 = ConvBlock(dim, bottleneck, out_channels, kernel_size=kernel_size, **opt)
        super().__init__(conv1, conv3)

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[1].out_channels)

    def shape(self, x):
        for layer in self:
            x = layer.shape(x)
        return x


@nitorchmodule
class DenseBlock(tnn.Sequential):

    def __init__(self,
                 dim,
                 in_channels,
                 nb_conv=4,
                 growth=12,
                 bottleneck=None,
                 kernel_size=3,
                 padding_mode='zeros',
                 groups=1,
                 bias=True,
                 dilation=1,
                 activation=tnn.ReLU,
                 batch_norm=True,
                 order='nac'):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int
            Number of channels in the input image.

        nb_conv : int, default=40
            Number of convolutions per block.

        growth : int, default=12
            Number of output channels in each convolution.

        bottleneck : int, optional
            The number of output channels after each 1d bottleneck
            will be `bottleneck*growth`. If None (default), no 1d conv
            is used.

        kernel_size : int or sequence[int], default=3
            Size of the convolution kernel.

        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.

        groups : int, default=1
            Number of blocked connections from input channels to
            output channels. Using this parameter is an alternative to
            the use of 'sequence' input/output channels. In that case,
            the number of input and output channels in each group is
            found by dividing the ``input_channels`` and ``output_channels``
            with ``groups``.

        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.

        dilation : int, default=1

        activation : str or type or callable, default=ReLU
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation

        batch_norm : bool or type or callable, default=True
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).

        order : permutation of 'nca', default='nac'
            Order in which to perform the normalization (n), convolution (c)
            and activation (a).

        """
        opt = dict(
            kernel_size=kernel_size,
            padding='auto',
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            batch_norm=batch_norm,
            order=order,
        )
        if bottleneck:
            Klass = DenseBottleneck
            opt['bottleneck'] = bottleneck * growth
        else:
            Klass = ConvBlock

        convs = []
        for l in range(nb_conv):
            cin = in_channels + l*growth
            convs.append(Klass(dim, cin, growth, **opt))

        super().__init__(*convs)

    def forward(self, x):
        for layer in self:
            output = layer(x)
            x = torch.cat([x, output], dim=1)
        return x

    out_channels = property(lambda self: self.in_channels
                                         + sum(l.out_channels for l in self))

    def shape(self, x):
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)
        x[1] = self.out_channels
        return x


@nitorchmodule
class UDenseBlock(tnn.Sequential):

    def __init__(
            self,
            dim,
            in_channels,
            growth=(16, 32, 64),
            nb_conv=(3, 4, 12),
            bottleneck=None,
            compression=2,
            kernel_size=3,
            stride=2,
            pool='conv',
            unpool='conv',
            activation=tnn.ReLU,
            batch_norm=True):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : int, default=(16, 32, 64)
            Number of channels at each scale.

        growth : [sequence of] int, default=(16, 32, 64)
            Number of output channels in each convolution

        nb_conv : [sequence of] int, default=(3, 4, 12)
            Number of convolutions per convolutional block.
            If a sequence, should be one value per scale.

        bottleneck : [sequence of] int, optional
            Use a bottleneck that maps to this number of channels
            before each convolution.
            If a sequence, should be one value per scale.

        compression : int, default=2
            THe number of channels is divided by this number after each
            up/down-sampling step.

        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.

        stride : int or sequence[int], default=2
            Stride per dimension.

        pool : {'max, 'down', 'conv', None}, default='conv'
            Downsampling method.

        unpool : {'up', 'conv', None}, default='conv'
            Upsampling method.

        activation : str or type or callable or None, default='relu'
            Activation function.

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        n_levels = max(py.make_list(growth),
                       py.make_list(nb_conv),
                       py.make_list(bottleneck))
        growth = py.make_list(growth, n_levels)
        nb_conv = py.make_list(nb_conv, n_levels)
        bottleneck = py.make_list(bottleneck, n_levels)

        opt = dict(
            dim=dim,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
        )

        encoder = []
        params = zip(growth[:-1], nb_conv[:-1], bottleneck[:-1])
        cin = in_channels
        for g, nbc, btn in params:
            block = DenseBlock(
                in_channels=cin,
                growth=g,
                nb_conv=nbc,
                bottleneck=btn,
                **opt)
            down = DownStep(
                dim=dim,
                in_channels=block.out_channels,
                out_channels=block.out_channels//compression,
                stride=stride,
                pool=pool)
            cin = down.out_channels
            encoder.append(nitorchmodule(tnn.Sequential)(block, down))

        growth = list(reversed(growth))
        nb_conv = list(reversed(nb_conv))
        bottleneck = list(reversed(bottleneck))

        decoder = []
        params = zip(growth[:-1], nb_conv[:-1], bottleneck[:-1])
        for i, (g, nbc, btn) in enumerate(params):
            if i > 0:
                cin = cin + encoder[-i][0].out_channels
            block = DenseBlock(
                in_channels=cin,
                growth=g,
                nb_conv=nbc,
                bottleneck=btn,
                **opt)
            up = UpStep(
                dim=dim,
                in_channels=block.out_channels,
                out_channels=block.out_channels//compression,
                stride=stride,
                unpool=unpool)
            cin = up.out_channels
            decoder.append(nitorchmodule(tnn.Sequential)(block, up))

        final = DenseBlock(
            in_channels=cin + encoder[0][0].out_channels,
            growth=growth[-1],
            nb_conv=nb_conv[-1],
            bottleneck=bottleneck[-1],
            **opt
        )

        super().__init__(*[*encoder, *decoder, final])

    def forward(self, x):
        """

        Parameters
        ----------
        x : (b, c, *spatial) tensor

        Returns
        -------
        x : (b, c, *spatial) tensor

        """

        nb_levels = len(self) // 2
        encoder = list(self)[:nb_levels]
        decoder = list(self)[nb_levels:-1]
        last = list(self)[-1]

        # encoder
        #   - save intermediate outputs for skip connections
        intermediates = []
        for layer in encoder:
            layer, pool = layer
            x = layer(x)
            intermediates.append(x)
            x = pool(x)

        intermediates.append(None)
        intermediates = intermediates[::-1]

        # decoder
        #   - add intermediate result at each layer
        for layer in decoder:
            skip = intermediates.pop(0)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            layer, pool = layer
            x = layer(x)
            x = pool(x, output_shape=intermediates[0].shape[2:])

        # last block
        x = torch.cat([x, intermediates.pop(0)], dim=1)
        x = last(x)

        return x
