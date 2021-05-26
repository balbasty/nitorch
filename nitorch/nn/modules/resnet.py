"""Residual networks

These networks use residual blocks, but are more targeted to dense prediction
than image classification.

References
----------
..[1] "Deep Residual Learning for Image Recognition"
      K He, X Zhang, S Ren, J Sun
      CVPR (2016)
..[2] "Identity Mappings in Deep Residual Networks"
      K He, X Zhang, S Ren, J Sun
      ECCV (2016)
"""

import torch.nn as tnn
import torch
from nitorch.core import py, utils
from ..base import Module, nitorchmodule
from .conv import ConvBlock
from .encode_decode import DownStep, UpStep


@nitorchmodule
class ResBlock(tnn.ModuleList):
    """Residual block.

    We follow He et al. (ECCV 2016) and perform all activations and
    normalizations in the alternate branch, keeping a full identity
    path open in the main branch.

    References
    ----------
    ..[1] "Deep Residual Learning for Image Recognition"
          K He, X Zhang, S Ren, J Sun
          CVPR (2016)
    ..[2] "Identity Mappings in Deep Residual Networks"
          K He, X Zhang, S Ren, J Sun
          ECCV (2016)

    """

    def __init__(self,
                 dim,
                 channels,
                 bottleneck=None,
                 nb_conv=2,
                 nb_res=2,
                 recurrent=False,
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

        channels : int
            Number of channels in the input image.

        bottleneck : int, optional
            If provided, a bottleneck architecture is used, where
            1d conv are used to project the input channels onto a
            lower-dimensional space, where the spatial convolutions
            are performed, before being mapped back to the original
            number of dimensions.

        nb_conv : int, default=2
            Number of convolutions

        nb_res : int, default=2
            Number of residual blocks

        recurrent : {'conv', 'res', 'conv+res', None}, default=None
            Use recurrent convolutions and/or recurrent residual blocks.

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

        if not isinstance(recurrent, str):
            recurrent = 'res' if recurrent else ''
        act_inplace = (order.index('a') > order.index('c') or
                       (batch_norm and order.index('a') > order.index('n')))
        conv_opt = dict(
            dim=dim,
            channels=channels,
            bottleneck=bottleneck,
            nb_conv=nb_conv,
            recurrent='conv' in recurrent,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            batch_norm=batch_norm,
            order=order,
            inplace=act_inplace
        )
        nb_res_inde = 1 if 'res' in recurrent else nb_res
        super().__init__([ConvBlock(**conv_opt) for _ in range(nb_res_inde)])
        self.recurrent = recurrent

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[-1].out_channels)

    def shape(self, x, **overload):
        if torch.is_tensor(x):
            return tuple(x.shape)
        else:
            return tuple(x)

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, channels, *spatial) tensor
            Input tensor
        **overload : dict
            Some parameters can be overloaded at call time

        Returns
        -------
        x : (batch, channels, *spatial) tensor
            Output tensor, with the same shape as the input tensor

        """
        nb_res = overload.pop('nb_res', len(self))
        if nb_res != len(self) and not self.recurrent:
            raise ValueError(
                f'Number of required blocks and registered '
                f'blocks not consistent: '
                f'{nb_res} vs. {len(self)}.')

        if self.recurrent:
            block = self[0]
            for _ in range(nb_res):
                identity = x
                x = block(x, **overload)
                x += identity
        else:
            for block in self:
                identity = x
                x = block(x, **overload)
                x += identity
        return x


@nitorchmodule
class AtrousBlock(tnn.Sequential):
    """Atrous residual block"""

    def __init__(self,
                 dim,
                 channels,
                 dilations=(1, 2, 4, 8, 16),
                 nb_conv=2,
                 nb_res=2,
                 recurrent=False,
                 kernel_size=3,
                 padding_mode='zeros',
                 groups=1,
                 bias=True,
                 activation=tnn.ReLU,
                 batch_norm=True,
                 order='nac',
                 inplace=True):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.

        dilations : sequence[int], default=(1, 2, 4, 8, 16)
            Number of dilation in each parallel path.

        nb_conv : int, default=2
            Number of convolutions

        nb_res : int, default=2
            Number of residual blocks

        recurrent : {'conv', 'res', 'atrous', 'conv+res', ...}, default=None
            Use recurrent convolutions and/or recurrent residual blocks
            and/or recurrent atrous blocks.

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

        inplace : bool, default=True
            Apply activation inplace if possible
            (i.e., not ``is_leaf and requires_grad``).

        """

        if not isinstance(recurrent, str):
            recurrent = 'res' if recurrent else ''
        conv_opt = dict(
            dim=dim,
            channels=channels,
            nb_conv=nb_conv,
            nb_res=nb_res,
            recurrent=recurrent,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            order=order,
            inplace=inplace,
        )
        super().__init__(*[ResBlock(**conv_opt, dilation=d) for d in dilations])

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, channels, *spatial) tensor
            Input tensor
        **overload : dict
            Some parameters can be overloaded at call time

        Returns
        -------
        x : (batch, channels, *spatial) tensor
            Output tensor, with the same shape as the input tensor

        """
        y = 0
        for block in self:
            identity = y
            y = block(x, **overload)
            y += identity
        y /= len(self)
        return y


@nitorchmodule
class ResEncodingBlock(tnn.Sequential):

    def __init__(
            self,
            dim,
            in_channels,
            out_channels=None,
            bottleneck=None,
            nb_conv=2,
            nb_res=2,
            recurrent=False,
            kernel_size=3,
            stride=2,
            pool=None,
            activation=tnn.ReLU,
            batch_norm=False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : sequence[int]
            Number of channels at the current scale.

        out_channels : sequence[int], default=in_channels*2
            Number of channels at the next scale.

        nb_conv : int, default=2
            Number of convolutions per convolutional block.

        nb_res : int, default=2
            Number of residual blocks

        recurrent : {'conv', 'res', 'conv+res', None}, default=None
            Use recurrent convolutions and/or recurrent residual blocks..

        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.

        stride : int or sequence[int], default=2
            Stride per dimension.

        pool : {'max, 'down', 'conv', None}, default='conv'
            Downsampling method.

        activation : str or type or callable or None, default='relu'
            Activation function.

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        out_channels = out_channels or in_channels * 2
        res = ResBlock(
            dim=dim,
            channels=in_channels,
            bottleneck=bottleneck,
            nb_conv=nb_conv,
            nb_res=nb_res,
            recurrent=recurrent,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
        )
        down = DownStep(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            pool=pool,
            stride=stride)
        super().__init__(res, down)

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[1].out_channels)

    def shape(self, x):
        for layer in self:
            x = layer.shape(x)
        return x


@nitorchmodule
class ResDecodingBlock(tnn.Sequential):

    def __init__(
            self,
            dim,
            in_channels,
            out_channels=None,
            bottleneck=None,
            nb_conv=2,
            nb_res=2,
            recurrent=False,
            kernel_size=3,
            stride=2,
            unpool=None,
            activation=tnn.ReLU,
            batch_norm=True):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : sequence[int]
            Number of channels at the current scale.

        out_channels : sequence[int], default=in_channels*2
            Number of channels at the next scale.

        nb_conv : int, default=2
            Number of convolutions per convolutional block

        nb_res : int, default=2
            Number of residual blocks

        recurrent : {'conv', 'res', 'conv+res', None}, default=None
            Use recurrent convolutions and/or recurrent residual blocks..

        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.

        stride : int or sequence[int], default=2
            Stride per dimension.

        unpool : {'conv', 'up', 0..7}, default=0
            Unpooling used to change resolution:
            'conv' : strided convolution
            'up' : upsampling (without filling) + channel-conv
            0..7 : upsampling (of a given order) + channel-conv

        activation : str or type or callable or None, default='relu'
            Activation function.

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        out_channels = out_channels or in_channels * 2
        res = ResBlock(
            dim=dim,
            channels=in_channels,
            bottleneck=bottleneck,
            nb_conv=nb_conv,
            nb_res=nb_res,
            recurrent=recurrent,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
        )
        down = UpStep(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            unpool=unpool,
            stride=stride)
        super().__init__(res, down)

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[1].out_channels)

    def shape(self, x):
        for layer in self:
            x = layer.shape(x)
        return x


@nitorchmodule
class UResBlock(tnn.Sequential):
    """U-Net with residual blocks"""

    def __init__(
            self,
            dim,
            channels=(16, 32, 64),
            bottleneck=None,
            nb_conv=2,
            nb_res=2,
            recurrent=False,
            kernel_size=3,
            stride=2,
            pool='conv',
            unpool='conv',
            activation=tnn.LeakyReLU(0.2),
            batch_norm=True):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        channels : sequence[int], default=(16, 32, 64)
            Number of channels at each scale.

        bottleneck : [sequence of] int, optional
            Use a bottleneck that maps to this number of channels
            before (and after) each convolution.
            If a sequence, should be one value per scale.

        nb_conv : [sequence of] int, default=2
            Number of convolutions per convolutional block.
            If a sequence, should be one value per scale.

        nb_res : [sequence of] int, default=2
            Number of residual blocks.
            If a sequence, should be one value per scale.

        recurrent : {'conv', 'res', 'conv+res', None}, default=None
            Use recurrent convolutions and/or recurrent residual blocks..

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
        nb_conv = py.make_list(nb_conv, len(channels))
        nb_res = py.make_list(nb_res, len(channels))

        resopt = dict(
            dim=dim,
            bottleneck=bottleneck,
            recurrent=recurrent,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
        )

        encoder = []
        params = zip(channels[:-1], channels[1:], nb_conv[:-1], nb_res[:-1])
        for cin, cout, nbc, nbr in params:
            resblock = ResEncodingBlock(
                in_channels=cin,
                out_channels=cout,
                nb_conv=nbc,
                nb_res=nbr,
                stride=stride,
                pool=pool,
                **resopt
            )
            encoder.append(resblock)

        channels = list(reversed(channels))
        nb_conv = list(reversed(nb_conv))
        nb_res = list(reversed(nb_res))

        decoder = []
        params = zip(channels[:-1], channels[1:], nb_conv[:-1], nb_res[:-1])
        for cin, cout, nbc, nbr in params:
            resblock = ResDecodingBlock(
                in_channels=cin,
                out_channels=cout,
                nb_conv=nbc,
                nb_res=nbr,
                stride=stride,
                unpool=unpool,
                **resopt
            )
            encoder.append(resblock)

        final = ResBlock(
            channels=channels[-1],
            **resopt
        )

        super().__init__(*encoder, *decoder, final)
        
    def forward(self, *x, return_all=False):
        """

        Parameters
        ----------
        x : (b, c, *spatial) tensor
            At least one tensor
            If multiple inputs are provided, they will be used as
            inputs at multiple scales.
        return_all: bool, default=False
            if True, return all intermediate outputs (at different scales)

        Returns
        -------
        x : (b, c, *spatial) tensor

        """

        nb_levels = len(self) // 2
        encoder = list(self)[:nb_levels]
        decoder = list(self)[nb_levels:-1]
        last = list(self)[-1]

        inputs = list(x)

        # encoder
        #   - save intermediate outputs for skip connections
        #   - add optional inputs at each scale
        x = 0
        intermediates = []
        for layer in encoder:
            if inputs:
                x += inputs.pop(0)
            layer, pool = layer
            x = layer(x)
            intermediates.append(x)
            x = pool(x)

        # there's no skipped connection in the bottleneck,
        # but there might be an input to add.
        if inputs:
            intermediates.append(inputs.pop(0))
        else:
            intermediates.append(0)
        intermediates = intermediates[::-1]

        # decoder
        #   - add intermediate result at each layer
        #   - save output at each scale if needed
        outputs = []
        for layer in decoder:
            x += intermediates.pop(0)
            layer, pool = layer
            x = layer(x)
            if return_all:
                outputs.append(x)
            x = pool(x, output_shape=intermediates[0].shape[2:])

        # last residual block
        x += intermediates.pop(0)
        x = last(x)

        return [x, *outputs[::-1]] if return_all else x
