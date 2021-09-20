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
..[3] "Bag of Tricks for Image Classification with Convolutional
       Neural Network"
      Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
      CVPR (2019)
"""

import torch.nn as tnn
import torch
from nitorch.core import py, utils
from ..base import Module, ModuleList, Sequential, nitorchmodule
from .conv import Conv, ConvGroup, ConvBlock, BottleneckConv
from .spatial import Resize
from .pool import MeanPool


@nitorchmodule
class ResBlock(ModuleList):
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
                 norm=True,
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

        norm : bool or type or callable, default=True
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
                       (norm and order.index('a') > order.index('n')))
        conv_opt = dict(
            dim=dim,
            channels=channels,
            bottleneck=channels//bottleneck if bottleneck else None,
            nb_conv=nb_conv,
            recurrent='conv' in recurrent,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            norm=norm,
            order=order,
            inplace=act_inplace
        )
        nb_res_inde = 1 if 'res' in recurrent else nb_res
        super().__init__([ConvGroup(**conv_opt) for _ in range(nb_res_inde)])
        self.recurrent = recurrent

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[-1].out_channels)

    def shape(self, x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        else:
            return tuple(x)

    def forward(self, x):
        """

        Parameters
        ----------
        x : (batch, channels, *spatial) tensor
            Input tensor

        Returns
        -------
        x : (batch, channels, *spatial) tensor
            Output tensor, with the same shape as the input tensor

        """
        blocks = [self[0]] * self.nb_res if self.recurrent else self
        for block in blocks:
            identity = x
            x = block(x)
            x += identity
        return x


@nitorchmodule
class ResDownStep(ModuleList):
    """Downsampling step for ResNets

    This layer performs a residual downsampling:
        strided_proj(x) + strided_conv(x)

    It implements strided_conv as option B in [2]:
        strided_conv = Conv(1) -> Conv(3, stride=2) -> Conv(1)
    Note that the 1d conv are only used if `bottleneck` is used.

    It implements strided_proj as option D in [2], but with a tweak:
    the kernel_size of the average pooling is 3 instead of two, and 'auto'
    padding is used. Otherwise, a half voxel shift is introduced between
    the two branches:
        strided_proj = Pool(3, stride=2) -> Proj(1)

    References
    ----------
    ..[1] "Deep Residual Learning for Image Recognition"
          K He, X Zhang, S Ren, J Sun
          CVPR (2016)
    ..[2] "Bag of Tricks for Image Classification with Convolutional
           Neural Network"
          Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
          CVPR (2019)

    """
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels=None,
                 bottleneck=1,
                 residual=True,
                 kernel_size=3,
                 stride=2,
                 bound='zeros',
                 groups=1,
                 bias=True,
                 dilation=1,
                 activation=tnn.ReLU,
                 norm=True,
                 order='nac'):
        """

        Parameters
        ----------
        dim : int
            Number of spatial dimensions

        in_channels : int
            Number of input channels

        out_channels : int, default=in_channels*2
            Number of output channels

        bottleneck : int, default=1
            Divide input channels by this number in the bottleneck

        residual : bool, default=True
            Residual connection. If False, only perform the
            convolutional branch.

        Other Parameters
        ----------------
        kernel_size : int, default=3
        stride : int, default=2
        bound : default='zeros'
        groups : int, default=1
        bias : bool, default=True
        dilation : int, default=1
        activation : default=ReLU
        norm : bool, default=True
        order : permutation of 'nac', default='nac'
        """
        out_channels = out_channels or (in_channels * 2)
        opt = dict(
            dim=dim,
            bound=bound,
            padding='same',
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            norm=norm,
            order=order,
        )
        if bottleneck:
            bottleneck = in_channels//bottleneck
            branch1 = BottleneckConv(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                bottleneck=bottleneck,
                stride=stride,
                **opt)
        else:
            branch1 = ConvBlock(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                **opt)
        # Branch 2 (pseudo-identity) is just a projection
        # (no activation, no bias, no batch norm)
        if residual:
            branch2 = nitorchmodule(tnn.Sequential)(
                MeanPool(dim,
                         kernel_size=3,
                         stride=stride,
                         padding='auto'),
                Conv(dim=dim,
                     kernel_size=1,
                     in_channels=in_channels,
                     out_channels=out_channels),
            )
            super().__init__([branch1, branch2])
        else:
            super().__init__([branch1])

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[0].out_channels)

    def shape(self, x):
        return self[0].shape(x)

    def forward(self, x):
        return sum(layer(x) for layer in self)


@nitorchmodule
class ResUpStep(ModuleList):
    """Downsampling step for ResNets

    This layer performs a residual upsampling:
        strided_deproj(x) + strided_deconv(x)

    The strided deconvolution is implemented similarly to the strided
    convolution (option B) in [2]:
        strided_deconv = Conv(1) -> Conv(3, stride=2, transposed) -> Conv(1)
    where all Conv have batch norm, pre-activation and a bias term.

    The identity branch is implemented using linear resampling [2]:
        strided_proj = Proj(1) -> Resize(2)
    where Proj is a plain linear projection (no BN, no activation, no bias).

    References
    ----------
    ..[1] "Deep Residual Learning for Image Recognition"
          K He, X Zhang, S Ren, J Sun
          CVPR (2016)
    ..[2] "Bag of Tricks for Image Classification with Convolutional
           Neural Network"
          Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
          CVPR (2019)

    """
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels=None,
                 bottleneck=1,
                 residual=True,
                 kernel_size=3,
                 stride=2,
                 padding_mode='zeros',
                 groups=1,
                 bias=True,
                 dilation=1,
                 activation=tnn.ReLU,
                 norm=True,
                 order='nac'):
        """

        Parameters
        ----------
        dim : int
        in_channels : int
        out_channels : int, default=in_channels*2
        bottleneck : int, default=1
        residual : bool, default=True
        kernel_size : int, default=3
        stride : int, default=2
        padding_mode : default='zeros'
        groups : int, default=1
        bias : bool, default=True
        dilation : int, default=1
        activation : default=ReLU
        norm : bool, default=True
        order : permutation of 'nac', default='nac'
        """
        out_channels = out_channels or (in_channels * 2)
        opt = dict(
            dim=dim,
            padding_mode=padding_mode,
            padding='auto',
            groups=groups,
            bias=bias,
            dilation=dilation,
            activation=activation,
            norm=norm,
            order=order,
            transposed=True,
        )
        if bottleneck:
            bottleneck = out_channels//bottleneck
            branch1 = BottleneckConv(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                bottleneck=bottleneck,
                stride=stride,
                **opt)
        else:
            branch1 = ConvBlock(kernel_size=kernel_size,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                **opt)
        # Branch 2 (pseudo-identity) is just a projection
        # (no activation, no bias, no batch norm)
        if residual:
            branch2 = nitorchmodule(tnn.Sequential)(
                Conv(dim=dim,
                     kernel_size=1,
                     in_channels=in_channels,
                     out_channels=out_channels),
                Resize(factor=2),
            )
            super().__init__([branch1, branch2])
        else:
            super().__init__([branch1])

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[0].out_channels)

    def shape(self, x, output_shape=None):
        return self[0].shape(x, output_shape=output_shape)

    def forward(self, x, output_shape=None):
        conv, (conv1, up) = self
        y = up(conv1(x), output_shape=output_shape)
        y += conv(x, output_shape=output_shape)
        return y


@nitorchmodule
class AtrousBlock(Sequential):
    """Atrous residual block"""

    def __init__(self,
                 dim,
                 channels,
                 dilations=(1, 2, 4, 8, 16),
                 nb_conv=2,
                 nb_res=2,
                 recurrent=False,
                 kernel_size=3,
                 bound='zeros',
                 groups=1,
                 bias=True,
                 activation=tnn.ReLU,
                 norm=True,
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

        dilation : sequence[int], default=(1, 2, 4, 8, 16)
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

        bound : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
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

        norm : bool or type or callable, default=True
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
            bound=bound,
            groups=groups,
            bias=bias,
            activation=activation,
            norm=norm,
            order=order,
            inplace=inplace,
        )
        super().__init__(*[ResBlock(**conv_opt, dilation=d) for d in dilations])

    def forward(self, x):
        """

        Parameters
        ----------
        x : (batch, channels, *spatial) tensor
            Input tensor

        Returns
        -------
        x : (batch, channels, *spatial) tensor
            Output tensor, with the same shape as the input tensor

        """
        y = 0
        for block in self:
            identity = y
            y = block(x)
            y += identity
        y /= len(self)
        return y


@nitorchmodule
class ResEncodingBlock(Sequential):

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
            residual_pool=True,
            activation=tnn.ReLU,
            norm=False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : sequence[int]
            Number of channels at the current scale.

        out_channels : sequence[int], default=in_channels*2
            Number of channels at the next scale.

        bottleneck : int, default=1

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

        activation : str or type or callable or None, default='relu'
            Activation function.

        norm : bool or type or callable, default=False
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
            norm=norm,
        )
        down = ResDownStep(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck=bottleneck,
            norm=norm,
            residual=residual_pool,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride)
        super().__init__(res, down)

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[1].out_channels)

    def shape(self, x):
        for layer in self:
            x = layer.shape(x)
        return x


@nitorchmodule
class ResDecodingBlock(Sequential):

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
            residual_unpool=True,
            activation=tnn.ReLU,
            norm=True):
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

        activation : str or type or callable or None, default='relu'
            Activation function.

        norm : bool or type or callable, default=False
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
            norm=norm,
        )
        up = ResUpStep(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck=bottleneck,
            norm=norm,
            kernel_size=kernel_size,
            activation=activation,
            residual=residual_unpool,
            stride=stride)
        super().__init__(res, up)

    def forward(self, x, output_shape=None):
        res, up = self
        x = res(x)
        x = up(x, output_shape=output_shape)
        return x

    def shape(self, x, output_shape=None):
        res, up = self
        x = res.shape(x)
        x = up.shape(x, output_shape=output_shape)
        return x


@nitorchmodule
class UResBlock(Sequential):
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
            residual_pool=True,
            residual_unpool=True,
            activation=tnn.LeakyReLU(0.2),
            norm=True):
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

        residual_pool : default=True
            If True, the downsampling is "residual": it adds a convolved
            version of the previous layer to a linearly projected version
            (which is equivalent to the identity shortcut in a
            single-resolution resnet). If False, only the convolved version is
            passed to the next scale.

        residual_unpool : default=True
            If True, the upsampling step is "residual": it adds a convolved
            version of the previous layer to a linearly projected version
            (which is equivalent to the identity shortcut in a
            single-resolution resnet). If False, only the convolved version is
            passed to the next scale.

        activation : str or type or callable or None, default='relu'
            Activation function.

        norm : bool or type or callable, default=False
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
            norm=norm,
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
                residual_pool=residual_pool,
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
                residual_unpool=residual_unpool,
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
            layer, unpool = layer
            x = layer(x)
            if return_all:
                outputs.append(x)
                outputs.append(x)
            x = unpool(x, output_shape=intermediates[0].shape[2:])

        # last residual block
        x += intermediates.pop(0)
        x = last(x)

        return [x, *outputs[::-1]] if return_all else x
