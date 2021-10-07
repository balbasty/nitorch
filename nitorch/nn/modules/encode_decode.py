"""Reusable layers for up-sampling/down-sampling"""

import torch.nn as tnn
import torch
from typing import Sequence, Optional, Union, Callable, Type
from nitorch.core import py, utils
from ..base import Module, nitorchmodule
from .pool import pool_map
from .conv import ConvBlock
from .spatial import Resize


ActivationLike = Union[str, Callable, Type]
NormalizationLike = Union[bool, str, Callable, Type]


class Subsample(Module):
    """Subsample the spatial dimensions:
        x -> x[..., o::s, o::s, o::s]
    """

    def __init__(self, offset=0, stride=2):
        """

        Parameters
        ----------
        offset : [sequence of] int, default=0
        stride : [sequence of] int, default=2
        """
        super().__init__()
        self.offset = offset
        self.stride = stride

    def forward(self, x):
        """

        Parameters
        ----------
        x : (b, c, **spatial) tensor

        Returns
        -------
        x : (b, c, **spatial_out) tensor

        """
        dim = x.dim() - 2
        offset = py.make_list(self.offset, dim)
        stride = py.make_list(self.stride, dim)
        slicer = [slice(o, ((sz-o)//st)*st, st) for o, st, sz in
                  zip(offset, stride, x.shape[2:])]
        slicer = [slice(None)]*2 + slicer
        return x[tuple(slicer)]

    def shape(self, x):
        """Output shape of the equivalent forward call.

        Parameters
        ----------
        x : (b, c, **spatial) tensor or sequence[int]
            A tensor or its shape.

        Returns
        -------
        shape : tuple[int]
            (b, c, **spatial_out)
            In each dimension, the output shape is `(size-offset)//stride`.

        """
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        dim = len(x) - 2
        offset = py.make_list(self.offset, dim)
        stride = py.make_list(self.stride, dim)

        x = x[:2] + [(xx-o)//s for xx, o, s in zip(x[2:], offset, stride)]
        return tuple(x)


class Upsample(Module):

    def __init__(self, offset=0, stride=2, output_padding=0,
                 output_shape=None, fill=True):
        """
        Only one of `output_padding` or `output_shape` should be provided.

        Parameters
        ----------
        offset : [sequence of] int, default=0
        stride : [sequence of] int, default=2
        output_padding : [sequence of] int, default=0
        output_shape : [sequence of] int, optional
        fill : bool, default=True
        """
        super().__init__()
        self.offset = offset
        self.stride = stride
        self.output_padding = output_padding
        self.output_shape = output_shape
        self.fill = fill

    def forward(self, x, output_padding=None, output_shape=None):
        """

        Parameters
        ----------
        x : (batch, channel, *in_spatial) tensor
        output_padding : [sequence of] int, default=self.output_padding
        output_shape : [sequence of] int, default=self.output_shape

        Returns
        -------
        x : (batch, channel, *out_spatial) tensor

        """
        dim = x.dim() - 2
        offset = py.make_list(self.offset, dim)
        stride = py.make_list(self.stride, dim)

        new_shape = self.shape(x, output_padding=output_padding,
                               output_shape=output_shape)
        y = x.new_zeros(new_shape)
        if self.fill:
            z = utils.unfold(y, stride)
            x = utils.unsqueeze(x, -1, dim)
            slicer = [slice(o, o+sz*st) for sz, st, o in
                      zip(x.shape[2:], stride, offset)]
            slicer = [slice(None)]*2 + slicer
            subz = z[tuple(slicer)]
            slicer = [slice(mx) for mx in subz.shape[2:]]
            slicer = [slice(None)]*2 + slicer
            subz.copy_(x[tuple(slicer)])
        else:
            slicer = [slice(o, None, s) for o, s in zip(offset, stride)]
            slicer = [slice(None)]*2 + slicer
            suby = y[tuple(slicer)]
            slicer = [slice(mx) for mx in suby.shape[2:]]
            slicer = [slice(None)]*2 + slicer
            suby.copy_(x[tuple(slicer)])

        return y

    def shape(self, x, output_padding=None, output_shape=None):
        """

        Parameters
        ----------
        x : (batch, channel, *in_spatial) tensor or sequence[int]
        output_padding : [sequence of] int, default=self.output_padding
        output_shape : [sequence of] int, default=self.output_shape

        Returns
        -------
        shape : tuple[int]

        """
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        if output_padding is not None and output_shape is not None:
            raise ValueError('Only one of `output_padding` or `output_shape` '
                             'should be provided.')
        elif output_padding is None and output_shape is None:
            output_padding = self.output_padding
            output_shape = self.output_shape
        dim = len(x) - 2
        offset = py.make_list(self.offset, dim)
        stride = py.make_list(self.stride, dim)
        output_padding = py.make_list(output_padding, dim)

        if output_shape:
            output_shape = py.make_list(output_shape, dim)
        else:
            output_shape = [sz*st + o + p for sz, st, o, p in
                            zip(x[2:], stride, offset, output_padding)]
        return (*x[:2], *output_shape)


@nitorchmodule
class DownStep(tnn.Sequential):
    """Down-sampling step

    Generic switch between downsampling modules. This function reduces
    spatial dimensions and changes the number of channels. By default,
    channel mapping is linear (no activation, no bias). Possible
    methods are:
    - Pooling + Channel-Conv
    - Strided conv

    """

    def __init__(
            self,
            dim,
            in_channels=None,
            out_channels=None,
            stride=2,
            pool=None,
            activation=None,
            norm=False,
            groups=1,
            bias=False,
            order='nac'):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int, optional if `pool`
            Number of input channels.

        out_channels : int, optional if `pool`
            Number of output channels.

        stride : int or sequence[int], default=2
            Up/Downsampling factor.

        activation : [sequence of] str or type or callable, default='relu'
            Activation function (if strided conv).

        norm : [sequence of] bool, default=False
            Batch normalization before each convolution (if strided conv).

        pool : pool_like or int or None, default=None
            Pooling used to change resolution:
            - pool_like ('max', 'mean', ...)' : moving window + channel-conv
            - int : strided convolution with this kernel size
            - None : strided convolution with same kernel size as stride size

        groups : [sequence of] int, default=1
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            channels.

        bias : bool, default=True
            Include a bias term in the convolution.

        order : permutation of 'nac', default='nac'
            Order in which to perform the normalization ('n'),
            activation ('a') and convolution ('c').

        """
        if not pool and (not in_channels or not out_channels):
            raise ValueError('Number of channels mandatory for strided conv')
        stride = py.make_list(stride, dim)

        opt = dict(dim=dim,
                   in_channels=in_channels,
                   out_channels=out_channels,
                   bias=bias,
                   activation=activation,
                   groups=groups,
                   norm=norm,
                   order=order)

        if pool in pool_map:
            Pool = pool_map(pool)
            modules = [Pool(dim, kernel_size=stride, stride=stride),
                       ConvBlock(kernel_size=1, stride=1, **opt)]
        else:
            pool = pool or stride
            modules = [ConvBlock(kernel_size=pool, stride=stride, **opt)]
        super().__init__(*modules)

    def shape(self, x):
        """

        Parameters
        ----------
        x : (batch, channel, *in_spatial) tensor

        Returns
        -------
        shape : tuple[int]

        """
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        dim = len(x) - 2
        stride = py.make_list(self.stride, dim)

        x = [x[0], self.out_channels] + [xx//s for xx, s in zip(x[2:], stride)]
        return tuple(x)

    def forward(self, x):
        if len(self) == 1:
            # strided conv
            conv, = self
            x = conv(x)
        else:
            # pool + conv
            pool, conv = self
            x = pool(x)
            x = conv(x)
        return x


@nitorchmodule
class UpStep(tnn.Sequential):
    """Up-sampling step

    Generic switch between upsampling modules. This function augments
    spatial dimensions and changes the number of channels. By default,
    channel mapping is linear (no activation, no bias). Possible
    methods are:
    - UnPooling + Channel-Conv
    - Upsampling + Channel-Conv
    - Resizing + Channel-Conv
    - Strided transposed conv

    """

    def __init__(
            self,
            dim,
            in_channels=None,
            out_channels=None,
            stride=2,
            output_padding=0,
            unpool=None,
            activation=None,
            norm=False,
            groups=1,
            bias=False,
            order='nac'):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int, optional if `pool`
            Number of input channels.

        out_channels : int, optional if `pool`
            Number of output channels.

        stride : int or sequence[int], default=2
            Upsampling factor.

        output_padding : [sequence of] int, defualt=0

        activation : str or type or callable, default='relu'
            Activation function

        norm : [sequence of] bool, default=False
            Batch normalization before each convolution (if strided conv).

        unpool : interpolation_like or int or None, default=None
            Pooling used to change resolution:
            - interpolation_like ('nearest', 'linear')' : upsampling + channel-conv
            - int : strided convolution with this kernel size
            - None : strided convolution with same kernel size as stride size

        groups : [sequence of] int, default=1
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.

        bias : bool, default=True
            Include a bias term in the convolution.

        order : permutation of 'nac', default='nac'
            Order in which to perform the normalization ('n'),
            activation ('a') and convolution ('c').

        """
        if not in_channels or not out_channels:
            raise ValueError('Number of channels mandatory')
        out_channels = out_channels or in_channels
        stride = py.make_list(stride, dim)

        opt = dict(dim=dim,
                   in_channels=in_channels,
                   out_channels=out_channels,
                   bias=bias,
                   activation=activation,
                   groups=groups,
                   norm=norm,
                   order=order)

        if unpool in ('nearest', 'linear'):
            modules = [Resize(factor=stride,
                              anchor='f',
                              interpolation=unpool,
                              bound='zero',
                              extrapolate=False,
                              output_padding=output_padding),
                       ConvBlock(kernel_size=1, stride=1, **opt)]
        else:
            unpool = unpool or stride
            modules = [ConvBlock(kernel_size=unpool, stride=stride, **opt)]

        super().__init__(*modules)

    stride = property(lambda self: self[0].stride)
    output_padding = property(lambda self: self[0].output_padding)

    def shape(self, x, output_shape=None):
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        dim = len(x) - 2
        stride = py.make_list(self.stride, dim)

        if output_shape:
            output_shape = py.make_list(output_shape, dim)
        else:
            output_shape = [sz*st for sz, st in zip(x[2:], stride)]
        return (*x[:2], *output_shape)

    def forward(self, x, output_shape=None):

        if len(self) == 1:
            # strided conv
            conv, = self
            x = conv(x, output_shape=output_shape)
        else:
            # resize + conv
            resize, conv = self
            factor = [1/s for s in py.make_list(self.stride)]
            x = resize(x, output_shape=output_shape, factor=factor)
            x = conv(x)
        return x
