"""Reusable layers for up-sampling/down-sampling"""

import torch.nn as tnn
import torch
from nitorch.core import py, utils
from ..base import Module, nitorchmodule
from .pool import Pool
from .conv import Conv
from .spatial import Resize


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

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (b, c, **spatial) tensor
        overload : dict
            `offset` and `stride` can be overloaded at call time

        Returns
        -------
        x : (b, c, **spatial_out) tensor

        """
        offset = overload.get('offset', self.offset)
        stride = overload.get('stride', self.stride)
        dim = x.dim() - 2
        offset = py.make_list(offset, dim)
        stride = py.make_list(stride, dim)
        slicer = [slice(o, ((sz-o)//st)*st, st) for o, st, sz in
                  zip(offset, stride, x.shape[2:])]
        slicer = [slice(None)]*2 + slicer
        return x[tuple(slicer)]

    def shape(self, x, **overload):
        """Output shape of the equivalent forward call.

        Parameters
        ----------
        x : (b, c, **spatial) tensor or sequence[int]
            A tensor or its shape.
        overload : dict
            `offset` and `stride` can be overloaded at call time

        Returns
        -------
        shape : tuple[int]
            (b, c, **spatial_out)
            In each dimension, the output shape is `(size-offset)//stride`.

        """
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        offset = overload.get('offset', self.offset)
        stride = overload.get('stride', self.stride)
        dim = len(x) - 2
        offset = py.make_list(offset, dim)
        stride = py.make_list(stride, dim)

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

    def forward(self, x, **overload):
        offset = overload.get('offset', self.offset)
        stride = overload.get('stride', self.stride)
        fill = overload.get('fill', self.fill)
        dim = x.dim() - 2
        offset = py.make_list(offset, dim)
        stride = py.make_list(stride, dim)

        new_shape = self.shape(x, **overload)
        y = x.new_zeros(new_shape)
        if fill:
            z = utils.unfold(y, stride)
            x = utils.unsqueeze(x, -1, dim)
            slicer = [slice(o, o+sz*st) for sz, st, o in
                      zip(x.shape[2:], stride, offset)]
            slicer = [slice(None)]*2 + slicer + [slice(None)]*dim
            z[tuple(slicer)].copy_(x)
        else:
            slicer = [slice(o, None, s) for o, s in zip(offset, stride)]
            slicer = [slice(None)]*2 + slicer
            y[tuple(slicer)] = x
        return y

    def shape(self, x, **overload):
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        offset = overload.get('offset', self.offset)
        stride = overload.get('stride', self.stride)
        output_padding = overload.get('output_padding', self.output_padding)
        output_shape = overload.get('output_shape', self.output_shape)
        dim = len(x) - 2
        offset = py.make_list(offset, dim)
        stride = py.make_list(stride, dim)
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
    - Subsampling + Channel-Conv
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
            batch_norm=False,
            groups=1,
            bias=False):
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

        batch_norm : [sequence of] bool, default=False
            Batch normalization before each convolution (if strided conv).

        pool : {'max', 'median', 'mean', 'sum', 'conv', None}, default=None
            Pooling used to change resolution:
            'max', 'median', 'mean', 'sum' : moving window + channel-conv
            'conv' : strided convolution
            None : subsampling + channel-conv

        groups : [sequence of] int, default=1
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.

        bias : bool, default=True
            Include a bias term in the convolution.

        """
        if not pool and (not in_channels or not out_channels):
            raise ValueError('Number of channels mandatory for strided conv')
        stride = py.make_list(stride, dim)

        if pool in ('max', 'median', 'mean', 'sum'):
            module = Pool(dim,
                          kernel_size=stride,
                          stride=stride)
        elif pool == 'conv':
            module = Conv(dim,
                          in_channels, out_channels,
                          kernel_size=stride,
                          bias=bias,
                          stride=stride,
                          activation=activation,
                          groups=groups,
                          batch_norm=batch_norm)
        else:
            module = Subsample(stride=stride)
        modules = [module]

        if pool != 'conv':
            modules.append(Conv(dim,
                                in_channels, out_channels,
                                kernel_size=1,
                                bias=bias,
                                activation=activation,
                                groups=groups,
                                batch_norm=batch_norm))

        super().__init__(*modules)

    def shape(self, x, **overload):
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        stride = overload.get('stride', self.stride)
        dim = len(x) - 2
        stride = py.make_list(stride, dim)

        x = [x[0], self.out_channels] + [xx//s for xx, s in zip(x[2:], stride)]
        return tuple(x)

    def forward(self, x, **overload):
        opt = ({'stride': overload.pop('stride')}
               if 'stride' in overload else {})
        if len(self) == 1:
            # strided conv
            conv, = self
            if 'stride' in opt:
                opt['kernel_size'] = opt['stride']
            x = conv(x, **overload, **opt)
        else:
            # pool|sub + conv
            pool, conv = self
            if not isinstance(pool, Subsample) and 'stride' in opt:
                opt['kernel_size'] = opt['stride']
            x = pool(x, **opt)
            x = conv(x, **overload)
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
            batch_norm=False,
            groups=1,
            bias=False):
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

        batch_norm : [sequence of] bool, default=False
            Batch normalization before each convolution (if strided conv).

        unpool : {'conv', 'up', 0..7}, default=0
            Unpooling used to change resolution:
            'conv' : strided convolution
            'up' : upsampling (without filling) + channel-conv
            0..7 : upsampling (of a given order) + channel-conv

        groups : [sequence of] int, default=1
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.

        bias : bool, default=True
            Include a bias term in the convolution.

        """
        if not in_channels or not out_channels:
            raise ValueError('Number of channels mandatory')
        out_channels = out_channels or in_channels
        stride = py.make_list(stride, dim)

        if unpool == 'conv':
            module = Conv(dim,
                          in_channels, out_channels,
                          kernel_size=stride,
                          bias=bias,
                          stride=stride,
                          transposed=True,
                          activation=activation,
                          groups=groups,
                          batch_norm=batch_norm,
                          output_padding=output_padding)
        elif isinstance(unpool, int):
            module = Resize(factor=stride,
                            anchor='f',
                            interpolation=unpool,
                            bound='zero',
                            extrapolate=False,
                            output_padding=output_padding)
        else:
            module = Upsample(stride=stride,
                              output_padding=output_padding)
        modules = [module]

        if unpool != 'conv':
            modules.append(Conv(dim,
                                in_channels, out_channels,
                                kernel_size=1,
                                bias=bias,
                                activation=activation,
                                groups=groups,
                                batch_norm=batch_norm))

        super().__init__(*modules)

    stride = property(lambda self: self[0].stride)
    output_padding = property(lambda self: self[0].output_padding)

    def shape(self, x, output_shape=None, **overload):
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        stride = overload.get('stride', self.stride)
        output_padding = overload.get('output_padding', self.output_padding)
        dim = len(x) - 2
        stride = py.make_list(stride, dim)
        output_padding = py.make_list(output_padding, dim)

        if output_shape:
            output_shape = py.make_list(output_shape, dim)
        else:
            output_shape = [sz*st + p for sz, st, p in
                            zip(x[2:], stride, output_padding)]
        return (*x[:2], *output_shape)

    def forward(self, x, output_shape=None, **overload):

        if output_shape:
            overload['output_padding'] = 0
            shape_nopad = self.shape(x, **overload)[2:]
            output_padding = [s1 - s0 for s1, s0 in
                              zip(output_shape, shape_nopad)]
            overload['output_padding'] = output_padding

        opt = ({'stride': overload.pop('stride')}
               if 'stride' in overload else {})
        if len(self) == 1:
            # strided conv
            conv, = self
            if 'stride' in opt:
                opt['kernel_size'] = opt['stride']
            x = conv(x, **overload, **opt)
        else:
            # pool|sub + conv
            pool, conv = self
            if not isinstance(pool, Upsample) and 'stride' in opt:
                opt['kernel_size'] = opt['stride']
            x = pool(x, **opt)
            x = conv(x, **overload)
        return x


