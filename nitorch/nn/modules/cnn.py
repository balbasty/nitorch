"""Convolutional neural networks (UNet, VAE, etc.)."""

import inspect
import math
from collections import OrderedDict
import torch
from typing import Sequence, Optional, Union, Callable, Type, TypeVar
from torch import nn as tnn
from nitorch.core import py, utils, linalg
from nitorch import io, spatial
from nitorch.nn.base import nitorchmodule, Module, Sequential
from ..activations import make_activation
from .conv import ConvBlock, Conv
from .norm import BatchNorm
from .pool import Pool, MaxPool
from .reduction import reductions, Reduction
from .spatial import Resize
from .encode_decode import Upsample
from .. import check


ActivationLike = Union[str, Callable, Type]
NormalizationLike = Union[bool, str, Callable, Type]
_T = TypeVar('_T')
ScalarOrSequence = Union[_T, Sequence[_T]]


def interleaved_cat(tensors, dim=0, groups=1):
    """Split tensors into `groups` chunks and concatenate them
    in an interleaved manner.

    If the input is [x, y, z] and `groups=2`, they are chunked into
    (x1, x2), (y1, y2), (z1, z2) and concatenated as [x1, y1, z1, x2,y2, z2]

    Parameters
    ----------
    tensors : sequence[tensor]
        Input tensors
    dim : int, default=0
        Dimension along which to split/concatenate
    groups : int, default=1
        Tensors are split into that many chunks

    Returns
    -------
    tensor

    """
    if groups == 1:
        return torch.cat(tensors, dim=dim)
    tensors = [t.chunk(groups, dim) for t in tensors]
    tensors = [chunk for chunks in zip(*tensors) for chunk in chunks]
    return torch.cat(tensors, dim=dim)


def expand_list(x, n, crop=False, default=None, use_default=False):
    """Expand ellipsis in a list by substituting it with the value
    on its left, repeated as many times as necessary. By default,
    a "virtual" ellipsis is present at the end of the list.

    expand_list([1, 2, 3],       5)            -> [1, 2, 3, 3, 3]
    expand_list([1, 2, ..., 3],  5)            -> [1, 2, 2, 2, 3]
    expand_list([1, 2, 3, 4, 5], 3, crop=True) -> [1, 2, 3]
    """
    x = list(x)
    if Ellipsis not in x:
        x.append(Ellipsis)
    idx_ellipsis = x.index(Ellipsis)
    if idx_ellipsis == 0 or use_default:
        fill_value = default
    else:
        fill_value = x[idx_ellipsis-1]
    k = len(x) - 1
    x = (x[:idx_ellipsis] + 
         [fill_value] * max(0, n-k) + 
         x[idx_ellipsis+1:])
    if crop:
        x = x[:n]
    return x


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

    def __init__(self, input_groups: int = 2, output_groups: int = 2):
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
        input = linalg.matvec(self.weight.t(), input)
        input = utils.movedim(input, -1, 1).reshape([batch, out_channels, *spatial])

        return input


@nitorchmodule
class StackedConv(tnn.ModuleList):
    """Multiple convolutions at the same resolution followed by 
    a up- or down- sampling, using either a strided convolution or 
    a pooling operation.

    By default, padding is used so that convolutions with stride 1
    preserve spatial dimensions. Strided convolutions do not use padding.

    (Norm? > [Grouped]Conv > Dropout?(Activation)? > Stitch?)* >
    (Norm? > [Grouped](StridedConv|Conv > Pool) > Dropout?(Activation)? > Stitch?)
    """
    
    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: ScalarOrSequence[int],
            kernel_size: ScalarOrSequence[int] = 3,
            stride: ScalarOrSequence[int] = 1,
            transposed: bool = False,
            pool: Optional[str] = None,
            activation: ScalarOrSequence[Optional[ActivationLike]] = tnn.ReLU,
            norm: ScalarOrSequence[NormalizationLike] = None,
            groups: ScalarOrSequence[Optional[int]] = None,
            stitch: ScalarOrSequence[int] = 1,
            bias: ScalarOrSequence[bool] = True,
            dropout: ScalarOrSequence[float] = 0,
            residual: bool = False,
            return_last: bool = False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int
            Number of input channels.
            
        out_channels : int or sequence[int]
            Number of output channels in each convolution.
            If a sequence, multiple convolutions are performed.
            
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
            
        activation : [sequence of] str or type or callable, default='relu'
            Activation function.
            
        norm : [sequence of] {'batch', 'instance', 'layer', 'group'}, default=None
            Normalization before each convolution.
            
        stride : int or sequence[int], default=1
            Up to one value per spatial dimension.
            `output_shape \approx input_shape // stride`
            
        transposed : bool, default=False
            Make the strided convolution a transposed convolution.
            
        pool : {'max', 'min', 'median', 'mean', 'sum', 'up', 'down', 'conv', None}, default=None
            Pooling used to change resolution:
                - 'max', 'min', 'median', 'mean', 'sum' : pooling
                - 'up' : bilinear upsmapling
                - 'down' : strided downsampling
                - 'conv' : learnable convolution with kernel size `stride`
                        and as many output as input channels.
                - None : the final convolution is a strided convolution
                    with kernel size `kernel_size`.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per convolution.

        bias : [sequence of] int, default=True
            Include a bias term in the convolution.

        dropout : float or sequence[float] or type or callable, default=0
            Apply dropout (if 0 < p <= 1)

        residual : bool, default=False
            Add residual connections between convolutions.
            This has no effect if only one convolution is performed.
            No residual connection is applied to the output of the last
            layer (strided conv or pool).

        return_last : {'single', 'cat', 'single+cat'} or bool, default=False
            Return the last output before up/downsampling on top of the
            real output (useful for skip connections).

            'single' and 'cat' are useful when the stacked convolution contains
            a single strided convolution and takes as input a skipped
            connection. 'single' only returns the first input argument
            whereas 'cat' returns all concatenated input arguments.
            `True` is equivalent to 'single'.

        """
        self.dim = dim
        self.residual = residual
        self.return_last = return_last

        out_channels = py.make_list(out_channels)
        in_channels = [in_channels] + out_channels[:-1]
        nb_layers = len(out_channels)
        
        stitch = map(lambda x: x or 1, py.make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(py.make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(py.make_list(activation), nb_layers, default='relu')
        norm = expand_list(py.make_list(norm), nb_layers, default=False)
        dropout = expand_list(py.make_list(dropout), nb_layers, default=0, use_default=True)        
        bias = expand_list(py.make_list(bias), nb_layers, default=True)
        
        if pool not in (None, 'up', 'conv') and transposed:
            raise ValueError('Cannot have both `pool` and `transposed`.')
      
        all_shapes = zip(
            in_channels, 
            out_channels, 
            activation,
            norm,
            dropout,
            groups, 
            stitch,
            bias)
        *all_shapes, final_shape = all_shapes
        
        # stacked conv (without strides)
        modules = []
        for d, (i, o, a, bn, do, g, s, b) in enumerate(all_shapes):
            modules.append(ConvBlock(
                dim, i, o, kernel_size,
                activation=a,
                norm=bn,
                dropout=do,
                padding='same',
                groups=g,
                bias=b))
            if s > 1:
                modules.append(Stitch(s, s))
        
        # last conv (strided if not pool)
        i, o, a, bn, do, g, s, b = final_shape
        modules.append(ConvBlock(
            dim, i, o, kernel_size,
            transposed=transposed and not pool,
            activation=a,
            norm=bn,
            dropout=do,
            stride=1 if pool else stride,
            padding='same' if pool or (stride == 1) else 'valid',
            groups=g,
            bias=b))
        
        # pooling
        if pool:
            if pool == 'up':
                modules.append(Resize(factor=stride, anchor='f'))
            elif pool == 'down':
                stride = [1/s for s in py.make_list(stride)]
                modules.append(Resize(factor=stride, anchor='f', interpolation=0))
            elif pool == 'conv':
                modules.append(ConvBlock(
                    dim, o, o, stride,
                    transposed=transposed,
                    activation=None,
                    norm=bn,
                    dropout=do,
                    stride=stride,
                    padding='valid',
                    groups=g,
                    bias=False))
            else:
                modules.append(Pool(
                    dim, stride,
                    stride=stride,
                    activation=None))

        # final stitch
        if s > 1:
            modules.append(Stitch(s, s))
                
        super().__init__(modules)
        
    @property
    def stride(self):
        for layer in reversed(self):
            if isinstance(self, (Pool, ConvBlock)):
                return layer.stride

    @property
    def in_channels(self):
        for layer in self:
            if isinstance(layer, ConvBlock):
                return layer.in_channels

    @property
    def out_channels(self):
        for layer in reversed(self):
            if isinstance(layer, ConvBlock):
                return layer.out_channels

    @property
    def out_channels_last(self):
        for layer in reversed(self):
            if isinstance(layer, ConvBlock):
                return layer.in_channels

    def shape(self, x):
        if torch.is_tensor(x):
            x = tuple(x.shape)
        for layer in self:
            if isinstance(layer, (ConvBlock, Pool, Resize)):
                x = layer.shape(x)
        return x
    
    def forward(self, *x, output_shape=None, return_last=None):
        """

        Parameters
        ----------
        x : (B, Ci, *spatial_in) tensor
            Input tensor.
            If multiple tensors are provided, they are concatenated
            along the channel dimension.

        Other parameters
        ----------------
        output_shape : sequence[int], optional
            Shape of the output tensor. Only useful if the last convolution
            is transposed, or if 'down' pooling is used.
        return_last : bool or {'single', 'cat', 'single+cat'}, optional
            Whether to return the last

        Returns
        -------
        output : (B, Co, *spatial_out) tensor
            Convolved tensor
        last : (B, C2, *spatial_in) tensor, if `return_last`
            Last output before the final up/downsampling
        last : (B, C2, *spatial_in) tensor, if `return_last`
            Last output before the final up/downsampling

        """
        def is_last(layer):
            if isinstance(layer, (Pool, Resize)):
                return True
            if isinstance(layer, ConvBlock):
                if not all(s == 1 for s in py.make_list(layer.stride)):
                    return True
            return False

        if return_last is None:
            return_last = self.return_last
        if not isinstance(return_last, str):
            return_last = 'single' if return_last else ''

        last = []
        if 'single' in return_last:
            last.append(x[0])
        x = torch.cat(x, 1) if len(x) > 1 else x[0]
        if 'cat' in return_last:
            last.append(x)
        for layer in self:
            if isinstance(layer, ConvBlock) and layer.transposed:
                kwargs = dict(output_shape=output_shape)
            elif isinstance(layer, Resize):
                kwargs = dict(output_shape=output_shape)
            else:
                kwargs = {}

            if self.residual:
                identity = x
                x = layer(x, **kwargs)
                x += identity
            else:
                x = layer(x, **kwargs)
            if return_last and not is_last(layer):
                last = [x]
                if 'single' in return_last and 'cat' in return_last:
                    last = last * 2

        return (x, *last) if return_last else x
    
        
class EncodingLayer(StackedConv):
    """An encoding layer is a layer that performs one downsampling step 
    (along with other operations such as batch norm, convolutions, 
     activation, etc.)
     
    (BatchNorm? > [Grouped]Conv > Activation? > Stitch?)* > 
    (BatchNorm? > [Grouped](StridedConv|(Conv > Pool)) > Activation? > Stitch?)
    """

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            pool=None,
            activation=tnn.ReLU,
            norm=None,
            groups=None,
            stitch=1,
            bias=True,
            residual=False,
            return_last=False):
        
        super().__init__(
            dim, 
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool=pool,
            activation=activation,
            norm=norm,
            groups=groups,
            stitch=stitch,
            bias=bias,
            residual=residual,
            return_last=return_last,
        )

        
class DecodingLayer(StackedConv):
    """An decoding layer is a layer that performs one upsampling step 
    (along with other operations such as batch norm, convolutions, 
     activation, etc.)
     
    (BatchNorm? > [Grouped]Conv > Activation? > Stitch?)* > 
    (BatchNorm? > [Grouped]TransposedConv > Activation? > Stitch?)
    """
    
    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            unpool=None,
            activation=tnn.ReLU,
            norm=None,
            groups=None,
            stitch=1,
            bias=True,
            residual=False,
            return_last=False):
        
        super().__init__(
            dim, 
            in_channels, 
            out_channels,
            transposed=True,
            kernel_size=kernel_size,
            stride=stride,
            pool=unpool,
            activation=activation,
            norm=norm,
            groups=groups,
            stitch=stitch,
            bias=bias,
            residual=residual,
            return_last=return_last,
        )


@nitorchmodule
class Encoder(tnn.Sequential):
    """Encoder network (for U-nets, VAEs, etc.).

    Notes
    -----
    .. The encoder is made of `n` encoding layer.
    .. Each encoding layer is made of `k >= 0` convolutions followed by one
       strided convolution _or_ pooling.
    .. The activation function that follows the very last convolution/pooling
       can be different from the other ones.
    .. If normalization is activated, it is performed before each
       convolution.
    .. Grouped convolutions and stitching units can be used at each layer.

    Examples
    --------
    Encoder(dim, 2, [8, 16])
        (B, 2) -> Conv(stride=2) + ReLU -> (B, 8)
               -> Conv(stride=2) + ReLU -> (B, 16)
    Encoder(dim, 2, [[8, 8], [16, 16], [32, 32]], activation=[..., None])
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
            in_channels,
            out_channels,
            skip_channels=None,
            kernel_size=3,
            stride=2,
            pool=None,
            activation=tnn.ReLU,
            norm=None,
            groups=None,
            stitch=1):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
            
        in_channels : int or sequence[int]
            Number of input channels.
            If a sequence, the first convolution is a grouped convolution.
            
        out_channels : sequence[int]
            Number of output channels in each encoding layer.
            Elements can be integers or nested sequences.
            If nested sequences, multiple convolutions are performed at the
            same resolution.
            
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
            
        activation : [sequence of] str or type or callable, default='relu'
            Activation function.
            
        norm : [sequence of] {'batch', 'instance', 'layer', 'group'}, default=None
            Normalization before each convolution.
            
        stride : int or sequence[int], default=2
            Spatial dimensions are divided by this number after each
            encoding layer.
            
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, use strided convolutions.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per layer. 
            If > 1, a grouped convolution is performed. 
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per layer. 

        """
        self.dim = dim
                
        out_channels = list(map(py.make_list, out_channels))
        in_channels = [in_channels] + [c[-1] for c in out_channels[:-1]]
        nb_layers = len(out_channels)
        
        stitch = map(lambda x: x or 1, py.make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(py.make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(py.make_list(activation), nb_layers, default='relu')
        norm = expand_list(py.make_list(norm), nb_layers, default=False)

        # deal with skipped connections (what a nightmare...)
        skip_channels = py.make_list(skip_channels or [])
        self.skip = len(skip_channels) > 0
        self.skip_groups = []
        for i in range(len(skip_channels)):
            self.skip_groups.append(groups[i])
            if len(in_channels) > i+1:
                in_channels[i] += skip_channels[i]

        all_shapes = zip(
            in_channels, 
            out_channels, 
            activation,
            norm,
            groups, 
            stitch)
                               
        modules = []
        for i, o, a, b, g, s in all_shapes:
            modules.append(EncodingLayer(
                dim, i, o, kernel_size,
                stride=stride,
                pool=pool,
                activation=a,
                norm=b,
                groups=g,
                stitch=[..., s]))
            # If the layer is grouped, all convolutions in that 
            # layer are grouped. However, I only stitch once 
            # (at the end of the encoding layer)

        super().__init__(*modules)

                               
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
    .. If Normalization is activated, it is performed before each
       convolution.
    .. The `skip_channels` option can be used so that the the input is a
       list of tensors that are concatenated after each transposed
       convolution. Padding is used to ensure that tensors can be
       concatenated.


    Examples
    --------
    Decoder(dim, 16, [8, 2])
        (B, 16) -> ConvT(stride=2) + ReLU -> (B, 8)
                -> ConvT(stride=2) + ReLU -> (B, 2)
    Decoder(dim, 32, [[16, 16], [8,8], 2], activation=[..., None])
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
            in_channels,
            out_channels,
            skip_channels=None,
            kernel_size=3,
            stride=2,
            unpool=None,
            activation=tnn.ReLU,
            norm=None,
            groups=None,
            stitch=1):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
            
        in_channels : int or sequence[int]
            Number of input channels.
            If a sequence, the first convolution is a grouped convolution.
            
        out_channels : sequence[int or sequence[int]]
            Number of channels in each decoding layer.
            There will be len(channels) layers.
            
        skip_channels : sequence[int or sequence[int]]
            Number of skipped channels per layer,
            If an element is a sequence, it comes from a grouped conv.
            
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.

        stride : int or sequence[int], default=2
            Stride per dimension.

        unpool : {'up', 'conv', None}, default=None
            - 'up': Linear upsampling
            - 'conv': strided convolution with kernel size `stride`
            - None: strided convolution with bias, act and `kernel size`

        activation : [sequence of] str or type or callable, default='relu'
            Activation function.
            
        norm : [sequence of] {'batch', 'instance', 'layer', 'group'}, default=None
            Normalization before each convolution.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per layer. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per layer.
        """                     
        self.dim = dim
                
        out_channels = list(map(py.make_list, out_channels))
        in_channels = [in_channels]
        in_channels += [c[-1] for c in out_channels[:-1]]
        nb_layers = len(out_channels)
        
        stitch = map(lambda x: x or 1, py.make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(py.make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(py.make_list(activation), nb_layers, default='relu')
        norm = expand_list(py.make_list(norm), nb_layers, default=False)

        # deal with skipped connections (what a nightmare...)
        skip_channels = py.make_list(skip_channels or [])
        self.skip = len(skip_channels) > 0
        self.skip_groups = []
        for i in range(len(skip_channels)):
            self.skip_groups.append(groups[i])
            if len(in_channels) > i+1:
                in_channels[i+1] += skip_channels[i]

        all_shapes = zip(
            in_channels, 
            out_channels, 
            activation,
            norm,
            groups, 
            stitch)
                               
        modules = []
        for i, o, a, b, g, s in all_shapes:
            modules.append(DecodingLayer(
                dim, i, o, kernel_size,
                stride=stride,
                unpool=unpool,
                activation=a,
                norm=b,
                groups=g,
                stitch=[..., s]))
            # If the layer is grouped, all convolutions in that 
            # layer are grouped. However, I only stitch once 
            # (at the end of the decoding layer)

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

        # Layers with skipped connections
        groups = list(self.skip_groups)
        for i, layer in enumerate(decoder):
            inp1 = inputs.pop(0)
            x = layer(x, output_shape=inp1.shape[2:])
            x = interleaved_cat((x, inp1), dim=1, groups=groups.pop(0))

        # Post-processing (convolutions without skipped connections)
        for layer in postproc:
            x = layer(x)

        return x


@nitorchmodule
class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(
            self,
            dim: int,
            in_channels: Union[int, Sequence[int]],
            out_channels: Union[int, Sequence[int]],
            encoder: Optional[Sequence[int]] = None,
            decoder: Optional[Sequence[int]] = None,
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 2,
            activation=tnn.ReLU,
            pool: Optional[str] = None,
            unpool: Optional[str] = None,
            norm=None,
            groups: Optional[int] = None,
            stitch: int = 1):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
            
        in_channels : int or sequence[int]
            Number of input channels.
            If a sequence, the first convolution is a grouped convolution.
            
        out_channels : int or sequence [int]
            Number of output channels.
            If a sequence, the last convolution is a grouped convolution.
            
        encoder : sequence[int], default=[16, 32, 32, 32]
            Number of channels in each encoding layer.
            
        decoder : sequence[int], default=[32, 32, 32, 32, 32, 16, 16]
            Number of channels in each decoding layer.
            If the number of decoding layer is larger than the number of
            encoding layers, stacked convolutions are appended to the
            UNet.
            
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
            
        activation : [sequence of] str or type or callable or None, default='relu'
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
                
        pool : {'max', 'min', 'median', 'mean', 'sum', 'down', None}, default=None
            Pooling to use in the encoder.
            If 'down', use strided convolution with same kernel size as stride.
            If None, use strided convolutions with same kernel size as other conv.

        unpool : {'up', None}, default=None
            If 'down', use strided convolution with same kernel size as stride.
            If None, use strided convolutions with same kernel size as other conv.

        norm : {'batch', 'instance', 'layer', 'group'}, default=None
            Normalization before each convolution.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per layer. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per layer.
        """
        self.dim = dim

        in_channels = py.make_list(in_channels)
        out_channels = py.make_list(out_channels)

        # defaults
        encoder = list(encoder or [16, 32, 32, 32])
        decoder = list(decoder or [32, 32, 32, 32, 32, 16, 16])

        # ensure as many upsampling steps as downsampling steps
        decoder = expand_list(decoder, len(encoder), crop=False)
        encoder = list(map(py.make_list, encoder))
        encoder_out = list(map(lambda x: x[-1], reversed(encoder)))
        encoder_out.append(sum(in_channels))
        decoder = list(map(py.make_list, decoder))
        stack = py.flatten(decoder[len(encoder):])
        decoder = decoder[:len(encoder)]

        nb_layers = len(encoder) + len(decoder) + len(stack)
        
        stitch = map(lambda x: x or 1, py.make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        stitch[-1] = 1  # do not stitch last layer
        groups = expand_list(py.make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        groups[0] = len(in_channels)    # first layer
        groups[-1] = len(out_channels)  # last layer
        activation = expand_list(py.make_list(activation), nb_layers, default='relu')

        range_e = slice(len(encoder))
        range_d = slice(len(encoder), len(encoder) + len(decoder))
        range_s = slice(len(encoder) + len(decoder), -1)
        stitch_encoder = stitch[range_e]
        stitch_decoder = stitch[range_d]
        stitch_stack = stitch[range_s]
        groups_encoder = groups[range_e]
        groups_decoder = groups[range_d]
        groups_stack = groups[range_s]
        activation_encoder = activation[range_e]
        activation_decoder = activation[range_d]
        activation_stack = activation[range_s]

        activation_final = activation[-1]   
        
        modules = []
        enc = Encoder(dim,
                      in_channels=in_channels,
                      out_channels=encoder,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation_encoder,
                      norm=norm,
                      pool=pool,
                      groups=groups_encoder,
                      stitch=stitch_encoder)
        modules.append(('encoder', enc))

        e_groups = reversed(groups_encoder)
        d_groups = groups_decoder[1:] + (groups_stack[:1] or [len(out_channels)])
        skip_repeat = [max(1, gd // ge) for ge, gd in 
                       zip(e_groups, d_groups)]
        skip_channels = [e * r for e, r in zip(encoder_out[1:], skip_repeat)]
        self.skip_repeat = [1] + skip_repeat
        
        dec = Decoder(dim,
                      in_channels=encoder_out[0],
                      out_channels=decoder,
                      skip_channels=skip_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      unpool=unpool,
                      activation=activation_decoder,
                      norm=norm,
                      groups=groups_decoder,
                      stitch=stitch_decoder)
        modules.append(('decoder', dec))

        last_decoder = decoder[-1][-1] + skip_channels[-1]
        if stack:
            stk = StackedConv(dim,
                              in_channels=last_decoder,
                              out_channels=stack,
                              kernel_size=kernel_size,
                              activation=activation_stack,
                              norm=norm,
                              groups=groups_stack,
                              stitch=stitch_stack)
        else:
            stk = Cat()
        modules.append(('stack', stk))

        input_final = stack[-1] if stack else last_decoder   
        input_final = py.make_list(input_final)
        if len(input_final) == 1:
            input_final = [input_final[0]//len(out_channels)] * len(out_channels)
        stk = ConvBlock(dim, input_final, out_channels,
                        kernel_size=kernel_size,
                        activation=activation_final,
                        norm=norm,
                        padding='same')
        modules.append(('final', stk))

        super().__init__(OrderedDict(modules))
        
    def forward(self, x):
        
        # encoder
        encoder_out = []
        for layer in self.encoder:
            x, y = layer(x, return_last=True)
            encoder_out.append(y)
        encoder_out.append(x)
        # repeat skipped values if decoder has split
        x = list(reversed(encoder_out))
        x = [x1.repeat([1, r] + [1]*self.dim) if r > 1. else x1
             for x1, r in zip(x, self.skip_repeat)]
        # decoder
        x = self.decoder(x)
        x = self.stack(x)
        x = self.final(x)
        return x


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
    .. If Normalization is activated, it is performed before each
       encoding convolution.

    """

    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            encoder: Optional[Sequence[int]] = None,
            stack: Optional[Sequence[int]] = None,
            kernel_size=3,
            stride=2,
            pool: Optional[str] = None,
            reduction='max',
            activation=None,
            norm=None,
            dropout: float = 0):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int
            Number of input channels.

        out_channels : int
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

        norm : {'batch', 'instance', layer'} or int, default=None
            Normalization before each convolution.
            If an int, group noramlization is used.

        dropout : sequence[float], default=0
            Dropout probability.

        """
        self.dim = dim

        encoder = list(encoder or [16, 32, 32, 32])
        stack = list(stack or [32, 16, 16])
        last_encoder = py.make_list(encoder[-1])[-1]
        stack = stack + [out_channels]
        if isinstance(reduction, str):
            reduction = reductions.get(reduction, None)
        reduction = reduction(keepdim=True) if inspect.isclass(reduction) \
                    else reduction if callable(reduction) \
                    else None
        if not isinstance(reduction, Reduction):
            raise TypeError('reduction must be a `Reduction` module.')

        nb_layers = len(encoder) + len(stack)
        if activation is None:
            activation = ['relu', ..., None]
        activation = expand_list(py.make_list(activation), nb_layers, default='relu')
        activation_encoder = activation[:len(encoder)]
        activation_stack = activation[len(encoder):]

        modules = []
        enc = Encoder(dim,
                      in_channels=in_channels,
                      out_channels=encoder,
                      kernel_size=kernel_size,
                      stride=stride,
                      pool=pool,
                      activation=activation_encoder,
                      norm=norm)
        modules.append(('encoder', enc))

        modules.append(('reduction', reduction))

        stk = StackedConv(dim,
                          in_channels=last_encoder,
                          out_channels=stack,
                          kernel_size=1,
                          activation=activation_stack,
                          dropout=dropout,
                         )
        modules.append(('stack', stk))

        super().__init__(OrderedDict(modules))

    def forward(self, input):
        """

        Parameters
        ----------
        input : (batch, in_channels, *spatial) tensor

        Returns
        -------
        output : (batch, out_channels) tensor

        """
        input = super().forward(input)
        input = input.reshape(input.shape[:2])
        return input


@nitorchmodule
class MRF(tnn.Sequential):
    """Categorical MRF network."""

    def __init__(self, dim, num_classes, num_filters=16, num_extra=1,
                 kernel_size=3, activation=tnn.LeakyReLU(0.2),
                 batch_norm=False, bias=False):
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
        activation : [sequence of] str or type or callable, default=`LeakyReLU(0.2)`
            Activation function.
        batch_norm : [sequence of] bool or type or callable, default=False
            Batch normalization before each convolution.
        bias : [sequence of] bool, default=False
            Adds learnable bias to the output.
        """
        self.dim = dim

        # preprocess parameters

        kernel_size = py.make_list(kernel_size, dim)
        if not all(k % 2 for k in kernel_size):
            raise ValueError(f'MRF kernel size must be odd. Got {kernel_size}.')

        activation = expand_list(py.make_list(activation), 2 + num_extra,
                                 default=tnn.LeakyReLU(0.2))
        mrf_activation, *activation = activation

        batch_norm = expand_list(py.make_list(batch_norm), 2 + num_extra,
                                 default=False)
        mrf_batch_norm, *batch_norm = batch_norm

        bias = expand_list(py.make_list(bias), 2 + num_extra, default=False)
        mrf_bias, *bias = bias

        # make layers
        modules = []

        module = ConvBlock(dim,
                           in_channels=num_classes,
                           out_channels=num_filters,
                           kernel_size=kernel_size,
                           activation=mrf_activation,
                           norm=mrf_batch_norm,
                           bias=mrf_bias,
                           padding='same')

        center = tuple(k//2 for k in kernel_size)
        center = (slice(None),) * 2 + center
        module.weight.data[center] = 0            # zero-centre
        modules.append(('mrf', module))

        module = StackedConv(
            dim,
            in_channels=num_filters,
            out_channels=[num_filters] * num_extra + [num_classes],
            kernel_size=1,
            activation=activation,
            norm=batch_norm,
            bias=bias)
        modules.append(('extra', module))

        # build model
        super().__init__(OrderedDict(modules))


class Cat(Module):

    def forward(self, *x, return_last=False, **k):
        last = x[0]
        x = torch.cat(x, 1) if len(x) > 1 else x[0]
        return (x, last) if return_last else x


class Add(Module):

    def forward(self, *x, return_last=False, **k):
        last = x[0]
        x = sum(x) if len(x) > 1 else x[0]
        return (x, last) if return_last else x


@nitorchmodule
class UNet2(tnn.Sequential):
    """Alternative U-Net.

     x -*1-> ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~> *1 -> y
    (?) ~~~> -*2-> -*1-> ~~~~~~~~~~~~~~~> *1 -> -*2^
    (?) ~~~~~~~~~~~~~~~> -*2-> -*1-> -*2^

    The difference with the `UNet` class are:
    - There is always at least one non-strided convolution at the top level
      (at the next levels, there can be only strided conv, like in `UNet`)
    - There are `len(encoder)` resolution levels and therefore
      `len(encoder) - 1` strided convolutions (instead of
      `len(encoder) + 1` levels and `len(encoder)` strided conv in `UNet`).
    - There is an option to perform multiple "convolution + activation"
      at each level.
    - There are a lot less fancy options (groups, stitches, pooling, etc).
    - There can be one input per level.
    """

    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            encoder: Optional[Sequence[int]] = None,
            decoder: Optional[Sequence[int]] = None,
            skip_decoder_level: int = 0,
            conv_per_layer: int = 1,
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 2,
            pool: Optional[str] = None,
            unpool: Optional[str] = None,
            activation=tnn.ReLU,
            norm=None):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : [sequence of] int
            Number of input channels.
            If a sequence, inputs are provided at multiple scales.

        out_channels : int
            Number of output channels.

        encoder : sequence[int], default=[16, 32, 32, 32]
            Number of channels in each encoding layer.

        decoder : sequence[int], optional
            Number of channels in each decoding layer.
            Default: symmetric of encoder

        conv_per_layer : int, default=1
            Number of convolutions per layer.

        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.

        stride : int or sequence[int], default=2
            Stride per dimension.

        pool : {'max, 'down', 'conv', None}, default=None
            Downsampling method.

        unpool : {'up', 'conv', None}, default=None
            Upsampling method.

        activation : str or type or callable or None, default='relu'
            Activation function.

        norm : {'batch', 'instance', 'layer'} or int, default=None
            Normalization before each convolution.
            In an int, group normalization is used.
        """
        self.dim = dim

        default_encoder = [16, 32, 32, 32]
        default_decoder = [32, 32, 32, 16, 16, 16]
        
        # defaults
        conv_per_layer = max(1, conv_per_layer)
        encoder = list(encoder or default_encoder)
        nb_scales = len(encoder)
        decoder = py.make_list(decoder or default_decoder,
                            n=nb_scales-1, crop=False)
        stack = decoder[len(encoder)-1-skip_decoder_level:]
        decoder = encoder[-1:] + decoder[:len(encoder)-1-skip_decoder_level]

        in_channels = py.make_list(in_channels, n=nb_scales, default=0)

        modules = []
        if not pool:
            first = ConvBlock(dim,
                              in_channels=in_channels[0],
                              out_channels=encoder[0],
                              kernel_size=kernel_size,
                              activation=activation,
                              norm=norm,
                              padding='same')
            modules.append(('first', first))

        modules_encoder = []
        for n in range(nb_scales-1):
            if pool:
                cin = encoder[n-1] if n > 0 else in_channels[0]
                cout = [encoder[n]] * conv_per_layer
            else:
                cin = encoder[n]
                cout = encoder[n+1]
                cout = [encoder[n]] * (conv_per_layer - 1) + [cout]
            if n > 0 or pool:
                cin += in_channels[n]
            modules_encoder.append(EncodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                pool=pool,
                activation=activation,
                norm=norm,
            ))
        enc = tnn.ModuleList(modules_encoder)
        modules.append(('encoder', enc))

        if pool and unpool:
            cin = encoder[-2]
            cout = [decoder[0]] * conv_per_layer
        elif pool:
            cin = encoder[-2]
            cout = decoder[1]
            cout = [decoder[0]] * (conv_per_layer - 1) + [cout]
        elif unpool:
            cin = decoder[0]
            cout = [decoder[0]] * conv_per_layer
        else:
            cin = decoder[0]
            cout = decoder[1]
            cout = [decoder[0]] * (conv_per_layer - 1) + [cout]
        cin += in_channels[-1]
        btk = DecodingLayer(
            dim,
            in_channels=cin,
            out_channels=cout,
            kernel_size=kernel_size,
            stride=stride,
            unpool=unpool,
            activation=activation,
            norm=norm,
        )
        modules.append(('bottleneck', btk))

        _, *decoder = decoder
        *encoder, _ = encoder

        modules_decoder = []
        for n in range(len(decoder)-1):
            if unpool:
                cin = decoder[n-1] if n > 0 else cout[-1]
                cout = [decoder[n]] * conv_per_layer
            else:
                cin = decoder[n]
                cout = decoder[n+1]
                cout = [decoder[n]] * (conv_per_layer - 1) + [cout]
            cin += encoder.pop(-1)
            modules_decoder.append(DecodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                unpool=unpool,
                activation=activation,
                norm=norm,
            ))
        dec = tnn.ModuleList(modules_decoder)
        modules.append(('decoder', dec))

        if unpool:
            cin = decoder[-2]
            cout = [decoder[-1]] * conv_per_layer
        else:
            cin = decoder[-1]
            cout = [decoder[-1]] * (conv_per_layer - 1)
        cin += encoder.pop(-1)
        for s in stack:
            cout += [s] * conv_per_layer
        if cout:
            stk = StackedConv(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
            )
            modules.append(('stack', stk))
            last_stack = cout[-1]
        else:
            modules.append(('stack', Cat()))
            last_stack = cin

        final = ConvBlock(dim, last_stack, out_channels,
                          kernel_size=kernel_size,
                          padding='same')
        modules.append(('final', final))

        super().__init__(OrderedDict(modules))
        self.verbose = False

    def forward(self, *x, return_all=False, verbose=None):
        """

        Parameters
        ----------
        x : (batch, in_channels, *spatial) tensor
            Input tensor
        return_all : bool, default=False
            Return outputs at all scales

        Returns
        -------
        x : [tuple of] (batch, out_channels, *spatial) tensor
            Output tensor
            If `return_all`, tuple of tensors ordered from finest to
            coarsest scale.

        """
        if verbose is None:
            verbose = self.verbose

        all_x = list(x)
        if hasattr(self, 'first'):
            x = all_x.pop(0)
            if verbose:
                print('first:', list(x.shape), end=' -> ', flush=True)
            x = self.first(x)
            if verbose:
                print(list(x.shape))

        # encoder
        buffers = []
        for n, layer in enumerate(self.encoder):
            if all_x and not (hasattr(self, 'first') and n == 0):
                x = [x, all_x.pop(0)]
            else:
                x = [x]
            if verbose:
                print('encoder:', *[list(xx.shape) for xx in x], end=' -> ', flush=True)
            x, buffer = layer(*x, return_last=True)
            if verbose:
                print(list(x.shape))
            buffers.append(buffer)

        x = [x, all_x.pop(0)] if all_x else [x]
        if verbose:
            print('bottleneck:', *[list(xx.shape) for xx in x], end=' -> ', flush=True)
        x = self.bottleneck(*x, output_shape=buffers[-1].shape[2:],
                            return_last=return_all)
        if return_all:
            x, tmp = x
            if verbose:
                print(list(x.shape))
            buffers.insert(0, tmp)
        elif verbose:
            print(list(x.shape))

        # decoder
        for layer in self.decoder:
            buffer = buffers.pop()
            if verbose:
                print('decoder:', list(x.shape), list(buffer.shape), end=' -> ', flush=True)
            x = layer(x, buffer,
                      output_shape=buffers[-1].shape[2:],
                      return_last=return_all)
            if return_all:
                x, tmp = x
                if verbose:
                    print(list(x.shape))
                buffers.insert(0, tmp)
            elif verbose:
                print(list(x.shape))

        buffer = buffers.pop()
        if verbose:
            print('stack:', list(x.shape), list(buffer.shape), end=' -> ', flush=True)
        x = self.stack(x, buffer)
        if verbose:
            print(list(x.shape))
        if verbose:
            print('final:', list(x.shape), end=' -> ', flush=True)
        x = self.final(x)
        if verbose:
            print(list(x.shape))
        if return_all:
            buffers.insert(0, x)
            return tuple(buffers)
        return x


@nitorchmodule
class UUNet(tnn.Sequential):
    """Iterative U-Net with two-way skip connections."""

    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            encoder: Optional[Sequence[int]] = None,
            decoder: Optional[Sequence[int]] = None,
            conv_per_layer: int = 1,
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 2,
            activation=tnn.ReLU,
            norm=None,
            residual: bool = False,
            nb_iter: int = 1):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        encoder : sequence[int], default=[16, 32, 32, 32]
            Number of channels in each encoding layer.

        decoder : sequence[int], optional
            Number of channels in each decoding layer.
            Default: symmetric of encoder

        conv_per_layer : int, default=2
            Number of convolutions per layer.

        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.

        activation : [sequence of] str or type or callable or None, default='relu'
            Activation function.

        norm : {'batch', 'instance', 'layer'}, default=None
            Normalization before each convolution.
            In an int, group normalizarion is used.

        residual : bool, default=False
            Use residual skipped connections
        """
        self.dim = dim
        self.residual = residual
        self.nb_iter = nb_iter

        # defaults
        conv_per_layer = max(1, conv_per_layer)
        encoder = list(encoder or [16, 32, 32, 32])
        decoder = py.make_list(decoder or list(reversed(encoder[:-1])),
                            n=len(encoder)-1, crop=False)
        stack = decoder[len(encoder)-1:]
        decoder = encoder[-1:] + decoder[:len(encoder)]
        activation, final_activation = py.make_list(activation, 2)

        modules = []
        first = ConvBlock(dim,
                          in_channels=in_channels,
                          out_channels=encoder[0],
                          kernel_size=kernel_size,
                          activation=activation,
                          norm=norm,
                          padding='same')
        modules.append(('first', first))

        modules_encoder = []
        for n in range(len(encoder)-1):
            cin = encoder[n] + decoder[-n-1]
            cout = encoder[n+1]
            cout = [encoder[n]] * (conv_per_layer - 1) + [cout]
            modules_encoder.append(EncodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            ))
        enc = tnn.ModuleList(modules_encoder)
        modules.append(('encoder', enc))

        cin = decoder[0]
        cout = decoder[1]
        cout = [decoder[0]] * (conv_per_layer - 1) + [cout]
        btk = DecodingLayer(
            dim,
            in_channels=cin,
            out_channels=cout,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            norm=norm,
        )
        modules.append(('bottleneck', btk))

        _, *decoder = decoder
        *encoder, _ = encoder

        modules_decoder = []
        for n in range(len(decoder)-1):
            cin = decoder[n] + encoder[-n-1]
            cout = decoder[n+1]
            cout = [decoder[n]] * (conv_per_layer - 1) + [cout]
            modules_decoder.append(DecodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            ))
        dec = tnn.ModuleList(modules_decoder)
        modules.append(('decoder', dec))

        cin = decoder[-1] + encoder[0]
        cout = [decoder[-1]] * (conv_per_layer - 1)
        for s in stack:
            cout += [s] * conv_per_layer
        if cout:
            stk = StackedConv(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
            )
            modules.append(('stack', stk))
            last_stack = cout[-1]
        else:
            modules.append(('stack', Cat()))
            last_stack = cin

        final = ConvBlock(dim, last_stack, out_channels,
                          kernel_size=kernel_size,
                          activation=final_activation,
                          padding='same')
        modules.append(('final', final))

        super().__init__(OrderedDict(modules))

    def forward_once(self, x):

        x = self.first(x)

        # encoder
        buffers = []
        for d, layer in enumerate(self.encoder):
            buffer_shape = list(x.shape)
            buffer_shape[1] = layer.in_channels - buffer_shape[1]
            buffer = x.new_zeros(buffer_shape)
            x, buffer = layer(x, buffer, return_last=True)
            buffers.append(buffer)

        # decoder
        for layer in self.decoder:
            buffer = buffers.pop()
            x = layer(x, buffer, output_shape=buffers[-1].shape[2:])

        x = self.stack(x, buffers.pop())
        x = self.final(x)
        return x

    def forward(self, x, nb_iter=None):

        if nb_iter is None:
            nb_iter = self.nb_iter
        if nb_iter == 1:
            return self.forward_once(x)

        buffers_encoder = [None] * len(self.encoder)
        buffers_decoder = [None] * len(self.decoder)

        x0 = self.first(x)
        for n_iter in range(nb_iter):
        
            x = x0
        
            # encoder
            for d, layer in enumerate(self.encoder):
                buffer = buffers_decoder[-d-1]
                if buffer is None:
                    buffer_shape = list(x.shape)
                    buffer_shape[1] = layer.in_channels - buffer_shape[1]
                    buffer = x.new_empty(buffer_shape).normal_(std=1/2.355)
                    buffers_decoder[-d-1] = buffer
                x, buffer = layer(x, buffer, return_last=True)
                if buffers_encoder[d] is None or not self.residual:
                    buffers_encoder[d] = buffer
                else:
                    buffers_encoder[d] = buffers_encoder[d] + buffer

            x = self.bottleneck(x, output_shape=buffers_encoder[-1].shape[2:])

            # decoder
            for d, layer in enumerate(self.decoder):
                buffer = buffers_encoder[-d-1]
                x, buffer = layer(x, buffer, return_last=True,
                                  output_shape=buffers_encoder[-d-2].shape[2:])
                if buffers_decoder[d] is None or not self.residual:
                    buffers_decoder[d] = buffer
                else:
                    buffers_decoder[d] = buffers_decoder[d] + buffer

            x = self.stack(x, buffers_encoder[0])
            if buffers_decoder[-1] is None or not self.residual:
                buffers_decoder[-1] = x
            else:
                buffers_decoder[-1] = buffers_decoder[-1] + x

        del x0
        x = self.final(x)
        return x


@nitorchmodule
class WNet(tnn.Sequential):
    """W-net (= cascaded U-Net) with skip connections between the nets."""

    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            encoder: Optional[Sequence[int]] = None,
            decoder: Optional[Sequence[int]] = None,
            encoder2: Optional[Sequence[int]] = None,
            decoder2: Optional[Sequence[int]] = None,
            conv_per_layer: int = 1,
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 2,
            activation=tnn.ReLU,
            norm=None,
            skip: bool = True):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        mid_channels : int, optional
            Number of output channels after the first U-Net.

        encoder : sequence[int], default=[16, 32, 32, 32]
            Number of channels in each encoding layer of the first U-Net.
            The length of `encoder` defines the number of resolution levels.
            The first value is the number of features after an initial
            (stride 1) convolution at the top level. Subsequence values
            are the number of features after each downsampling step.
            The last value is the number of features in the bottleneck.

        decoder : sequence[int], optional
            Number of channels in each decoding layer of the first U-Net.
            Default: symmetric of encoder (e.g., [32, 32, 16]).
            If more than `len(encoder)-1` values are provided,
            additional (stride 1) convolutions are performed at the final
            level.

        encoder2 : sequence[int], optional
            Number of channels in each encoding layer of the second U-Net.
            The length of `encoder2` defines the number of resolution levels.
            Default: same as encoder, without the top level (e.g., [32, 32, 32])

        decoder2 : sequence[int], optional
            Number of channels in each encoding layer of the second U-Net.
            The length of `encoder2` defines the number of resolution levels.
            Default: symmetric of encoder2, plus the top level
            (e.g., [32, 32, 16])

        conv_per_layer : int, default=2
            Number of convolutions per layer.

        kernel_size : int, default=3
            Kernel size (in all dimensions).
            If a list/tuple of two elements, the second element is
            the final kernel size.

        activation : str or type or callable or None, default='relu'
            Activation function.
            If a list/tuple of two elements, the second element is
            the final activation.

        norm : {'batch', 'instance', 'layer'} or int, default=False
            Normalization before each convolution.

        skip : bool, default=True
            Add skip connections between the two U-Nets.
        """
        self.dim = dim
        self.skip = skip
        encoder1 = encoder
        decoder1 = decoder

        # defaults
        conv_per_layer = max(1, conv_per_layer)
        default_encoder = [16, 32, 32, 32]
        encoder1 = list(encoder1 or default_encoder)
        default_decoder = list(reversed(encoder1[:-1]))
        decoder1 = py.make_list(decoder1 or default_decoder, 
                             n=len(encoder1) - 1, crop=False)
        default_encoder2 = encoder1[1:]
        encoder2 = list(encoder2 or default_encoder2)
        default_decoder2 = decoder1
        decoder2 = py.make_list(decoder2 or default_decoder2, 
                             n=len(encoder2), crop=False)

        stack1 = decoder1[len(encoder1) - 1:]
        decoder1 = decoder1[:len(encoder1) - 1]
        stack2 = decoder2[len(encoder2):]
        decoder2 = decoder2[:len(encoder2)]
        activation, final_activation = py.make_list(activation, 2)
        kernel_size, final_kernel_size = py.make_list(kernel_size, 2)

        modules = OrderedDict()

        # --- initial feature extraction --------------------------------
        modules['first'] = ConvBlock(
            dim,
            in_channels=in_channels,
            out_channels=encoder1[0],
            kernel_size=kernel_size,
            activation=activation,
            norm=norm,
            padding='same')

        # --- first unet -----------------------------------------------
        modules_encoder = []
        for n in range(len(encoder1) - 1):
            cin = encoder1[n]
            cout = encoder1[n + 1]
            cout = [encoder1[n]] * (conv_per_layer - 1) + [cout]
            modules_encoder.append(EncodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            ))
        modules['encoder1'] = tnn.ModuleList(modules_encoder)

        cin = encoder1[-1]
        cout = decoder1[0]
        cout = [encoder1[-1]] * (conv_per_layer - 1) + [cout]
        modules['bottleneck1'] = DecodingLayer(
            dim,
            in_channels=cin,
            out_channels=cout,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            norm=norm,
        )

        modules_decoder = []
        *encoder1, bottleneck1 = encoder1
        for n in range(len(decoder1) - 1):
            cin = decoder1[n] + encoder1[-n - 1]
            cout = decoder1[n + 1]
            cout = [decoder1[n]] * (conv_per_layer - 1) + [cout]
            modules_decoder.append(DecodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            ))
        modules['decoder1'] = tnn.ModuleList(modules_decoder)

        # --- second unet ----------------------------------------------
        modules_encoder = []
        # first level -> connects first unet
        cin = decoder1[-1] + encoder1[0]
        cout = [decoder1[-1]] * (conv_per_layer - 1)
        for s in stack1:
            cout += [s] * conv_per_layer
        cout += [encoder2[0]]
        modules_encoder.append(EncodingLayer(
            dim,
            in_channels=cin,
            out_channels=cout,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            norm=norm,
        ))
        # next levels -> skip connections
        for n in range(len(encoder2) - 1):
            skip_channels = decoder1[-n-2]
            in_channels = modules_encoder[-1].out_channels
            cin = in_channels + (skip_channels if skip else 0)
            cout = encoder2[n + 1]
            cout = [encoder2[n]] * (conv_per_layer - 1) + [cout]
            modules_encoder.append(EncodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            ))
        modules['encoder2'] = tnn.ModuleList(modules_encoder)

        cin = encoder2[-1] + (bottleneck1 if skip else 0)
        cout = decoder2[0]
        cout = [encoder2[-1]] * (conv_per_layer - 1) + [cout]
        modules['bottleneck2'] = DecodingLayer(
            dim,
            in_channels=cin,
            out_channels=cout,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            norm=norm,
        )

        modules_decoder = []
        *encoder2, bottleneck2 = encoder2
        for n in range(len(decoder2) - 1):
            cin = decoder2[n] + encoder2[-n - 1]
            cout = decoder2[n + 1]
            cout = [decoder2[n]] * (conv_per_layer - 1) + [cout]
            modules_decoder.append(DecodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            ))
        modules['decoder2'] = tnn.ModuleList(modules_decoder)

        # stack -> connects
        cin = decoder2[-1] + (stack1[-1] if stack1 else decoder1[-1])
        cout = [decoder2[-1]] * (conv_per_layer - 1)
        for s in stack2:
            cout += [s] * conv_per_layer
        if cout:
            modules['stack2'] = StackedConv(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
            )
            last_stack = cout[-1]
        else:
            modules['stack2'] = Cat()
            last_stack = cin

        # --- final layer ----------------------------------------------
        modules['final'] = ConvBlock(
            dim, last_stack, out_channels,
            kernel_size=final_kernel_size,
            activation=final_activation,
            padding='same')

        # --- middle output --------------------------------------------
        if mid_channels:
            modules['middle'] = ConvBlock(
                dim, modules['encoder2'][0].out_channels_last, mid_channels,
                kernel_size=final_kernel_size,
                activation=final_activation,
                padding='same')

        super().__init__(modules)

    def forward(self, x, **overload):

        x = self.first(x)

        # encoder1
        buffers_encoder = []
        for layer in self.encoder1:
            x, buffer = layer(x, return_last=True)
            buffers_encoder.append(buffer)

        # bottleneck 1
        oshape = buffers_encoder[-1].shape[2:]
        x, buffer = self.bottleneck1(x, return_last=True, output_shape=oshape)

        # decoder1
        buffers_decoder = [buffer] if self.skip else []
        for layer in self.decoder2:
            buffer = buffers_encoder.pop()
            if buffers_encoder:
                oshape = buffers_encoder[-1].shape[2:]
            else:
                oshape = None
            x, buffer = layer(x, buffer, return_last=True, output_shape=oshape)
            if self.skip:
                buffers_decoder.append(buffer)

        # encoder2
        buffers_decoder = buffers_decoder + buffers_encoder
        buffers_encoder = []
        buffer_middle = None
        for layer in self.encoder2:
            buffer = [buffers_decoder.pop()] if buffers_decoder else []
            if buffer_middle is None and hasattr(self, 'middle'):
                x, buffer, inmiddle = layer(x, *buffer, return_last='single+cat')
                buffer_middle = self.middle(inmiddle)
                del inmiddle
            else:
                x, buffer = layer(x, *buffer, return_last=True)
            buffers_encoder.append(buffer)

        # bottleneck 2
        buffer = [buffers_decoder.pop()] if buffers_decoder else []
        oshape = buffers_encoder[-1].shape[2:]
        x = self.bottleneck2(x, *buffer, output_shape=oshape)

        # decoder2
        for layer in self.decoder2:
            buffer = buffers_encoder.pop()
            if buffers_encoder:
                oshape = buffers_encoder[-1].shape[2:]
            else:
                oshape = None
            x = layer(x, buffer, output_shape=oshape)

        x = self.stack2(x, *buffers_encoder)
        x = self.final(x)
        return x if buffer_middle is None else (x, buffer_middle)


class SEWNet(Module):
    """SEW: Siamese-Encoder W-Net

    A first (siamese) U-Net is used to extract "features" from grouped
    inputs. The generated features are then concatenated and fed to a second
    U-Net.

    Optionally, skip-connections between the encoding U-Net and decoding
    U-Net can be used.

    Schematically, it goes like this:
    ```
    Input1  ~~~[FeatUNet]~~~>  Feat1 \
                                      ~~[Cat]~~>  Feat  ~~~[MainNet]~~~> Output
    Input2  ~~~[FeatUNet]~~~>  Feat2 /
    ```
    """

    def __init__(
            self,
            dim: int,
            nb_twins: int,
            in_channels: int,
            out_channels: int,
            feat_channels: Optional[int] = None,
            encoder: Optional[Sequence[int]] = None,
            decoder: Optional[Sequence[int]] = None,
            encoder2: Optional[Sequence[int]] = None,
            decoder2: Optional[Sequence[int]] = None,
            skip: bool = False,
            **kwargs):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.

        nb_twins : int
            Number of repeats of the siamese network.

        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        feat_channels : int, optional
            Number of output "feature" channels after the first U-Net.
            By default, the last value of `encoder` is used.

        encoder : sequence[int], default=[16, 32, 32, 32]
            Number of channels in each encoding layer of the first U-Net.
            The length of `encoder` defines the number of resolution levels.
            The first value is the number of features after an initial
            (stride 1) convolution at the top level. Subsequence values
            are the number of features after each downsampling step.
            The last value is the number of features in the bottleneck.

        decoder : sequence[int], optional
            Number of channels in each decoding layer of the first U-Net.
            Default: symmetric of encoder (e.g., [32, 32, 16]).
            If more than `len(encoder)-1` values are provided,
            additional (stride 1) convolutions are performed at the final
            level.

        encoder2 : sequence[int], optional
            Number of channels in each encoding layer of the second U-Net.
            The length of `encoder2` defines the number of resolution levels.
            Default: same as encoder, without the top level (e.g., [32, 32, 32])

        decoder2 : sequence[int], optional
            Number of channels in each encoding layer of the second U-Net.
            The length of `encoder2` defines the number of resolution levels.
            Default: symmetric of encoder2, plus the top level
            (e.g., [32, 32, 16])

        skip : bool, default=True
            Add skip connections between the two U-Nets.

        Other Parameters
        ----------------
        conv_per_layer : int, default=2
            Number of convolutions per layer.

        kernel_size : int, default=3
            Kernel size (in all dimensions).
            If a list/tuple of two elements, the second element is
            the final kernel size.

        pool : {'max, 'down', 'conv', None}, default=None
            Downsampling method.

        unpool : {'up', 'conv', None}, default=None
            Upsampling method.

        activation : str or type or callable or None, default='relu'
            Activation function.

        norm : {'batch', 'instance', 'layer'} or int, default=None
            Normalization before each convolution.

        """
        super().__init__()

        self.dim = dim
        self.skip = skip
        self.nb_twins = nb_twins

        encoder1 = encoder
        decoder1 = decoder

        # defaults
        default_encoder = [16, 32, 32, 32]
        encoder1 = list(encoder1 or default_encoder)
        default_decoder = list(reversed(encoder1[:-1]))
        decoder1 = py.make_list(decoder1 or default_decoder,
                             n=len(encoder1) - 1, crop=False)
        default_encoder2 = encoder1[1:]
        encoder2 = list(encoder2 or default_encoder2)
        default_decoder2 = decoder1
        decoder2 = py.make_list(decoder2 or default_decoder2,
                             n=len(encoder2), crop=False)
        if not feat_channels:
            feat_channels = decoder1.pop()

        # feature extraction
        self.siamese = UNet2(dim, in_channels, feat_channels,
                             encoder=encoder1, decoder=decoder1, **kwargs)

        # compute number of input channels (per scale)
        siamese_out = [feat_channels]
        if skip:
            siamese_out.insert(1, self.siamese.bottleneck.out_channels)
            for layer in self.siamese.decoder[:-1]:
                siamese_out.insert(1, layer.out_channels)
        siamese_out = [n*nb_twins for n in siamese_out]

        # feature fusion
        self.fusion = UNet2(dim, siamese_out, out_channels,
                            encoder=encoder2, decoder=decoder2, **kwargs)

    def forward(self, *x, return_feat=False, verbose=False):
        """

        Parameters
        ----------
        *x : (batch, in_channels, *spatial) tensor
            `nb_twins` input tensors.
        return_feat : bool, default=False
            Return output after the first unet

        Returns
        -------
        out : (batch, out_channels, *spatial) tensor
            Output tensor
        feat : (batch, feat_channels, *spatial) tensor, if `return_feat`
            Feature tensor
        """
        # check inputs
        for y in x[1:]:
            check.shape(x[0], y)
        if len(x) != self.nb_twins:
            raise ValueError(f'Expected {self.nb_layers} inputs but got '
                             f'{len(x)}.')

        # treat repeats as batches
        x = torch.cat(x)

        # first unet
        x = self.siamese(x, return_all=self.skip, verbose=verbose)
        x = py.make_list(x)

        # convert batch to channels
        x = [xx.chunk(self.nb_twins) for xx in x]
        feat = x[0] if return_feat else None
        x = [torch.cat(xx, dim=1) for xx in x]

        # second unet
        x = self.fusion(*x, verbose=verbose)

        return (x, feat) if return_feat else x


class NeuriteResMerge(Sequential):
    """Residual merge

    This is a utility class for flexible residual paths:
        - If `in_channels` and `out_channels` are equal, this is a simple
          add + activation.
        - If `in_channels` and `out_channels` differ, the identity
          path goes through a convolution that matches the number of
          output channels.
    """
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 padding='same',
                 dropout=0,
                 activation='elu'):
        """

        Parameters
        ----------
        dim : int
            Number of spatial dimensions
        in_channels : int
            Number of channels in the identity path
        out_channels : int
            Number of channels in the conv path
        kernel_size : int or sequence[int], default=3
        dilation : int or sequence[int], default=1
        padding : {'same', 'valid'} or int ir sequence[int], default='same'
        dropout : float in 0..1, default=0
        activation : str or class or callable, default='elu'
        """
        res_block = []
        if (in_channels > 1 and out_channels > 1
                and (in_channels != out_channels)):
            res_block.append(['conv', Conv(
                dim, in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )])
            if activation:
                res_block.append(
                    ('activation', make_activation(activation)))
            if dropout:
                res_block.append(('dropout', tnn.Dropout(dropout)))
        res_block.append(('add', Add()))
        if activation:
            res_block.append(('activation', make_activation(activation)))
        super().__init__(OrderedDict(res_block))

    def forward(self, identity, branch):
        """

        Parameters
        ----------
        identity : (batch, in_channels, *spatial)
        branch : (batch, out_channels, *spatial)

        Returns
        -------
        out : (batch, out_channels, *spatial)

        """
        for layer in self:
            if isinstance(layer, Add):
                identity = layer(identity, branch)
            else:
                identity = layer(identity)
        return identity


class NeuriteConv(Sequential):
    """A single convolutional "sandwich". """
    def __init__(self,
                 dim,                           # int
                 in_channels,                   # int
                 out_channels,                  # int
                 kernel_size=3,                 # int or sequence[int]
                 dilation=1,                    # int
                 padding='same',                # {'valid', 'same'} or int or sequence[int]
                 activation='elu',              # str or class or callable
                 residual=False,                # bool
                 dropout=0,                     # float in 0..1
                 batch_norm=None,               # bool or 'str' or class or callable
                 ):

        conv_block = []
        # --------------------------------------------------------------
        # Convolution
        # --------------------------------------------------------------
        conv_block.append(('conv', Conv(
            dim, in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )))
        # --------------------------------------------------------------
        # Residual path
        # --------------------------------------------------------------
        # Voxelmorph variant of the residual path:
        # `out = Activation(Conv(x) + [Conv](x))`
        if residual:
            conv_block.append(('residual', NeuriteResMerge(
                dim, in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                activation=None,
                dropout=0,
            )))
        # --------------------------------------------------------------
        # Activation
        # --------------------------------------------------------------
        if activation:
            conv_block.append(
                ('activation', make_activation(activation)))
        # --------------------------------------------------------------
        # Dropout
        # --------------------------------------------------------------
        if dropout:
            conv_block.append(
                ('dropout', tnn.Dropout(dropout)))
        # --------------------------------------------------------------
        # Batch Norm
        # --------------------------------------------------------------
        if batch_norm:
            conv_block.append(('norm', BatchNorm(
                dim, out_channels, eps=1e-3, momentum=0.01)))
        super().__init__(OrderedDict(conv_block))

    def forward(self, x):
        children_names = [name for name, _ in self.named_children()]
        identity = x if ('residual' in children_names) else None
        for layer in self:
            if isinstance(layer, NeuriteResMerge):
                x, identity = layer(identity, x), None
            else:
                x = layer(x)
        return x


class NeuriteConvMulti(Sequential):
    """Multiple sequential convolutions"""
    def __init__(self,
                 dim,                           # int
                 in_channels,                   # int
                 out_channels,                  # int or sequence[int]
                 nb_conv=1,                     # int
                 kernel_size=3,                 # int or sequence[int]
                 dilation=1,                    # int
                 padding='same',                # {'valid', 'same'} or int or sequence[int]
                 activation='elu',              # str or class or callable
                 residual=False,                # bool or 'vxm'
                 dropout=0,                     # float in 0..1
                 batch_norm=None,               # bool or 'str' or class or callable
                 ):
        all_out_channels = list(py.ensure_list(out_channels, nb_conv))

        # --------------------------------------------------------------
        # Choose residual variant
        # --------------------------------------------------------------
        # . VoxelMorph's UNet performs residual connections around each
        #   convolution.
        # . Neurite's UNet performs residual connections around blocks
        #   of sequential convolutions.
        residual_inner = residual == 'vxm'
        residual_outer = residual and not residual_inner
        in_channels_first = in_channels

        blocks = []
        for i in range(nb_conv):
            # ----------------------------------------------------------
            # One convolution
            # ----------------------------------------------------------
            out_channels = all_out_channels.pop(0)
            has_activation = i < nb_conv-1 or not residual_outer
            blocks.append(NeuriteConv(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                activation=activation if has_activation else None,
                residual=residual_inner,
                dropout=dropout,
            ))
            in_channels = out_channels
        blocks = [('conv_multi', tnn.Sequential(*blocks))]
        # --------------------------------------------------------------
        # Global residual path
        # --------------------------------------------------------------
        # This type of residual connection is used in Neurite's UNet
        if residual_outer:
            blocks.append(('residual', NeuriteResMerge(
                dim=dim,
                in_channels=in_channels_first,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                dropout=dropout,
                activation=activation,
            )))
        # --------------------------------------------------------------
        # Batch Norm
        # --------------------------------------------------------------
        # We do as in Neurite and only perform one BatchNorm for the
        # whole block
        if batch_norm:
            blocks.append(('norm', BatchNorm(
                dim, out_channels, eps=1e-3, momentum=0.01)))
        super().__init__(OrderedDict(blocks))

    def forward(self, x):
        children_names = [name for name, _ in self.named_children()]
        identity = x if ('residual' in children_names) else None
        for layer in self:
            if isinstance(layer, NeuriteResMerge):
                x, identity = layer(identity, x), None
            else:
                x = layer(x)
        return x


class NeuriteEncoder(Sequential):
    def __init__(self,
                 dim,
                 in_channels,
                 nb_levels=5,
                 kernel_size=3,
                 nb_feat=16,
                 feat_mult=1,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 residual=False,
                 nb_conv_per_level=1,
                 dropout=0,
                 batch_norm=None):

        if isinstance(nb_feat, (list, tuple)):
            nb_feat = list(nb_feat)  # trigger a copy + ensure popable

        encoder_module = []
        for level in range(nb_levels):
            level_block = []

            # ----------------------------------------------------------
            # Sequential convolutions (with activation, norm, dropout)
            # ----------------------------------------------------------
            if not isinstance(nb_feat, list):
                out_channels = int(round(nb_feat * feat_mult ** level))
            else:
                out_channels = nb_feat[:nb_conv_per_level]
                nb_feat = nb_feat[nb_conv_per_level:]
            dilation = dilation_rate_mult ** level

            level_block.append(('conv_block', NeuriteConvMulti(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                nb_conv=nb_conv_per_level,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                activation=activation,
                residual=residual,
                dropout=dropout,
            )))
            if isinstance(out_channels, list):
                in_channels = out_channels[-1]
            else:
                in_channels = out_channels
            if batch_norm:
                # Batch Norm must be done after the conv because
                # we propagate pre-norm features in the skip connection
                level_block.append(('batch_norm', BatchNorm(
                    dim, in_channels, eps=1e-3, momentum=0.01
                )))

            # ----------------------------------------------------------
            # Max pooling
            # ----------------------------------------------------------
            if level < nb_levels - 1:
                level_block.append(('pool', MaxPool(
                    dim, pool_size, padding=padding, ceil=True)))

            level_block = tnn.Sequential(OrderedDict(level_block))
            encoder_module.append(level_block)
        super().__init__(*encoder_module)

    def forward(self, x, return_skip=False):
        out = []
        for n, level in enumerate(self):
            for i, blocks in enumerate(level):
                x = blocks(x)
                if return_skip and i == 0 and n != len(self) - 1:
                    # first block is the conv block -> save out
                    out.append(x)
        return (x, *reversed(out)) if out else x


class NeuriteDecoder(Sequential):
    def __init__(self,
                 dim,                           # int
                 in_channels,                   # int or list[int]
                 out_channels,                  # int
                 nb_levels=5,                   # int
                 kernel_size=3,                 # int or sequence[int]
                 nb_feat=16,                    # int or sequence[int]
                 feat_mult=1,                   # int
                 pool_size=2,                   # int or sequence[int]
                 padding='same',                # {'valid', 'same'} or int or sequence[int]
                 dilation_rate_mult=1,          # int
                 activation='elu',              # str or class or callable
                 final_activation='softmax',    # str ot class or callable
                 final_kernel_size=1,           # int or sequence[int]
                 residual=False,                # bool or 'vxm'
                 nb_conv_per_level=1,           # int
                 dropout=0,                     # float in 0..1
                 batch_norm=None,               # bool or 'str' or class or callable
                 ):

        if isinstance(nb_feat, (list, tuple)):
            nb_feat = list(nb_feat)  # trigger a copy + ensure popable

        decoder_module = []
        in_channels = list(py.ensure_list(in_channels))
        nb_channels_in = in_channels.pop(0)

        # --------------------------------------------------------------
        # Decoding blocks
        # --------------------------------------------------------------
        for level in range(nb_levels-1):
            level_block = []

            # ----------------------------------------------------------
            # Upsample and Concatenate
            # ----------------------------------------------------------
            level_block.append(('up', Upsample(stride=pool_size)))
            if in_channels:
                level_block.append(('cat', Cat()))
                nb_channels_in = nb_channels_in + in_channels.pop(0)

            # ----------------------------------------------------------
            # Convolution blocks
            # ----------------------------------------------------------
            if not isinstance(nb_feat, list):
                nb_channels_out = int(round(nb_feat * feat_mult ** (nb_levels - 2 - level)))
            else:
                nb_channels_out = nb_feat[:nb_conv_per_level]
                nb_feat = nb_feat[nb_conv_per_level:]
            dilation = dilation_rate_mult ** (nb_levels - 2 - level)

            level_block.append(('conv_block', NeuriteConvMulti(
                dim=dim,
                in_channels=nb_channels_in,
                out_channels=nb_channels_out,
                nb_conv=nb_conv_per_level,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                activation=activation,
                residual=residual,
                dropout=dropout,
                batch_norm=batch_norm,
            )))
            if isinstance(nb_channels_out, list):
                nb_channels_in = nb_channels_out[-1]
            else:
                nb_channels_in = nb_channels_out

            level_block = tnn.Sequential(OrderedDict(level_block))
            decoder_module.append(level_block)
        decoder_module = [('down_path', tnn.Sequential(*decoder_module))]

        # --------------------------------------------------------------
        # Additional convolutions
        # --------------------------------------------------------------
        # This is not handled in Neurite, but it commonly used in
        # voxelmorph's UNet.
        if isinstance(nb_feat, list) and nb_feat:
            # take care of remaining convolutions
            decoder_module.append(('conv_block', NeuriteConvMulti(
                dim=dim,
                in_channels=nb_channels_in,
                out_channels=nb_feat,
                nb_conv=len(nb_feat),
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                activation=activation,
                residual=residual,
                dropout=dropout,
                batch_norm=batch_norm,
            )))
            nb_channels_in = nb_feat[-1]

        # --------------------------------------------------------------
        # Final convolution (features to output channels)
        # --------------------------------------------------------------
        decoder_module.append(('final_conv', NeuriteConv(
            dim=dim,
            in_channels=nb_channels_in,
            out_channels=out_channels,
            kernel_size=final_kernel_size,
            padding=padding,
            activation=final_activation,
        )))
        super().__init__(OrderedDict(decoder_module))

    def forward(self, x, *skip):
        skip = list(skip)
        down_path, *final_convs = self.children()
        for level in down_path:
            for block in level:
                if skip and isinstance(block, Upsample):
                    x = block(x, output_shape=skip[0].shape[2:])
                elif isinstance(block, Cat):
                    x = block(skip.pop(0), x)
                else:
                    x = block(x)
        for layer in final_convs:
            x = layer(x)
        return x


class NeuriteUNet(Module):
    """
    UNet with the exact same architecture as in Neurite, so that we
    can load pre-trained weights from SynthStuff.
    """
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 nb_levels=5,
                 skip_decoder_levels=0,
                 kernel_size=3,
                 nb_feat=16,
                 feat_mult=1,
                 pool_size=2,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 residual=False,
                 final_activation='softmax',
                 final_kernel_size=1,
                 nb_conv_per_level=1,
                 dropout=0,
                 batch_norm=None):
        """

        Parameters
        ----------
        dim : int
            Number of spatial dimensions
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels/classes
        nb_levels : int, default=5
            Number of levels in the UNet
        skip_decoder_levels : int, default=0
            Number of levels to skip during decoding.
            If used, effectively reduces the size of the output tensor.
        kernel_size : int or sequence[int], default=3
            Kernel size
        nb_feat : int or sequence[int], default=16
            Either the number of channels after each convolution in
            the UNet (if a sequence) or a fixed number of features
            that gets multipled by `feat_mult` at each level (if an int).
        feat_mult : int, default=1
            How to expand the number of features at each coarser level.
        pool_size : int, default=2
            Kernel size of the max pooling operation.
        padding : {'valid', 'same'} or int or sequence[int], default='same'
            Padding method used in conv and pool/
        dilation_rate_mult : int, default=1
            How to expand the dilation of the kernel at each coarser level.
        activation : str or class or callable, default='elu'
            Activation function after each convolution.
        residual : bool, default=False
            Make the convolutional blocks residual.
        final_activation : str or class or callable, default='softmax'
            Final activation function, after the last convolution.
        nb_conv_per_level : int, default=1
            Number of sequential convolutions at each level.
        dropout : float in (0..1), default=0
            Channel dropout probability.
        batch_norm : {'norm', 'instance', 'layer'} or class or callable, default=None
            Normalization to use.
        """
        super().__init__()

        if isinstance(nb_feat, (list, tuple)):
            nb_feat_encoder = nb_feat[:nb_levels * nb_conv_per_level]
            nb_feat_decoder = nb_feat[nb_levels * nb_conv_per_level:]
            in_decoder = nb_feat_encoder[::-nb_conv_per_level]
        else:
            nb_feat_encoder = nb_feat_decoder = nb_feat
            in_decoder = [nb_feat * feat_mult ** (nb_levels - level - 1)
                          for level in range(nb_levels)]

        self.encoder = NeuriteEncoder(
            dim,
            in_channels=in_channels,
            nb_levels=nb_levels,
            kernel_size=kernel_size,
            nb_feat=nb_feat_encoder,
            feat_mult=feat_mult,
            pool_size=pool_size,
            padding=padding,
            dilation_rate_mult=dilation_rate_mult,
            activation=activation,
            residual=residual,
            nb_conv_per_level=nb_conv_per_level,
            dropout=dropout,
            batch_norm=batch_norm)

        self.decoder = NeuriteDecoder(
            dim,
            in_channels=in_decoder,
            out_channels=out_channels,
            nb_levels=nb_levels - skip_decoder_levels,
            kernel_size=kernel_size,
            nb_feat=nb_feat_decoder,
            feat_mult=feat_mult,
            pool_size=pool_size,
            padding=padding,
            dilation_rate_mult=dilation_rate_mult,
            activation=activation,
            final_activation=final_activation,
            final_kernel_size=final_kernel_size,
            residual=residual,
            nb_conv_per_level=nb_conv_per_level,
            dropout=dropout,
            batch_norm=batch_norm)

    def forward(self, x):
        return self.decoder(*self.encoder(x, return_skip=True))

