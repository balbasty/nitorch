"""Convolutional neural networks (UNet, VAE, etc.)."""

import inspect
import math
from collections import OrderedDict
import torch
from torch import nn as tnn
from nitorch.core.py import make_list, flatten
from nitorch.core.linalg import matvec
from nitorch.core.utils import movedim
from .base import nitorchmodule, Module
from .conv import Conv
from .pool import Pool
from .reduction import reductions, Reduction


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


def expand_list(x, n, crop=False, default=None):
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
    if idx_ellipsis == 0:
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
class Down(tnn.ModuleList):

    def __init__(
            self,
            dim,
            in_channels=None,
            out_channels=None,
            stride=2,
            kernel_size=None,
            pool=None,
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1,
            bias=True):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int, optional if `pool`
            Number of input channels (if strided conv).

        out_channels : int, optional if `pool`
            Number of output channels (if strided conv).

        stride : int or sequence[int], default=2
            Up/Downsampling factor.

        kernel_size : int or sequence[int], default=`stride`
            Kernel size per dimension (if strided conv).

        activation : [sequence of] str or type or callable, default='relu'
            Activation function (if strided conv).

        batch_norm : [sequence of] bool, default=False
            Batch normalization before each convolution (if strided conv).

        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, use a strided convolution.

        groups : [sequence of] int, default=`stitch`
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.

        stitch : int, default=1
            Number of stitched tasks.

        bias : bool, default=True
            Include a bias term in the convolution.

        """
        if not pool and (not in_channels or not out_channels):
            raise ValueError('Number of channels mandatory for strided conv')
        stitch = stitch or 1
        groups = groups or stitch
        stride = make_list(stride, dim)
        kernel_size = make_list(kernel_size, dim)
        kernel_size = [k or s for k, s in zip(kernel_size, stride)]

        if pool:
            module = Pool(dim,
                          kernel_size=kernel_size,
                          stride=stride,
                          activation=activation)
        else:
            module = Conv(dim,
                          in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          activation=activation,
                          groups=groups,
                          bias=bias,
                          batch_norm=batch_norm)
        modules = [module]
        if stitch:
            modules.append(Stitch(stitch, stitch))
        super().__init__(modules)


@nitorchmodule
class Up(tnn.ModuleList):

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            stride=2,
            kernel_size=None,
            output_padding=None,
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1,
            bias=True):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.

        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        stride : int or sequence[int], default=2
            Up/Downsampling factor.

        kernel_size : int or sequence[int], default=`stride`
            Kernel size per dimension.

        output_padding : int or sequence[int], default=0
            Padding to add to the output.

        activation : [sequence of] str or type or callable, default='relu'
            Activation function (if strided conv).

        batch_norm : [sequence of] bool, default=False
            Batch normalization before each convolution (if strided conv).

        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, use a strided convolution.

        groups : [sequence of] int, default=`stitch`
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.

        stitch : int, default=1
            Number of stitched tasks.

        bias : bool, default=True
            Include a bias term in the convolution.

        """
        stitch = stitch or 1
        groups = groups or stitch
        stride = make_list(stride, dim)
        kernel_size = make_list(kernel_size, dim)
        kernel_size = [k or s for k, s in zip(kernel_size, stride)]

        module = Conv(dim,
                      in_channels, out_channels,
                      transposed=True,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation,
                      groups=groups,
                      bias=bias,
                      batch_norm=batch_norm,
                      output_padding=output_padding)
        modules = [module]
        if stitch:
            modules.append(Stitch(stitch, stitch))
        super().__init__([module])


@nitorchmodule
class StackedConv(tnn.ModuleList):
    """Multiple convolutions at the same resolution followed by 
    a up- or down- sampling, using either a strided convolution or 
    a pooling operation.

    By default, padding is used so that convolutions with stride 1
    preserve spatial dimensions.

    (BatchNorm? > [Grouped]Conv > Activation? > Stitch?)* > 
    (BatchNorm? > [Grouped](StridedConv|Conv > Pool) > Activation? > Stitch?)
    """
    
    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            transposed=False,
            pool=None,
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1,
            output_padding=0,
            bias=True,
            residual=False,
            return_last=False):
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
            
        batch_norm : [sequence of] bool, default=False
            Batch normalization before each convolution.
            
        stride : int or sequence[int], default=2
            Up to one value per spatial dimension.
            `output_shape \approx input_shape // stride`
            
        transposed : bool, default=False
            Make the strided convolution a transposed convolution.
            
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, the final convolution is a strided convolution.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per convolution. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per convolution.

        bias : [sequence of] int, default=True
            Include a bias term in the convolution.

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

        out_channels = make_list(out_channels)
        in_channels = [in_channels] + out_channels[:-1]
        nb_layers = len(out_channels)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)
        bias = expand_list(make_list(bias), nb_layers, default=True)
        
        if pool and transposed:
            raise ValueError('Cannot have both `pool` and `transposed`.')
      
        all_shapes = zip(
            in_channels, 
            out_channels, 
            activation,
            batch_norm,
            groups, 
            stitch,
            bias)
        *all_shapes, final_shape = all_shapes
        
        # stacked conv (without strides)
        modules = []
        for d, (i, o, a, bn, g, s, b) in enumerate(all_shapes):
            modules.append(Conv(
                dim, i, o, kernel_size,
                activation=a,
                batch_norm=bn,
                padding='auto',
                groups=g,
                bias=b))
            if s > 1:
                modules.append(Stitch(s, s))
        
        # last conv (strided if not pool)
        i, o, a, bn, g, s, b = final_shape
        modules.append(Conv(
            dim, i, o, kernel_size,
            transposed=transposed,
            activation=None if pool else a,
            batch_norm=bn,
            stride=1 if pool else stride,
            padding='auto',
            groups=g,
            output_padding=output_padding,
            bias=b))
        
        # pooling
        if pool:
            modules.append(Pool(
                dim, kernel_size, 
                stride=stride,
                activation=a))

        # final stitch
        if s > 1:
            modules.append(Stitch(s, s))
                
        super().__init__(modules)
        
    @property
    def stride(self):
        for layer in reversed(self):
            if isinstance(self, (Pool, Conv)):
                return layer.stride
            
    @property
    def output_padding(self):
        for layer in reversed(self):
            if isinstance(self, Conv):
                return layer.output_padding
        return 0

    @property
    def in_channels(self):
        for layer in self:
            if isinstance(layer, Conv):
                return layer.in_channels

    @property
    def out_channels(self):
        for layer in reversed(self):
            if isinstance(layer, Conv):
                return layer.out_channels

    @property
    def out_channels_last(self):
        for layer in reversed(self):
            if isinstance(layer, Conv):
                return layer.in_channels

    def shape(self, x):
        if torch.is_tensor(x):
            x = tuple(x.shape)
        for layer in reversed(self):
            if isinstance(layer, (Conv, Pool)):
                x = layer.shape(x)
        return x
    
    def forward(self, *x, **overload):
        """

        Parameters
        ----------
        x : (B, Ci, *spatial_in) tensor

        Other parameters
        ----------------
        output_padding : int or sequence[int], optional
        residual : bool, optional
        return_last : [sequence of] bool or str, optional

        Returns
        -------
        output : (B, Co, *spatial_out) tensor
            Convolved tensor
        last : (B, C2, *spatial_in) tensor, if `return_last`
            Last output before the final up/downsampling

        """
        def is_last(layer):
            if isinstance(layer, Pool):
                return True
            if isinstance(layer, Conv):
                if not all(s == 1 for s in make_list(layer.stride)):
                    return True
            return False

        output_padding = overload.get('output_padding', self.output_padding)
        residual = overload.get('residual', self.residual)
        return_last = overload.get('return_last', self.return_last)
        if not isinstance(return_last, str):
            return_last = 'single' if return_last else ''

        last = []
        if 'single' in return_last:
            last.append(x[0])
        x = torch.cat(x, 1) if len(x) > 1 else x[0]
        if 'cat' in return_last:
            last.append(x)
        for layer in self:
            if isinstance(layer, Conv) and layer.transposed:
                kwargs = dict(output_padding=output_padding)
            else:
                kwargs = {}

            if residual:
                x = x + layer(x, **kwargs)
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
            batch_norm=False,
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
            batch_norm=batch_norm,
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
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1,
            output_padding=0,
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
            activation=activation,
            batch_norm=batch_norm,
            groups=groups,
            stitch=stitch,
            bias=bias,
            residual=residual,
            return_last=return_last,
            output_padding=output_padding,
        )


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
            batch_norm=False,
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
            
        batch_norm : [sequence of] bool or type or callable, default=False
            Batch normalization before each convolution.
            
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
                
        out_channels = list(map(make_list, out_channels))
        in_channels = [in_channels] + [c[-1] for c in out_channels[:-1]]
        nb_layers = len(out_channels)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)

        # deal with skipped connections (what a nightmare...)
        skip_channels = make_list(skip_channels or [])
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
            batch_norm,
            groups, 
            stitch)
                               
        modules = []
        for i, o, a, b, g, s in all_shapes:
            modules.append(EncodingLayer(
                dim, i, o, kernel_size,
                stride=stride,
                pool=pool,
                activation=a,
                batch_norm=b,
                groups=g,
                stitch=[..., s]))
            # If the layer is grouped, all convolutions in that 
            # layer are grouped. However, I only stitch once 
            # (at the end of the encoding layer)

        super().__init__(modules)

                               
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
            activation=tnn.ReLU,
            batch_norm=False,
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
            
        activation : [sequence of] str or type or callable, default='relu'
            Activation function.
            
        batch_norm : [sequence of] bool or type or callable, default=False
            Batch normalization before each convolution.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per layer. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per layer.
        """                     
        self.dim = dim
                
        out_channels = list(map(make_list, out_channels))
        in_channels = [in_channels]
        in_channels += [c[-1] for c in out_channels[:-1]]
        nb_layers = len(out_channels)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)

        # deal with skipped connections (what a nightmare...)
        skip_channels = make_list(skip_channels or [])
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
            batch_norm,
            groups, 
            stitch)
                               
        modules = []
        for i, o, a, b, g, s in all_shapes:
            modules.append(DecodingLayer(
                dim, i, o, kernel_size,
                stride=stride,
                activation=a,
                batch_norm=b,
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
            grp1 = groups.pop(0)
            inp1 = inputs.pop(0)
            oshape = layer.shape(x)[2:]
            ishape = inp1.shape[2:]
            pad = [i-o for o, i in zip(oshape, ishape)]
            x = layer(x, output_padding=pad)
            x = interleaved_cat((x, inp1), dim=1, groups=grp1)

        # Post-processing (convolutions without skipped connections)
        for layer in postproc:
            x = layer(x)

        return x


@nitorchmodule
class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            encoder=None,
            decoder=None,
            kernel_size=3,
            stride=2,
            activation=tnn.ReLU,
            pool=None,
            batch_norm=False,
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
                
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling to use in the encoder. If None, use strided convolutions.
            
        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
            
        groups : [sequence of] int, default=`stitch`
            Number of groups per layer. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : [sequence of] int, default=1
            Number of stitched tasks per layer.
        """
        self.dim = dim

        in_channels = make_list(in_channels)
        out_channels = make_list(out_channels)

        # defaults
        encoder = list(encoder or [16, 32, 32, 32])
        decoder = list(decoder or [32, 32, 32, 32, 32, 16, 16])

        # ensure as many upsampling steps as downsampling steps
        decoder = expand_list(decoder, len(encoder), crop=False)
        encoder = list(map(make_list, encoder))
        encoder_out = list(map(lambda x: x[-1], reversed(encoder)))
        encoder_out.append(sum(in_channels))
        decoder = list(map(make_list, decoder))
        stack = flatten(decoder[len(encoder):])
        decoder = decoder[:len(encoder)]

        nb_layers = len(encoder) + len(decoder) + len(stack)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        stitch[-1] = 1  # do not stitch last layer
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        groups[0] = len(in_channels)    # first layer
        groups[-1] = len(out_channels)  # last layer
        activation = expand_list(make_list(activation), nb_layers, default='relu')

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
                      batch_norm=batch_norm,
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
                      activation=activation_decoder,
                      batch_norm=batch_norm,
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
                              batch_norm=batch_norm,
                              groups=groups_stack,
                              stitch=stitch_stack)
        else:
            stk = Cat()
        modules.append(('stack', stk))

        input_final = stack[-1] if stack else last_decoder   
        input_final = make_list(input_final)
        if len(input_final) == 1:
            input_final = [input_final[0]//len(out_channels)] * len(out_channels)
        stk = Conv(dim, input_final, out_channels,
                   kernel_size=kernel_size,
                   activation=activation_final,
                   batch_norm=batch_norm,
                   padding='auto')
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
    .. If batch normalization is activated, it is performed before each
       encoding convolution.

    """

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            encoder=None,
            stack=None,
            kernel_size=3,
            stride=2,
            pool=None,
            reduction='max',
            activation='relu',
            batch_norm=False):
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

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        self.dim = dim

        encoder = list(encoder or [16, 32, 32, 32])
        stack = list(stack or [32, 16, 16])
        last_encoder = make_list(encoder[-1])[-1]
        stack = stack + [out_channels]
        if isinstance(reduction, str):
            reduction = reductions.get(reduction, None)
        reduction = reduction(keepdim=True) if inspect.isclass(reduction) \
                    else reduction if callable(reduction) \
                    else None
        if not isinstance(reduction, Reduction):
            raise TypeError('reduction must be a `Reduction` module.')

        nb_layers = len(encoder) + len(stack)
        activation = expand_list(make_list(activation), nb_layers, default='relu')
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
                      batch_norm=batch_norm)
        modules.append(('encoder', enc))

        modules.append(('reduction', reduction))

        stk = StackedConv(dim,
                          in_channels=last_encoder,
                          out_channels=stack,
                          kernel_size=1,
                          activation=activation_stack)
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

        kernel_size = make_list(kernel_size, dim)
        if not all(k % 2 for k in kernel_size):
            raise ValueError(f'MRF kernel size must be odd. Got {kernel_size}.')

        activation = expand_list(make_list(activation), 2 + num_extra,
                                 default=tnn.LeakyReLU(0.2))
        mrf_activation, *activation = activation

        batch_norm = expand_list(make_list(batch_norm), 2 + num_extra,
                                 default=False)
        mrf_batch_norm, *batch_norm = batch_norm

        bias = expand_list(make_list(bias), 2 + num_extra, default=False)
        mrf_bias, *bias = bias

        # make layers
        modules = []

        module = Conv(dim,
                      in_channels=num_classes,
                      out_channels=num_filters,
                      kernel_size=kernel_size,
                      activation=mrf_activation,
                      batch_norm=mrf_batch_norm,
                      bias=mrf_bias,
                      padding='auto')

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
            batch_norm=batch_norm,
            bias=bias)
        modules.append(('extra', module))

        # build model
        super().__init__(OrderedDict(modules))


class Cat(Module):

    def forward(self, *x, return_last=False, **k):
        last = x[0]
        x = torch.cat(x, 1) if len(x) > 1 else x[0]
        return (x, last) if return_last else x

    
@nitorchmodule
class UNet2(tnn.Sequential):
    """U-Net."""

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            encoder=None,
            decoder=None,
            conv_per_layer=1,
            kernel_size=3,
            stride=2,
            activation=tnn.ReLU,
            batch_norm=False,
            nb_iter=1):
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
            Either one or two activation function.
            If two functions are provided, the second one is the final
            activation function, and the first is used in all previous layers.

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.
        """
        self.dim = dim
        self.nb_iter = nb_iter

        in_channels = make_list(in_channels)
        out_channels = make_list(out_channels)

        # defaults
        conv_per_layer = max(1, conv_per_layer)
        encoder = list(encoder or [16, 32, 32, 32])
        decoder = make_list(decoder or list(reversed(encoder[:-1])),
                            n=len(encoder)-1)
        stack = decoder[len(encoder)-1:]
        decoder = encoder[-1:] + decoder[:len(encoder)]
        activation, final_activation = make_list(activation, 2)

        modules = []
        first = Conv(dim,
                     in_channels=in_channels,
                     out_channels=encoder[0],
                     kernel_size=kernel_size,
                     activation=activation,
                     batch_norm=batch_norm,
                     padding='auto')
        modules.append(('first', first))

        modules_encoder = []
        for n in range(len(encoder)-1):
            cin = encoder[n]
            cout = encoder[n+1]
            cout = [encoder[n]] * (conv_per_layer - 1) + [cout]
            modules_encoder.append(EncodingLayer(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                batch_norm=batch_norm,
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
            batch_norm=batch_norm,
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
                batch_norm=batch_norm,
            ))
        if conv_per_layer > 1:
            cin = decoder[-1] + encoder[0]
            cout = [decoder[-1]] * (conv_per_layer - 1)
            modules_decoder.append(StackedConv(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
            ))
            last_decoder = decoder[-1]
        else:
            modules_decoder.append(Cat())
            last_decoder = decoder[-1] + encoder[0]
        dec = tnn.ModuleList(modules_decoder)
        modules.append(('decoder', dec))

        if stack:
            stk = StackedConv(dim,
                              in_channels=last_decoder,
                              out_channels=stack,
                              kernel_size=kernel_size,
                              activation=activation,
                              batch_norm=batch_norm)
            modules.append(('stack', stk))
            last_stack = stack[-1]
        else:
            modules.append(('stack', Cat()))
            last_stack = last_decoder

        final = Conv(dim, last_stack, out_channels,
                     kernel_size=kernel_size,
                     activation=final_activation,
                     padding='auto')
        modules.append(('final', final))

        super().__init__(OrderedDict(modules))

    def forward(self, x):

        x = self.first(x)

        # encoder
        buffers = []
        for d, layer in enumerate(self.encoder):
            buffer_shape = list(x.shape)
            buffer_shape[1] = layer.in_channels - buffer_shape[1]
            buffer = x.new_zeros(buffer_shape)
            x, buffer = layer(x, buffer, return_last=True)
            buffers.append(buffer)

        pad = self.get_padding(buffers[-1].shape, x.shape, self.bottleneck)
        x = self.bottleneck(x, output_padding=pad)

        # decoder
        for d, layer in enumerate(self.decoder):
            buffer = buffers[-d-1]
            if d < len(self.decoder) - 1:
                pad = self.get_padding(buffers[-d-2].shape, x.shape, layer)
            else:
                pad = 0
            x = layer(x, buffer, output_padding=pad)

        x = self.stack(x)
        x = self.final(x)
        return x

    def get_padding(self, outshape, inshape, layer):
        outshape = outshape[2:]
        shape = layer.shape(inshape)[2:]
        padding = [o - i for o, i in zip(outshape, shape)]
        return padding


@nitorchmodule
class UUNet(tnn.Sequential):
    """Iterative U-Net with two-way skip connections."""

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            encoder=None,
            decoder=None,
            conv_per_layer=1,
            kernel_size=3,
            stride=2,
            activation=tnn.ReLU,
            batch_norm=False,
            residual=False,
            nb_iter=1):
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

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.

        residual : bool, default=False
            Use residual skipped connections
        """
        self.dim = dim
        self.residual = residual
        self.nb_iter = nb_iter

        # defaults
        conv_per_layer = max(1, conv_per_layer)
        encoder = list(encoder or [16, 32, 32, 32])
        decoder = make_list(decoder or list(reversed(encoder[:-1])),
                            n=len(encoder)-1)
        stack = decoder[len(encoder)-1:]
        decoder = encoder[-1:] + decoder[:len(encoder)]
        activation, final_activation = make_list(activation, 2)

        modules = []
        first = Conv(dim,
                     in_channels=in_channels,
                     out_channels=encoder[0],
                     kernel_size=kernel_size,
                     activation=activation,
                     batch_norm=batch_norm,
                     padding='auto')
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
                batch_norm=batch_norm,
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
            batch_norm=batch_norm,
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
                batch_norm=batch_norm,
            ))
        if conv_per_layer > 1:
            cin = decoder[-1] + encoder[0]
            cout = [decoder[-1]] * (conv_per_layer - 1)
            modules_decoder.append(StackedConv(
                dim,
                in_channels=cin,
                out_channels=cout,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
            ))
            last_decoder = decoder[-1]
        else:
            modules_decoder.append(Cat())
            last_decoder = decoder[-1] + encoder[0]
        dec = tnn.ModuleList(modules_decoder)
        modules.append(('decoder', dec))

        if stack:
            stk = StackedConv(dim,
                              in_channels=last_decoder,
                              out_channels=stack,
                              kernel_size=kernel_size,
                              activation=activation,
                              batch_norm=batch_norm)
            modules.append(('stack', stk))
            last_stack = stack[-1]
        else:
            modules.append(('stack', Cat()))
            last_stack = last_decoder

        final = Conv(dim, last_stack, out_channels,
                     kernel_size=kernel_size,
                     activation=final_activation,
                     padding='auto')
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

        pad = self.get_padding(buffers[-1].shape, x.shape, self.bottleneck)
        x = self.bottleneck(x, output_padding=pad)

        # decoder
        for d, layer in enumerate(self.decoder):
            buffer = buffers[-d-1]
            if d < len(self.decoder) - 1:
                pad = self.get_padding(buffers[-d-2].shape, x.shape, layer)
            else:
                pad = 0
            x = layer(x, buffer, output_padding=pad)

        x = self.stack(x)
        x = self.final(x)
        return x

    def get_padding(self, outshape, inshape, layer):
        outshape = outshape[2:]
        shape = layer.shape(inshape)[2:]
        padding = [o - i for o, i in zip(outshape, shape)]
        return padding

    def forward(self, x, **overload):

        nb_iter = overload.get('nb_iter', self.nb_iter)
#         if nb_iter == 1:
#             return self.forward_once(x)

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

            pad = self.get_padding(buffers_encoder[-1].shape, x.shape, self.bottleneck)
            x = self.bottleneck(x, output_padding=pad)

            # decoder
            for d, layer in enumerate(self.decoder):
                buffer = buffers_encoder[-d-1]
                if d < len(self.decoder) - 1:
                    pad = self.get_padding(buffers_encoder[-d-2].shape, x.shape, layer)
                else:
                    pad = 0
                x, buffer = layer(x, buffer, return_last=True, output_padding=pad)
                if buffers_decoder[d] is None or not self.residual:
                    buffers_decoder[d] = buffer
                else:
                    buffers_decoder[d] = buffers_decoder[d] + buffer
        
        del x0
        x = self.stack(x)
        x = self.final(x)
        return x


@nitorchmodule
class WNet(tnn.Sequential):
    """W-net (= cascaded U-Net) with skip connections between the nets."""

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            mid_channels=None,
            encoder=None,
            decoder=None,
            encoder2=None,
            decoder2=None,
            conv_per_layer=1,
            kernel_size=3,
            stride=2,
            activation=tnn.ReLU,
            batch_norm=False,
            skip=True):
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

        batch_norm : bool or type or callable, default=False
            Batch normalization before each convolution.

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
        decoder1 = make_list(decoder1 or default_decoder, n=len(encoder1) - 1)
        default_encoder2 = encoder1[1:]
        encoder2 = list(encoder2 or default_encoder2)
        default_decoder2 = list(reversed(encoder2[:-1])) + [encoder1[0]]
        decoder2 = make_list(decoder2 or default_decoder2, n=len(encoder2))

        stack1 = decoder1[len(encoder1) - 1:]
        decoder1 = decoder1[:len(encoder1)]
        stack2 = decoder2[len(encoder2):]
        decoder2 = decoder2[:len(encoder2)]
        activation, final_activation = make_list(activation, 2)
        kernel_size, final_kernel_size = make_list(kernel_size, 2)

        modules = OrderedDict()

        # --- initial feature extraction --------------------------------
        modules['first'] = Conv(
            dim,
            in_channels=in_channels,
            out_channels=encoder1[0],
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            padding='auto')

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
                batch_norm=batch_norm,
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
            batch_norm=batch_norm,
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
                batch_norm=batch_norm,
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
            batch_norm=batch_norm,
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
                batch_norm=batch_norm,
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
            batch_norm=batch_norm,
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
                batch_norm=batch_norm,
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
                batch_norm=batch_norm,
            )
            last_stack = cout[-1]
        else:
            modules['stack2'] = Cat()
            last_stack = cin

        # --- final layer ----------------------------------------------
        modules['final'] = Conv(
            dim, last_stack, out_channels,
            kernel_size=final_kernel_size,
            activation=final_activation,
            padding='auto')

        # --- middle output --------------------------------------------
        if mid_channels:
            modules['middle'] = Conv(
                dim, modules['encoder2'][0].out_channels_last, mid_channels,
                kernel_size=final_kernel_size,
                activation=final_activation,
                padding='auto')

        super().__init__(modules)

    def get_padding(self, outshape, inshape, layer):
        outshape = outshape[2:]
        shape = layer.shape(inshape)[2:]
        padding = [o - i for o, i in zip(outshape, shape)]
        return padding

    def forward(self, x, **overload):

        x = self.first(x)

        # encoder1
        buffers_encoder = []
        for layer in self.encoder1:
            x, buffer = layer(x, return_last=True)
            buffers_encoder.append(buffer)

        # bottleneck 1
        pad = self.get_padding(buffers_encoder[-1].shape, x.shape,
                               self.bottleneck1)
        x, buffer = self.bottleneck1(x, return_last=True, output_padding=pad)

        # decoder1
        buffers_decoder = [buffer] if self.skip else []
        for layer in self.decoder2:
            buffer = buffers_encoder.pop()
            if buffers_encoder:
                pad = self.get_padding(buffers_encoder[-1].shape,
                                       x.shape, layer)
            else:
                pad = 0
            x, buffer = layer(x, buffer, return_last=True, output_padding=pad)
            if self.skip:
                buffers_decoder.append(buffer)

        # encoder2
        buffers_decoder = buffers_decoder + buffers_encoder
        buffers_encoder = []
        buffer_middle = None
        for layer in self.encoder2:
            buffer = [buffers_decoder.pop()] if buffers_decoder else []
            x, buffer, inmiddle = layer(x, *buffer, return_last='single+cat')
            if buffer_middle is None and hasattr(self, 'middle'):
                buffer_middle = self.middle(inmiddle)
            del inmiddle
            buffers_encoder.append(buffer)

        # bottleneck 2
        buffer = [buffers_decoder.pop()] if buffers_decoder else []
        pad = self.get_padding(buffers_encoder[-1].shape, x.shape,
                               self.bottleneck2)
        x = self.bottleneck2(x, *buffer, output_padding=pad)

        # decoder2
        for layer in self.decoder2:
            buffer = buffers_encoder.pop()
            if buffers_encoder:
                pad = self.get_padding(buffers_encoder[-1].shape,
                                       x.shape, layer)
            else:
                pad = 0
            x = layer(x, buffer, output_padding=pad)

        x = self.stack2(x, *buffers_encoder)
        x = self.final(x)
        return x if buffer_middle is None else x, buffer_middle

