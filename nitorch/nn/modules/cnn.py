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


def expand_list(x, n, crop=False, default=None):
    """
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
class StackedConv(tnn.ModuleList):
    """Multiple convolutions at the same resolution followed by 
    a up- or down- sampling, using either a strided convolution or 
    a pooling operation.
    
    (BatchNorm? > [Grouped]Conv > Activation? > Stitch?)* > 
    (BatchNorm? > [Grouped](StridedConv|Conv > Pool) > Activation? > Stitch?)
    """
    
    def __init__(
            self,
            dim,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            transposed=False,
            pool=None,
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1,
            output_padding=0):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        input_channels : int
            Number of input channels.
            
        output_channels : int or sequence[int]
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

        """
        self.dim = dim
        
        
        output_channels = make_list(output_channels)
        input_channels = [input_channels] + output_channels[:-1]
        nb_layers = len(output_channels)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)
        
        if pool and transposed:
            raise ValueError('Cannot have both `pool` and `transposed`.')
      
        all_shapes = zip(
            input_channels, 
            output_channels, 
            activation,
            batch_norm,
            groups, 
            stitch)
        *all_shapes, final_shape = all_shapes
        
        # stacked conv (without strides)
        modules = []
        for d, (i, o, a, b, g, s) in enumerate(all_shapes):
            modules.append(Conv(
                dim, i, o, kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                padding='auto',
                groups=g))
            if s > 1:
                modules.append(Stitch(s, s))
        
        # last conv (strided if not pool)
        i, o, a, b, g, s = final_shape
        modules.append(Conv(
            dim, i, o, kernel_size,
            transposed=transposed,
            activation=None if pool else a,
            batch_norm=b,
            stride=1 if pool else stride,
            padding='auto',
            groups=g,
            output_padding=output_padding))
        
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
            
    def shape(self, x):
        if torch.is_tensor(x):
            x = tuple(x.shape)
        for layer in self:
            if isinstance(layer, (Conv, Pool)):
                x = layer.shape(x)
        return x
    
    def forward(self, x, **kwargs):
        output_padding = kwargs.get('output_padding', self.output_padding)
        
        for layer in self:
            if isinstance(layer, Conv) and layer.transposed:
                x = layer(x, output_padding=output_padding)
            else:
                x = layer(x)
        return x
    
        
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
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            pool=None,
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1):
        
        super().__init__(
            dim, 
            input_channels, 
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool=pool,
            activation=activation,
            batch_norm=batch_norm,
            groups=groups,
            stitch=stitch,
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
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1):
        
        super().__init__(
            dim, 
            input_channels, 
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            batch_norm=batch_norm,
            groups=groups,
            stitch=stitch,
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
            activation=tnn.ReLU,
            batch_norm=False,
            groups=None,
            stitch=1):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension.
            
        input_channels : int or sequence[int]
            Number of input channels.
            If a sequence, the first convolution is a grouped convolution.
            
        output_channels : sequence[int]
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
                
        output_channels = list(map(make_list, output_channels))
        input_channels = [input_channels] + [c[-1] for c in output_channels[:-1]]
        nb_layers = len(output_channels)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)

        all_shapes = zip(
            input_channels, 
            output_channels, 
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
            skip_channels=None,
            skip_groups=1,
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
            
        input_channels : int or sequence[int]
            Number of input channels.
            If a sequence, the first convolution is a grouped convolution.
            
        output_channels : sequence[int or sequence[int]]
            Number of channels in each decoding layer.
            There will be len(channels) layers.
            
        skip_channels : sequence[int or sequence[int]]
            Number of skipped channels per layer,
            If an element is a sequence, it comes from a grouped conv.
            
        skip_groups : sequence[int]
            Number of groups per input skipped connections.
            
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
                
        output_channels = list(map(make_list, output_channels))
        input_channels = [input_channels]
        input_channels += [c[-1] for c in output_channels[:-1]]
        nb_layers = len(output_channels)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)

        # deal with skipped connections (what a nightmare...)
        skip_groups = make_list(skip_groups or [1], len(output_channels))
        skip_channels = make_list(skip_channels or [])
        self.skip = len(skip_channels) > 0
        self.skip_repeat = []
        self.skip_groups = []
        for i in range(len(skip_channels)):
            self.skip_repeat.append(max(groups[i] // skip_groups[i], 1))
            self.skip_groups.append(groups[i])
            if len(input_channels) > i+1:
                input_channels[i+1] += skip_channels[i] * self.skip_repeat[-1]

        all_shapes = zip(
            input_channels, 
            output_channels, 
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
        repeat = list(self.skip_repeat)
        groups = list(self.skip_groups)
        for i, layer in enumerate(decoder):
            rep1 = repeat.pop(0)
            grp1 = groups.pop(0)
            inp1 = inputs.pop(0)
            
            if rep1 > 1:
                inp1 = torch.cat([inp1]*rep1, dim=1)
            oshape = layer.shape(x)[2:]
            ishape = inp1.shape[2:]
            pad = [i-o for o, i in zip(oshape, ishape)]
            x = layer(x, output_padding=pad)
            x = interleaved_cat((x, inp1), dim=1, groups=grp1)

        # Post-processing (convolutions without skipped connections)
        for layer in postproc:
            x = layer(x)

        return x


# @nitorchmodule
# class StackedConv(tnn.Sequential):
#     """Stacked convolutions, without up/down-sampling."""

#     def __init__(
#             self,
#             dim,
#             input_channels,
#             output_channels,
#             kernel_size=3,
#             stride=1,
#             transposed=False,
#             activation=tnn.ReLU,
#             final_activation='same',
#             batch_norm=False,
#             groups=None,
#             final_group='same',
#             stitch=1,
#             final_stitch='same'):
#         """

#         Parameters
#         ----------
#         dim : {1, 2, 3}
#             Dimension.
#         input_channels : int
#             Number of input channels.
#         output_channels : sequence[int]
#             Number of output channels in each convolutional layer.
#         kernel_size : int or sequence[int], default=3
#             Kernel size per dimension.
#         stride : int or sequence[int], default=1
#             Stride in the final convolution.
#         activation : str or type or callable, default='relu'
#             Activation function.
#         final_activation : str or type or callable, default='same'
#             Final activation function. If 'same', same as `activation`.
#         batch_norm : bool or type or callable, default=False
#             Batch normalization before each convolution.
#         groups : int or sequence[int], default=`stitch`
#             Number of groups per layer. If > 1, a grouped convolution
#             is performed, which is equivalent to `groups` independent
#             layers.
#         final_group : int, default='same'
#             Final number of groups
#         stitch : int or sequence[int], default=1
#             Number of stitched tasks per layer.
#         final_stitch : int, default='same'
#             Final stitch operation.
#         """
#         self.dim = dim

#         stitch = make_list(stitch, len(output_channels))
#         stitch = [stch or 1 for stch in stitch]
#         groups = make_list(groups, len(output_channels))
#         groups = [group or stch for group, stch in zip(groups, stitch)]
#         output_channels = list(output_channels)
#         input_channels = [input_channels] + output_channels[:-1]

#         modules = []
#         kernel_size = make_list(kernel_size, dim)
#         nb_module = len(input_channels)

#         final_stride, stride = (stride, 1)
#         final_transposed, transposed = (transposed, False)
#         final_group = groups[-1] if final_group == 'same' else final_group
#         final_stitch = groups[-1] if final_stitch == 'same' else final_stitch

#         all_shapes = zip(input_channels, output_channels, groups, stitch)
#         for l, (i, o, g, s) in enumerate(all_shapes):
#             pad = [(k-1)//2 for k in kernel_size]
#             if l == nb_module-1:
#                 if final_activation != 'same':
#                     activation = final_activation
#                 stride = final_stride
#                 transposed = final_transposed
#                 s = final_stitch or 1
#                 g = final_group or s
#             modules.append(Conv(dim, i, o,
#                                 kernel_size=kernel_size,
#                                 padding=pad,
#                                 stride=stride,
#                                 activation=activation,
#                                 batch_norm=batch_norm,
#                                 transposed=transposed,
#                                 groups=g))
#             if s != 1:
#                 modules.append(Stitch(s, s))
#         super().__init__(*modules)


@nitorchmodule
class UNet(tnn.Sequential):
    """Fully-convolutional U-net."""

    def __init__(
            self,
            dim,
            input_channels,
            output_channels,
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
            
        input_channels : int or sequence[int]
            Number of input channels.
            If a sequence, the first convolution is a grouped convolution.
            
        output_channels : int or sequence [int]
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
            
        activation : str or type or callable or None, default='relu'
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
            
        groups : int or sequence[int], default=`stitch`
            Number of groups per layer. If > 1, a grouped convolution
            is performed, which is equivalent to `groups` independent
            layers.
            
        stitch : int or sequence[int], default=1
            Number of stitched tasks per layer.
        """
        self.dim = dim

        input_channels = make_list(input_channels)
        output_channels = make_list(output_channels)

        # defaults
        encoder = list(encoder or [16, 32, 32, 32])
        decoder = list(decoder or [32, 32, 32, 32, 32, 16, 16])

        # ensure as many upsampling steps as downsampling steps
        decoder = expand_list(decoder, len(encoder), crop=False)
        encoder = list(map(make_list, encoder))
        encoder_out = list(map(lambda x: x[-1], reversed(encoder)))
        encoder_out.append(sum(input_channels))
        decoder = list(map(make_list, decoder))
        stack = flatten(decoder[len(encoder):])
        decoder = decoder[:len(encoder)]
        last_decoder = decoder[-1][-1] + sum(input_channels)

        nb_layers = len(encoder) + len(decoder) + len(stack)
        
        stitch = map(lambda x: x or 1, make_list(stitch))
        stitch = expand_list(stitch, nb_layers, default=1)
        stitch[-1] = 1  # do not stitch last layer
        groups = expand_list(make_list(groups), nb_layers)
        groups = [g or s for g, s in zip(groups, stitch)]
        groups[0] = 1   # do not group first layer
        groups[-1] = 1  # do not group lasy layer
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=False)

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
        batch_norm_encoder = batch_norm[range_e]
        batch_norm_decoder = batch_norm[range_d]
        batch_norm_stack = batch_norm[range_s]
        
        activation_final = activation[-1]
        batch_norm_final = batch_norm[-1]
        input_final = stack[-1]            
        
        modules = []
        enc = Encoder(dim,
                      input_channels=input_channels,
                      output_channels=encoder,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation_encoder,
                      batch_norm=batch_norm_encoder,
                      pool=pool,
                      groups=groups_encoder,
                      stitch=stitch_encoder)
        modules.append(('encoder', enc))

        skip_groups = list(reversed(groups_encoder[1:-1])) + [len(input_channels)]
        dec = Decoder(dim,
                      input_channels=encoder_out[0],
                      output_channels=decoder,
                      skip_channels=encoder_out[1:],
                      skip_groups=skip_groups,
                      kernel_size=kernel_size,
                      stride=stride,
                      activation=activation_decoder,
                      batch_norm=batch_norm_decoder,
                      groups=groups_decoder,
                      stitch=stitch_decoder)
        modules.append(('decoder', dec))

        stk = StackedConv(dim,
                          input_channels=last_decoder,
                          output_channels=stack,
                          kernel_size=kernel_size,
                          activation=activation_stack,
                          batch_norm=batch_norm_stack,
                          groups=groups_stack,
                          stitch=stitch_stack)
        modules.append(('stack', stk))

        stk = Conv(dim, input_final, output_channels,
                   kernel_size=kernel_size,
                   activation=activation_final,
                   batch_norm=batch_norm_final)
        modules.append(('final', stk))

        super().__init__(OrderedDict(modules))
        
    def forward(self, x):
        
        # encoder
        encoder_out = [x]
        for layer in self.encoder:
            x = layer(x)
            encoder_out.append(x)
        # decoder
        x = list(reversed(encoder_out))
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
