"""Convolution layers."""

from copy import copy
import math
import inspect
import random
import torch
from torch import nn as tnn
from nitorch.core.py import make_tuple
from nitorch.core import py, utils
from .base import nitorchmodule, Module
from .norm import BatchNorm
from ..activations import _map_activations

# NOTE:
# My version of Conv allows parameters to be overridden at eval time.
# This probably clashes with these parameters being declared as __constants__
# in torch.nn._ConvND. I think this __constants__ mechanics is only used
# In TorchScript, so it's fine for now.
#
# After some googling, I think it is alright as long as the *attribute*
# is not mutated...
# Some references:
# .. https://pytorch.org/docs/stable/jit.html#frequently-asked-questions
# .. https://discuss.pytorch.org/t/why-do-we-use-constants-or-final/70331/2
#
# Note that optional submodules can also be added to __constants__ in a
# hacky way:
# https://discuss.pytorch.org/t/why-do-we-use-constants-or-final/70331/4


_native_padding_mode = ('zeros', 'reflect', 'replicate', 'circular')


def _guess_output_shape(inshape, dim, kernel_size, stride=1, dilation=1,
                        padding=0, output_padding=0, transposed=False):
    """Guess the output shape of a convolution"""
    kernel_size = make_tuple(kernel_size, dim)
    stride = make_tuple(stride, dim)
    padding = make_tuple(padding, dim)
    output_padding = make_tuple(output_padding, dim)
    dilation = make_tuple(dilation, dim)

    N = inshape[0]
    C = inshape[1]
    shape = [N, C]
    for L, S, Pi, D, K, Po in zip(inshape[2:], stride, padding,
                                  dilation, kernel_size, output_padding):
        if transposed:
            shape += [(L - 1) * S - 2 * Pi + D * (K - 1) + Po + 1]
        else:
            shape += [math.floor((L + 2 * Pi - D * (K - 1) - 1) / S + 1)]
    return tuple(shape)


def _get_conv_class(dim, transposed=False):
    """Return the appropriate torch Conv class"""
    if transposed:
        if dim == 1:
            ConvKlass = nitorchmodule(tnn.ConvTranspose1d)
        elif dim == 2:
            ConvKlass = nitorchmodule(tnn.ConvTranspose2d)
        elif dim == 3:
            ConvKlass = nitorchmodule(tnn.ConvTranspose3d)
        else:
            raise NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
    else:
        if dim == 1:
            ConvKlass = nitorchmodule(tnn.Conv1d)
        elif dim == 2:
            ConvKlass = nitorchmodule(tnn.Conv2d)
        elif dim == 3:
            ConvKlass = nitorchmodule(tnn.Conv3d)
        else:
            raise NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
    return ConvKlass


class SimpleConv(Module):
    """Simple convolution.
    
    We merely wrap torch's Conv class, with a single entry point for
    any number of spatial dimensions and optional transposed conv.
    
    Furthermore, this class allows parameters to be mutated after
    the instance has been created (using getters/setters) or even 
    at call time.
    """
    
    def __init__(self, 
                 dim, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 stride=1, 
                 padding='auto', 
                 padding_mode='zeros',
                 dilation=1,
                 groups=1,
                 bias=True,
                 transposed=False,
                 output_padding=0):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int
            Number of channels in the input image.
            
        out_channels : int
            Number of channels produced by the convolution.
            
        kernel_size : int or tuple[int]
            Size of the convolution kernel
            
        stride : int or sequence[int], default=1
            Stride of the convolution.
            
        padding : int or sequence[int], default='auto'
            Zero-padding added to all three sides of the input.
            If ``'auto'``, padding such that the output shape is the 
            same as the input shape (up to strides) is used.
            
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
            
        dilation : int or tuple[int], default=1
            Spacing between kernel elements.
            
        groups : int, default=1
            Number of groups in a grouped convolution.
            The number of input channels and of output channels 
            must be divisible by this number.
            
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            
        transposed : bool, default=False:
            Transposed convolution.
            
        output_padding : int or tuple[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if ``transposed is True``.
        """
        super().__init__()
        self.dim = dim
        self.transposed = transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # deal with padding
        pre_padding = 0
        pre_padding_mode = None
        post_padding = 0
        if (padding == 'auto' or padding_mode not in _native_padding_mode):
            pre_padding = padding
            pre_padding_mode = padding_mode
            padding = 0
            padding_mode = 'zeros'
        if not transposed:
            post_padding = output_padding
            output_padding = {}
        else:
            output_padding = {'output_padding': output_padding}
        self._pre_padding = pre_padding
        self._padding_mode = pre_padding_mode
        self._post_padding = post_padding
            
        # instantiate underlying conv class
        klass = _get_conv_class(dim, transposed)
        self.conv = klass(in_channels, out_channels, 
                          kernel_size=py.make_list(kernel_size, dim),
                          stride=py.make_list(stride, dim),
                          padding=py.make_list(padding, dim),
                          padding_mode=padding_mode,
                          dilation=py.make_list(dilation, dim),
                          groups=groups,
                          bias=bias,
                          **output_padding)
    
    def reset_parameters(self, method='kaiming', a=None, dist='uniform',
                         johnshift=False):
        """Initialize the values of the weights and bias
        
        Parameters
        ----------
        method : {'kaiming', 'xavier', None}, default='kaiming'
            Initialization method.
        a : float, default=sqrt(5)
            If method is 'kaiming', negative slope of the activation function 
            (0 for ReLU, some non-zero value for leaky ReLU)
            If method is 'xavier', gain.
            If method is None, FWHM of the distribution.
        dist : {'uniform', 'normal'}, default='uniform'
            Distribution to sample.
        johnshift : bool, default=True
            Each filter has one of its central input weights centered
            about one instead of zeros, making them close to an identity.
        """
        method = (method or '').lower()
        if method.startswith('k'):
            a = math.sqrt(5) if a is None else a
        else:
            a = 1. if a is None else a
            
        if method.startswith('k') and dist.startswith('u'):
            fn = tnn.init.kaiming_uniform_
        elif method.startswith('k') and dist.startswith('n'):
            fn = tnn.init.kaiming_normal_
        elif method.startswith('x') and dist.startswith('u'):
            fn = tnn.init.xavier_uniform_
        elif method.startswith('x') and dist.startswith('n'):
            fn = tnn.init.xavier_normal_
        elif not method and dist.startswith('u'):
            fn = lambda x, a: tnn.init.uniform_(x, a=-a/2, b=a/2)
        elif not method and dist.startswith('n'):
            fn = lambda x, a: tnn.init.normal_(x, std=a/2.355)
        else:
            raise ValueError(f'Unknown method {method} or dist {dist}.')

        fn(self.conv.weight, a=a)

        if johnshift:
            # we make filters closer to an identity transform by
            # "opening" a path for each channel (i.e., initializing
            # a weight with a value close to one, instead of zero).
            # This trick was found by John Ashburner.
            with torch.no_grad():
                center = tuple(k//2 for k in self.conv.weight.shape[2:])
                center = self.conv.weight[(slice(None), slice(None), *center)]
                nfilters = min(center.shape[0], center.shape[1])
                cin = list(range(center.shape[0]))
                random.shuffle(cin)
                cin = cin[:nfilters]
                cout = list(range(center.shape[1]))
                random.shuffle(cout)
                cout = cout[:nfilters]
                center[cin, cout] += 1

        if self.bias is not None:
            fan_in, _ = tnn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            if dist.startswith('u'):
                tnn.init.uniform_(self.bias, -bound, bound)
            else:
                tnn.init.normal_(self.bias, std=2*bound/2.355)
    
    def _set_padding(self, padding, padding_mode):
        if padding != 'auto' and padding_mode in _native_padding_mode:
            self.conv.padding_mode = padding_mode
            self.conv.padding = make_tuple(padding, self.dim)
            self._pre_padding = 0
            self._padding_mode = ''
        else:
            self._pre_padding = padding
            self._padding_mode = padding_mode
            self.conv.padding = 0

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def stride(self):
        return self.conv.stride

    @stride.setter
    def stride(self, value):
        self.conv.stride = make_tuple(value, self.dim)

    @property
    def padding(self):
        return self._pre_padding or self.conv.padding

    @padding.setter
    def padding(self, value):
        self._set_padding(value, self.padding_mode)

    @property
    def output_padding(self):
        return self._post_padding or self.conv.output_padding

    @output_padding.setter
    def output_padding(self, value):
        if self.transposed:
            self.conv.output_padding = make_tuple(value, self.dim)
        else:
            self._post_padding = make_tuple(value, self.dim)

    @property
    def dilation(self):
        return self.conv.dilation

    @dilation.setter
    def dilation(self, value):
        self.conv.dilation = make_tuple(value, self.dim)

    @property
    def groups(self):
        return self.conv.dilation

    @groups.setter
    def groups(self, value):
        self.conv.groups = value

    @property
    def padding_mode(self):
        if self._pre_padding:
            return self._padding_mode
        else:
            return self.conv.padding_mode

    @padding_mode.setter
    def padding_mode(self, value):
        self._set_padding(self.padding, value)

    @property
    def groups(self):
        return self.conv.groups

    @groups.setter
    def groups(self, value):
        self.conv.groups = value
    
    def forward(self, x, **overload):
        
        stride = overload.get('stride', self.stride)
        padding = overload.get('padding', self.padding)
        padding_mode = overload.get('padding_mode', self.padding_mode)
        output_padding = overload.get('output_padding', self.output_padding)
        dilation = overload.get('dilation', self.dilation)
        
        kernel_size = make_tuple(self.kernel_size, self.dim)
        stride = make_tuple(stride, self.dim)
        output_padding = make_tuple(output_padding, self.dim)
        dilation = make_tuple(dilation, self.dim)

        if padding == 'auto':
            padding = [((k-1)*d)//2 for k, d in zip(kernel_size, dilation)]
        padding = make_tuple(padding, self.dim)
        
        # perform pre-padding
        if padding_mode not in _native_padding_mode:
            x = utils.pad(x, padding, mode=padding_mode, side='both')
            padding = 0
        
        # call native convolution
        self.stride, stride0 = (stride, self.stride)
        self.padding, padding0 = (padding, self.padding)
        self.padding_mode, padding_mode0 = (padding_mode, self.padding_mode)
        self.output_padding, output_padding0 = (output_padding, self.output_padding)
        self.dilation, dilation0 = (dilation, self.dilation)
        
        x = self.conv(x)
        
        self.stride = stride0
        self.padding = padding0
        self.padding_mode = padding_mode0
        self.output_padding = output_padding0
        self.dilation = dilation0
        
        # perform post-padding
        if not self.transposed and output_padding:
            x = utils.pad(x, output_padding, side='right')
        
        return x
            
    def shape(self, x, **overload):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
        overload : dict
            Some parameters defined at build time can be overridden:
            `stride`, `padding`, `output_padding`, `dilation`.

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        if torch.is_tensor(x):
            inshape = tuple(x.shape)
        else:
            inshape = x
        
        stride = overload.get('stride', self.stride)
        padding = overload.get('padding', self.padding)
        output_padding = overload.get('output_padding', self.output_padding)
        dilation = overload.get('dilation', self.dilation)
        transposed = self.transposed
        kernel_size = make_tuple(self.kernel_size, self.dim)

        stride = make_tuple(stride, self.dim)
        output_padding = make_tuple(output_padding, self.dim)
        dilation = make_tuple(dilation, self.dim)

        if padding == 'auto':
            padding = [((k-1)*d)//2 for k, d in zip(kernel_size, dilation)]
        padding = make_tuple(padding, self.dim)

        shape = _guess_output_shape(
            inshape, self.dim,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            output_padding=output_padding,
            transposed=transposed)
        shape = list(shape)
        shape[1] = self.out_channels
        
        return tuple(shape)
    
    def __str__(self):
        s = [f'{self.in_channels}', f'{self.out_channels}']
        s += [f'kernel_size={self.kernel_size}']
        if self.transposed:
            s += [f'transposed=True']
        if any(s > 1 for s in self.stride):
            s += [f'stride={self.stride}']
        if any(s > 1 for s in self.dilation):
            s += [f'dilation={self.dilation}']
        if self.groups > 1:
            s += [f'groups={self.groups}']
        s = ', '.join(s)
        return f'SimpleConv({s})'
    
    __repr__ = __str__


@nitorchmodule
class GroupedConv(tnn.ModuleList):
    """Simple imbalanced grouped convolution.
    
    Same as SimpleConv, but allows to have groups with non-equal number of channels.
    """

    # TODO:
    #    Maybe keep the `groups` option and simply let in_channels and 
    #    out_channels be lists *optionnally* if the user wants imbalanced
    #    groups
    
    def __init__(self, dim, in_channels, out_channels, *args, groups=None, **kwargs):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : sequence[int]
            Number of channels in the input image.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a scalar, must be divisible by the number of groups.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.
            
        stride : int or sequence[int], default=1:
            Stride of the convolution.
            
        padding : int or sequence[int], default='auto'
            Zero-padding added to all three sides of the input.
            If 'auto', padding such that the output shape is the 
            same as the input shape (up to strides) is used.
            
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
            
        dilation : int or sequence[int], default=1
            Spacing between kernel elements.
            
        groups : int, default=None
            Number of groups. Default is the maximum of the lengths 
            of ``in_channels`` and ``out_channels``.
            
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            
        transposed : bool, default=False:
            Transposed convolution.
            
        output_padding : int or tuple[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if ``transposed is True``.
        """
        in_channels = py.make_list(in_channels)
        out_channels = py.make_list(out_channels)
        nb_groups = groups or max(len(in_channels), len(out_channels))
        if len(in_channels) == 1:
            in_channels = [in_channels[0]//nb_groups] * nb_groups
        if len(out_channels) == 1:
            out_channels = [out_channels[0]//nb_groups] * nb_groups
        if len(in_channels) != nb_groups or len(out_channels) != nb_groups:
            raise ValueError(f'The number of input and output groups '
                             f'must be the same: {len(in_channels)} vs '
                             f'{len(out_channels)}')

        modules = [SimpleConv(dim, i, o, *args, **kwargs)
                   for i, o in zip(in_channels, out_channels)]
        super().__init__(modules)
        self.in_channels = sum(in_channels)
        self.out_channels = sum(out_channels)

    def forward(self, input, **overload):
        """Perform a grouped convolution with imbalanced channels.

        Parameters
        ----------
        input : (B, sum(in_channels), *in_spatial) tensor

        Returns
        -------
        output : (B, sum(out_channels), *out_spatial) tensor

        """
        out_shape = (input.shape[0], self.out_channels, *input.shape[2:])
        output = input.new_empty(out_shape)
        input = input.split([c.in_channels for c in self], dim=1)
        out_channels = [c.out_channels for c in self]
        for d, (layer, inp) in enumerate(zip(self, input)):
            slicer = [slice(None)] * (self.dim + 2)
            slicer[1] = slice(sum(out_channels[:d]), sum(out_channels[:d+1]))
            output[slicer] = layer(inp, **overload)
        return output

    
    def reset_parameters(self, *args, **kwargs):
        for layer in self:
            layer.reset_parameters(*args, **kwargs)
    
    @property
    def weight(self):
        return [layer.conv.weight for layer in self]

    @property
    def bias(self):
        return [layer.conv.bias for layer in self]

    @property
    def dim(self):
        for layer in self:
            return layer.dim
        
    @property
    def transposed(self):
        for layer in self:
            return layer.transposed
        
    @property
    def groups(self):
        return len(self.children())

    @property
    def stride(self):
        for layer in self:
            return layer.stride

    @stride.setter
    def stride(self, value):
        for layer in self:
            layer.stride = value

    @property
    def padding(self):
        for layer in self:
            return layer.padding

    @padding.setter
    def padding(self, value):
        for layer in self:
            layer.padding = value

    @property
    def output_padding(self):
        for layer in self:
            return layer.output_padding

    @output_padding.setter
    def output_padding(self, value):
        for layer in self:
            layer.output_padding = value

    @property
    def dilation(self):
        for layer in self:
            return layer.dilation

    @dilation.setter
    def dilation(self, value):
        for layer in self:
            layer.dilation = value

    @property
    def padding_mode(self):
        for layer in self:
            return layer.padding_mode

    @padding_mode.setter
    def padding_mode(self, value):
        for layer in self:
            layer.padding_mode = value

    def shape(self, x, **overload):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
        overload : dict
            All parameters defined at build time can be overridden
            at call time, except `dim`, `in_channels`, `out_channels`
            and `kernel_size`.

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        
        for layer in self:
            shape = layer.shape(x, **overload)
            break
        shape = list(shape)
        shape[1] = self.out_channels
        return tuple(shape)
    
    def __str__(self):
        in_channels = [l.in_channels for l in self]
        out_channels = [l.out_channels for l in self]
        in_channels = ', '.join(in_channels)
        out_channels = ', '.join(out_channels)
        s = [f'[{self.in_channels}]', f'[{self.out_channels}]']
        s += [f'kernel_size={self.kernel_size}']
        if self.transposed:
            s += [f'transposed=True']
        if any(s > 1 for s in self.stride):
            s += [f'stride={self.stride}']
        if any(s > 1 for s in self.dilation):
            s += [f'dilation={self.dilation}']
        if self.groups > 1:
            s += [f'groups={self.groups}']
        s = ', '.join(s)
        return f'SimpleConv({s})'
    
    __repr__ = __str__


def _defer_property(prop, module, setter=False):
    """Return a 'property' objet that links to a submodule property

    prop (str) : property name
    module (str): module name
    setter (bool, default=False) : define a setter
    returns (property) : property object

    """
    if setter:
        return property(lambda self: getattr(getattr(self, module), prop),
                        lambda self, val: setattr(getattr(self, module), prop, val))
    else:
        return property(lambda self: getattr(getattr(self, module), prop))


class Conv(Module):
    """Convolution layer (with batch norm and activation).

    Applies a convolution over an input signal.
    Optionally: apply an activation function to the output.

    """
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 groups=1,
                 bias=True,
                 transposed=False,
                 output_padding=0,
                 activation=None,
                 batch_norm=False,
                 order='nca',
                 inplace=True):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a sequence, grouped convolutions are used.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.
            
        stride : int or sequence[int], default=1:
            Stride of the convolution.
            
        padding : int or sequence[int] or 'auto', default=0
            Zero-padding added to all three sides of the input.
            
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
            
        dilation : int or sequence[int], default=1
            Spacing between kernel elements.
            
        groups : int, default=1
            Number of blocked connections from input channels to
            output channels. Using this parameter is an alternative to
            the use of 'sequence' input/output channels. In that case,
            the number of input and output channels in each group is
            found by dividing the ``input_channels`` and ``output_channels``
            with ``groups``.
            
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            
        transposed : bool, default=False:
            Transposed convolution.
            
        output_padding : int or sequence[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if `transposed is True`.
            
        activation : str or type or callable, optional
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
                
        batch_norm : bool or type or callable, optional
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).

        order : permutation of 'nca', default='nca'
            Order in which to perform the normalization (n), convolution (c)
            and activation (a).

        inplace : bool, default=True
            Apply activation inplace if possible
            (i.e., not ``is_leaf and requires_grad``).

        """
        super().__init__()

        # Store dimension
        self.inplace = inplace
        self.order = self._fix_order(order)

        # Build modules
        opt_conv = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            transposed=transposed,
            output_padding=output_padding,
            bias=bias)
        conv = self._build_conv(dim, in_channels, out_channels, groups, **opt_conv)
        batch_norm = self._build_batch_norm(batch_norm, conv, order)
        activation = self._build_activation(activation)

        # Assign submodules in order so that they are nicely
        # ordered during pretty printing
        for o in order:
            if o == 'n':
                self.batch_norm = batch_norm
            elif o == 'c':
                self.conv = conv
            elif o == 'a':
                self.activation = activation

        # Use appropriate weight initialization when possible
        self._init_weights(conv, activation)

    @staticmethod
    def _fix_order(order):
        if 'n' not in order:
            order = order + 'n'
        if 'c' not in order:
            order = order + 'c'
        if 'a' not in order:
            order = order + 'a'
        return order

    @staticmethod
    def _init_weights(conv, activation):
        if isinstance(activation, tnn.ReLU):
            conv.reset_parameters(a=0)
        elif isinstance(activation, tnn.LeakyReLU):
            conv.reset_parameters(a=activation.negative_slope)
        else:
            conv.reset_parameters()

    @staticmethod
    def _build_conv(dim, in_channels, out_channels, groups, **opt_conv):

        # Check if "manual" grouped conv are required
        in_channels = py.make_list(in_channels)
        out_channels = py.make_list(out_channels)
        if len(in_channels) > 1 and groups > 1:
            raise ValueError('Cannot use both `groups` and multiple '
                             'input channels, as both define grouped '
                             'convolutions.')
        if len(in_channels) == 1 and len(out_channels) == 1:
            ConvKlass = SimpleConv
            in_channels = in_channels[0]
            out_channels = out_channels[0]
            opt_conv['groups'] = groups
        else:
            ConvKlass = GroupedConv
            opt_conv['groups'] = max(len(in_channels), len(out_channels))

        conv = ConvKlass(dim, in_channels, out_channels, **opt_conv)
        return conv

    @staticmethod
    def _build_activation(activation):
        #   an activation can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        #   it is useful to accept both these cases as they allow to either:
        #       * have a learnable activation specific to this module
        #       * have a learnable activation shared with other modules
        #       * have a non-learnable activation
        if isinstance(activation, str):
            activation = _map_activations.get(activation.lower(), None)
        activation = (activation() if inspect.isclass(activation)
                      else activation if callable(activation)
                      else None)
        return activation

    @staticmethod
    def _build_batch_norm(batch_norm, conv, order):
        #   an normalization can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        dim = conv.dim
        in_channels = (conv.in_channels if order.index('n') < order.index('c')
                       else conv.out_channels)
        if isinstance(batch_norm, bool) and batch_norm:
            batch_norm = BatchNorm(dim, in_channels)
        batch_norm = (batch_norm(dim, in_channels)
                      if inspect.isclass(batch_norm)
                      else batch_norm if callable(batch_norm)
                      else None)
        return batch_norm

    weight = _defer_property('weight', 'conv')
    bias = _defer_property('bias', 'conv')
    dim = _defer_property('dim', 'conv')
    in_channels = _defer_property('in_channels', 'conv')
    out_channels = _defer_property('out_channels', 'conv')
    transposed = _defer_property('transposed', 'conv')
    stride = _defer_property('stride', 'conv', setter=True)
    padding = _defer_property('padding', 'conv', setter=True)
    output_padding = _defer_property('output_padding', 'conv', setter=True)
    dilation = _defer_property('dilation', 'conv', setter=True)
    padding_mode = _defer_property('padding_mode', 'conv', setter=True)
    groups = _defer_property('groups', 'conv', setter=True)
            
    def forward(self, x, **overload):
        """Forward pass.

        Parameters
        ----------
        x : (batch, in_channel, *in_spatial) tensor
            Input tensor
        overload : dict
            Some parameters defined at build time can be overridden
            at call time: ['stride', 'padding', 'output_padding',
            'dilation', 'padding_mode']

        Returns
        -------
        x : (batch, out_channel, *out_spatial) tensor
            Convolved tensor

        Notes
        -----
        The output shape of an input tensor can be guessed using the
        method `shape`.

        """
        order = overload.pop('order', self.order)
        order = self._fix_order(order)
        batch_norm = overload.pop('batch_norm', self.batch_norm)
        batch_norm = self._build_batch_norm(batch_norm, self.conv, order)
        activation = overload.pop('activation', self.activation)
        activation = self._build_activation(activation)

        # make sure we can use inplace
        activation = copy(activation)
        if (activation and self.inplace and 
                hasattr(activation, 'inplace') and
                not (x.is_leaf and x.requires_grad)):
            activation.inplace = True

        # BatchNorm + Convolution + Activation
        for o in order:
            if o == 'n' and batch_norm:
                x = batch_norm(x)
            elif o == 'c':
                x = self.conv(x, **overload)
            elif o == 'a' and activation:
                x = activation(x)
        return x

    def shape(self, x, **overload):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
        overload : dict
            All parameters defined at build time can be overridden
            at call time, except `dim`, `in_channels`, `out_channels`
            and `kernel_size`.

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        return self.conv.shape(x, **overload)


class ConvBlock(Module):
    """A block of multiple convolutions with the same number of input
    and output channels. Usually used inside a ResBlock."""

    def __init__(self,
                 dim,
                 channels,
                 nb_conv=2,
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

        nb_conv : int, default=2
            Number of convolutions

        recurrent : bool, default=False
            Use recurrent convolutions (weights are shared between blocks)

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
        super().__init__()

        conv_opt = dict(
            dim=dim,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding='auto',
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            order=order,
            inplace=inplace,
        )
        nb_conv_inde = 1 if recurrent else nb_conv
        convs = [Conv(**conv_opt) for _ in range(nb_conv_inde)]
        self.convs = tnn.ModuleList(convs)
        self.recurrent = recurrent
        self.nb_conv = nb_conv

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

        nb_conv = overload.pop('nb_conv', self.nb_conv)
        if nb_conv != len(self.convs) and not self.recurrent:
            raise ValueError(f'Number of required convolutions and registered '
                             f'convolutions not consistent: '
                             f'{nb_conv} vs. {len(self.convs)}.')

        if self.recurrent:
            conv = self.convs[0]
            for _ in range(nb_conv):
                x = conv(x, **overload)
        else:
            for conv in self.convs:
                x = conv(x, **overload)
        return x


class ResBlock(Module):
    """Residual block"""

    def __init__(self,
                 dim,
                 channels,
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

        super().__init__()

        conv_opt = dict(
            dim=dim,
            channels=channels,
            nb_conv=nb_conv,
            recurrent='conv' in recurrent,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            order=order,
            inplace=inplace,
        )
        nb_res_inde = 1 if 'res' in recurrent else nb_res
        blocks = [ConvBlock(**conv_opt) for _ in range(nb_res_inde)]
        self.blocks = tnn.ModuleList(blocks)
        self.recurrent = recurrent

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
        nb_res = overload.pop('nb_res', self.nb_res)
        if nb_res != len(self.blocks) and not self.recurrent:
            raise ValueError(
                f'Number of required blocks and registered '
                f'blocks not consistent: '
                f'{nb_res} vs. {len(self.blocks)}.')

        if self.recurrent:
            block = self.blocks[0]
            for _ in range(nb_res):
                x += block(x, **overload)
        else:
            for block in self.blocks:
                x += block(x, **overload)
        return x