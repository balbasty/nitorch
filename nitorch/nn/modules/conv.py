"""Convolution layers."""

import torch
from torch import nn as tnn
from .base import nitorchmodule, Module
from .norm import BatchNorm
from ..activations import _map_activations
from nitorch.core.py import make_tuple, rep_sequence, getargs, make_list
from nitorch.core import py
from copy import copy
import math
import inspect

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


@nitorchmodule
class GroupedConv(tnn.ModuleList):
    """Simple imbalanced grouped convolution
       (without activation, batch norm, etc.)
    """

    def __init__(self, dim, in_channels, out_channels, *args, transposed=False,
                 **kwargs):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
        in_channels : sequence[int]
            Number of channels in the input image.
        out_channels : sequence[int]
            Number of channels produced by the convolution.
        kernel_size : int or tuple[int]
            Size of the convolution kernel
        stride : int or tuple[int], default=1:
            Stride of the convolution.
        padding : int or tuple[int], default=0
            Zero-padding added to all three sides of the input.
        output_padding : int or tuple[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if `transposed is True`.
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
        dilation : int or tuple[int], default=1
            Spacing between kernel elements.
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
        transposed : bool, default=False:
            Transposed convolution.
        """
        in_channels = py.make_list(in_channels)
        out_channels = py.make_list(out_channels)
        if len(in_channels) != len(out_channels):
            raise ValueError(f'The number of input and output groups '
                             f'must be the same: {len(in_channels)} vs '
                             f'{len(out_channels)}')

        klass = _get_conv_class(dim, transposed)
        modules = [klass(i, o, *args, **kwargs)
                   for i, o in zip(in_channels, out_channels)]
        super().__init__(modules)
        self.in_channels = len(in_channels)
        self.out_channels = len(out_channels)

    def forward(self, input):
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
        output = output.split([c.out_channels for c in self], dim=1)
        for layer, inp, out in zip(self, input, output):
            out[...] = layer(inp)
        return output

    @property
    def dim(self):
        for layer in self:
            return len(layer.kernel_size)

    @property
    def stride(self):
        for layer in self:
            return layer.stride

    @stride.setter
    def stride(self, value):
        for layer in self:
            layer.stride = make_tuple(value, self.dim)

    @property
    def padding(self):
        for layer in self:
            return layer.padding

    @padding.setter
    def padding(self, value):
        for layer in self:
            layer.padding = make_tuple(value, self.dim)
            layer._padding_repeated_twice = rep_sequence(value, 2, interleaved=True)

    @property
    def output_padding(self):
        for layer in self:
            return layer.output_padding

    @output_padding.setter
    def output_padding(self, value):
        for layer in self:
            layer.output_padding = make_tuple(value, self.dim)

    @property
    def dilation(self):
        for layer in self:
            return layer.dilation

    @dilation.setter
    def dilation(self, value):
        for layer in self:
            layer.dilation = make_tuple(value, self.dim)

    @property
    def padding_mode(self):
        for layer in self:
            return layer.padding_mode

    @padding_mode.setter
    def padding_mode(self, value):
        for layer in self:
            layer.padding_mode = value


class Conv(Module):
    """Convolution layer (with activation).

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
                 output_padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 groups=1,
                 bias=True,
                 transposed=False,
                 activation=None,
                 batch_norm=False,
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
        kernel_size : int or tuple[int]
            Size of the convolution kernel
        stride : int or tuple[int], default=1:
            Stride of the convolution.
        padding : int or tuple[int], default=0
            Zero-padding added to all three sides of the input.
        output_padding : int or tuple[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if `transposed is True`.
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
        dilation : int or tuple[int], default=1
            Spacing between kernel elements.
        groups : int, default=1
            Number of blocked connections from input channels to
            output channels. Using this parameter is an alternative to
            the use of 'sequence' input/output channels. In that case,
            the number of input and output channels in each group is
            found by dividing the `input_channels` and `output_channels`
            with `groups`.
        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
        transposed : bool, default=False:
            Transposed convolution.
        activation : str or type or callable, optional
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
        batch_norm : bool or callable, optional
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
        inplace : bool, default=True
            Apply activation inplace if possible
            (i.e., not `is_leaf and requires_grad`).

        """
        super().__init__()

        # Store dimension
        self.dim = dim
        self.inplace = inplace

        # Check if "manual" grouped conv are required
        in_channels = py.make_list(in_channels)
        out_channels = py.make_list(out_channels)
        if len(in_channels) != len(out_channels):
            raise ValueError(f'The number of input and output groups '
                             f'must be the same: {len(in_channels)} vs '
                             f'{len(out_channels)}')
        if len(in_channels) > 1 and groups > 1:
            raise ValueError('Cannot use both `groups` and multiple '
                             'input channels, as both define grouped '
                             'convolutions.')

        opt_conv = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=bias)

        if len(in_channels) == 1:
            ConvKlass = _get_conv_class(dim, transposed)
            conv = ConvKlass(in_channels[0], out_channels[0],
                                  **opt_conv, groups=groups)
        else:
            conv = GroupedConv(dim, in_channels, out_channels,
                                    **opt_conv, transposed=transposed)

        # Select batch norm
        if isinstance(batch_norm, bool) and batch_norm:
                batch_norm = BatchNorm(self.dim, conv.in_channels)
        self.batch_norm = batch_norm() if inspect.isclass(batch_norm) \
                          else batch_norm if callable(batch_norm) \
                          else None

        # Set conv attribute after batch_norm so that they are nicely
        # ordered during pretty printing
        self.conv = conv

        # Add activation
        #   an activation can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        #   it is useful to accept both these cases as they allow to either:
        #       * have a learnable activation specific to this module
        #       * have a learnable activation shared with other modules
        #       * have a non-learnable activation
        if isinstance(activation, str):
            activation = _map_activations.get(activation.lower(), None)
        self.activation = activation() if inspect.isclass(activation) \
                          else activation if callable(activation) \
                          else None

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

        conv = copy(self.conv)
        stride = overload.get('stride', conv.stride)
        padding = overload.get('padding', conv.padding)
        output_padding = overload.get('output_padding', conv.output_padding)
        dilation = overload.get('dilation', conv.dilation)
        padding_mode = overload.get('padding_mode', conv.padding_mode)

        # Override constructed parameters
        conv.stride = make_tuple(stride, self.dim)
        conv.padding = make_tuple(padding, self.dim)
        conv._padding_repeated_twice = rep_sequence(conv.padding, 2,
                                                    interleaved=True)
        conv.output_padding = make_tuple(output_padding, self.dim)
        conv.dilation = make_tuple(dilation, self.dim)
        conv.padding_mode = padding_mode

        # Batch norm
        batch_norm = overload.get('batch_norm', self.batch_norm)
        batch_norm = batch_norm() if inspect.isclass(batch_norm)    \
                     else batch_norm if callable(batch_norm)        \
                     else None

        # Activation
        activation = overload.get('activation', self.activation)
        if isinstance(activation, str):
            activation = _map_activations.get(activation.lower(), None)
        activation = activation() if inspect.isclass(activation)    \
                     else activation if callable(activation)        \
                     else None

        if self.inplace \
                and hasattr(activation, 'inplace') \
                and not (x.is_leaf and x.requires_grad):
            activation.inplace = True

        # BatchNorm + Convolution + Activation
        if batch_norm is not None:
            x = batch_norm(x)
        x = conv(x)
        if activation is not None:
            x = activation(x)
        return x

    def shape(self, x, **overload):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : (batch, in_channel, *in_spatial) tensor
            Input tensor
        overload : dict
            All parameters defined at build time can be overridden
            at call time, except `dim`, `in_channels`, `out_channels`
            and `kernel_size`.

        Returns
        -------
        shape : tuple[int]
            Output shape

        """

        stride = overload.get('stride', self.conv.stride)
        padding = overload.get('padding', self.conv.padding)
        output_padding = overload.get('output_padding', self.conv.output_padding)
        dilation = overload.get('dilation', self.conv.dilation)
        transposed = self.conv.transposed
        kernel_size = self.conv.kernel_size

        stride = make_tuple(stride, self.dim)
        padding = make_tuple(padding, self.dim)
        output_padding = make_tuple(output_padding, self.dim)
        dilation = make_tuple(dilation, self.dim)

        N = x.shape[0]
        C = self.conv.out_channels
        shape = [N, C]
        for L, S, Pi, D, K, Po in zip(x.shape[2:], stride, padding,
                                      dilation, kernel_size, output_padding):
            if transposed:
                shape += [(L - 1) * S - 2 * Pi + D * (K - 1) + Po + 1]
            else:
                shape += [math.floor((L + 2 * Pi - D * (K - 1) - 1) / S + 1)]
        return tuple(shape)
