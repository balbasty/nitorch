"""Convolution layers."""

import torch
from torch import nn as tnn
from ._base import nitorchmodule
from ._norm import BatchNorm
from ..activations import _map_activations
from ...core.pyutils import make_tuple, rep_sequence, getargs
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


@nitorchmodule
class Conv(tnn.Module):
    """Convolution layer (with activation).

    Applies a convolution over an input signal.
    Optionally: apply an activation function to the output.

    """
    def __init__(self, dim, *args, **kwargs):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Dimension of the convolving kernel
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int or tuple[int]
            Size of the convolution kernel
        stride : int or tuple[int], default=1:
            Stride of the convolution.
        padding : int or tuple[int], default=0
            Zero-padding added to all three sides of the input.
        output_padding : int or tuple[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if `transposed is True`.
      ``(out_padT, out_padH, out_padW)``. Default: 0
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
        dilation : int or tuple[int], default=1
            Spacing between kernel elements.
        groups : int, default=1
            Number of blocked connections from input channels to
            output channels.
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
        try_inplace : bool, default=True
            Apply activation inplace if possible
            (i.e., not (is_leaf and requires_grad).

        """
        super().__init__()

        # Get additional arguments that are not present in torch's conv
        transposed, activation, batch_norm, try_inplace = getargs(
            [('transposed', 10, False),
             ('activation', 11, None),
             ('batch_norm', 12, None),
             ('try_inplace', 13, True)],
            args, kwargs, consume=True)

        # Store dimension
        self.dim = dim
        self.try_inplace = try_inplace

        # Select Conv
        if transposed:
            if dim == 1:
                conv = nitorchmodule(tnn.ConvTranspose1d)(*args, **kwargs)
            elif dim == 2:
                conv = nitorchmodule(tnn.ConvTranspose2d)(*args, **kwargs)
            elif dim == 3:
                conv = nitorchmodule(tnn.ConvTranspose3d)(*args, **kwargs)
            else:
                NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
        else:
            if dim == 1:
                conv = nitorchmodule(tnn.Conv1d)(*args, **kwargs)
            elif dim == 2:
                conv = nitorchmodule(tnn.Conv2d)(*args, **kwargs)
            elif dim == 3:
                conv = nitorchmodule(tnn.Conv3d)(*args, **kwargs)
            else:
                NotImplementedError('Conv is only implemented in 1, 2, or 3D.')

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
            All parameters defined at build time can be overridden
            at call time, except `dim`, `in_channels`, `out_channels`
            and `kernel_size`.

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

        if self.try_inplace \
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


class ConvZeroCentre(Conv):
    """Same as Conv, but constrains convolution filter to have zero centre.

    """
    def __init__(self, dim, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)

    def forward(self, x, **overload):
        # zero centre
        k = self.conv.kernel_size
        if self.dim == 3:
            self.conv.weight.data[:, :, k[0] // 2, k[1] // 2, k[2] // 2].clamp_(0, 0)
        elif self.dim == 2:
            self.conv.weight.data[:, :, k[0] // 2, k[1] // 2].clamp_(0, 0)
        else:
            self.conv.weight.data[:, :, k[0] // 2].clamp_(0, 0)
        # parent class' forward pass
        x = super().forward(x, **overload)

        return x