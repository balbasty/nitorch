"""Pooling layers"""

import inspect
import math
import torch
from nitorch.core.py import make_list, make_tuple
from nitorch.spatial import pool
from nitorch.nn.activations import _map_activations
from nitorch.nn.base import Module


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


class Pool(Module):
    """Generic Pooling layer."""

    reduction = 'max'

    def __init__(self, dim, kernel_size=3, stride=None, padding=0,
                 dilation=1, reduction=None, activation=None,
                 return_indices=False):
        """

        Parameters
        ----------
        dim : int
            Dimensionality
        kernel_size : int or sequence[int], default=3
            Kernel/Window size
        stride : int or sequence[int], default=kernel_size
            Step/stride
        padding : int or sequence[int], default=0
            Zero-padding
        dilation : int or sequence[int], default=1
            Dilation
        reduction : {'min', 'max', 'mean', 'sum', 'median'}, default='max'
            Pooling type
        activation : str or type or callable, optional
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation
        return_indices : bool, default=False
            Return indices of the min/max/median elements.
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.reduction = reduction or self.reduction
        self.return_indices = return_indices

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
        self.activation = (activation() if inspect.isclass(activation)
                           else activation if callable(activation)
                           else None)
            
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
        if torch.is_tensor(x):
            inshape = tuple(x.shape)
        else:
            inshape = x
        
        stride = overload.get('stride', self.stride)
        padding = overload.get('padding', self.padding)
        dilation = overload.get('dilation', self.dilation)
        kernel_size = overload.get('kernel_size', self.kernel_size)

        stride = make_tuple(stride, self.dim)
        padding = make_tuple(padding, self.dim)
        kernel_size = make_tuple(kernel_size, self.dim)
        dilation = make_tuple(dilation, self.dim)

        return _guess_output_shape(
            inshape, self.dim, kernel_size,
            stride=stride, dilation=dilation, padding=padding)

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to pool
        overload : dict
            Most parameters defined at build time can be overriden at
            call time

        Returns
        -------
        x : (batch, channel, *spatial_out) tensor
            Pooled tensor
        indices : (batch, channel, *spatial_out, dim) tensor, if `return_indices`
            Indices of input elements.

        """

        dim = self.dim
        kernel_size = make_list(overload.get('kernel_size', self.kernel_size), dim)
        stride = make_list(overload.get('stride', self.stride), dim)
        padding = make_list(overload.get('padding', self.padding), dim)
        dilation = make_list(overload.get('dilation', self.dilation), dim)
        reduction = overload.get('reduction', self.reduction)
        return_indices = overload.get('return_indices', self.return_indices)

        # Activation
        activation = overload.get('activation', self.activation)
        if isinstance(activation, str):
            activation = _map_activations.get(activation.lower(), None)
        activation = (activation() if inspect.isclass(activation)
                      else activation if callable(activation)
                      else None)

        x = pool(dim, x,
                 kernel_size=kernel_size,
                 stride=stride,
                 dilation=dilation,
                 padding=padding,
                 reduction=reduction,
                 return_indices=return_indices)

        if activation:
            x = activation(x)
        return x


class MaxPool(Module):
    """Generic max pooling layer"""
    reduction = 'max'


class MinPool(Pool):
    """Generic min pooling layer"""
    reduction = 'min'


class SumPool(Pool):
    """Generic sum pooling layer"""
    reduction = 'sum'


class MedianPool(Pool):
    """Generic median pooling layer"""
    reduction = 'median'


class MeanPool(Module):
    """Generic mean pooling layer"""
    reduction = 'mean'

    def forward(self, x, w=None, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to pool
        w : (batch, channel, *spatial) tensor, optional
            Tensor of weights. Its shape must be broadcastable to the
            shape of `x`.
        overload : dict
            Most parameters defined at build time can be overriden at
            call time

        Returns
        -------
        x : (batch, channel, *spatial_out)
            Pooled tensor

        """

        if w is None:
            return super().forward(x, **overload)
        else:
            overload['reduction'] = 'sum'
            activation = overload.pop('activation', self.activation)
            overload['activation'] = 'None'
            sumpool = lambda t: super().forward(t, **overload)
            w = w.expand(x.shape)
            x = sumpool(x*w) / sumpool(w)

            if isinstance(activation, str):
                activation = _map_activations.get(activation.lower(), None)
            activation = (activation() if inspect.isclass(activation)
                          else activation if callable(activation)
                          else None)
            if activation is not None:
                x = activation(x)

            return x


