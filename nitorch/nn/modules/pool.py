"""Pooling layers"""

import inspect
import math
import torch
from nitorch.core.py import make_list, make_tuple
from nitorch.spatial import pool, compute_conv_shape
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
                 return_indices=False, ceil=False):
        """

        Parameters
        ----------
        dim : int
            Dimensionality
        kernel_size : int or sequence[int], default=3
            Kernel/Window size
        stride : int or sequence[int], default=kernel_size
            Step/stride
        padding : {'valid', 'same'} or int or sequence[int], default=0
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
        ceil : bool, default=False
            Whether to take the ceil rather than the floor when computing
            the output shape. Ceil ensures that all input elements
            belong to at least one tile, but is not implemented with
            all reduction methods.
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = make_list(kernel_size, dim)
        self.stride = make_list(stride, dim)
        self.padding = padding
        self.dilation = make_list(dilation, dim)
        self.reduction = reduction or self.reduction
        self.return_indices = return_indices
        self.ceil = ceil

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
            
    def shape(self, x):
        """Compute output shape of the equivalent ``forward`` call.

        Parameters
        ----------
        x : tuple or (batch, in_channel, *in_spatial) tensor
            Input tensor or its shape
            (Only used by min/max/median)

        Returns
        -------
        shape : tuple[int]
            Output shape

        """
        if torch.is_tensor(x):
            inshape = tuple(x.shape)
        else:
            inshape = x

        batch, channel, *inshape = inshape
        return (batch, channel, *compute_conv_shape(
            inshape, self.kernel_size, stride=self.stride,
            dilation=self.dilation, padding=self.padding, ceil=self.ceil))

    def forward(self, x, return_indices=None):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to pool
        return_indices : bool, default=self.return_indices

        Returns
        -------
        x : (batch, channel, *spatial_out) tensor
            Pooled tensor
        indices : (batch, channel, *spatial_out, dim) tensor, if `return_indices`
            Indices of input elements.

        """
        return_indices = self.return_indices
        if return_indices is None:
            return_indices = self.return_indices

        x = pool(self.dim, x,
                 kernel_size=self.kernel_size,
                 stride=self.stride,
                 dilation=self.dilation,
                 padding=self.padding,
                 reduction=self.reduction,
                 return_indices=return_indices,
                 ceil=self.ceil)
        if return_indices:
            x, ind = x

        if self.activation:
            x = self.activation(x)
        return (x, ind) if return_indices else x

    def extra_repr(self):
        s = [f'kernel_size={list(self.kernel_size)}']
        if any(x and x != k for x, k in zip(self.stride, self.kernel_size)):
            s += [f'stride={list(self.stride)}']
        if any(x > 1 for x in self.dilation):
            s += [f'dilation={list(self.dilation)}']
        s = ', '.join(s)
        return s


class MaxPool(Pool):
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


class MeanPool(Pool):
    """Generic mean pooling layer"""
    reduction = 'mean'

    def forward(self, x, w=None):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to pool
        w : (batch, channel, *spatial) tensor, optional
            Tensor of weights. Its shape must be broadcastable to the
            shape of `x`.
        return_indices : bool, default=self.return_indices

        Returns
        -------
        x : (batch, channel, *spatial_out)
            Pooled tensor

        """
        if w is None:
            return super().forward(x)
        else:
            sumpool = SumPool(self.dim, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation)
            w = w.expand(x.shape)
            x = sumpool(x*w) / sumpool(w)

            if self.activation is not None:
                x = self.activation(x)

            return x


pool_map = {
    'max': MaxPool,
    'mean': MeanPool,
    'median': MedianPool,
    'sum': SumPool,
    'min': MinPool,
}
