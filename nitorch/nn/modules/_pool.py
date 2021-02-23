"""Pooling layers"""

import inspect
from nitorch.core.py import make_list
from nitorch.spatial import pool
from nitorch.nn.activations import _map_activations
from ._base import Module


class Pool(Module):
    """Generic Pooling layer."""

    reduction = 'max'

    def __init__(self, dim, kernel_size=3, stride=None, padding=0,
                 dilation=1, reduction=None, activation=None):
        """

        Parameters
        ----------
        dim : int
            Dimensionality
        kernel_size : int or list[int], default=3
            Kernel/Window size
        stride : int or list[int], default=kernel_size
            Step/stride
        padding : int, default=0
            Zero-padding
        dilation : int, default=1
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
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.reduction = reduction or self.reduction

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
        x : (batch, channel, *spatial_out)
            Pooled tensor

        """

        dim = self.dim
        kernel_size = make_list(overload.get('kernel_size', self.kernel_size), dim)
        stride = make_list(overload.get('stride', self.stride), dim)
        padding = make_list(overload.get('padding', self.padding), dim)
        dilation = make_list(overload.get('dilation', self.dilation), dim)
        reduction = overload.get('reduction', self.reduction)

        # Activation
        activation = overload.get('activation', self.activation)
        if isinstance(activation, str):
            activation = _map_activations.get(activation.lower(), None)
        activation = (activation() if inspect.isclass(activation)
                      else activation if callable(activation)
                      else None)

        x = pool(dim, x, kernel_size=kernel_size, stride=stride,
                 dilation=dilation, padding=padding, reduction=reduction)
        if activation is not None:
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


