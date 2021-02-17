"""Pooling layers"""

import torch.nn as tnn
import torch
from ._base import Module
from ._reduction import reductions
from ...core.py import make_list
from copy import copy


class Pool(Module):
    """Generic Pooling layer."""

    reduction = 'max'

    def __init__(self, dim, kernel_size=3, stride=None, padding=0,
                 dilation=1, reduction=None):
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
        reduction : {'min', 'max', 'mean', 'sum'}, default='max'
            Pooling type
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.reduction = reduction or self.reduction

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

        # extract patches
        stride = [(s or k) for k, s in zip(kernel_size, stride)]
        for d, (k, s) in enumerate(zip(kernel_size, stride)):
            if padding[d] != 0:
                raise ValueError('Padding not implemented for reduction '
                                 '{}'.format(padding[d]))
            if dilation[d] != 1:
                raise ValueError('Dilation not implemented for reduction '
                                 '{}'.format(dilation[d]))
            x = x.unfold(dimension=d + 2, size=k, step=s)

        # flatten patches
        x = x.reshape([*(x.shape[:dim+2]), -1])

        # reduce
        reduce_fn = reductions[reduction](dim=-1)
        x = reduce_fn(x)
        return x


class MaxPool(Module):
    """Generic max pooling layer"""

    def __init__(self, dim, kernel_size=3, stride=None, padding=0,
                 dilation=1):
        super().__init__(dim, kernel_size, stride, padding, dilation,
                         reduction='max')

        if dim == 1:
            self.pool = tnn.MaxPool1d
        elif dim == 2:
            self.pool = tnn.MaxPool2d
        elif dim == 3:
            self.pool = tnn.MaxPool3d
        else:
            self.pool = None

        if self.pool is not None:
            self.pool = self.pool(kernel_size, stride, padding, dilation)

    def forward(self, x, **overload):

        if self.pool is None:
            return super().forward(x, **overload)

        dim = self.dim
        kernel_size = make_list(overload.get('kernel_size', self.kernel_size), dim)
        stride = make_list(overload.get('stride', self.stride))
        padding = make_list(overload.get('padding', self.padding))
        dilation = make_list(overload.get('dilation', self.dilation))

        pool = copy(self.pool)
        pool.kernel_size = kernel_size
        pool.stride = stride
        pool.padding = padding
        pool.dilation = dilation

        return pool(x)


class MeanPool(Module):
    """Generic mean pooling layer"""

    def __init__(self, dim, kernel_size=3, stride=None, padding=0,
                 dilation=1):
        super().__init__(dim, kernel_size, stride, padding, dilation,
                         reduction='mean')

        if dim == 1:
            self.pool = tnn.AvgPool1d
        elif dim == 2:
            self.pool = tnn.AvgPool2d
        elif dim == 3:
            self.pool = tnn.AvgPool3d
        else:
            self.pool = None

        if self.pool is not None:
            self.pool = self.pool(kernel_size, stride, padding, dilation)

    def forward(self, x, w=None, **overload):

        if self.pool is None and w is None:
            return super().forward(x, **overload)

        dim = self.dim
        kernel_size = make_list(overload.get('kernel_size', self.kernel_size), dim)
        stride = make_list(overload.get('stride', self.stride))
        padding = make_list(overload.get('padding', self.padding))
        dilation = make_list(overload.get('dilation', self.dilation))

        if w is not None:
            pool = SumPool(dim=dim, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
            w = w.expand(x.shape)
            return pool(x*w) / pool(w)
        else:
            pool = copy(self.pool)
            pool.kernel_size = kernel_size
            pool.stride = stride
            pool.padding = padding
            pool.dilation = dilation
            return pool(x)


class MinPool(Pool):
    reduction = 'min'


class SumPool(Pool):
    reduction = 'sum'
