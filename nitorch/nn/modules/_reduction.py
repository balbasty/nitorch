"""Various reduction layers."""

import torch
import torch.nn as tnn
from ._base import nitorchmodule, Module
from ...core.py import make_list
from ...core import math


class Reduction(Module):
    """Base class for reductions."""

    def __init__(self, dim=None, keepdim=False, omitnan=False):
        """

        Parameters
        ----------
        dim : int or list[int], default=`2:`
            Dimensions to reduce. By default, all but batch and channel.
        keepdim : bool, default=False
            Keep reduced dimensions.
        omitnan : bool, default=False
            Discard nans when reducing.
        """

        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.omitnan = omitnan

    def forward(self, x, reduce, nanreduce, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to reduce
        reduce : callable
            Reduction function
        nanreduce : callable
            Reduction function that omits nans
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        reduced : tensor
            Reduced tensor

        """

        dim = overload.get('dim', self.dim)
        keepdim = overload.get('keepdim', self.keepdim)
        omitnan = overload.get('omitnan', self.omitnan)

        if dim is None:
            dim = list(range(2, x.dim()))
        if isinstance(dim, slice):
            start = dim.start if dim.start is not None else 2
            stop = dim.stop if dim.stop is not None else x.dim()
            step = dim.step if dim.step is not None else 1
            dim = list(range(start, stop, step))

        reduce_fn = nanreduce if omitnan else reduce
        return reduce_fn(x, dim, keepdim=keepdim)


def _max(input, dim=None, keepdim=False):
    """Multi-dimensional max reduction."""
    if not isinstance(dim, (list, tuple)):
        return torch.max(input, dim=dim, keepdim=keepdim).values

    nb_dim = input.dim()
    dim = make_list(dim)
    dim = list(reversed(sorted([nb_dim+d if d < 0 else d for d in dim])))
    for d in dim:
        input = torch.max(input, dim=d, keepdim=keepdim).values
    return input


def _min(input, dim=None, keepdim=False):
    """Multi-dimensional min reduction."""
    if not isinstance(dim, (list, tuple)):
        return torch.min(input, dim=dim, keepdim=keepdim).values

    nb_dim = input.dim()
    dim = make_list(dim)
    dim = list(reversed(sorted([nb_dim+d if d < 0 else d for d in dim])))
    for d in dim:
        input = torch.min(input, dim=d, keepdim=keepdim).values
    return input


class MaxReduction(Reduction):
    """Maximum"""
    def forward(self, x, **overload):
        return super().forward(x, _max, math.nanmax, **overload)


class MinReduction(Reduction):
    """Minimum"""
    def forward(self, x, **overload):
        return super().forward(x, _min, math.nanmin, **overload)


class SumReduction(Reduction):
    """Sum"""
    def forward(self, x, **overload):
        return super().forward(x, torch.sum, math.nansum, **overload)


class MeanReduction(Reduction):
    """Mean"""
    def forward(self, x, **overload):
        return super().forward(x, torch.mean, math.nanmean, **overload)


def _forceomitnan(klass):
    """Decorator to force `omitnan = False`"""

    init = klass.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.omitnan = False

    klass.__init__ = __init__
    return klass


reductions = {
    'max': MaxReduction,
    'min': MinReduction,
    'sum': SumReduction,
    'mean': MeanReduction,
    'nanmax': _forceomitnan(MaxReduction),
    'nanmin': _forceomitnan(MinReduction),
    'nansum': _forceomitnan(SumReduction),
    'nanmean': _forceomitnan(MeanReduction),
}
