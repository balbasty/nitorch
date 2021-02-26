"""Various reduction layers."""

from .base import Module
from nitorch.core import math


class Reduction(Module):
    """Base class for reductions."""
    reduction = math.max
    omitnan = False

    def __init__(self, dim=None, keepdim=False, omitnan=None, reduction=None):
        """

        Parameters
        ----------
        dim : int or list[int], default=`2:`
            Dimensions to reduce. By default, all but batch and channel.
        keepdim : bool, default=False
            Keep reduced dimensions.
        omitnan : bool, default=False
            Discard nans when reducing.
        reduction : str or  callable, optional
            Reduction function
        """

        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.omitnan = omitnan if omitnan is not None else type(self).omitnan
        self.reduction = reduction or type(self).reduction

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to reduce
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
        reduction = overload.get('reduction', self.reduction)

        if isinstance(reduction, str):
            if not reduction in reduction_functions:
                raise ValueError(f'Unknown reducton {reduction}')
            reduction = reduction_functions[reduction]

        if dim is None:
            dim = list(range(2, x.dim()))
        if isinstance(dim, slice):
            start = dim.start if dim.start is not None else 2
            stop = dim.stop if dim.stop is not None else x.dim()
            step = dim.step if dim.step is not None else 1
            dim = list(range(start, stop, step))

        print(reduction)
        return reduction(x, dim, keepdim=keepdim, omitnan=omitnan)


class MaxReduction(Reduction):
    """Maximum"""
    reduction = math.max


class MinReduction(Reduction):
    """Minimum"""
    reduction = math.min


class MedianReduction(Reduction):
    """Median"""
    reduction = math.median


class SumReduction(Reduction):
    """Sum"""
    reduction = math.sum


class MeanReduction(Reduction):
    """Mean"""
    reduction = math.mean


class VarReduction(Reduction):
    """Mean"""
    reduction = math.var


class StdReduction(Reduction):
    """Mean"""
    reduction = math.std


class NanMinReduction(MinReduction):
    omitnan = True


class NanMaxReduction(MaxReduction):
    omitnan = True


class NanSumReduction(SumReduction):
    omitnan = True


class NanMeanReduction(MeanReduction):
    omitnan = True


class NanVarReduction(VarReduction):
    omitnan = True


class NanStdReduction(StdReduction):
    omitnan = True


reduction_functions = {
    'max': math.max,
    'min': math.min,
    'median': math.median,
    'sum': math.sum,
    'mean': math.mean,
    'nanmax': math.nanmax,
    'nanmin': math.nanmin,
    'nansum': math.nansum,
    'nanmean': math.nanmean,
}


reductions = {
    'max': MaxReduction,
    'min': MinReduction,
    'median': MedianReduction,
    'sum': SumReduction,
    'mean': MeanReduction,
    'nanmax': NanMaxReduction,
    'nanmin': NanMinReduction,
    'nansum': NanSumReduction,
    'nanmean': NanMeanReduction,
}
