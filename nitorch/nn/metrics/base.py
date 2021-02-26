"""
Base class for metrics.

Reduction mechanisms are implemented here.
"""

import torch.nn as tnn
from nitorch.core.math import nansum, nanmean


class Metric(tnn.Module):
    """Base class for metrics."""

    def __init__(self, reduction='mean', *args, **kwargs):
        """

        Parameters
        ----------
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.


        """
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, x, **kwargs):
        reduction = kwargs.get('reduction', self.reduction)
        if reduction is None:
            return x
        elif isinstance(reduction, str):
            reduction = reduction.lower()
            if reduction == 'mean':
                return nanmean(x)
            elif reduction == 'sum':
                return nansum(x)
            elif reduction == 'none':
                return x
            else:
                raise ValueError('Unknown reduction {}'.format(reduction))
        elif callable(reduction):
            return reduction(x)
        else:
            raise TypeError("reduction should be a callable or in "
                            "('none', 'sum', 'mean'). Got {}."
                            .format(type(reduction)))
