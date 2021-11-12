"""
Base class for losses.

Reduction mecanisms are implemented here.
"""

import torch.nn as tnn
from nitorch.core.math import nansum, nanmean, sum, mean


class Loss(tnn.Module):
    """Base class for losses."""

    def __init__(self, reduction='mean'):
        """

        Parameters
        ----------
        reduction : {'mean', 'sum', 'none'} or callable, default='mean'
            Type of reduction to apply.


        """
        super().__init__()
        self.reduction = reduction or 'none'

    def reduce(self, x):
        reduction = self.reduction
        if reduction is None:
            return x
        elif isinstance(reduction, str):
            reduction = reduction.lower()
            if reduction == 'mean':
                return mean(x)
            elif reduction == 'sum':
                return sum(x)
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

    def forward(self, x):
        return self.reduce(x)
