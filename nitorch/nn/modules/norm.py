"""Batch norm layer"""
import torch
from torch import nn as tnn
from nitorch.nn.base import Module, nitorchmodule


class BatchNorm(Module):
    """Batch normalization layer."""
    def __init__(self, dim, nb_channels, *args, **kwargs):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Spatial dimension
        nb_channels : int
            Number of channels in the input image
        eps : float, default=1e-5
            A value added to the denominator for numerical stability
        momentum : float, default=0.1
             The value used for the running_mean and running_var computation.
            Can be set to ``None`` for cumulative moving average
            (i.e. simple average).
        affine : bool, default=True
            When ``True``, this module has learnable affine parameters.
        track_running_stats : bool, default=True
            If``True``, track the running mean and variance.
            If ``False``, do not track such statistics and use batch
            statistics instead in both training and eval modes if the
            running mean and variance are ``None``

        """
        super().__init__()

        # Store dimension
        self.dim = dim

        # Select Layer
        if dim == 1:
            self.batchnorm = nitorchmodule(tnn.BatchNorm1d)(nb_channels, *args, **kwargs)
        elif dim == 2:
            self.batchnorm = nitorchmodule(tnn.BatchNorm2d)(nb_channels, *args, **kwargs)
        elif dim == 3:
            self.batchnorm = nitorchmodule(tnn.BatchNorm3d)(nb_channels, *args, **kwargs)
        else:
            NotImplementedError('BatchNorm is only implemented in 1, 2, or 3D.')

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Input tensor

        Returns
        -------
        x : (batch, channel, *spatial) tensor
            Normalized tensor

        """
        return self.batchnorm(x)

    in_channels = property(lambda self: self.num_features)
    out_channels = property(lambda self: self.num_features)

    @staticmethod
    def shape(x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        return tuple(x)

    def __str__(self):
        s = [f'{self.batchnorm.num_features}']
        if self.batchnorm.eps != 1e-5:
            s += [f'eps={self.batchnorm.eps}']
        if self.batchnorm.momentum != 0.1:
            s += [f'momentum={self.batchnorm.momentum}']
        if not self.batchnorm.affine:
            s += [f'affine=False']
        if not self.batchnorm.track_running_stats:
            s += [f'track_running_stats=False']
        s = ', '.join(s)
        return f'BatchNorm({s})'
    
    __repr__ = __str__
