"""Batch norm layer"""
from torch import nn as tnn
from .base import Module, nitorchmodule


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
