"""Batch norm layer"""
import torch
from torch import nn as tnn
from nitorch.nn.base import Module, nitorchmodule


class BatchNorm(Module):
    """Batch normalization layer.

    BatchNorm computes statistics across [batch, *spatial].
    """
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
            self.norm = nitorchmodule(tnn.BatchNorm1d)(nb_channels, *args, **kwargs)
        elif dim == 2:
            self.norm = nitorchmodule(tnn.BatchNorm2d)(nb_channels, *args, **kwargs)
        elif dim == 3:
            self.norm = nitorchmodule(tnn.BatchNorm3d)(nb_channels, *args, **kwargs)
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
        return self.norm(x)

    in_channels = property(lambda self: self.norm.num_features)
    out_channels = property(lambda self: self.norm.num_features)

    @staticmethod
    def shape(x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        return tuple(x)

    def __str__(self):
        s = [f'{self.norm.num_features}']
        if self.norm.eps != 1e-5:
            s += [f'eps={self.norm.eps}']
        if self.norm.momentum != 0.1:
            s += [f'momentum={self.norm.momentum}']
        if not self.norm.affine:
            s += [f'affine=False']
        if not self.norm.track_running_stats:
            s += [f'track_running_stats=False']
        s = ', '.join(s)
        return f'BatchNorm({s})'
    
    __repr__ = __str__


class InstanceNorm(Module):
    """Instance normalization layer.

    InstanceNorm computes statistics across [*spatial].
    """

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
            self.norm = nitorchmodule(tnn.InstanceNorm1d)(nb_channels, *args, **kwargs)
        elif dim == 2:
            self.norm = nitorchmodule(tnn.InstanceNorm2d)(nb_channels, *args, **kwargs)
        elif dim == 3:
            self.norm = nitorchmodule(tnn.InstanceNorm3d)(nb_channels, *args, **kwargs)
        else:
            NotImplementedError('InstanceNorm is only implemented in 1, 2, or 3D.')

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
        return self.norm(x)

    in_channels = property(lambda self: self.norm.num_features)
    out_channels = property(lambda self: self.norm.num_features)

    @staticmethod
    def shape(x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        return tuple(x)

    def __str__(self):
        s = [f'{self.norm.num_features}']
        if self.norm.eps != 1e-5:
            s += [f'eps={self.norm.eps}']
        if self.norm.momentum != 0.1:
            s += [f'momentum={self.norm.momentum}']
        if not self.norm.affine:
            s += [f'affine=False']
        if not self.norm.track_running_stats:
            s += [f'track_running_stats=False']
        s = ', '.join(s)
        return f'InstanceNorm({s})'

    __repr__ = __str__


class GroupNorm(Module):
    """Group normalization layer.

    GroupNorm computes statistics across [channels//groups, *spatial].

    .. `groups=nb_channels` is equivalent to InstanceNorm
    .. `groups=1` is equivalent to (a certain type of) LayerNorm
    """

    def __init__(self, dim, nb_channels, groups, *args, **kwargs):
        """
        Parameters
        ----------
        dim : {1, 2, 3}
            Spatial dimension
        nb_channels : int
            Number of channels in the input image
        groups : int
            Number of groups to separate the channels into
        eps : float, default=1e-5
            A value added to the denominator for numerical stability
        affine : bool, default=True
            When ``True``, this module has learnable affine parameters.

        """
        # Note that `dim` is not used, but we keep it for consistency
        # with [Batch,Instance]Norm.
        super().__init__()

        # Store dimension
        self.dim = dim
        self.norm = nitorchmodule(tnn.GroupNorm)(groups, nb_channels, *args, **kwargs)

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
        return self.norm(x)

    in_channels = property(lambda self: self.norm.num_features)
    out_channels = property(lambda self: self.norm.num_features)

    @staticmethod
    def shape(x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        return tuple(x)

    def __str__(self):
        s = [f'{self.norm.num_features}//{self.norm.groups}']
        if self.norm.eps != 1e-5:
            s += [f'eps={self.norm.eps}']
        if not self.norm.affine:
            s += [f'affine=False']
        s = ', '.join(s)
        return f'GroupNorm({s})'

    __repr__ = __str__


class LayerNorm(GroupNorm):
    """Layer normalization layer.

    LayerNorm computes statistics across [channels, *spatial].

    This layer targets computed vision tasks and is less versatile than
    PyTorch's LayerNorm class. We actually use `GroupNorm` to implement it.
    """

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
        affine : bool, default=True
            When ``True``, this module has learnable affine parameters.

        """
        super().__init__(dim, nb_channels, nb_channels, *args, **kwargs)

    def __str__(self):
        s = []
        if self.batchnorm.eps != 1e-5:
            s += [f'eps={self.norm.eps}']
        if not self.norm.affine:
            s += [f'affine=False']
        s = ', '.join(s)
        return f'LayerNorm({s})'

    __repr__ = __str__


def norm_from_name(name):
    """Return a normalization Class from its name.

    Parameters
    ----------
    name : str
        Normalization name. Registered functions are:
        - 'batch'    : BatchNorm    == Normalize across [batch, *spatial]
        - 'instance' : InstanceNorm == Normalize across [*spatial]
        - 'layer'    : LayerNorm    == Normalize across [channel, *spatial]
        - 'group'    : GroupNorm    == Normalize across [channel//groups, *spatial]

    Returns
    -------
    Norm : type(Module)
        A normalization class

    """
    name = name.lower()
    if name == 'batch':
        return BatchNorm
    if name == 'instance':
        return InstanceNorm
    if name == 'layer':
        return LayerNorm
    if name == 'group':
        return GroupNorm
    raise KeyError(f'Normalization {name} is not registered.')


def make_norm_from_name(name, dim, nb_channels, *args, **kwargs):
    """Return an instantiated normalization from its name.

    Parameters
    ----------
    name : str or int
        Normalization name. Registered functions are:
        - 'batch'    : BatchNorm    == Normalize across [batch, *spatial]
        - 'instance' : InstanceNorm == Normalize across [*spatial]
        - 'layer'    : LayerNorm    == Normalize across [channel, *spatial]
        - int        : GroupNorm    == Normalize across [channel//groups, *spatial]
    dim : int
        Number of spatial dimensions
    nb_channels : int
        Number of input channels
    eps : float, default=1e-5
        A value added to the denominator for numerical stability
    affine : bool, default=True
        When ``True``, this module has learnable affine parameters.


    Returns
    -------
    activation : Module
        An instantiated activation module

    """
    groups = 0
    if isinstance(name, int):
        groups = name
        name = 'group'
    klass = norm_from_name(name)
    if groups:
        return klass(dim, nb_channels, groups, *args, **kwargs)
    return klass(dim, nb_channels, *args, **kwargs)
