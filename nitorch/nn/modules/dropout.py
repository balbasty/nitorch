"""Dropout layer"""
from torch import nn as tnn
from .base import Module, nitorchmodule


class Dropout(Module):
    """Dropout layer (only applied during training).
    
    Dropout is commonly applied both before, and after, the activation function:

    https://stats.stackexchange.com/a/317313

    This class is written as to enable this usage. This is achieved by simply pointing the class to
    the activation function and specifying if Dropout should be applied before or after calling it.

    """
    def __init__(self, activation=None, before_activation=True, **kwargs):
        """
        Parameters
        ----------
        activation : callable, optional
            activation function
        before_activation : bool, default=True
            apply dropout before, or after, activation
        p : float, default=0.5
            Probability of an element to be zeroed.
        inplace  : default=False
            If set to True, will do this operation in-place.

        """
        super().__init__()

        # activation        
        if activation is None:
            activation = lambda x: activation(x)
        self.activation = activation
        self.before_activation = before_activation

        # Add Layer
        self.dropout = nitorchmodule(tnn.Dropout)(**kwargs)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Input tensor

        Returns
        -------
        x : (batch, channel, *spatial) tensor
            Output tensor with dropout applied

        """
        if self.training:
            return self.activation(self.dropout(x)) \
                if self.before_activation \
                else self.dropout(self.activation(x))
        else:
            return self.activation(x)

    def __str__(self):
        s = ['']
        if self.dropout.p != 0.5:
            s += [f'p={p}']
        if not self.before_activation:
            s += [f'before_activation=False']
        s = ', '.join(s)
        return f'Dropout({s})'

    __repr__ = __str__
