from nitorch.core.math import softmax, log_softmax
from nitorch.core import py
from torch import nn


class SoftMax(nn.Module):

    def __init__(self, dim=1, implicit=False):
        """

        Parameters
        ----------
        dim : int, default=1
            Dimension along which to take the softmax
        implicit : bool or (bool, bool), default=False
            The first value relates to the input tensor and the second
            relates to the output tensor.
            - implicit[0] == True assumes that an additional (hidden) channel
              with value zero exists.
            - implicit[1] == True drops the last class from the
              softmaxed tensor.
        """
        super().__init__()
        self.implicit = py.make_list(implicit, 2)
        self.dim = dim

    def forward(self, input):
        """

        Parameters
        ----------
        input : (batch, classes [-1], *spatial) tensor

        Returns
        -------
        input : (batch, classes [-1], *spatial) tensor
            Softmaxed tensor

        """
        return softmax(input, dim=self.dim, implicit=self.implicit,
                       implicit_index=0)


class LogSoftMax(nn.Module):

    def __init__(self, dim=1, implicit=False):
        """

        Parameters
        ----------
        dim : int, default=1
            Dimension along which to take the softmax
        implicit : bool or (bool, bool), default=False
            The first value relates to the input tensor and the second
            relates to the output tensor.
            - implicit[0] == True assumes that an additional (hidden) channel
              with value zero exists.
            - implicit[1] == True drops the last class from the
              softmaxed tensor.
        """
        super().__init__()
        self.implicit = py.make_list(implicit, 2)
        self.dim = dim

    def forward(self, input):
        """

        Parameters
        ----------
        input : (batch, classes [-1], *spatial) tensor

        Returns
        -------
        input : (batch, classes [-1], *spatial) tensor
            Softmaxed tensor

        """
        return log_softmax(input, dim=self.dim, implicit=self.implicit,
                           implicit_index=0)
