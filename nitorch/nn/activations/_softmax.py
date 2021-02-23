from nitorch.core.math import softmax
from nitorch.nn.modules import Module
from nitorch.core import py


class SoftMax(Module):

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

    def forward(self, input, **overload):
        """

        Parameters
        ----------
        input : (batch, classes [-1], *spatial) tensor
        overload : dict
            `dim` and `implicit` can be overloaded at call time.

        Returns
        -------
        input : (batch, classes [-1], *spatial) tensor
            Softmaxed tensor

        """
        dim = overload.get('dim', self.dim)
        implicit = overload.get('implicit', self.implicit)
        implicit = py.make_list(implicit, 2)

        return softmax(input, dim=dim, implicit=implicit)
