import torch.nn as tnn
from nitorch.core import py, utils
from ..base import Module


def _defer_property(prop, module, setter=False):
    """Return a 'property' objet that links to a submodule property

    prop (str) : property name
    module (str): module name
    setter (bool, default=False) : define a setter
    returns (property) : property object

    """
    if setter:
        return property(lambda self: getattr(getattr(self, module), prop),
                        lambda self, val: setattr(getattr(self, module), prop, val))
    else:
        return property(lambda self: getattr(getattr(self, module), prop))


class Linear(Module):
    """Add options to torch.nn.Linear

    The fully connected layer is applied to the channels dimension
    by default (instead of between the last spatial dimension).
    """

    def __init__(self, in_channels, out_channels, bias=True, dim=1):
        super().__init__()
        self.linear = tnn.Linear(in_channels, out_channels, bias)
        self.dim = dim

    def forward(self, x, **overload):
        dim = overload.get('dim', self.dim)
        x = utils.movedim(x, dim, -1)
        x = self.linear(x)
        x = utils.movedim(x, -1, dim)
        return x

    in_channels = _defer_property('in_features', 'linear')
    out_channels = _defer_property('out_features', 'linear')
    bias = _defer_property('bias', 'linear')
