import torch
from nitorch import core
from .variables import RandomVariable


class Dirac(RandomVariable, torch.Tensor):
    """A Dirac distribution is equivalent to a deterministic parameter."""

    # TODO: fail if the value is a random variable

    def __new__(cls, value=None):
        if value is None:
            value = torch.Tensor()
        return torch.Tensor._make_subclass(cls, value)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format))
            memo[id(self)] = result
            return result

    @property
    def mean(self):
        return torch.Tensor(self)

    @property
    def variance(self):
        return 0

    def moment(self, order):
        return self.mean ** order

    def pdf(self, value):
        return core.constants.inf * (torch.Tensor(self) == value)

    def logpdf(self, value):
        inf = torch.as_tensor(core.constants.inf, device=self.device)
        return torch.where(torch.Tensor(self) == value, inf, -inf)

    def sample(self, shape=None):
        shape = tuple(shape or [])
        if shape:
            samp = torch.Tensor(self)
            samp = samp.reshape((1,) * len(shape) + samp.shape)
            samp = samp.expand(shape + self.shape)
            return samp.clone()
        else:
            torch.Tensor(self).clone()
