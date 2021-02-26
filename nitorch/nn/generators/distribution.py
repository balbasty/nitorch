from ..modules.base import Module
import torch


class RandomDistribution(Module, torch.distributions.Distribution):

    def __init__(self, distribution, *args, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.args = args
        self.kwargs = kwargs

    def forward(self):
        args = list(self.args)
        kwargs = dict(self.kwargs)

        for i, value in args:
            if callable(value):
                args[i] = value()

        for i, (key, value) in kwargs.items():
            if callable(value):
                kwargs[key] = value()

        dist = self.distribution(*args, **kwargs)
        return dist
