import torch
from enum import Enum


class InterpolationType(Enum):
    nearest = zeroth = 0
    linear = first = 1
    quadratic = second = 2
    cubic = third = 3
    fourth = 4
    fifth = 5
    sixth = 6
    seventh = 7


@torch.jit.script
class Spline:

    def __init__(self, order: InterpolationType = InterpolationType.linear):
        self.order = order

    def weight(self, x):
        if self.order == 0:
            x = x.abs()
            one = torch.ones(x.shape, dtype=x.dtype, device=x.device)
            zero = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            return torch.where(x < 0.5, one, zero)

    def weight_(self, x):
        if self.order == 0:
            x.abs_()
            x[x < 0.5].zeros_()
            x[x > 0.5].fill_(1)
            return x.neg_().add_(1)

    def fastweight(self, x):
        if self.order == 0:
            return torch.ones(x.shape, dtype=x.dtype, device=x.device)

    def grad(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)

    def fastgrad(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)

    def hess(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)

    def fasthess(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)

    def bounds(self, x):
        if self.order == 0:
            upp = low = x.round()
            return upp, low