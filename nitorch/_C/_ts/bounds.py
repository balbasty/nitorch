import torch
from enum import Enum
from typing import Optional
Tensor = torch.Tensor
from .utils import floor_div_int


class BoundType(Enum):
    zero = zeros = 0
    replicate = nearest = 1
    dct1 = mirror = 2
    dct2 = reflect = 3
    dst1 = antimirror = 4
    dst2 = antireflect = 5
    dft = wrap = 6


class ExtrapolateType(Enum):
    no = 0     # threshold: (0, n-1)
    yes = 1
    hist = 2   # threshold: (-0.5, n-0.5)


@torch.jit.script
class Bound:

    def __init__(self, bound_type: int = 3):
        self.type = bound_type

    def index(self, i, n: int):
        if self.type in (0, 1):  # zero / replicate
            return i.clamp(min=0, max=n-1)
        elif self.type == 6:  # dft
            return i.remainder(n)
        else:
            return self.index_(i.clone(), n)

    def index_(self, i, n: int):
        if self.type in (0, 1):  # zero / replicate
            return i.clamp_(min=0, max=n-1)
        elif self.type in (3, 5):  # dct2 / dst2
            n2 = n * 2
            pre = (i < 0)
            i[pre].neg_().sub_(1)
            i.remainder_(n2)
            i[pre].add_(n2 - 1)
            i[i >= n].neg_().add_(n2 - 1)
            return i
        elif self.type == 2:  # dct1
            if n == 1:
                return i.zero_()
            else:
                n2 = (n - 1) * 2
                i[i < 0].neg_()
                i.remainder_(n2)
                i[i >= n].neg_().add(n2)
                return i
        elif self.type == 4:  # dst1
            n2 = 2 * (n + 1)
            i[i == 0].zero_()
            i[i < 0].neg_().sub_(2)
            i.remainder_(n2)
            i[i == n].fill_(n - 1)
            i[i > n].neg_().add_(n2 - 2)
            return i
        elif self.type == 6:  # dft
            return i.remainder_(n)
        else:
            return i

    def transform(self, i, n: int) -> Optional[Tensor]:
        if self.type == 4:  # dst1
            if n == 1:
                return None
            x = torch.ones(i.shape, dtype=torch.int8, device=i.device)
            n2 = 2 * (n + 1)
            i = i.clone()
            i[i < 0].neg_().add_(n - 1)
            i.remainder_(n2)
            x[i.remainder(n + 1) == n].zero_()
            x[floor_div_int(i, n+1).remainder_(2) > 0].fill_(-1)
            return x
        elif self.type == 5:  # dst2
            x = torch.ones(i.shape, dtype=torch.int8, device=i.device)
            i = torch.where(i < 0, n - 1 - i, i)
            x[floor_div_int(i, n).remainder_(2) > 0].fill_(-1)
            return x
        elif self.type == 0:  # zero
            x = torch.ones(i.shape, dtype=torch.int8, device=i.device)
            outbounds = ((i < 0) | (i >= n))
            x[outbounds].zero_()
            return x
        else:
            return None
