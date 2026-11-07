import torch
from enum import Enum
from typing import Optional
from .utils import floor_div_int
Tensor = torch.Tensor


class BoundType(Enum):
    zero = zeros = 0
    replicate = nearest = 1
    dct1 = mirror = 2
    dct2 = reflect = 3
    dst1 = antimirror = 4
    dst2 = antireflect = 5
    dft = wrap = 6
    # Displacement fields: along axis d, component d is dst2 and the other
    # components dct2. Pull/push split the components (`component_bounds`
    # in nitorch._C.grid); on its own, it is the dst2 of the normal
    # component.
    sliding = 7


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
        elif self.type in (3, 5, 7):  # dct2 / dst2 / sliding
            n2 = n * 2
            i = torch.where(i < 0, (-i-1).remainder(n2).neg().add(n2 - 1),
                            i.remainder(n2))
            i = torch.where(i >= n, -i + (n2 - 1), i)
            return i
        elif self.type == 2:  # dct1
            if n == 1:
                return torch.zeros(i.shape, dtype=i.dtype, device=i.device)
            else:
                n2 = (n - 1) * 2
                i = i.abs().remainder(n2)
                i = torch.where(i >= n, -i + n2, i)
                return i
        elif self.type == 4:  # dst1
            n2 = 2 * (n + 1)
            first = torch.zeros([1], dtype=i.dtype, device=i.device)
            last = torch.full([1], n - 1, dtype=i.dtype, device=i.device)
            i = torch.where(i < 0, -i - 2, i)
            i = i.remainder(n2)
            i = torch.where(i > n, -i + (n2 - 2), i)
            i = torch.where(i == -1, first, i)
            i = torch.where(i == n, last, i)
            return i
        elif self.type == 6:  # dft
            return i.remainder(n)
        else:
            return i

    def transform(self, i, n: int) -> Optional[Tensor]:
        if self.type == 4:  # dst1
            # (no special case for n == 1: the extension is then
            # `0, x0, 0, -x0` with period 4, not a replicated border)
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            # (as `periodic1` in `src/impl/bounds_common.h`)
            n2 = 2 * (n + 1)
            i = torch.where(i < 0, -i + (n-1), i)
            i = i.remainder(n2)
            # the zeros (ghost samples -1 and n) are those with
            # i % (n+1) == n (the first voxel, i == 0, is not one of them)
            x = torch.where(i.remainder(n + 1) == n, zero, one)
            # floor division (`i / (n+1)` is a true division, which
            # flipped the sign of in-bounds samples)
            x = torch.where(floor_div_int(i, n + 1).remainder(2) > 0, -x, x)
            return x
        elif self.type in (5, 7):  # dst2 / sliding (normal component)
            i = torch.where(i < 0, n - 1 - i, i)
            x = torch.ones([1], dtype=torch.int8, device=i.device)
            # floor division (`i / n` is a true division, which flipped
            # the sign of in-bounds samples)
            x = torch.where(floor_div_int(i, n).remainder(2) > 0, -x, x)
            return x
        elif self.type == 0:  # zero
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            outbounds = ((i < 0) | (i >= n))
            x = torch.where(outbounds, zero, one)
            return x
        else:
            return None
