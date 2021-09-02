import torch
import math
from typing import List, Optional
from .utils import movedim1
from .pushpull import pad_list_int


@torch.jit.script
def get_poles2(order: int) -> List[float]:
    empty: List[float] = []
    if order in (0, 1):
        return empty
    if order == 2:
        return [math.sqrt(8.) - 3.]
    if order == 3:
        return [math.sqrt(3.) - 2.]
    if order == 4:
        return [math.sqrt(664. - math.sqrt(438976.)) + math.sqrt(304.) - 19.,
                math.sqrt(664. + math.sqrt(438976.)) - math.sqrt(304.) - 19.]
    if order == 5:
        return [math.sqrt(67.5 - math.sqrt(4436.25)) + math.sqrt(26.25) - 6.5,
                math.sqrt(67.5 + math.sqrt(4436.25)) - math.sqrt(26.25) - 6.5]
    if order == 6:
        return [-0.488294589303044755130118038883789062112279161239377608394,
                -0.081679271076237512597937765737059080653379610398148178525368,
                -0.00141415180832581775108724397655859252786416905534669851652709]
    if order == 7:
        return [-0.5352804307964381655424037816816460718339231523426924148812,
                -0.122554615192326690515272264359357343605486549427295558490763,
                -0.0091486948096082769285930216516478534156925639545994482648003]
    raise NotImplementedError


@torch.jit.script
def get_gain(poles: List[float]) -> float:
    lam: float = 1.
    for pole in poles:
        lam *= (1. - pole) * (1. - 1./pole)
    return lam


@torch.jit.script
def dft_initial(inp, pole: float, dim: int = -1, keepdim: bool = False):

    max_iter: int = int(math.ceil(-30./math.log(abs(pole))))
    max_iter = min(max_iter, inp.shape[dim])

    poles = torch.as_tensor(pole, dtype=inp.dtype, device=inp.device)
    poles = poles.pow(torch.arange(1, max_iter+1, dtype=inp.dtype, device=inp.device))
    poles = poles.flip(0)

    inp = movedim1(inp, dim, 0)
    inp0 = inp[0]
    inp = inp[-max_iter:]
    inp = movedim1(inp, 0, -1)
    out = torch.matmul(inp.unsqueeze(-2), poles.unsqueeze(-1)).squeeze(-1)
    out = out + inp0.unsqueeze(-1)
    if keepdim:
        out = movedim1(out, -1, dim)
    else:
        out = out.squeeze(-1)

    pole = pole ** max_iter
    out = out / (1 - pole)
    return out


@torch.jit.script
def dct2_initial(inp, pole: float, dim: int = -1, keepdim: bool = False):

    n = inp.shape[dim]
    max_iter: int = int(math.ceil(-30./math.log(abs(pole))))

    if max_iter < n:

        poles = torch.as_tensor(pole, dtype=inp.dtype, device=inp.device)
        poles = poles.pow(torch.arange(1, max_iter, dtype=inp.dtype, device=inp.device))

        inp = movedim1(inp, dim, 0)
        inp0 = inp[0]
        inp = inp[1:max_iter]
        inp = movedim1(inp, 0, -1)
        out = torch.matmul(inp.unsqueeze(-2), poles.unsqueeze(-1)).squeeze(-1)
        out = out + inp0.unsqueeze(-1)
        if keepdim:
            out = movedim1(out, -1, dim)
        else:
            out = out.squeeze(-1)

    else:

        polen = pole ** (n - 1)
        inp0 = inp[0] + polen * inp[-1]
        inp = inp[1:-1]
        inp = movedim1(inp, 0, -1)

        poles = torch.as_tensor(pole, dtype=inp.dtype, device=inp.device)
        poles = poles.pow(torch.arange(1, n-1, dtype=inp.dtype, device=inp.device))
        poles = poles + (polen * polen) / poles

        out = torch.matmul(inp.unsqueeze(-2), poles.unsqueeze(-1)).squeeze(-1)
        out = out + inp0.unsqueeze(-1)
        if keepdim:
            out = movedim1(out, -1, dim)
        else:
            out = out.squeeze(-1)

        pole = pole ** max_iter
        out = out / (1 - pole * pole)

    return out


@torch.jit.script
def dft_final(inp, pole: float, dim: int = -1, keepdim: bool = False):

    max_iter: int = int(math.ceil(-30./math.log(abs(pole))))
    max_iter = min(max_iter, inp.shape[dim])

    poles = torch.as_tensor(pole, dtype=inp.dtype, device=inp.device)
    poles = poles.pow(torch.arange(1, max_iter+1, dtype=inp.dtype, device=inp.device))

    inp = movedim1(inp, dim, 0)
    inp0 = inp[-1]
    inp = inp[:max_iter]
    inp = movedim1(inp, 0, -1)
    out = torch.matmul(inp.unsqueeze(-2), poles.unsqueeze(-1)).squeeze(-1)
    out = out + inp0.unsqueeze(-1)
    if keepdim:
        out = movedim1(out, -1, dim)
    else:
        out = out.squeeze(-1)

    pole = pole ** max_iter
    out = out / (pole - 1)
    return out


@torch.jit.script
def dct2_final(inp, pole: float, dim: int = -1, keepdim: bool = False):
    inp = movedim1(inp, dim, 0)
    out = pole * inp[-2] + inp[-1]
    out = out * (pole / (pole*pole - 1))
    if keepdim:
        out = movedim1(out.unsqueeze(0), 0, dim)
    return out


@torch.jit.script
def dft_filter(inp, poles: List[float], dim: int = -1, inplace: bool = False):

    if not inplace:
        inp = inp.clone()

    if inp.shape[dim] == 1:
        return inp

    gain = get_gain(poles)
    inp *= gain
    inp = movedim1(inp, dim, 0)
    n = inp.shape[0]

    for pole in poles:
        inp[0] = dft_initial(inp, pole, dim=0)

        for i in range(1, n-1):
            inp[i].add_(inp[i-1], alpha=pole)

        inp[-1] = dft_final(inp, pole, dim=0)

        for i in range(n-2, -1, -1):
            inp[i].neg_().add_(inp[i+1]).mul_(pole)

    inp = movedim1(inp, 0, dim)
    return inp


@torch.jit.script
def dct2_filter(inp, poles: List[float], dim: int = -1, inplace: bool = False):

    if not inplace:
        inp = inp.clone()

    if inp.shape[dim] == 1:
        return inp

    gain = get_gain(poles)
    inp *= gain
    inp = movedim1(inp, dim, 0)
    n = inp.shape[0]

    for pole in poles:
        inp[0] = dct2_initial(inp, pole, dim=0)

        for i in range(1, n-1):
            inp[i].add_(inp[i-1], alpha=pole)

        inp[-1] = dct2_final(inp, pole, dim=0)

        for i in range(n-2, -1, -1):
            inp[i].neg_().add_(inp[i+1]).mul_(pole)

    inp = movedim1(inp, 0, dim)
    return inp


@torch.jit.script
def spline_coeff(inp, bound: List[int], order: List[int], dim: Optional[int],
                 inplace: bool = False):
    if not inplace:
        inp = inp.clone()

    if dim is None:
        dim = inp.dim()

    bound = pad_list_int(bound, dim)
    order = pad_list_int(order, dim)

    for d, b, o in zip(range(dim), bound, order):

        poles = get_poles2(o)
        if b == 6:    # dft
            inp = dft_filter(inp, poles, dim=-dim+d, inplace=True)
        elif b == 3:  # dct2 (but is it really dct2, or dct1 ?)
            inp = dct2_filter(inp, poles, dim=-dim+d, inplace=True)
        else:
            raise NotImplementedError

    return inp