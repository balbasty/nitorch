import torch
from .bounds import Bound, ExtrapolateType
from .utils import sub2ind, ind2sub
from typing import List, Optional
Tensor = torch.Tensor


@torch.jit.script
def make_sign(signx: Optional[Tensor],
              signy: Optional[Tensor],
              signz: Optional[Tensor]) -> Optional[Tensor]:
    sign = signx
    if signy is not None:
        if sign is None:
            sign = signy
        else:
            sign = sign * signy
    if signz is not None:
        if sign is None:
            sign = signz
        else:
            sign = sign * signz
    return sign


@torch.jit.script
def pull3d(inp, g, bound: List[Bound], extrapolate: int = 1):
    """
    inp: (B, C, iX, iY, iZ) tensor
    g: (B, oX, oY, oZ, 3) tensor
    bound: List{3}[Bound] tensor
    extrapolate: ExtrapolateType
    returns: (B, C, oX, oY, oZ) tensor
    """
    dim = 3
    boundx, boundy, boundz = bound
    oshape = g.shape[-dim-1:-1]
    g = g.reshape([g.shape[0], 1, -1, dim])
    gx, gy, gz = torch.unbind(g, -1)
    shape = inp.shape[-dim:]
    nx, ny, nz = shape

    # mask of inbounds voxels
    mask: Optional[Tensor] = None
    if extrapolate in (0, 2):  # no / hist
        tiny = 1e-5
        threshold = tiny
        if extrapolate == 2:
            threshold = 0.5 + tiny
        mask = ((gx > -threshold) & (gx < nx - 1 + threshold) &
                (gy > -threshold) & (gy < ny - 1 + threshold) &
                (gz > -threshold) & (gy < nz - 1 + threshold))

    # corners
    gx0 = gx.floor().long()
    gy0 = gy.floor().long()
    gz0 = gz.floor().long()
    gx1 = gx0 + 1
    gy1 = gy0 + 1
    gz1 = gz0 + 1

    # multiplicative transform (if dst-like bound)
    signx1 = boundx.transform(gx1, nx)
    signy1 = boundy.transform(gy1, ny)
    signz1 = boundz.transform(gz1, nz)
    signx0 = boundx.transform(gx0, nx)
    signy0 = boundy.transform(gy0, ny)
    signz0 = boundz.transform(gz0, nz)

    # wrap indices
    gx1 = boundx.index_(gx1, nx)
    gy1 = boundx.index_(gy1, ny)
    gz1 = boundx.index_(gz1, nz)
    gx0 = boundx.index_(gx0, nx)
    gy0 = boundx.index_(gy0, ny)
    gz0 = boundx.index_(gz0, nz)
    gx -= gx0
    gy -= gy0
    gz -= gz0

    # gather
    inp = inp.reshape(inp.shape[:2] + [-1])
    ## corner 000
    idx = sub2ind(torch.stack([gx0, gy0, gz0]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out = inp.gather(-1, idx)
    sign = make_sign(signx0, signy0, signz0)
    if sign is not None:
        out *= sign
    out *= (1 - gx) * (1 - gy) * (1 - gz)
    ## corner 001
    idx = sub2ind(torch.stack([gx0, gy0, gz1]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx0, signy0, signz1)
    if sign is not None:
        out1 *= sign
    out1 *= (1 - gx) * (1 - gy) * gz
    out += out1
    ## corner 010
    idx = sub2ind(torch.stack([gx0, gy1, gz0]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx0, signy1, signz0)
    if sign is not None:
        out1 *= sign
    out1 *= (1 - gx) * gy * (1 - gz)
    out += out1
    ## corner 011
    idx = sub2ind(torch.stack([gx0, gy1, gz1]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx0, signy1, signz1)
    if sign is not None:
        out1 *= sign
    out1 *= (1 - gx) * gy * gz
    out += out1
    ## corner 100
    idx = sub2ind(torch.stack([gx1, gy0, gz0]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx1, signy0, signz0)
    if sign is not None:
        out1 *= sign
    out1 *= gx * (1 - gy) * (1 - gz)
    out += out1
    ## corner 101
    idx = sub2ind(torch.stack([gx1, gy0, gz1]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx1, signy0, signz1)
    if sign is not None:
        out1 *= sign
    out1 *= gx * (1 - gy) * gz
    out += out1
    ## corner 110
    idx = sub2ind(torch.stack([gx1, gy1, gz0]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx1, signy1, signz0)
    if sign is not None:
        out1 *= sign
    out1 *= gx * gy * (1 - gz)
    out += out1
    ## corner 111
    idx = sub2ind(torch.stack([gx1, gy1, gz1]), shape)
    idx = idx.expand([max(idx.shape[0], inp.shape[0]), inp.shape[1], idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = make_sign(signx1, signy1, signz1)
    if sign is not None:
        out1 *= sign
    out1 *= gx * gy * gz
    out += out1

    if mask is not None:
        out *= mask
    out = out.reshape(out.shape[:2] + oshape)
    return out
