"""
This file implements differentiable histograms and joint histograms.
It also exposes functions that enable the explicit computation of
first and (approximate) second derivatives, for use in optimization-based
algorithms.
"""
from nitorch._C._ts import iso0, iso1, nd, utils as ts_utils
from nitorch._C._ts.pushpull import make_bound, make_spline
from nitorch._C._ts.bounds import Bound
from nitorch._C.grid import bound_to_nitorch, inter_to_nitorch
from nitorch.spatial import smooth, grid_grad, grid_pull, grid_push, grid_count
from nitorch.core import kernels
import torch
from typing import List
from . import py, dtypes


# ======================================================================
#                                UTILS
# ======================================================================
# I copied that from nitorch.core.utils because the main histc function
# needs it.


def movedim(input, source, destination):
    """Moves the position of one or more dimensions

    Other dimensions that are not explicitly moved remain in their
    original order and appear at the positions not specified in
    destination.

    Parameters
    ----------
    input : tensor
        Input tensor
    source : int or sequence[int]
        Initial positions of the dimensions
    destination : int or sequence[int]
        Output positions of the dimensions.

        If a single destination is provided:
        - if it is negative, the last source dimension is moved to
          `destination` and all other source dimensions are moved to its left.
        - if it is positive, the first source dimension is moved to
          `destination` and all other source dimensions are moved to its right.

    Returns
    -------
    output : tensor
        Tensor with moved dimensions.

    """
    input = torch.as_tensor(input)
    dim = input.dim()
    source = py.make_list(source)
    destination = py.make_list(destination)
    if len(destination) == 1:
        # we assume that the user wishes to keep moved dimensions
        # in the order they were provided
        destination = destination[0]
        if destination >= 0:
            destination = list(range(destination, destination+len(source)))
        else:
            destination = list(range(destination+1-len(source), destination+1))
    if len(source) != len(destination):
        raise ValueError('Expected as many source as destination positions.')
    source = [dim + src if src < 0 else src for src in source]
    destination = [dim + dst if dst < 0 else dst for dst in destination]
    if len(set(source)) != len(source):
        raise ValueError(f'Expected source positions to be unique but got '
                         f'{source}')
    if len(set(destination)) != len(destination):
        raise ValueError(f'Expected destination positions to be unique but got '
                         f'{destination}')

    # compute permutation
    positions_in = list(range(dim))
    positions_out = [None] * dim
    for src, dst in zip(source, destination):
        positions_out[dst] = src
        positions_in[src] = None
    positions_in = filter(lambda x: x is not None, positions_in)
    for i, pos in enumerate(positions_out):
        if pos is None:
            positions_out[i], *positions_in = positions_in

    return input.permute(*positions_out)


# ======================================================================
#               EXTENSIONS TO TORCHSCRIPT-PUSHPULL
# ======================================================================
# Computing the Hessian of a histogram-based function requires an
# additional utility that computes G'HG, where G is equivalent to the
# "grad" function and H is the Hessian of the loss with respect to the
# histogram. We call this utility "grad2".


@torch.jit.script
def iso1_grad2_1d(inp, g, bound: List[Bound], extrapolate: int = 1, abs: bool = False):
    """
    inp: (B, C, iX, iX) tensor
    g: (B, oX, 1) tensor
    bound: List{1}[Bound] tensor
    extrapolate: ExtrapolateType
    returns: (B, C, oX, 1) tensor
    """
    dim = 1
    boundx = bound[0]
    oshape = list(g.shape[-dim-1:-1])
    g = g.reshape([g.shape[0], 1, -1, dim])
    gx = g.squeeze(-1)
    batch = max(inp.shape[0], gx.shape[0])
    channel = inp.shape[1]
    shape = list(inp.shape[-dim:])
    nx = shape[0]

    # mask of inbounds voxels
    mask = ts_utils.inbounds_mask_1d(extrapolate, gx, nx)

    # corners
    # (upper weight, lower corner, upper corner, lower sign, upper sign)
    gx, gx0, gx1, signx0, signx1 = iso1.get_weights_and_indices(gx, nx, boundx)

    # gather
    inp = inp.reshape(list(inp.shape[:2]) + [-1])
    out = torch.empty([batch, channel] + list(g.shape[-2:]),
                      dtype=inp.dtype, device=inp.device)
    outx = out.squeeze(-1)
    # - corner 00
    idx = ts_utils.sub2ind_list([gx0, gx0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    torch.gather(inp, -1, idx, out=outx)
    # - corner 01
    idx = ts_utils.sub2ind_list([gx0, gx1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signx1])
    if sign is not None:
        out1 *= sign
    outx.add_(out1, alpha=1 if abs else -1)
    # - corner 10
    idx = ts_utils.sub2ind_list([gx1, gx0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signx1])
    if sign is not None:
        out1 *= sign
    outx.add_(out1, alpha=1 if abs else -1)
    # - corner 11
    idx = ts_utils.sub2ind_list([gx1, gx1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.add_(out1)

    if mask is not None:
        out *= mask.unsqueeze(-1)
    out = out.reshape(list(out.shape[:2]) + oshape + [dim])
    return out


@torch.jit.script
def iso1_grad2_diag_1d(inp, g, bound: List[Bound], extrapolate: int = 1, abs: bool = False):
    """
    inp: (B, C, iX) tensor
    g: (B, oX, 1) tensor
    bound: List{1}[Bound] tensor
    extrapolate: ExtrapolateType
    returns: (B, C, oX, 1) tensor
    """
    dim = 1
    boundx = bound[0]
    oshape = list(g.shape[-dim-1:-1])
    g = g.reshape([g.shape[0], 1, -1, dim])
    gx = g.squeeze(-1)
    batch = max(inp.shape[0], gx.shape[0])
    channel = inp.shape[1]
    shape = list(inp.shape[-dim:])
    nx = shape[0]

    # mask of inbounds voxels
    mask = ts_utils.inbounds_mask_1d(extrapolate, gx, nx)

    # corners
    # (upper weight, lower corner, upper corner, lower sign, upper sign)
    gx, gx0, gx1, signx0, signx1 = iso1.get_weights_and_indices(gx, nx, boundx)

    # gather
    inp = inp.reshape(list(inp.shape[:2]) + [-1])
    out = torch.empty([batch, channel] + list(g.shape[-2:]),
                      dtype=inp.dtype, device=inp.device)
    outx = out.squeeze(-1)
    # - corner 00
    idx = gx0
    idx = idx.expand([batch, channel, idx.shape[-1]])
    torch.gather(inp, -1, idx, out=outx)
    # - corner 11
    idx = gx1
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.add_(out1)

    if mask is not None:
        out *= mask.unsqueeze(-1)
    out = out.reshape(list(out.shape[:2]) + oshape + [dim])
    return out


@torch.jit.script
def iso1_grad2_2d(inp, g, bound: List[Bound], extrapolate: int = 1,
                  abs: bool = False):
    """
    inp: (B, C, iX, iY, iX, iY) tensor
    g: (B, oX, oY, 2) tensor
    bound: List{2}[Bound] tensor
    extrapolate: ExtrapolateType
    returns: (B, C, oX, oY, 2) tensor
    """
    dim = 2
    boundx, boundy = bound
    oshape = list(g.shape[-dim-1:-1])
    g = g.reshape([g.shape[0], 1, -1, dim])
    gx, gy = torch.unbind(g, -1)
    batch = max(inp.shape[0], gx.shape[0])
    channel = inp.shape[1]
    shape = list(inp.shape[-dim:])
    nx, ny = shape

    # mask of inbounds voxels
    mask = ts_utils.inbounds_mask_2d(extrapolate, gx, gy, nx, ny)

    # corners
    # (upper weight, lower corner, upper corner, lower sign, upper sign)
    gx, gx0, gx1, signx0, signx1 = iso1.get_weights_and_indices(gx, nx, boundx)
    gy, gy0, gy1, signy0, signy1 = iso1.get_weights_and_indices(gy, ny, boundy)

    # gather
    inp = inp.reshape(list(inp.shape[:2]) + [-1])
    out = torch.empty([batch, channel] + list(g.shape[-2:]),
                      dtype=inp.dtype, device=inp.device)
    outx, outy = out.unbind(-1)
    # - corner 00-00
    idx = ts_utils.sub2ind_list([gx0, gy0, gx0, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    torch.gather(inp, -1, idx, out=outx)
    outy.copy_(outx)
    outx *= (1 - gy) ** 2
    outy *= (1 - gx) ** 2
    # - corner 00-01
    idx = ts_utils.sub2ind_list([gx0, gy0, gx0, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signy0, signx0, signy1])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, (1 - gy) * gy)
    outy.addcmul_(out1, (1 - gx) ** 2 * (-1 if not abs else 1))
    # - corner 00-10
    idx = ts_utils.sub2ind_list([gx0, gy0, gx1, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signy0, signx1, signy0])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, (1 - gy) ** 2 * (-1 if not abs else 1))
    outy.addcmul_(out1, (1 - gx) * gx)
    # - corner 00-11
    idx = ts_utils.sub2ind_list([gx0, gy0, gx1, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signy0, signx1, signy1])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, (1 - gy) * gy * (-1 if not abs else 1))
    outy.addcmul_(out1, (1 - gx) * gx * (-1 if not abs else 1))
    # - corner 01-00
    idx = ts_utils.sub2ind_list([gx0, gy1, gx0, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signy1, signx0, signy0])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy * (1 - gy))
    outy.addcmul_(out1, (1 - gx) ** 2 * (-1 if not abs else 1))
    # - corner 01-01
    idx = ts_utils.sub2ind_list([gx0, gy1, gx0, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.addcmul_(out1, gy ** 2)
    outy.addcmul_(out1, (1 - gx) ** 2)
    # - corner 01-10
    idx = ts_utils.sub2ind_list([gx0, gy1, gx1, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signy1, signx1, signy0])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy * (1 - gy) * (-1 if not abs else 1))
    outy.addcmul_(out1, gx * (1 - gx) * (-1 if not abs else 1))
    # - corner 01-11
    idx = ts_utils.sub2ind_list([gx0, gy1, gx1, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx0, signy1, signx1, signy1])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy ** 2 * (-1 if not abs else 1))
    outy.addcmul_(out1, gx * (1 - gx))
    # - corner 10-00
    idx = ts_utils.sub2ind_list([gx1, gy0, gx0, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx1, signy0, signx0, signy0])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, (1 - gy) ** 2 * (-1 if not abs else 1))
    outy.addcmul_(out1, gx * (1 - gx))
    # - corner 10-01
    idx = ts_utils.sub2ind_list([gx1, gy0, gx0, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx1, signy0, signx0, signy1])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy * (1 - gy) * (-1 if not abs else 1))
    outy.addcmul_(out1, gx * (1 - gx) * (-1 if not abs else 1))
    # - corner 10-10
    idx = ts_utils.sub2ind_list([gx1, gy0, gx1, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.addcmul_(out1, (1 - gy) ** 2)
    outy.addcmul_(out1, gx ** 2)
    # - corner 10-11
    idx = ts_utils.sub2ind_list([gx1, gy0, gx1, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx1, signy0, signx1, signy1])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy * (1 - gy))
    outy.addcmul_(out1, gx ** 2 * (-1 if not abs else 1))
    # - corner 11-00
    idx = ts_utils.sub2ind_list([gx1, gy1, gx0, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx1, signy1, signx0, signy0])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy * (1 - gy) * (-1 if not abs else 1))
    outy.addcmul_(out1, gx * (1 - gx) * (-1 if not abs else 1))
    # - corner 11-01
    idx = ts_utils.sub2ind_list([gx1, gy1, gx0, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx1, signy1, signx0, signy1])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy ** 2 * (-1 if not abs else 1))
    outy.addcmul_(out1, gx * (1 - gx))
    # - corner 11-10
    idx = ts_utils.sub2ind_list([gx1, gy1, gx1, gy0], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    sign = ts_utils.make_sign([signx1, signy1, signx1, signy0])
    if sign is not None:
        out1 *= sign
    outx.addcmul_(out1, gy * (1 - gy))
    outy.addcmul_(out1, gx ** 2 * (-1 if not abs else 1))
    # - corner 11-11
    idx = ts_utils.sub2ind_list([gx1, gy1, gx1, gy1], shape*2)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.addcmul_(out1, gy ** 2)
    outy.addcmul_(out1, gx ** 2)

    if mask is not None:
        out *= mask.unsqueeze(-1)
    out = out.reshape(list(out.shape[:2]) + oshape + [dim])
    return out


@torch.jit.script
def iso1_grad2_diag_2d(inp, g, bound: List[Bound], extrapolate: int = 1,
                       abs: bool = False):
    """
    inp: (B, C, iX, iY) tensor
    g: (B, oX, oY, 2) tensor
    bound: List{2}[Bound] tensor
    extrapolate: ExtrapolateType
    returns: (B, C, oX, oY, 2) tensor
    """
    dim = 2
    boundx, boundy = bound
    oshape = list(g.shape[-dim-1:-1])
    g = g.reshape([g.shape[0], 1, -1, dim])
    gx, gy = torch.unbind(g, -1)
    batch = max(inp.shape[0], gx.shape[0])
    channel = inp.shape[1]
    shape = list(inp.shape[-dim:])
    nx, ny = shape

    # mask of inbounds voxels
    mask = ts_utils.inbounds_mask_2d(extrapolate, gx, gy, nx, ny)

    # corners
    # (upper weight, lower corner, upper corner, lower sign, upper sign)
    gx, gx0, gx1, signx0, signx1 = iso1.get_weights_and_indices(gx, nx, boundx)
    gy, gy0, gy1, signy0, signy1 = iso1.get_weights_and_indices(gy, ny, boundy)

    # gather
    inp = inp.reshape(list(inp.shape[:2]) + [-1])
    out = torch.empty([batch, channel] + list(g.shape[-2:]),
                      dtype=inp.dtype, device=inp.device)
    outx, outy = out.unbind(-1)
    # - corner 00-00
    idx = ts_utils.sub2ind_list([gx0, gy0], shape)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    torch.gather(inp, -1, idx, out=outx)
    outy.copy_(outx)
    outx *= (1 - gy) ** 2
    outy *= (1 - gx) ** 2
    # - corner 01-01
    idx = ts_utils.sub2ind_list([gx0, gy1], shape)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.addcmul_(out1, gy ** 2)
    outy.addcmul_(out1, (1 - gx) ** 2)
    # - corner 10-10
    idx = ts_utils.sub2ind_list([gx1, gy0], shape)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.addcmul_(out1, (1 - gy) ** 2)
    outy.addcmul_(out1, gx ** 2)
    # - corner 11-11
    idx = ts_utils.sub2ind_list([gx1, gy1], shape)
    idx = idx.expand([batch, channel, idx.shape[-1]])
    out1 = inp.gather(-1, idx)
    outx.addcmul_(out1, gy ** 2)
    outy.addcmul_(out1, gx ** 2)

    if mask is not None:
        out *= mask.unsqueeze(-1)
    out = out.reshape(list(out.shape[:2]) + oshape + [dim])
    return out


# ======================================================================
#                          HISTOGRAM UTILITIES
# ======================================================================
# These functions compute forward and backward passes of (1d) histogram
# computation. Inputs are assumed to be already converted to "bin indices"
# (that is, the function
#    ```x.mul_(bins / (max - min)).add_(bins / (1 - max / min)).sub_(0.5)```
#  has already been applied)


def _histc_forward(x, bins, w=None, order=0, bound='replicate', extrapolate=True):
    """Build histogram.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    x : (b, n) tensor
    bins : int
    w : ([b], n) tensor, optional
    order : int, default=0
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True

    Returns
    -------
    h : (b, bins) tensor

    """
    order = inter_to_nitorch(order, 'int')
    bound = make_bound([bound_to_nitorch(bound, 'int')])
    extrapolate = 1 if extrapolate else 2

    if w is None:
        w = x.new_ones([1]).expand(x.shape)
    while w.dim() < 2:
        w = w.unsqueeze(0)

    args = (w.unsqueeze(-2), x.unsqueeze(-1), [bins], bound, extrapolate)
    if order == 0:
        h = iso0.push1d(*args)
    elif order == 1:
        h = iso1.push1d(*args)
    else:
        order = make_spline([order])
        args = (*args[:-1], order, args[-1])
        h = nd.push(*args)
    h = h.squeeze(1)

    return h


def _histc_backward(g, x, w=None, order=0, bound='replicate',
                    extrapolate=True, gradx=True, gradw=False):
    """Compute derivative of the histogram.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    g : (b, bins) tensor
    x : (b, n) tensor
    w : ([b], n) tensor, optional
    order : int, default=0
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True
    gradx : bool, default=True
    gradw : bool, default=False

    Returns
    -------
    gx : (b, n) tensor, if gradx
    gw : ([b], n) tensor, if gradw

    """
    order = inter_to_nitorch(order, 'int')
    bound = make_bound([bound_to_nitorch(bound, 'int')])
    extrapolate = 1 if extrapolate else 2

    out = []

    if gradx:
        args = (g.unsqueeze(1), x.unsqueeze(-1), bound, extrapolate)
        if order == 0:
            gx = iso0.grad(*args)
        elif order == 1:
            gx = iso1.grad1d(*args)
        else:
            args = (*args[:-1], make_spline([order]), args[-1])
            gx = nd.grad(*args)
        gx = gx.squeeze(1).squeeze(-1)
        if w is not None:
            gx *= w
        out.append(gx)

    if gradw:
        args = (g.unsqueeze(1), x.unsqueeze(-1), bound, extrapolate)
        if order == 0:
            gw = iso0.pull1d(*args)
        elif order == 1:
            gw = iso1.pull1d(*args)
        else:
            args = (*args[:-1], make_spline([order]), args[-1])
            gw = nd.pull(*args)
        gw = gw.squeeze(1)

        wshape = w.shape
        while w.dim() < 2:
            w = w.unsqueeze(0)
        if w.numel() == 1:
            gw = gw.sum(keepdim=True)
        elif w.shape[0] == 1:
            gw = gw.sum(dim=0, keepdim=True)
        elif w.shape[1] == 1:
            gw = gw.sum(dim=1, keepdim=True)
        if len(wshape) < 2:
            gw = gw[0]
        if len(wshape) < 1:
            gw = gw[0]

        out.append(gw)

    return out[0] if len(out) == 1 else tuple(out)


_SPLINE_SQ_CACHE = {}


def _spline_sq_integrals(order):
    """Return (w, d) = (integral of B_k^2, integral of (B_k')^2).

    Computed once (exactly, by piecewise Gauss-Legendre quadrature) using
    nitorch's own spline weight/grad functions, then cached.
    w = mean_t sum_i B_k(t-i)^2 and d = mean_t sum_i B_k'(t-i)^2.
    """
    order = int(order)
    if order not in _SPLINE_SQ_CACHE:
        import numpy as np
        from nitorch._C._ts.splines import Spline
        nodes, weights = np.polynomial.legendre.leggauss(16)
        nodes = torch.as_tensor(nodes, dtype=torch.double)
        weights = torch.as_tensor(weights, dtype=torch.double)
        # B_k is piecewise polynomial with knots on the half-integer grid
        half = (order + 1) / 2
        edges = torch.arange(-half, half, 0.5, dtype=torch.double)
        t = (edges[:, None] + 0.25 * (nodes[None, :] + 1)).reshape(-1)
        q = (weights.repeat(len(edges)) * 0.25)
        spline = Spline(order)
        w = (spline.weight(t).square() * q).sum().item()
        d = (spline.grad(t).square() * q).sum().item() if order > 0 else 0.
        _SPLINE_SQ_CACHE[order] = (w, d)
    return _SPLINE_SQ_CACHE[order]


def _hess_order_scale(order):
    """Factors that rescale order-1 Gauss-Newton Hessians to order `order`.

    For a single dimension, returns d_k / d_1.
    For a list [kx, ky], returns the factors for the x and y components:
    (d_kx*w_ky)/(d_1*w_1) and (w_kx*d_ky)/(w_1*d_1).
    """
    w1, d1 = _spline_sq_integrals(1)
    if isinstance(order, (list, tuple)):
        (wx, dx), (wy, dy) = [_spline_sq_integrals(o) for o in order]
        return [(dx * wy) / (d1 * w1), (wx * dy) / (w1 * d1)]
    return _spline_sq_integrals(order)[1] / d1


_ORDER0_MSG = ('Gauss-Newton Hessian of a histogram with spline order 0 is '
               'not defined (the derivative of B0 is zero almost '
               'everywhere). Use order >= 1.')


def _histc_backward2(h, x, w=None, order=0, bound='replicate',
                     extrapolate=True):
    """Compute (Fisher's) second derivative of the histogram.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    h : (b, bins, [bins]) tensor
    x : (b, n) tensor
    w : ([b], n) tensor, optional
    order : int, default=0
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True

    Returns
    -------
    hx : (b, n) tensor

    Notes
    -----
    Only an order-1 Hessian kernel is implemented. For order != 1, the
    order-1 result is rescaled (per component) so that its expectation
    over the sub-bin position matches the expected Gauss-Newton curvature
    of the order-k histogram. This is an approximate, position-averaged
    Hessian. Order 0 is not supported (zero derivative a.e.).

    """
    order = inter_to_nitorch(order, 'int')
    bound = make_bound([bound_to_nitorch(bound, 'int')])
    extrapolate = 1 if extrapolate else 2

    is_diag = h.dim() == 2

    args = (h.unsqueeze(1), x.unsqueeze(-1), bound, extrapolate)
    if order == 0:
        raise NotImplementedError(_ORDER0_MSG)
    if is_diag:
        gx = iso1_grad2_diag_1d(*args)
    else:
        gx = iso1_grad2_1d(*args)
    if order != 1:
        gx = gx * _hess_order_scale(order)
    gx = gx.squeeze(1).squeeze(-1)
    if w is not None:
        gx *= w if is_diag else w.unsqueeze(-1)

    return gx


# ======================================================================
#                        JOINT HISTOGRAM UTILITIES
# ======================================================================
# These functions compute forward and backward passes of (2d) histogram
# computation. Inputs are assumed to be already converted to "bin indices"
# (that is, the function
#    ```x.mul_(bins / (max - min)).add_(bins / (1 - max / min)).sub_(0.5)```
#  has already been applied)


def _jhistc_forward(x, bins, w=None, order=0, bound='replicate', extrapolate=True):
    """Build joint histogram.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    x : (b, n, 2) tensor
    bins : int or (int, int)
    w : ([b], n) tensor, optional
    order : int or (int, int), default=0
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True

    Returns
    -------
    h : (b, bins, bins) tensor

    """
    bins = py.make_list(bins, 2)
    extrapolate = 1 if extrapolate else 2
    opt = dict(shape=bins, interpolation=order,
               bound=bound, extrapolate=extrapolate)
    x = x.unsqueeze(-3)                     # make 2d spatial
    if w is None:
        h = grid_count(x, **opt)
    else:
        w = w.unsqueeze(-2).unsqueeze(-2)   # make 2d spatial + add channel
        h = grid_push(w, x, **opt)
        h = h.squeeze(-3)                   # drop channel
    return h

    # order = py.make_list(order, 2)
    # order = [inter_to_nitorch(o, 'int') for o in order]
    # bound = make_bound([bound_to_nitorch(bound, 'int')]*2)
    # extrapolate = 1 if extrapolate else 2
    # bins = py.make_list(bins, 2)
    #
    # if w is None:
    #     w = x.new_ones([1]).expand(x.shape[:-1])
    # while w.dim() < 2:
    #     w = w.unsqueeze(0)
    #
    # args = (w.unsqueeze(-2).unsqueeze(-2), x.unsqueeze(1), bins, bound, extrapolate)
    # if all(o == 0 for o in order):
    #     h = iso0.push2d(*args)
    # elif all(o == 1 for o in order):
    #     h = iso1.push2d(*args)
    # else:
    #     args = (*args[:-1], make_spline(order), args[-1])
    #     h = nd.push(*args)
    # h = h.squeeze(1).squeeze(1)
    #
    # return h


def _jhistc_backward(g, x, w=None, order=0, bound='replicate',
                     extrapolate=True, gradx=True, gradw=False):
    """Compute derivative of the joint histogram.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    g : (b, bins, bins) tensor
    x : (b, n, 2) tensor
    w : ([b], n) tensor, optional
    order : int, default=0
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True
    gradx : bool, default=True
    gradw : bool, default=False

    Returns
    -------
    gx : (b, n, 2) tensor, if gradx
    gw : ([b], n) tensor, if gradw

    """
    extrapolate = 1 if extrapolate else 2
    opt = dict(interpolation=order, bound=bound, extrapolate=extrapolate)
    x = x.unsqueeze(-3)                     # make 2d spatial
    g = g.unsqueeze(-3)                     # add channel dimension
    out = []
    if gradx:
        gx = grid_grad(g, x, **opt)
        gx = gx.squeeze(-3).squeeze(-3)
        if w is not None:
            gx *= w.unsqueeze(-1)
        out.append(gx)
    if gradw and w is not None:
        gw = grid_pull(g, x, **opt)
        gw = gw.squeeze(-2).squeeze(-2)     # drop spatial + channel
        out.append(gw)
    elif gradw:
        out.append(None)
    return out[0] if len(out) == 1 else tuple(out)

    # order = py.make_list(order, 2)
    # order = [inter_to_nitorch(o, 'int') for o in order]
    # bound = make_bound([bound_to_nitorch(bound, 'int')]*2)
    # extrapolate = 1 if extrapolate else 2
    #
    # out = []
    #
    # if gradx:
    #     args = (g.unsqueeze(1), x.unsqueeze(1), bound, extrapolate)
    #     if all(o == 0 for o in order):
    #         gx = iso0.grad(*args)
    #     elif all(o == 1 for o in order):
    #         gx = iso1.grad2d(*args)
    #     else:
    #         args = (*args[:-1], make_spline(order), args[-1])
    #         gx = nd.grad(*args)
    #     gx = gx.squeeze(1).squeeze(1)
    #     if w is not None:
    #         gx *= w.unsqueeze(-1)
    #     out.append(gx)
    #
    # if gradw:
    #     args = (g.unsqueeze(1), x.unsqueeze(1), bound, extrapolate)
    #     if all(o == 0 for o in order):
    #         gw = iso0.pull2d(*args)
    #     elif all(o == 1 for o in order):
    #         gw = iso1.pull2d(*args)
    #     else:
    #         args = (*args[:-1], make_spline(order), args[-1])
    #         gw = nd.pull(*args)
    #     gw = gw.squeeze(1).squeeze(1)
    #
    #     wshape = w.shape
    #     while w.dim() < 2:
    #         w = w.unsqueeze(0)
    #     if w.numel() == 1:
    #         gw = gw.sum(keepdim=True)
    #     elif w.shape[0] == 1:
    #         gw = gw.sum(dim=0, keepdim=True)
    #     elif w.shape[1] == 1:
    #         gw = gw.sum(dim=1, keepdim=True)
    #     if len(wshape) < 2:
    #         gw = gw[0]
    #     if len(wshape) < 1:
    #         gw = gw[0]
    #
    #     out.append(gw)
    #
    # return out[0] if len(out) == 1 else tuple(out)


def _smooth_matrix(bins, fwhm, bound, dtype=None, device=None):
    """Dense matrix `K` of the 1D histogram smoothing: `smooth(h) = K @ h`.

    `K[c, a]` is the response in bin `c` to a unit count in bin `a`.
    With a 'replicate' bound, `K` is neither symmetric nor
    mass-preserving near the edges, so its adjoint is `K.T`, not `K`.
    """
    eye = torch.eye(bins, dtype=dtype, device=device)
    if not fwhm:
        return eye
    return smooth(eye, fwhm=fwhm, bound=bound, dim=1).transpose(0, 1)


def _corner_columns(k, bound):
    """Columns of `k` at the lower/upper corners of each linear cell.

    Cell `i` (i = -1, ..., n-1) spans [i, i+1]. Returns `(k0, k1)`, each
    (n, n+1), with `k0[:, i+1] = sign * k[:, index(i)]` and
    `k1[:, i+1] = sign * k[:, index(i+1)]`, using the same index/sign
    mapping as the order-1 push/pull kernels.
    """
    n = k.shape[-1]
    i = torch.arange(-1, n, device=k.device)
    cols = []
    for j in (i, i + 1):
        sign = bound.transform(j, n)
        col = k[:, bound.index(j, n)]
        if sign is not None:
            col = col * sign.to(col.dtype)
        cols.append(col)
    return tuple(cols)


def _jhistc_backward2_diag_x(h, u, v, kx, ky, hx, hy, bx, by):
    """x-component of the exact per-sample GN diagonal (order 1).

    For a sample at bin coordinates (u, v), the derivative of the
    (smoothed) joint histogram with respect to u is the separable
    vector `dp[c, d] = phi'[c] * phi[d]`, with
        phi'[c] = kx[c, a1] - kx[c, a0]
        phi[d]  = (1 - t) * ky[d, b0] + t * ky[d, b1]
    ((a0, a1), (b0, b1) the cell corners, t = v - floor(v)). This
    returns, for each sample,
        sum_cd h[c, d] dp[c, d]^2
      + sum_c hx[c] (sum_d dp[c, d])^2
      + sum_d hy[d] (sum_c dp[c, d])^2
    i.e. `dp^T M dp` for the histogram-space matrix
    `M = diag(h) + (hx 1^T)-blocks + (1 hy^T)-blocks`, without
    approximating `M` (or its smoothing) by its diagonal.
    """
    kx0, kx1 = _corner_columns(kx, bx)
    ky0, ky1 = _corner_columns(ky, by)
    nx, ny = kx.shape[-1], ky.shape[-1]
    dk2 = (kx1 - kx0).square()                       # (nx, nx+1)
    i = u.floor().long().clamp_(-1, nx - 1) + 1      # (b, n) in [0, nx]
    j = v.floor().long().clamp_(-1, ny - 1) + 1      # (b, n) in [0, ny]
    t1 = (v - v.floor())
    t0 = 1 - t1
    w00, w01, w11 = t0 * t0, 2 * t0 * t1, t1 * t1
    flat = i * (ny + 1) + j

    def gather2(tab):
        return tab.reshape([tab.shape[0], -1]).gather(-1, flat)

    out = 0
    if h is not None:
        t00 = torch.einsum('ci,bcd,dj->bij', dk2, h, ky0 * ky0)
        t01 = torch.einsum('ci,bcd,dj->bij', dk2, h, ky0 * ky1)
        t11 = torch.einsum('ci,bcd,dj->bij', dk2, h, ky1 * ky1)
        out = w00 * gather2(t00) + w01 * gather2(t01) + w11 * gather2(t11)
    if hx is not None:
        # (sum_d phi[d])^2 = 1 in the interior; computed exactly anyway
        s0, s1 = ky0.sum(0)[j], ky1.sum(0)[j]
        mx = torch.einsum('ci,bc->bi', dk2, hx).gather(-1, i)
        out = out + mx * (t0 * s0 + t1 * s1).square()
    if hy is not None:
        # sum_c phi'[c] = 0 when smoothing preserves mass, so this term
        # (the fixed marginal) only survives near the histogram edges.
        ex = (kx1 - kx0).sum(0).square()[i]
        y00 = (hy @ (ky0 * ky0)).gather(-1, j)
        y01 = (hy @ (ky0 * ky1)).gather(-1, j)
        y11 = (hy @ (ky1 * ky1)).gather(-1, j)
        out = out + ex * (w00 * y00 + w01 * y01 + w11 * y11)
    return out


def _jhistc_backward2_diag(h, x, kx=None, ky=None, hx=None, hy=None,
                           bound='replicate', extrapolate=True):
    """Exact per-sample diagonal of `J^T M J` for an order-1 joint histogram.

    Parameters
    ----------
    h : (b, nx, ny) tensor or None
        Bin-diagonal part of the histogram-space matrix `M`.
    x : (b, n, 2) tensor
        Bin coordinates.
    kx, ky : (nx, nx), (ny, ny) tensor, optional
        Smoothing matrices (the histogram is `kx @ hist @ ky.T`).
    hx : (b, nx) tensor, optional
        Coefficients of the x-marginal term `sum_c hx[c] (sum_d dp[c,d])^2`
    hy : (b, ny) tensor, optional
        Coefficients of the y-marginal term `sum_d hy[d] (sum_c dp[c,d])^2`

    Returns
    -------
    out : (b, n, 2) tensor

    """
    ref = h if h is not None else (hx if hx is not None else hy)
    backend = dict(dtype=x.dtype, device=x.device)
    if h is not None:
        nx, ny = h.shape[-2:]
    else:
        nx = kx.shape[-1] if kx is not None else hx.shape[-1]
        ny = ky.shape[-1] if ky is not None else hy.shape[-1]
    kx = torch.eye(nx, **backend) if kx is None else kx.to(**backend)
    ky = torch.eye(ny, **backend) if ky is None else ky.to(**backend)
    bx, by = make_bound([bound_to_nitorch(bound, 'int')]*2)
    extrapolate = 1 if extrapolate else 2
    batch = max(x.shape[0], ref.shape[0])
    x = x.expand([batch, *x.shape[1:]])
    u, v = x.unbind(-1)
    exp = lambda t: None if t is None else t.expand([batch, *t.shape[1:]])
    h, hx, hy = exp(h), exp(hx), exp(hy)
    out_x = _jhistc_backward2_diag_x(h, u, v, kx, ky, hx, hy, bx, by)
    hT = None if h is None else h.transpose(-1, -2)
    out_y = _jhistc_backward2_diag_x(hT, v, u, ky, kx, hy, hx, by, bx)
    out = torch.stack([out_x, out_y], -1)
    mask = ts_utils.inbounds_mask_2d(extrapolate, u, v, nx, ny)
    if mask is not None:
        out = out * mask.unsqueeze(-1)
    return out


def _jhistc_backward2(h, x, w=None, order=0, bound='replicate',
                      extrapolate=True, kx=None, ky=None, hx=None, hy=None):
    """Compute (Fisher's) second derivative of the joint histogram.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    h : (b, bins, bins, [bins, bins]) tensor
    x : (b, n, 2) tensor
    w : ([b], n) tensor, optional
    order : int, default=0
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True
    kx, ky : (bins, bins) tensor, optional
        Smoothing matrices (diagonal case only), see `_smooth_matrix`.
    hx, hy : (b, bins) tensor, optional
        Marginal terms of the histogram-space matrix (diagonal case only),
        see `_jhistc_backward2_diag`.

    Returns
    -------
    hx : (b, n, 2) tensor

    Notes
    -----
    In the diagonal case (`h` is (b, bins, bins)), the per-sample
    diagonal of `J^T M J` is exact for order 1, including the
    smoothing and the histogram edges (where the histogram does not
    depend on the sample and the result is zero).
    Only an order-1 Hessian kernel is implemented. For order != 1, the
    order-1 result is rescaled (per component) so that its expectation
    over the sub-bin position matches the expected Gauss-Newton curvature
    of the order-k histogram. This is an approximate, position-averaged
    Hessian. Order 0 is not supported (zero derivative a.e.).

    """
    order = py.make_list(order, 2)
    order = [inter_to_nitorch(o, 'int') for o in order]

    is_diag = h is None or h.dim() == 3

    if any(o == 0 for o in order):
        raise NotImplementedError(_ORDER0_MSG)
    if is_diag:
        gx = _jhistc_backward2_diag(h, x, kx, ky, hx, hy, bound, extrapolate)
    else:
        bound = make_bound([bound_to_nitorch(bound, 'int')]*2)
        extrapolate = 1 if extrapolate else 2
        args = (h.unsqueeze(1), x.unsqueeze(1), bound, extrapolate)
        gx = iso1_grad2_2d(*args)
        gx = gx.squeeze(1).squeeze(1)
    if any(o != 1 for o in order):
        sx, sy = _hess_order_scale(order)
        scl = torch.as_tensor([sx, sy], dtype=gx.dtype, device=gx.device)
        gx = gx * scl
    if w is not None:
        w = w.unsqueeze(-1)
        if is_diag:
            # the histogram is linear in w, so the GN term scales with w^2
            gx = gx * w.square()
        else:
            gx *= w.unsqueeze(-1)

    return gx


def _jhistc_backward_hess(g, x, w=None, order=1, bound='replicate',
                          extrapolate=True):
    """Second derivative of `<g, hist(x)>` with respect to each sample.

    The input must already be a soft mapping to bins indices.

    Parameters
    ----------
    g : (b, bins, bins) tensor
    x : (b, n, 2) tensor
    w : ([b], n) tensor, optional
    order : int, default=1
    bound : {'zero', 'nearest'}, default='nearest'
    extrapolate : bool, default=True

    Returns
    -------
    hx : (b, n, 2) tensor
        `[sum_ab g[a, b] B''(u - a) B(v - b), sum_ab g[a, b] B(u - a) B''(v - b)]`

    Notes
    -----
    This is the (voxel-diagonal) term of the Hessian that Gauss-Newton
    drops. For order 1, `B''` is a sum of Diracs at the bin edges; we
    return its average over the sub-bin position instead, which is
    exactly the order-2 `B''` (since `B1 * B0 = B2`).
    """
    order = py.make_list(order, 2)
    order = [inter_to_nitorch(o, 'int') for o in order]
    if any(o == 0 for o in order):
        raise NotImplementedError(_ORDER0_MSG)
    bound = make_bound([bound_to_nitorch(bound, 'int')]*2)
    extrapolate = 1 if extrapolate else 2
    shape = list(g.shape[-2:])
    batch = max(g.shape[0], x.shape[0])
    g = g.reshape([g.shape[0], 1, -1]).expand([batch, 1, -1])
    x = x.expand([batch, *x.shape[1:]])

    out = []
    for d in range(2):
        orders = list(order)
        orders[d] = max(orders[d], 2)
        weights, _, hesss, coords, signs = nd.get_weights(
            x.unsqueeze(1), bound, make_spline(orders), shape, hess=True)
        weights = [list(wd) for wd in weights]
        weights[d] = hesss[d]
        out1 = 0
        for i in range(orders[0] + 1):
            for j in range(orders[1] + 1):
                idx = coords[0][i] * shape[1] + coords[1][j]
                val = g.gather(-1, idx) * weights[0][i] * weights[1][j]
                sign = ts_utils.make_sign([signs[0][i], signs[1][j]])
                if sign is not None:
                    val = val * sign
                out1 = out1 + val
        out.append(out1.squeeze(1))
    out = torch.stack(out, -1)

    mask = nd.inbounds_mask(extrapolate, x, shape)
    if mask is not None:
        out = out * mask.squeeze(1).unsqueeze(-1)
    if w is not None:
        # the histogram is linear in w
        out = out * w.unsqueeze(-1)
    return out


# ======================================================================
#                              HISTOGRAM
# ======================================================================


def _preproc_hist(x, weights, dim):
    x = torch.as_tensor(x)
    if weights is not None:
        dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
        weights = torch.as_tensor(weights, dtype=dtype, device=x.device).expand(x.shape)
    inshape = x.shape
    if dim is None:
        x = x.reshape([1, -1])
        batch = []
        if weights is not None:
            weights = weights.reshape([1, -1])
    else:
        dim = py.make_list(dim)
        odim = list(range(-len(dim), 0))
        x = movedim(x, dim, odim)
        batch = x.shape[:-len(dim)]
        pool = x.shape[-len(dim):]
        x = x.reshape([-1, py.prod(pool)])
        if weights is not None:
            weights = weights.reshape([-1, py.prod(pool)])
    return x, weights, batch, inshape


def _postproc_hist(h, batch, inshape, dim, keepdim):
    n = h.shape[-1]
    if keepdim:
        oshape = list(inshape)
        for d in dim:
            oshape[d] = 1
        oshape += [n]
    else:
        oshape = [*batch, n]
    h = h.reshape(oshape)
    return h


def _postproc_grad(g, batch, inshape, dim):
    g = g.reshape([*batch, *inshape])
    if dim is not None:
        dim = py.make_list(dim)
        odim = list(range(-len(dim), 0))
        g = movedim(g, odim, dim)
    return g


def _intensity_to_bin(x, min, max, batch, n):
    # compute limits
    if min is None:
        min = x.min(dim=-1, keepdim=True).values
    else:
        min = torch.as_tensor(min)
        min = min.expand(batch).reshape([-1, 1])
    if max is None:
        max = x.max(dim=-1, keepdim=True).values
    else:
        max = torch.as_tensor(max)
        max = max.expand(batch).reshape([-1, 1])

    # convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    if not dtypes.dtype(x.dtype).is_floating_point:
        ftype = torch.get_default_dtype()
        x = x.to(ftype)
    x = x.clone()
    x = x.mul_(n / (max - min)).add_(n / (1 - max / min)).sub_(0.5)

    return x, min, max


def histc_forward(x, n=64, min=None, max=None, dim=None, keepdim=False, weights=None,
                  order=1, bound='replicate', extrapolate=False, dtype=None,
                  return_minmax=False):
    """Batched + differentiable histogram computation

    Parameters
    ----------
    x : tensor
        Input tensor.
    n : int, default=64
        Number of bins.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to the input batch shape.
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to the input batch shape.
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    keepdim : bool, default=False
        Keep singleton dimensions.
    weights : tensor, optional
        Observation weights
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.
    dtype : torch.dtype, optional
        Output data type.
        Default: same as x unless it is not a floating point type, then
        `torch.get_default_dtype()`

    Returns
    -------
    h : (..., n) tensor
        Count histogram

    """
    # reshape as [batch, pool]
    x, weights, batch, inshape = _preproc_hist(x, weights, dim)

    # compute limits + convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    x, min, max = _intensity_to_bin(x, min, max, batch, n)
    min = min.reshape(batch)
    max = max.reshape(batch)

    # push data into the histogram
    h = _histc_forward(x, n, weights, order, bound, extrapolate)

    # reshape
    h = h.to(dtype)
    h = _postproc_hist(h, batch, inshape, dim, keepdim)
    return (h, min, max) if return_minmax else h


def histc_backward(g, x, min=None, max=None, dim=None, weights=None,
                   order=1, bound='replicate', extrapolate=False):
    """Batched + differentiable histogram computation

    Parameters
    ----------
    g : (..., n) tensor
        Gradient wrt histogram
    x : tensor
        Input tensor.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to the input batch shape.
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to the input batch shape.
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    weights : tensor, optional
        Observation weights
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.

    Returns
    -------
    g : (*x.shape) tensor
        Gradient with respect to the input tensor
    w : (*weights.shape) tensor, if weights is not None
        Gradient with respect to the weight tensor

    """
    n = g.shape[-1]

    # reshape as [batch, pool]
    x, weights, batch, inshape = _preproc_hist(x, weights, dim)
    g = g.reshape([-1, n])

    # compute limits + convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    x, min, max = _intensity_to_bin(x, min, max, batch, n)

    # pull histogram gradient
    g = g.to(x.dtype)
    g = _histc_backward(g, x, weights, order, bound, extrapolate)
    if weights is not None:
        g, weights = g

    # backward pass of linear transform
    g = g.mul_(n / (max - min)).add_(n / (1 - max / min)).sub_(0.5)

    # reshape
    g = _postproc_grad(g, batch, inshape, dim)
    return g


class _HistC_AutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, n=64, min=None, max=None, dim=None, keepdim=False,
                weights=None, order=1, bound='replicate', extrapolate=False,
                dtype=None):
        # reshape as [batch, pool]
        x, weights, batch, inshape = _preproc_hist(x, weights, dim)

        # compute limits + convert intensities to coordinates
        # (min -> -0.5  // max -> n-0.5)
        x, min, max = _intensity_to_bin(x, min, max, batch, n)

        # push data into the histogram
        h = _histc_forward(x, n, weights, order, bound, extrapolate)

        # reshape
        h = h.to(dtype)
        h = _postproc_hist(h, batch, inshape, dim, keepdim)

        ctx.save_for_backward(x, min, max)
        if weights is not None:
            ctx.save_for_backward(weights)
        ctx.opt = (order, bound, extrapolate)

        return h

    @staticmethod
    def backward(ctx, g):
        n = g.shape[-1]
        g = g.reshape([-1, n])
        x, min, max, *weights = ctx.saved_tensors
        order, bound, extrapolate = ctx.opt
        weights = weights[0] if weights else None
        g = g.to(x.dtype)
        g = _histc_backward(g, x, weights, order, bound, extrapolate)
        if weights is not None:
            g, weights = g

        return [g] + [None] * 5 + [weights] + [None] * 4


def histc(x, n=64, min=None, max=None, dim=None, keepdim=False, weights=None,
          order=1, bound='replicate', extrapolate=False, dtype=None):
    """Batched + differentiable histogram computation

    Parameters
    ----------
    x : tensor
        Input tensor.
    n : int, default=64
        Number of bins.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to the input batch shape.
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to the input batch shape.
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    keepdim : bool, default=False
        Keep singleton dimensions.
    weights : tensor, optional
        Observation weights
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.
    dtype : torch.dtype, optional
        Output data type.
        Default: same as x unless it is not a floating point type, then
        `torch.get_default_dtype()`

    Returns
    -------
    h : (..., n) tensor
        Count histogram

    """
    return _HistC_AutoGrad.apply(x, n, min, max, dim, keepdim, weights,
                                 order, bound, extrapolate, dtype)


class HistCount:

    def __init__(self, bins=64, order=1, bound='replicate', extrapolate=False):
        self.bins = bins
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, w=None, min=None, max=None, return_minmax=False):
        """

        Parameters
        ----------
        x : (..., n) tensor
        w : (..., n) tensor, optional
        min : (...) tensor_like, optional
        max : (...) tensor_like, optional

        Returns
        -------
        h : (..., bins) tensor

        """
        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-1]
        xbatch = x.shape[:-1]
        if w is not None:
            x, w = torch.broadcast_tensors(x, w)
            w = w.reshape([-1, n])
            batch = x.shape[:-1]
        else:
            batch = xbatch
        x = x.reshape([-1, n])

        if min is None:
            min = x.min(-1).values
        else:
            min = torch.as_tensor(min, **backend).expand(xbatch).reshape([-1, 1])
        if max is None:
            max = x.max(-1).values
        else:
            max = torch.as_tensor(max, **backend).expand(xbatch).reshape([-1, 1])

        x = x.clone()
        x = x.mul_(self.bins / (max - min)).add_(self.bins / (1 - max / min)).sub_(0.5)
        min = min.reshape(xbatch)
        max = max.reshape(xbatch)

        # push data into the histogram
        h = _histc_forward(x, self.bins, w, self.order, self.bound, self.extrapolate)

        # reshape
        h = h.reshape(h, [*batch, self.bins])
        return (h, min, max) if return_minmax else h

    def backward(self, g, x, w=None, min=None, max=None):
        """

        Parameters
        ----------
        g : (..., bins) tensor
        x : (..., n) tensor
        w : (..., n) tensor, optional
        min : (...) tensor_like, optional
        max : (...) tensor_like, optional

        Returns
        -------
        g : (..., n) tensor

        """

        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-1]
        xbatch = x.shape[:-1]
        if w is not None:
            x, w = torch.broadcast_tensors(x, w)
            w = w.reshape([-1, n])
            batch = x.shape[:-1]
        else:
            batch = xbatch
        x = x.reshape([-1, n])
        g = g.reshape([-1, self.bins])

        if min is None:
            min = x.min(-1, keepdim=True).values
        else:
            min = torch.as_tensor(min, **backend).expand(xbatch).reshape([-1, 1])
        if max is None:
            max = x.max(-1, keepdim=True).values
        else:
            max = torch.as_tensor(max, **backend).expand(xbatch).reshape([-1, 1])

        x = x.clone()
        x = x.mul_(self.bins / (max - min)).add_(self.bins / (1 - max / min)).sub_(0.5)

        # push data into the histogram
        g = _histc_backward(g, x, w, self.order, self.bound, self.extrapolate)
        g = g.mul_(self.bins / (max - min))

        # reshape
        g = g.reshape([*batch, n])
        return g

    def backward2(self, h, x, w=None, min=None, max=None):
        """

        Parameters
        ----------
        h : (..., bins, [bins]) tensor
        x : (..., n) tensor
        w : (..., n) tensor, optional
        min : (...) tensor_like, optional
        max : (...) tensor_like, optional

        Returns
        -------
        h : (..., n) tensor

        """

        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-1]
        xbatch = x.shape[:-1]
        if w is not None:
            x, w = torch.broadcast_tensors(x, w)
            w = w.reshape([-1, n])
            batch = x.shape[:-1]
        else:
            batch = xbatch
        x = x.reshape([-1, n])
        if h.shape[:-1] == batch:
            h = h.reshape([-1, self.bins])
        elif h.shape[:-2] == batch:
            h = h.reshape([-1, self.bins, self.bins])
        else:
            raise ValueError('Don\'t know what to do with input shape')

        if min is None:
            min = x.min(-1, keepdim=True).values
        else:
            min = torch.as_tensor(min, **backend).expand(xbatch).reshape([-1, 1])
        if max is None:
            max = x.max(-1, keepdim=True).values
        else:
            max = torch.as_tensor(max, **backend).expand(xbatch).reshape([-1, 1])

        x = x.clone()
        x = x.mul_(self.bins / (max - min)).add_(self.bins / (1 - max / min)).sub_(0.5)

        # push data into the histogram
        h = _histc_backward2(h, x, w, self.order, self.bound, self.extrapolate)
        h = h.mul_((self.bins / (max - min)).square_())

        # reshape
        h = h.reshape([*batch, n])
        return h


# ======================================================================
#                            JOINT HISTOGRAM
# ======================================================================

class JointHistCount:

    def __init__(self, bins=64, order=1, bound='replicate', extrapolate=False,
                 fwhm=1):
        self.bins = py.make_list(bins, 2)
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate
        self.fwhm = py.make_list(fwhm, 2)

    def smoothing_matrices(self, dtype=None, device=None):
        """Dense matrices (kx, ky) of the histogram smoothing, such that
        the smoothed histogram is `kx @ h @ ky.T`."""
        return tuple(_smooth_matrix(b, f, self.bound, dtype, device)
                     for b, f in zip(self.bins, self.fwhm))

    def forward(self, x, w=None, min=None, max=None, return_minmax=False):
        """

        Parameters
        ----------
        x : (..., n, 2) tensor
        w : (..., n) tensor, optional
        min : (..., 2) tensor_like, optional
        max : (..., 2) tensor_like, optional

        Returns
        -------
        h : (..., *bins) tensor

        """
        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-2]
        xbatch = x.shape[:-2]
        if w is not None:
            _, w = torch.broadcast_tensors(x[..., 0], w)
            batch = w.shape[:-1]
            x = x.expand([*batch, *x.shape[-2:]])
            w = w.reshape([-1, n])
        else:
            batch = xbatch
        x = x.reshape([-1, n, 2])

        if min is None:
            min = x.min(-2, keepdim=True).values
        else:
            min = torch.as_tensor(min, **backend).expand([*batch, 2]).reshape([-1, 1, 2])
        if max is None:
            max = x.max(-2, keepdim=True).values
        else:
            max = torch.as_tensor(max, **backend).expand([*batch, 2]).reshape([-1, 1, 2])

        x = x.clone()
        bins = torch.as_tensor(self.bins, **backend)
        x = x.mul_(bins / (max - min)).add_(bins / (1 - max / min)).sub_(0.5)
        min = min.reshape([*batch, 2])
        max = max.reshape([*batch, 2])

        # push data into the histogram
        h = _jhistc_forward(x, self.bins, w, self.order, self.bound, self.extrapolate)

        # smooth:
        if any(self.fwhm):
            h = smooth(h, fwhm=self.fwhm, bound=self.bound, dim=2)

        # reshape
        h = h.reshape([*batch, *self.bins])
        return (h, min, max) if return_minmax else h

    def backward(self, g, x, w=None, min=None, max=None):
        """

        Parameters
        ----------
        g : (..., *bins) tensor
        x : (..., n, 2) tensor
        w : (..., n) tensor, optional
        min : (...) tensor_like, optional
        max : (...) tensor_like, optional

        Returns
        -------
        g : (..., n, 2) tensor

        """
        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-2]
        xbatch = x.shape[:-2]
        if w is not None:
            _, w = torch.broadcast_tensors(x[..., 0], w)
            batch = w.shape[:-1]
            x = x.expand([*batch, *x.shape[-2:]])
            w = w.reshape([-1, n])
        else:
            batch = xbatch
        x = x.reshape([-1, n, 2])

        if min is None:
            min = x.min(-2, keepdim=True).values
        else:
            min = torch.as_tensor(min, **backend).expand([*xbatch, 2]).reshape([-1, 1, 2])
        if max is None:
            max = x.max(-2, keepdim=True).values
        else:
            max = torch.as_tensor(max, **backend).expand([*xbatch, 2]).reshape([-1, 1, 2])

        x = x.clone()
        bins = torch.as_tensor(self.bins, **backend)
        x = x.mul_(bins / (max - min)).add_(bins / (1 - max / min)).sub_(0.5)
        min = min.reshape([*xbatch, 2])
        max = max.reshape([*xbatch, 2])

        g = g.reshape([-1, *self.bins])

        # smooth backward (adjoint of the forward smoothing: kx.T @ g @ ky,
        # which differs from smoothing `g` again near the edges)
        if any(self.fwhm):
            kx, ky = self.smoothing_matrices(**backend)
            g = kx.transpose(0, 1).matmul(g).matmul(ky)

        # push data into the histogram
        g = _jhistc_backward(g, x, w, self.order, self.bound, self.extrapolate)
        g = g.mul_(bins / (max - min))

        # reshape
        g = g.reshape([*batch, n, 2])
        return g

    def backward_hess(self, g, x, w=None, min=None, max=None):
        """Second derivative of `<g, hist(x)>` with respect to each sample.

        If `g` is the gradient of a loss with respect to the histogram,
        this is the voxel-diagonal term of the Hessian of the loss that
        Gauss-Newton drops (`sum_ab g[a, b] d2 hist[a, b] / dx_n^2`).
        For spline order 1, it is averaged over the sub-bin position
        (see `_jhistc_backward_hess`).

        Parameters
        ----------
        g : (..., *bins) tensor
        x : (..., n, 2) tensor
        w : (..., n) tensor, optional
        min : (...) tensor_like, optional
        max : (...) tensor_like, optional

        Returns
        -------
        h : (..., n, 2) tensor

        """
        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-2]
        xbatch = x.shape[:-2]
        if w is not None:
            _, w = torch.broadcast_tensors(x[..., 0], w)
            batch = w.shape[:-1]
            x = x.expand([*batch, *x.shape[-2:]])
            w = w.reshape([-1, n])
        else:
            batch = xbatch
        x = x.reshape([-1, n, 2])

        if min is None:
            min = x.min(-2, keepdim=True).values
        else:
            min = torch.as_tensor(min, **backend).expand([*xbatch, 2]).reshape([-1, 1, 2])
        if max is None:
            max = x.max(-2, keepdim=True).values
        else:
            max = torch.as_tensor(max, **backend).expand([*xbatch, 2]).reshape([-1, 1, 2])

        x = x.clone()
        bins = torch.as_tensor(self.bins, **backend)
        x = x.mul_(bins / (max - min)).add_(bins / (1 - max / min)).sub_(0.5)

        g = g.reshape([-1, *self.bins])

        # smooth backward (adjoint of the forward smoothing)
        if any(self.fwhm):
            kx, ky = self.smoothing_matrices(**backend)
            g = kx.transpose(0, 1).matmul(g).matmul(ky)

        h = _jhistc_backward_hess(g, x, w, self.order, self.bound,
                                  self.extrapolate)
        h = h.mul_((bins / (max - min)).square_())

        # reshape
        h = h.reshape([*batch, n, 2])
        return h

    def backward2(self, h, x, w=None, min=None, max=None, hx=None, hy=None):
        """Per-sample diagonal of the Gauss-Newton term `J^T M J`.

        `J` is the Jacobian of the (smoothed) histogram with respect to
        the sample values and `M` is a histogram-space matrix.

        Parameters
        ----------
        h : (..., *bins, [*bins]) tensor
            Bin-diagonal part of `M` (or the full `M`).
        x : (..., n, 2) tensor
        w : (..., n) tensor, optional
        min : (...) tensor_like, optional
        max : (...) tensor_like, optional
        hx : (..., bins[0]) tensor, optional
            Marginal part of `M` along the first axis:
            `M[(a, b), (c, d)] += delta(a, c) * hx[a]`
            (only when `h` is bin-diagonal).
        hy : (..., bins[1]) tensor, optional
            Marginal part of `M` along the second axis:
            `M[(a, b), (c, d)] += delta(b, d) * hy[b]`
            (only when `h` is bin-diagonal).

        Returns
        -------
        h : (..., n, 2) tensor

        Notes
        -----
        When `h` is bin-diagonal, the result is the exact diagonal of
        `J^T M J` for spline order 1, smoothing included (the marginal
        terms are not approximated by their bin-diagonal).

        """
        backend = dict(dtype=x.dtype, device=x.device)
        n = x.shape[-2]
        xbatch = x.shape[:-2]
        if w is not None:
            _, w = torch.broadcast_tensors(x[..., 0], w)
            batch = w.shape[:-1]
            x = x.expand([*batch, *x.shape[-2:]])
            w = w.reshape([-1, n])
        else:
            batch = xbatch
        x = x.reshape([-1, n, 2])

        if h.shape[:-2] == batch:
            is_diag = True
        elif h.shape[:-4] == batch:
            is_diag = False
        else:
            raise ValueError('Don\'t know what to do with that shape')

        if min is None:
            min = x.min(-2, keepdim=True).values
        else:
            min = torch.as_tensor(min, **backend).expand([*xbatch, 2]).reshape([-1, 1, 2])
        if max is None:
            max = x.max(-2, keepdim=True).values
        else:
            max = torch.as_tensor(max, **backend).expand([*xbatch, 2]).reshape([-1, 1, 2])

        x = x.clone()
        bins = torch.as_tensor(self.bins, **backend)
        x = x.mul_(bins / (max - min)).add_(bins / (1 - max / min)).sub_(0.5)
        min = min.reshape([*xbatch, 2])
        max = max.reshape([*xbatch, 2])

        if is_diag:
            h = h.reshape([-1, *self.bins])
            if hx is not None:
                hx = hx.reshape([-1, self.bins[0]])
            if hy is not None:
                hy = hy.reshape([-1, self.bins[1]])
        else:
            if hx is not None or hy is not None:
                raise ValueError('hx/hy are only supported with a '
                                 'bin-diagonal h')
            h = h.reshape([-1, *self.bins, *self.bins])

        if is_diag:
            # The smoothing is handled exactly inside the kernel: the
            # diagonal of K^T M K is *not* enough, since each sample
            # touches several (signed) bins at once.
            kx = ky = None
            if any(self.fwhm):
                kx, ky = self.smoothing_matrices(**backend)
            h = _jhistc_backward2(h, x, w, self.order, self.bound,
                                  self.extrapolate, kx=kx, ky=ky,
                                  hx=hx, hy=hy)
        else:
            # full histogram-space Hessian: K^T H K
            if any(self.fwhm):
                kx, ky = self.smoothing_matrices(**backend)
                h = torch.einsum('ca,db,ncdef,eg,fh->nabgh',
                                 kx, ky, h, kx, ky)
            h = _jhistc_backward2(h, x, w, self.order, self.bound,
                                  self.extrapolate)
        h = h.mul_((bins / (max - min)).square_())

        # reshape
        h = h.reshape([*batch, n, 2])
        return h
