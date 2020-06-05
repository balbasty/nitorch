# -*- coding: utf-8 -*-
"""Various utilities.

Created on Fri Apr 24 14:45:24 2020

@author: yael.balbastre@gmail.com
"""

# TODO:
#   . Directly use pytorch's pad when possible
#   . check time/memory footprint
#   . Implement modifiers for Dirichlet/Sliding boundaries

import torch

__all__ = ['pad', 'same_storage', 'shiftdim']


def divergence_3d(dat, vx=None, which='forward', bound='constant'):
    """ Computes the divergence of volumetric data.

    Args:
        dat (torch.tensor()): A 4D tensor (3, D, H, W).
        vx (tuple(float), optional): Voxel size. Defaults to (1, 1, 1).
            Note, the voxel size should be ordered as (W, H, D).
        which (string, optional): Gradient type:
            . 'forward': Forward difference (next - centre)
            . 'backward': Backward difference (centre - previous)
            . 'central': Central difference ((next - previous)/2)
            Defaults to 'forward'.
        bound (string, optional): Boundary conditions, defaults to 'constant' (zero).

    Returns:
        div (torch.tensor()): Divergence (D, H, W).

    """
    if vx is None:
        vx = (1,) * 3

    if which == 'forward':
        # Pad + reflected forward difference
        d = pad(dat[0, ...], (1, 0, 0, 0, 0, 0,), mode=bound)
        d = d[:-1, :, :] - d[1:, :, :]
        h = pad(dat[1, ...], (0, 0, 1, 0, 0, 0,), mode=bound)
        h = h[:, :-1, :] - h[:, 1:, :]
        w = pad(dat[2, ...], (0, 0, 0, 0, 1, 0,), mode=bound)
        w = w[:, :, :-1] - w[:, :, 1:]
    elif which == 'backward':
        # Pad + reflected backward difference
        d = pad(dat[0, ...], (0, 1, 0, 0, 0, 0,), mode=bound)
        d = d[:-1, :, :] - d[1:, :, :]
        h = pad(dat[1, ...], (0, 0, 0, 1, 0, 0,), mode=bound)
        h = h[:, :-1, :] - h[:, 1:, :]
        w = pad(dat[2, ...], (0, 0, 0, 0, 0, 1,), mode=bound)
        w = w[:, :, :-1] - w[:, :, 1:]
    elif which == 'central':
        # Pad + reflected central difference
        d = pad(dat[0, ...], (1, 1, 0, 0, 0, 0,), mode=bound)
        d = 0.5 * (d[:-2, :, :] - d[2:, :, :])
        h = pad(dat[1, ...], (0, 0, 1, 1, 0, 0,), mode=bound)
        h = 0.5 * (h[:, :-2, :] - h[:, 2:, :])
        w = pad(dat[2, ...], (0, 0, 0, 0, 1, 1,), mode=bound)
        w = 0.5 * (w[:, :, :-2] - w[:, :, 2:])
    else:
        raise ValueError('Undefined divergence')

    return d / vx[2] + h / vx[1] + w / vx[0]


def gradient_3d(dat, vx=None, which='forward', bound='reflect2'):
    """ Computes the gradient of volumetric data.

    Args:
        dat (torch.tensor()): A tensor (D, H, W).
        vx (tuple(float), optional): Voxel size. Defaults to (1, 1, 1).
            Note, the voxel size should be ordered as (W, H, D).
        which (string, optional): Gradient type:
            . 'forward': Forward difference (next - centre)
            . 'backward': Backward difference (centre - previous)
            . 'central': Central difference ((next - previous)/2)
            Defaults to 'forward'.
        bound (string, optional): Boundary conditions, defaults to 'reflect2'.

    Returns:
          grad (torch.tensor()): Gradient (3, D, H, W).

    """
    if vx is None:
        vx = (1,) * 3

    if which == 'forward':
        # Pad + forward difference
        dat = pad(dat, (0, 1, 0, 1, 0, 1,), mode=bound)
        gd = -dat[:-1, :-1, :-1] + dat[1:, :-1, :-1]
        gh = -dat[:-1, :-1, :-1] + dat[:-1, 1:, :-1]
        gw = -dat[:-1, :-1, :-1] + dat[:-1, :-1, 1:]
    elif which == 'backward':
        # Pad + backward difference
        dat = pad(dat, (1, 0, 1, 0, 1, 0,), mode=bound)
        gd = -dat[:-1, 1:, 1:] + dat[1:, 1:, 1:]
        gh = -dat[1:, :-1, 1:] + dat[1:, 1:, 1:]
        gw = -dat[1:, 1:, :-1] + dat[1:, 1:, 1:]
    elif which == 'central':
        # Pad + central difference
        dat = pad(dat, (1, 1, 1, 1, 1, 1,), mode=bound)
        gd = 0.5 * (-dat[:-2, 1:-1, 1:-1] + dat[2:, 1:-1, 1:-1])
        gh = 0.5 * (-dat[1:-1, :-2, 1:-1] + dat[1:-1, 2:, 1:-1])
        gw = 0.5 * (-dat[1:-1, 1:-1, :-2] + dat[1:-1, 1:-1, 2:])
    else:
        raise ValueError('Undefined gradient')

    return torch.stack((gd / vx[2], gh / vx[1], gw / vx[0]), dim=0)


def softmax(Z, dim=-1, get_ll=False, W=1):
    """ SoftMax (safe).

    Args:
        Z (torch.tensor): Tensor with values.
        dim (int, optional): Dimension to take softmax, defaults to last dimensions (-1).
        get_ll (bool, optional): Compute log-likelihood, defaults to False.
        W (torch.tensor, optional): Observation weights, defaults to 1 (no weights).

    Returns:
        Z (torch.tensor): Soft-maxed tensor with values.

    """
    Z_max, _ = torch.max(Z, dim=dim)
    Z = torch.exp(Z - Z_max[:, None])
    Z_sum = torch.sum(Z, dim=dim)
    if get_ll:
        # Compute log-likelihood
        ll = torch.sum((torch.log(Z_sum) + Z_max)*W, dtype=torch.float64)
    else:
        ll = None
    Z = Z / Z_sum[:, None]
    return Z, ll


def same_storage(x, y):
    # type: (torch.Tensor, torch.Tensor) -> bool
    """Return true if `x` and `y` share the same underlying storage."""
    return x.storage().data_ptr() == y.storage().data_ptr()


def shiftdim(x, n=None):
    # type: (torch.Tensor, int) -> torch.Tensor
    """Shift the dimensions of x by n.

    When N is positive, `shiftdim` shifts the dimensions to the left and wraps
    the N leading dimensions to the end.  When N is negative, `shiftdim`
    shifts the dimensions to the right and pads with singletons.

    When N is None, `shiftdim` removes all leading singleton
    dimensions. The number of removed dimensions is returned as well.


    Args:
        x (torch.Tensor): Input tensor.
        n (int): Shift. Defaults to None.

    Returns:
        x (torch.Tensor): Output tensor.
        n (int, if n is None): Number of removed dimensions

    """
    if n is None:
        shape = torch.as_tensor(x.size())
        n = (shape != 1).nonzero()
        if n.numel() == 0:
            n = x.dim()
            x = x.reshape([])
        else:
            n = n[0]
            x = x.reshape(shape[n:].tolist())
        return x, n
    elif n < 0:
        x = x.reshape((1,)*(-n) + x.size())
    elif n > 0:
        n = n % x.dim()
        x = x.permute(tuple(range(n, x.dim())) + tuple(range(n)))
    return x


def _bound_circular(i, n):
    return i % n


def _bound_replicate(i, n):
    return i.clamp(min=0, max=n-1)


def _bound_reflect2(i, n):
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i


def _bound_reflect1(i, n):
    if n == 1:
        return torch.zeros(i.size(), dtype=i.dtype, device=i.device)
    else:
        n2 = (n-1)*2
        pre = (i < 0)
        i[pre] = -i[pre]
        i = i % n2
        post = (i >= n)
        i[post] = n2 - i[post]
        return i


_bounds = {
    'circular': _bound_circular,
    'replicate': _bound_replicate,
    'reflect': _bound_reflect1,
    'reflect1': _bound_reflect1,
    'reflect2': _bound_reflect2,
    }


_modifiers = {
    'circular': lambda x, i, n: x,
    'replicate': lambda x, i, n: x,
    'reflect': lambda x, i, n: x,
    'reflect1': lambda x, i, n: x,
    'reflect2': lambda x, i, n: x,
    }


def pad(inp, padsize, mode='constant', value=0, side=None):
    # type: (torch.Pad, tuple[int], str, any, str) -> torch.Tensor
    """Pad a tensor.

    This function is a bit more generic than torch's native pad, but probably
    a bit slower:
        . works with any input type
        . works with arbitrarily large padding size
        . crops the tensor for negative padding values
        . implements additional padding modes
    When used with defaults parameters (side=None), it behaves
    exactly like `torch.nn.functional.pad`

    Boundary modes are:
        . 'circular'
            -> corresponds to the boundary condition of an FFT
        . 'reflect' or 'reflect1'
            -> corresponds to the boundary condition of a DCT-I
        . 'reflect2'
            -> corresponds to the boundary condition of a DCT-II
        . 'replicate'
            -> replicates border values
        . 'constant'
            -> pads with a constant value (defaults to 0)

    Side modes are 'pre', 'post', 'both' or None. If side is not None,
    inp.dim() values (or less) should be provided. If side is None,
    twice as many values should be provided, indicating different padding sizes
    for the 'pre' and 'post' sides. If the number of padding values is less
    than the dimension of the input tensor, zeros are prepended.

    Args:
        inp (tensor): Input tensor.
        padsize (tuple): Amount of padding in each dimension.
        mode (string,optional): 'constant', 'replicate', 'reflect1',
            'reflect2', 'circular'.
            Defaults to 'constant'.
        value (optional): Value to pad with in mode 'constant'.
            Defaults to 0.
        side: Use padsize to pad on left side ('pre'), right side ('post') or
            both sides ('both'). If None, the padding side for the left and
            right sides should be provided in alternate order.
            Defaults to None.

    Returns:
        Padded tensor.

    """
    # Argument checking
    if mode not in tuple(_bounds.keys()) + ('constant',):
        raise ValueError('Padding mode should be one of {}. Got {}.'
                         .format(tuple(_bounds.keys()) + ('constant',), mode))
    if side == 'both':
        padpre = padsize
        padpost = padsize
    elif side == 'pre':
        padpre = padsize
        padpost = (0,) * len(padpre)
    elif side == 'post':
        padpost = padsize
        padpre = (0,) * len(padpost)
    else:
        if len(padsize) % 2:
            raise ValueError('Padding length must be divisible by 2')
        padpre = padsize[::2]
        padpost = padsize[1::2]
    padpre = (0,) * max(0, inp.dim()-len(padpre)) + padpre
    padpost = (0,) * max(0, inp.dim()-len(padpost)) + padpost
    if inp.dim() != len(padpre) or inp.dim() != len(padpost):
        raise ValueError('Padding length too large')

    padpre = torch.as_tensor(padpre)
    padpost = torch.as_tensor(padpost)

    # Pad
    if mode == 'constant':
        return _pad_constant(inp, padpre, padpost, value)
    else:
        bound = _bounds[mode]
        modifier = _modifiers[mode]
        return _pad_bound(inp, padpre, padpost, bound, modifier)


def _pad_constant(inp, padpre, padpost, value):
    idim = torch.as_tensor(inp.shape)
    ndim = idim.numel()
    # First: crop input if needed
    start = (-padpre).clamp(min=0)
    length = idim - start - (-padpost).clamp(min=0)
    for d in range(ndim):
        inp = inp.narrow(d, start[d].item(), length[d].item())
    # Second: pad with constant tensors
    padpre = padpre.clamp(min=0)
    padpost = padpost.clamp(min=0)
    for d in range(ndim):
        idim = torch.as_tensor(inp.shape)
        padpredim = idim
        padpredim[d] = padpre[d]
        padpredim = [x.item() for x in padpredim]
        pre = torch.full(padpredim, value,
                         dtype=inp.dtype, device=inp.device)
        padpostdim = idim
        padpostdim[d] = padpost[d]
        padpostdim = [x.item() for x in padpostdim]
        post = torch.full(padpostdim, value,
                          dtype=inp.dtype, device=inp.device)
        inp = torch.cat((pre, inp, post), dim=d)
    return inp


def _pad_bound(inp, padpre, padpost, bound, modifier):
    begin = -padpre
    end = tuple(d+p for d, p in zip(inp.size(), padpost))
    idx = tuple(range(b, e) for (b, e) in zip(begin, end))
    idx = tuple(bound(torch.as_tensor(i, device=inp.device),
                      torch.as_tensor(n, device=inp.device))
                for (i, n) in zip(idx, inp.shape))
    for d in range(inp.dim()):
        inp = inp.index_select(d, idx[d])
    return inp
