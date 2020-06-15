# -*- coding: utf-8 -*-
"""Various utilities.

Created on Fri Apr 24 14:45:24 2020

@author: yael.balbastre@gmail.com
"""

# TODO:
#   . Directly use pytorch's pad when possible (done for constant)
#   . check time/memory footprint
#   . Implement modifiers for Dirichlet/Sliding boundaries


import torch
from torch.nn import functional as F


__all__ = ['pad', 'same_storage', 'shiftdim', 'softmax']


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
    # Uses torch.nn.functional.pad
    # Convert pre and post to a single list
    padpre = padpre.tolist()
    padpost = padpost.tolist()
    padding = padpre * 2
    padding[1::2] = padpost[::-1]
    padding[::2] = padpre[::-1]
    # Apply padding
    inp = F.pad(inp, padding, mode='constant', value=value)
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
