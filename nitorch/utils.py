# -*- coding: utf-8 -*-
"""Various utilities.

Created on Fri Apr 24 14:45:24 2020

@author: yael.balbastre@gmail.com
"""

# TODO:
#   . Directly use pytorch's pad when possible
#   . check time/memory footprint

import torch

__all__ = ['pad']


def bound_circular(i, n):
    nonneg = (i >= 0)
    i[nonneg] = i[nonneg] % n
    i[~nonneg] = (n + (i[~nonneg] % n) % n)
    return i


def bound_replicate(i, n):
    pre = (i < 0)
    post = (i >= n)
    i[pre] = 0
    i[post] = n-1
    return i


def bound_symmetric(i, n):
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i


bounds = {
    'circular': bound_circular,
    'replicate': bound_replicate,
    'symmetric': bound_symmetric,
    }


modifiers = {
    'circular': lambda x, i, n: x,
    'replicate': lambda x, i, n: x,
    'symmetric': lambda x, i, n: x,
    }


def padvec(x, n, value=0):
    """Pad value at the end of a vector so that it reaches length n."""
    x = torch.as_tensor(x).flatten()
    padsize = max(0, n-x.numel())
    padvec = torch.full((padsize,), value, dtype=x.dtype, device=x.device)
    x = torch.cat((x, padvec))
    return x


def pad(inp, padsize, mode='constant', value=0):
    """Pad a tensor.

    This function is a bit more generic than torch's native pad (it works in
    any dimension), but probably a bit slower:
        . works in any dimensions
        . crops the tensor for negative padding values
        . augments the dimension of the input tensor if needed

    Note that:
        . 'circular'  corresponds to the boundary condition of an FFT
        . 'symmetric' corresponds to the boundary condition of a DCT-II

    Args:
        inp (tensor): Input tensor.
        padsize (tuple): Amount of padding in each dimension (pre, post).
        mode (string): 'constant', 'replicate', 'symmetric', 'circular'
        value: Value to pad with in mode 'constant'.

    Returns:
        Padded tensor.

    """
    # Check mode
    if mode not in tuple(bounds.keys()) + ('constant',):
        raise ValueError('Padding mode should be one of {}. Got {}.'
                         .format(tuple(bounds.keys()) + ('constant',), mode))
    # Compute output dimensions
    inp = torch.as_tensor(inp)
    idim = torch.tensor(inp.shape)
    padsize = torch.as_tensor(padsize, dtype=torch.int64).flatten()
    padpre = padsize[slice(0, padsize.numel(), 2)]
    padpost = padsize[slice(1, padsize.numel(), 2)]
    ndim = max(idim.numel(), padpre.numel(), padpost.numel())
    idim = padvec(idim, ndim, 1)
    padpre = padvec(padpre, ndim, 0)
    padpost = padvec(padpost, ndim, 0)
    inp = inp.reshape(idim.tolist())

    if mode == 'constant':
        return pad_constant(inp, padpre, padpost, value)
    else:
        bound = bounds[mode]
        modifier = modifiers[mode]
        return pad_bound(inp, padpre, padpost, bound, modifier)


def pad_constant(inp, padpre, padpost, value):
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


def pad_bound(inp, padpre, padpost, bound, modifier):
    idim = torch.as_tensor(inp.shape)
    begin = -padpre
    end = idim + padpost
    idx = tuple(range(b, e) for (b, e) in zip(begin, end))
    idx = tuple(bound(torch.as_tensor(i, device=inp.device), n)
                for (i, n) in zip(idx, idim))
    for d in range(idim.numel()):
        inp = inp.index_select(d, idx[d])
    return inp