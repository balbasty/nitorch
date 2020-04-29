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

__all__ = ['pad', 'same_storage']


def same_storage(x, y):
    """Return true if `x` and `y` share the same underlying storage."""
    return x.storage().data_ptr() == y.storage().data_ptr()


def bound_circular(i, n):
    return i % n


def bound_replicate(i, n):
    return i.clamp(min=0, max=n-1)


def bound_symmetric(i, n):
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i


def bound_reflect(i, n):
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


bounds = {
    'circular': bound_circular,
    'replicate': bound_replicate,
    'symmetric': bound_symmetric,
    'reflect': bound_reflect,
    }


modifiers = {
    'circular': lambda x, i, n: x,
    'replicate': lambda x, i, n: x,
    'symmetric': lambda x, i, n: x,
    'reflect': lambda x, i, n: x,
    }


def padvec(x, n, value=0):
    """Pad value at the end of a vector so that it reaches length n."""
    x = torch.as_tensor(x).flatten()
    padsize = max(0, n-x.numel())
    padvec = torch.full((padsize,), value, dtype=x.dtype, device=x.device)
    x = torch.cat((x, padvec))
    return x


def pad(inp, padsize, mode='constant', value=0, side=None, nn=True):
    """Pad a tensor.

    This function is a bit more generic than torch's native pad, but probably
    a bit slower:
        . works in any dimensions
        . with any input type
        . with arbitrarily large padding size
        . crops the tensor for negative padding values
        . augments the dimension of the input tensor if needed
    When used with defaults parameters (side=None, nn=True), it behaves
    exactly like `torch.nn.functional.pad`

    Note that:
        . 'circular'  corresponds to the boundary condition of an FFT
        . 'reflect'   corresponds to the boundary condition of a DCT-I
        . 'symmetric' corresponds to the boundary condition of a DCT-II

    Args:
        inp (tensor): Input tensor.
        padsize (tuple): Amount of padding in each dimension.
        mode (string,optional): 'constant', 'replicate', 'reflect',
            'symmetric', 'circular'.
            Defaults to 'constant'.
        value (optional): Value to pad with in mode 'constant'.
            Defaults to 0.
        side: Use padsize to pad on left side ('pre'), right side ('post') or
            both sides ('both'). If None, the padding side for the left and
            right sides should be provided in alternate order.
            Defaults to None.
        nn (bool,optional): Neural network mode - assumes that the first two
            dimensions are channels and should not be padded.
            Defaults to True.

    Returns:
        Padded tensor.

    """
    # Check mode
    if mode not in tuple(bounds.keys()) + ('constant',):
        raise ValueError('Padding mode should be one of {}. Got {}.'
                         .format(tuple(bounds.keys()) + ('constant',), mode))
    # Compute output dimensions
    inp = torch.as_tensor(inp)
    idim = torch.as_tensor(inp.size())
    padsize = torch.as_tensor(padsize, dtype=torch.int64).flatten()
    if side == 'both':
        padpre = padsize
        padpost = padsize
    elif side == 'pre':
        padpre = padsize
        padpost = torch.zeros(padpre.size(), dtype=padpre.dtype,
                              device=padpre.device)
    elif side == 'post':
        padpost = padsize
        padpre = torch.zeros(padpost.size(), dtype=padpost.dtype,
                             device=padpost.device)
    else:
        padpre = padsize[::2]
        padpost = padsize[1::2]
    if nn:
        padpre = torch.cat((torch.zeros(2, dtype=padpre.dtype,
                                        device=padpre.device), padpre))
        padpost = torch.cat((torch.zeros(2, dtype=padpost.dtype,
                                         device=padpost.device), padpost))
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