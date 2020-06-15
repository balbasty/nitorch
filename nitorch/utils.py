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


__all__ = ['divergence_3d', 'gradient_3d', 'pad', 'same_storage',
           'shiftdim', 'softmax']


def divergence_3d(dat, vx=None, which='forward', bound='constant'):
    """ Computes the divergence of volumetric data.

    Args:
        dat (torch.tensor()): A 4D tensor (3, W, H, D).
        vx (tuple(float), optional): Voxel size. Defaults to (1, 1, 1).
        which (string, optional): Gradient type:
            . 'forward': Forward difference (next - centre)
            . 'backward': Backward difference (centre - previous)
            . 'central': Central difference ((next - previous)/2)
            Defaults to 'forward'.
        bound (string, optional): Boundary conditions, defaults to 'constant' (zero).

    Returns:
        div (torch.tensor()): Divergence (W, H, D).

    """
    if vx is None:
        vx = (1,) * 3
    if type(vx) is not torch.Tensor:
        vx = torch.tensor(vx, dtype=dat.dtype, device=dat.device)
    half = torch.tensor(0.5, dtype=dat.dtype, device=dat.device)

    if which == 'forward':
        # Pad + reflected forward difference
        w = pad(dat[0, ...], (1, 0, 0, 0, 0, 0,), mode=bound)
        w = w[:-1, :, :] - w[1:, :, :]
        h = pad(dat[1, ...], (0, 0, 1, 0, 0, 0,), mode=bound)
        h = h[:, :-1, :] - h[:, 1:, :]
        d = pad(dat[2, ...], (0, 0, 0, 0, 1, 0,), mode=bound)
        d = d[:, :, :-1] - d[:, :, 1:]
    elif which == 'backward':
        # Pad + reflected backward difference
        w = pad(dat[0, ...], (0, 1, 0, 0, 0, 0,), mode=bound)
        w = w[:-1, :, :] - w[1:, :, :]
        h = pad(dat[1, ...], (0, 0, 0, 1, 0, 0,), mode=bound)
        h = h[:, :-1, :] - h[:, 1:, :]
        d = pad(dat[2, ...], (0, 0, 0, 0, 0, 1,), mode=bound)
        d = d[:, :, :-1] - d[:, :, 1:]
    elif which == 'central':
        # Pad + reflected central difference
        w = pad(dat[0, ...], (1, 1, 0, 0, 0, 0,), mode=bound)
        w = half * (w[:-2, :, :] - w[2:, :, :])
        h = pad(dat[1, ...], (0, 0, 1, 1, 0, 0,), mode=bound)
        h = half * (h[:, :-2, :] - h[:, 2:, :])
        d = pad(dat[2, ...], (0, 0, 0, 0, 1, 1,), mode=bound)
        d = half * (d[:, :, :-2] - d[:, :, 2:])
    else:
        raise ValueError('Undefined divergence')

    return w / vx[0] + h / vx[1] + d / vx[2]


def gradient_3d(dat, vx=None, which='forward', bound='constant'):
    """ Computes the gradient of volumetric data.

    Args:
        dat (torch.tensor()): A 3D tensor (W, H, D).
        vx (tuple(float), optional): Voxel size. Defaults to (1, 1, 1).
        which (string, optional): Gradient type:
            . 'forward': Forward difference (next - centre)
            . 'backward': Backward difference (centre - previous)
            . 'central': Central difference ((next - previous)/2)
            Defaults to 'forward'.
        bound (string, optional): Boundary conditions, defaults to 'constant'.

    Returns:
          grad (torch.tensor()): Gradient (3, W, H, D).

    """
    if vx is None:
        vx = (1,) * 3
    if type(vx) is not torch.Tensor:
        vx = torch.tensor(vx, dtype=dat.dtype, device=dat.device)
    half = torch.tensor(0.5, dtype=dat.dtype, device=dat.device)

    if which == 'forward':
        # Pad + forward difference
        dat = pad(dat, (0, 1, 0, 1, 0, 1,), mode=bound)
        gw = -dat[:-1, :-1, :-1] + dat[1:, :-1, :-1]
        gh = -dat[:-1, :-1, :-1] + dat[:-1, 1:, :-1]
        gd = -dat[:-1, :-1, :-1] + dat[:-1, :-1, 1:]
    elif which == 'backward':
        # Pad + backward difference
        dat = pad(dat, (1, 0, 1, 0, 1, 0,), mode=bound)
        gw = -dat[:-1, 1:, 1:] + dat[1:, 1:, 1:]
        gh = -dat[1:, :-1, 1:] + dat[1:, 1:, 1:]
        gd = -dat[1:, 1:, :-1] + dat[1:, 1:, 1:]
    elif which == 'central':
        # Pad + central difference
        dat = pad(dat, (1, 1, 1, 1, 1, 1,), mode=bound)
        gw = half * (-dat[:-2, 1:-1, 1:-1] + dat[2:, 1:-1, 1:-1])
        gh = half * (-dat[1:-1, :-2, 1:-1] + dat[1:-1, 2:, 1:-1])
        gd = half * (-dat[1:-1, 1:-1, :-2] + dat[1:-1, 1:-1, 2:])
    else:
        raise ValueError('Undefined gradient')

    return torch.stack((gw / vx[0], gh / vx[1], gd / vx[2]), dim=0)


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


def _check_adjoint(which='central', vx=None, dtype=torch.float32,
                  dim=64, device='cpu', bound='constant'):
    """ Check adjointness of gradient and divergence operators.
        For any variables u and v, of suitable size, then with gradu = grad(u),
        divv = div(v) the following should hold: sum(gradu(:).*v(:)) - sum(u(:).*divv(:)) = 0
        (to numerical precision).

    See also:
          https://regularize.wordpress.com/2013/06/19/
          how-fast-can-you-calculate-the-gradient-of-an-image-in-matlab/

    Example:
        _check_adjoint(which='forward', dtype=torch.float64, bound='constant',
                       vx=(3.5986, 2.5564, 1.5169), dim=(32, 64, 20))

    """
    if vx is None:
        vx = (1,) * 3
    if type(vx) is not torch.Tensor:
        vx = torch.tensor(vx, dtype=dtype, device=device)
    if type(dim) is int:
        dim = (dim,) * 3

    torch.manual_seed(0)
    # Check adjointness of..
    if which == 'forward' or which == 'backward' or which == 'central':
        # ..various gradient operators
        u = torch.rand(dim[0], dim[1], dim[2], dtype=dtype, device=device)
        v = torch.rand(3, dim[0], dim[1], dim[2], dtype=dtype, device=device)
        gradu = gradient_3d(u, vx=vx, which=which, bound=bound)
        divv = divergence_3d(v, vx=vx, which=which, bound=bound)
        val = torch.sum(gradu*v, dtype=torch.float64) - torch.sum(divv*u, dtype=torch.float64)
    # Print okay? (close to zero)
    print('val={}'.format(val))


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


def padlist(x, n):
    """Repeat the last element of a list-like object to match a target length.

    If the input length is grater than ``n``, the list is cropped.

    Args:
        x (scalar or list or tuple): Input argument
        n (int): Target length

    Returns:
        x (list or tuple): Padded argument of length n.
            If the input argument is not a list or tuple, the output
            type is ``tuple``.

    """
    if not isinstance(x, list) and not isinstance(x, tuple):
        x = (x,)
    if len(x) == 0:
        raise TypeError('Input argument cannot be empty')
    return_type = type(x)
    x = list(x)
    x = x[:min(len(x), n)]
    x += [x[-1]] * (n-len(x))
    return return_type(x)


def replist(x, n, interleaved=False):
    """Replicate a list-like object.

    Args:
        x (scalar or list or tuple): Input argument
        n (int): Number of replicates
        interleaved (bool, optional): Interleaved replication.
            Default: False

    Returns:
        x (list or tuple): Replicated list
            If the input argument is not a list or tuple, the output
            type is ``tuple``.

    """
    if not isinstance(x, list) and not isinstance(x, tuple):
        x = (x,)
    if len(x) == 0:
        raise TypeError('Input argument cannot be empty')
    return_type = type(x)
    x = list(x)
    if interleaved:
        x = [elem for sub in zip(*([x]*n)) for elem in sub]
    else:
        x = x * n
    return return_type(x)


def getargs(kpd, args=[], kwargs={}, consume=False):
    """Read and remove argument from args/kwargs input.

    Args:
        kpd (list of tuple): List of (key, position, default) tuples with:
            key (str): argument name
            position (int): argument position
            default (optional): default value
        args (optional): list of positional arguments
        kwargs (optional): list of keyword arguments
        consume (bool, optional): consume arguments from args/kwargs

    Returns:
        values (list): List of values

    """

    def raise_error(key):
        import inspect
        caller = inspect.stack()[1].function
        raise TypeError("{}() got multiple values for \
                        argument '{}}'".format(caller, key))

    # Sort argument by reverse position
    kpd = [(i,) + e for i, e in enumerate(kpd)]
    kpd = sorted(kpd, key=lambda x: x[2], reverse=True)

    values = []
    for elem in kpd:
        i = elem[0]
        key = elem[1]
        position = elem[2]
        default = elem[3] if len(elem) > 3 else None

        value = default
        if len(args) >= position:
            value = args[-1]
            if consume:
                del args[-1]
            if key in kwargs.keys():
                raise_error(key)
        elif key in kwargs.keys():
            value = kwargs[key]
            if consume:
                del kwargs[key]
        values.append((i, value))

    values = [v for _, v in sorted(values)]
    return values
