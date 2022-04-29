# -*- coding: utf-8 -*-
"""Convolution kernels.

@author: yael.balbastre@gmail.com
"""

# TODO:
# . Implement Sinc kernel
# . Use inplace operations if gradients not required

import torch
from . import utils

__all__ = ['smooth', 'energy', 'energy1d', 'energy2d', 'energy3d',
           'make_separable', 'imgrad']


def make_separable(ker, channels):
    """Transform a single-channel kernel into a multi-channel separable kernel.

    Args:
        ker (torch.tensor): Single-channel kernel (1, 1, D, H, W).
        channels (int): Number of input/output channels.

    Returns:
        ker (torch.tensor): Multi-channel group kernel (1, 1, D, H, W).

    """
    ndim = torch.as_tensor(ker.shape).numel()
    repetitions = (channels,) + (1,)*(ndim-1)
    ker = ker.repeat(repetitions)
    return ker


def _integrate_poly(l, h, *args):
    """Integrate a polynomial on an interval.

    k = _integrate_poly(l, h, a, b, c, ...)
    integrates the polynomial a+b*x+c*x^2+... on [l,h]

    All inputs should be `torch.Tensor`
    """
    # NOTE: operations are not performed inplace (+=, *=) so that autograd
    # can backpropagate.
    # TODO: (maybe) use inplace if gradients not required
    zero = torch.zeros(tuple(), dtype=torch.bool)
    k = torch.zeros(l.shape, dtype=l.dtype, device=l.device)
    hh = h
    ll = l
    for i in range(len(args)):
        if torch.any(args[i] != zero):
            k = k + (args[i]/(i+1))*(hh-ll)
        hh = hh * h
        ll = ll * l
    return k


def _dirac1d(fwhm, basis, x):
    if x is None:
        x = torch.ones(1, dtype=fwhm.dtype, device=fwhm.device)
    return (x == 1).to(x.dtype), x


def _gauss1d(fwhm, basis, x):
    if basis:
        return _gauss1d1(fwhm, x)
    else:
        return _gauss1d0(fwhm, x)


def _rect1d(fwhm, basis, x):
    if basis:
        return _rect1d1(fwhm, x)
    else:
        return _rect1d0(fwhm, x)


def _triangle1d(fwhm, basis, x):
    if basis:
        return _triangle1d1(fwhm, x)
    else:
        return _triangle1d0(fwhm, x)


def _gauss1d0(w, x):
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+0.5).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 1./(sqrttwo*s)
    ker = 0.5*((w1*(x+0.5)).erf() - (w1*(x-0.5)).erf())
    ker = ker.clamp(min=0)
    return ker, x


def _gauss1d1(w, x):
    import math
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    sqrtpi = torch.tensor(math.pi, dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+1).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 0.5*sqrttwo/s
    w2 = -0.5/s.pow(2)
    w3 = s/(sqrttwo*sqrtpi)
    ker = 0.5*((w1*(x+1)).erf()*(x+1)
               + (w1*(x-1)).erf()*(x-1)
               - 2*(w1*x).erf()*x) \
        + w3*((w2*(x+1).pow(2)).exp()
              + (w2*(x-1).pow(2)).exp()
              - 2*(w2*x.pow(2)).exp())
    ker = ker.clamp(min=0)
    return ker, x


def _rect1d0(w, x):
    if x is None:
        lim = torch.floor((w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    ker = torch.max(torch.min(x+0.5, w/2) - torch.max(x-0.5, -w/2), zero)
    ker = ker/w
    return ker, x


def _rect1d1(w, x):
    if x is None:
        lim = torch.floor((w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-w/2, -one),   zero)
    neg_upp = torch.max(torch.min(x+w/2,  zero), -one)
    pos_low = torch.min(torch.max(x-w/2,  zero),  one)
    pos_upp = torch.max(torch.min(x+w/2,  one),   zero)
    ker = _integrate_poly(neg_low, neg_upp, one,  one) \
        + _integrate_poly(pos_low, pos_upp, one, -one)
    ker = ker/w
    return ker, x


def _triangle1d0(w, x):
    if x is None:
        lim = torch.floor((2*w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-0.5, -w),     zero)
    neg_upp = torch.max(torch.min(x+0.5,  zero), -w)
    pos_low = torch.min(torch.max(x-0.5,  zero),  w)
    pos_upp = torch.max(torch.min(x+0.5,  w),     zero)
    ker = _integrate_poly(neg_low, neg_upp, one,  1/w) \
        + _integrate_poly(pos_low, pos_upp, one, -1/w)
    ker = ker/w
    return ker, x


def _triangle1d1(w, x):
    if x is None:
        lim = torch.floor((2*w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_neg_low = torch.min(torch.max(x,   -one),   zero)
    neg_neg_upp = torch.max(torch.min(x+w,  zero), -one)
    neg_pos_low = torch.min(torch.max(x,    zero),  one)
    neg_pos_upp = torch.max(torch.min(x+w,  one),   zero)
    pos_neg_low = torch.min(torch.max(x-w, -one),   zero)
    pos_neg_upp = torch.max(torch.min(x,    zero), -one)
    pos_pos_low = torch.min(torch.max(x-w,  zero),  one)
    pos_pos_upp = torch.max(torch.min(x,    one),   zero)
    ker = _integrate_poly(neg_neg_low, neg_neg_upp, 1+x/w,  1+x/w-1/w, -1/w) \
        + _integrate_poly(neg_pos_low, neg_pos_upp, 1+x/w, -1-x/w-1/w,  1/w) \
        + _integrate_poly(pos_neg_low, pos_neg_upp, 1-x/w,  1-x/w+1/w,  1/w) \
        + _integrate_poly(pos_pos_low, pos_pos_upp, 1-x/w, -1+x/w+1/w, -1/w)
    ker = ker/w
    return ker, x


_smooth_switcher = {
    'dirac': _dirac1d,
    'gauss': _gauss1d,
    'rect': _rect1d,
    'triangle': _triangle1d,
    -1: _dirac1d,
    0: _rect1d,
    1: _triangle1d,
    2: _gauss1d,
    }


def smooth(types='gauss', fwhm=1, basis=1, x=None, sep=True, dtype=None, device=None):
    """Create a smoothing kernel.

    Creates a (separable) smoothing kernel with fixed (i.e., not learned)
    weights. These weights are obtained by analytically convolving a
    smoothing function (e.g., Gaussian) with a basis function that encodes
    the underlying image (e.g., trilinear).
    Note that `smooth` is fully differentiable with respect to `fwhm`.
    If the kernel is evaluated at all integer coordinates from its support,
    its elements are ensured to sum to one.
    The returned kernel is a `torch.Tensor`.

    The returned kernel is intended for volumes ordered as (B, C, D, H, W).
    However, the fwhm elements should be ordered as (W, H, D).
    For more information about ordering conventions in nitorch, see
    `nitorch.spatial?`.

    Parameters
    ----------
    types : str or int or sequence[str or int]
        Smoothing function (integrates to one).
        - -1 or 'dirac' : Dirac function
        -  0 or 'rect'  : Rectangular function (0th order B-spline)
        -  1 or 'tri'   : Triangular function (1st order B-spline)
        -  2 or 'gauss' : Gaussian
    fwhm : int or sequence[int], default=1
        Full-width at half-maximum of the smoothing function 
        (in voxels), in each dimension.
    basis : int, default=1
        Image encoding basis (B-spline order)
    x : tuple or vector_like, optional
        Coordinates at which to evaluate the kernel. 
        If None, evaluate at all integer coordinates from its support 
        (truncated support for 'gauss').
    sep : bool, default=True
        Return separable 1D kernels. 
        If False, the 1D kernels are combined to form an N-D kernel.
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    tuple or (channel_in, channel_out, *kernel_size) tensor
        If `sep is False` or all input parameters are scalar,
        a single kernel is returned. 
        Else, a tuple of kernels is returned.


    """
    # Convert to tensors
    fwhm = torch.as_tensor(fwhm, dtype=dtype, device=device).flatten()
    if not fwhm.is_floating_point():
        fwhm = fwhm.type(torch.float)
    dtype = fwhm.dtype
    device = fwhm.device
    return_tuple = True
    if not isinstance(x, tuple):
        return_tuple = (len(fwhm.shape) > 0)
        x = (x,)
    x = tuple(torch.as_tensor(x1, dtype=dtype, device=device).flatten()
              if x1 is not None else None for x1 in x)
    if type(types) not in (list, tuple):
        types = [types]
    types = list(types)

    # Ensure all sizes are consistant
    nker = max(fwhm.numel(), len(x), len(types))
    fwhm = torch.cat((fwhm, fwhm[-1].repeat(max(0, nker-fwhm.numel()))))
    x = x + (x[-1],)*max(0, nker-len(x))
    types += (types[-1],)*max(0, nker-len(types))

    # Loop over dimensions
    ker = tuple()
    x = list(x)
    for d in range(nker):
        ker1, x[d] = _smooth_switcher[types[d]](fwhm[d], basis, x[d])
        shape = [1, ] * nker
        shape[d] = ker1.numel()
        ker1 = ker1.reshape(shape)
        ker1 = ker1[None, None, ...]  # Cout = 1, Cin = 1
        ker += (ker1, )

    # Make N-D kernel
    if not sep:
        ker1 = ker
        ker = ker1[0]
        for d in range(1, nker):
            ker = ker * ker1[d]
    elif not return_tuple:
        ker = ker[0]

    return ker


def energy(dim, absolute=0, membrane=0, bending=0, lame=(0, 0), vs=None,
           displacement=False, dtype=None, device=None):
    """Generate a convolution kernel for a mixture of differential energies.

    This function builds a convolution kernel that embeds a mixture of
    differential energies. In practice, this energy can be computed as
    E = <f,k*f>, where f is the image, k is the kernel and <.,.> is the
    Eucldean dot product.

    Possible energies are:
        . absolute = sum of squared absolute values
        . membrane = sum of squared first derivatives
        . bending  = sum of squared second derivatives (diagonal terms only)
        . lame     = linear elastic energy
            [0] sum of divergences  (Lame's 1st parameter, lambda)
            [1] sum of shears       (Lame's 2nd parameter, mu)
    Note: The lame parameters should be entered in the opposite order from SPM
          SPM: (mu, lambda) / nitorch: (lambda, mu)

    The returned kernel is intended for volumes ordered as (B, C, W, H, D).
    For more information about ordering conventions in nitorch, see
    `nitorch.spatial?`.

    The returned kernel is ordered as (C, C, W, H, D).
    If displacement is False, C == 1. Use `make_separable` to transform it
    into a multi-channel kernel.
    If displacement is True, C == dim and components (i.e., channels)
    are ordered as (W, H, D).

    Args:
        dim (int): Dimension of the problem (1, 2, or 3)
        absolute (float, optional): Defaults to 0.
        membrane (float, optional): Defaults to 0.
        bending (float, optional): Defaults to 0.
        lame (tuple[float], optional): Defaults to (0, 0).
        vs (tuple[float], optional): Defaults to ones.
        displacement (bool, optional): True if input field is a displacement
            field. Defaults to True if `linearelastic != (0, 0)`, else False.
        dtype (torch.dtype, optional)
        device (torch.device, optional)

    Returns:
        ker (torch.Tensor): Kernel weights.

    """
    # Check arguments
    absolute = torch.as_tensor(absolute)
    if absolute.numel() != 1:
        raise ValueError('The absolute energy must be parameterised by '
                         'exactly one parameter. Received {}.'
                         .format(absolute.numel()))
    absolute = absolute.flatten()[0]
    membrane = torch.as_tensor(membrane)
    if membrane.numel() != 1:
        raise ValueError('The membrane energy must be parameterised by '
                         'exactly one parameter. Received {}.'
                         .format(membrane.numel()))
    membrane = membrane.flatten()[0]
    bending = torch.as_tensor(bending)
    if bending.numel() != 1:
        raise ValueError('The bending energy must be parameterised by '
                         'exactly one parameter. Received {}.'
                         .format(bending.numel()))
    bending = bending.flatten()[0]
    lame = torch.as_tensor(lame)
    if lame.numel() != 2:
        raise ValueError('The linear-elastic energy must be parameterised by '
                         'exactly two parameters. Received {}.'
                         .format(lame.numel()))
    lame = lame.flatten()

    if (lame != 0).any():
        displacement = True

    # Kernel size
    if bending != 0:
        kdim = 5
    elif membrane != 0 or (lame != 0).any():
        kdim = 3
    elif absolute != 0:
        kdim = 1
    else:
        kdim = 0

    # Compute 1/vs^2
    if vs is None:
        vs = (1,) * dim
    else:
        if len(vs) != dim:
            raise ValueError('There must be as many voxel sizes as dimensions')
    vs = tuple(1./(v*v) for v in vs)

    # Accumulate energies
    ker = torch.zeros((1, 1) + (kdim,)*dim, dtype=dtype, device=device)
    if absolute != 0:
        kpad = ((kdim-1)//2,)*dim
        ker1 = _energy_absolute(dim, vs, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        if displacement:
            ker2 = ker1
            ker1 = torch.zeros((dim, dim) + (kdim,)*dim,
                               dtype=dtype, device=device)
            for d in range(dim):
                ker1[d, d, ...] = ker2/vs[d]
        ker = ker + absolute*ker1

    if membrane != 0:
        kpad = ((kdim-3)//2,)*dim
        ker1 = _energy_membrane(dim, vs, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        if displacement:
            ker2 = ker1
            ker1 = torch.zeros((dim, dim) + (kdim,)*dim,
                               dtype=dtype, device=device)
            for d in range(dim):
                ker1[d, d, ...] = ker2/vs[d]
        ker = ker + membrane*ker1

    if bending != 0:
        ker1 = _energy_bending(dim, vs, dtype, device)
        if displacement:
            ker2 = ker1
            ker1 = torch.zeros((dim, dim) + (kdim,)*dim,
                               dtype=dtype, device=device)
            for d in range(dim):
                ker1[d, d, ...] = ker2/vs[d]
        ker = ker + bending*ker1

    if lame[0] != 0:
        kpad = ((kdim-3)//2,)*dim
        ker1 = _energy_linearelastic(dim, vs, 1, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        ker = ker + lame[0]*ker1

    if lame[1] != 0:
        kpad = ((kdim-3)//2,)*dim
        ker1 = _energy_linearelastic(dim, vs, 2, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        ker = ker + lame[1]*ker1

    return ker


def energy1d(*args, **kwargs):
    """energy(1, absolute, membrane, bending, linearelastic, vs)."""
    return(energy(1, *args, **kwargs))


def energy2d(*args, **kwargs):
    """energy(2, absolute, membrane, bending, linearelastic, vs)."""
    return(energy(2, *args, **kwargs))


def energy3d(*args, **kwargs):
    """energy(3, absolute, membrane, bending, linearelastic, vs)."""
    return(energy(3, *args, **kwargs))


def _energy_absolute(dim, vs, dtype=torch.float, device='cpu'):
    ker = torch.ones(1, dtype=dtype, device=device)
    ker = ker.reshape((1,)*dim)
    ker = ker[None, None, ...]
    return ker


def _energy_membrane(dim, vs, dtype=torch.float, device='cpu'):
    ker = torch.zeros((3,)*dim, dtype=dtype, device=device)
    ker[(1,)*dim] = 2.*sum(vs)
    for d in range(dim):
        ker[tuple(0 if i == d else 1 for i in range(dim))] = -vs[d]
        ker[tuple(2 if i == d else 1 for i in range(dim))] = -vs[d]
    ker = ker[None, None, ...]
    return ker


def _energy_bending(dim, vs, dtype=torch.float, device='cpu'):
    centre = 6.*sum(v*v for v in vs)
    for d in range(dim):
        for dd in range(d+1, dim):
            centre = centre + 8.*vs[d]*vs[dd]
    ker = torch.zeros((5,)*dim, dtype=dtype, device=device)
    ker[(2,)*dim] = centre
    for d in range(dim):
        ker[tuple(1 if i == d else 2 for i in range(dim))] = -4.*vs[d]*sum(vs)
        ker[tuple(3 if i == d else 2 for i in range(dim))] = -4.*vs[d]*sum(vs)
        ker[tuple(0 if i == d else 2 for i in range(dim))] = vs[d]*vs[d]
        ker[tuple(4 if i == d else 2 for i in range(dim))] = vs[d]*vs[d]
        for dd in range(d+1, dim):
            ker[tuple(1 if i in (d, dd) else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
            ker[tuple(3 if i in (d, dd) else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
            ker[tuple(1 if i == d else 3 if i == dd else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
            ker[tuple(3 if i == d else 1 if i == dd else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
    ker = ker[None, None, ...]
    return ker


def _energy_linearelastic(dim, vs, lame, dtype=torch.float, device='cpu'):
    ker = torch.zeros((dim, dim) + (3,)*dim, dtype=dtype, device=device)
    for d in range(dim):
        ker[(d, d) + (1,)*dim] = 2.
        if lame == 2:
            ker[(d, d) + (1,)*dim] += 2.*sum(vs)/vs[d]
            for dd in range(dim):
                ker[(d, d) + tuple(0 if i == dd else 1 for i in range(dim))] \
                    = -vs[dd]/vs[d] if d != dd else -2.
                ker[(d, d) + tuple(2 if i == dd else 1 for i in range(dim))] \
                    = -vs[dd]/vs[d] if d != dd else -2.
        else:
            ker[(d, d) + tuple(0 if i == d else 1 for i in range(dim))] \
                = -1.
            ker[(d, d) + tuple(2 if i == d else 1 for i in range(dim))] \
                = -1.
        for dd in range(d+1, dim):
            ker[(d, dd) + tuple(0 if i in (d, dd) else 1
                                for i in range(dim))] = -0.25
            ker[(d, dd) + tuple(2 if i in (d, dd) else 1
                                for i in range(dim))] = -0.25
            ker[(d, dd) + tuple(0 if i == d else 2 if i == dd else 1
                                for i in range(dim))] = 0.25
            ker[(d, dd) + tuple(2 if i == d else 0 if i == dd else 1
                                for i in range(dim))] = 0.25
            ker[dd, d, ...] = ker[d, dd, ...]
    return ker


def imgrad(dim, vs=None, which='central', dtype=None, device=None):
    """Kernel that computes the first order gradients of a tensor.

    The returned kernel is intended for volumes ordered as (B, C, W, H, D).
    For more information about ordering conventions in nitorch, see
    `nitorch.spatial?`.

    The returned kernel is ordered as (C, 1, W, H, D), and the output
    components (i.e., channels) are ordered as (W, H, D).

    Args:
        dim (int): Dimension.
        vs (tuple[float], optional): Voxel size. Defaults to 1.
        which (str, optional): Gradient types (one or more):
            . 'forward': forward gradients (next - centre)
            . 'backward': backward gradients (centre - previous)
            . 'central': central gradients ((next - previous)/2)
            Defaults to 'central'.
        dtype (torch.dtype, optional): Data type. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.

    Returns:
        ker (torch.tensor): Kernel that can be used to extract image gradients
            (dim*len(which), 1, W, H, D)

    """
    if vs is None:
        vs = (1,) * dim
    elif len(vs) != dim:
        raise ValueError('There must be as many voxel sizes as dimensions')
    coord = (0, 2) if which == 'central' else \
            (1, 2) if which == 'forward' else \
            (0, 1)
    ker = torch.zeros((dim, 1) + (3,) * dim, dtype=dtype, device=device)
    for d in range(dim):
        sub = tuple(coord[0] if dd == d else 1 for dd in range(0, dim, -1))
        ker[(d, 0) + sub] = -1./(vs[d]*(coord[1]-coord[0]))
        sub = tuple(coord[1] if dd == d else 1 for dd in range(0, dim, -1))
        ker[(d, 0) + sub] = 1./(vs[d]*(coord[1]-coord[0]))
    ker = ker.reshape((dim, 1) + (3,)*dim)
    return ker


def greens(ker, shape, bound='circular'):
    """Compute the Greens function of a real and symmetric convolution.

    The convolution kernel is inverted in Frequency domain.


    Args:
        ker (array_like): Input kernel (Cout, Cin, Kw, Kh, Kd).
        shape (vector_like): Shape of the convolved image [W, H, D].
        bound (string, optional): Boundary conditions. Defaults to 'circular'.

    Returns:
        greens (array_like): Fourier (or other frequency) transform of the
            Greens function (Cout, Cin, W, H, D).

    (Adapted from John Ashburner's `spm_shoot_greens`)

    """
    if bound not in ('circular', 'fft'):
        raise ValueError('Only circular boundary condition is implemented.')
    ker = torch.as_tensor(ker)
    kdim = torch.as_tensor(ker.size()[2:])
    shape = torch.as_tensor(shape).flatten()
    ndim = max(kdim.numel(), shape.numel())
    kdim = utils.pad(kdim, ndim-kdim.numel(), side='post', value=1)
    shape = utils.pad(shape, ndim-shape.numel(), side='post', value=1)

    # TODO: I should do a bit more shape checking

    # Frequency transform
    pad = (shape-kdim)/2.
    pad = tuple(p for pair in zip(pad.floor(), pad.ceil()) for p in pair)
    ker = utils.pad(ker, pad)
    # TODO: fftshift
    ker = ker.rfft(ndim)

    # Note: PyTorch does not handle complex numbers (yet).
    # The real and imaginary part are therefore two dimensions of the
    # output tensor.
    # Since our kernels are symmetric, we know that the Fourier coefficients
    # are purely real. However, I am not sure that removing a dimensions
    # and then adding it is very efficient.
    # To make sure that the imaginary component is zero (while keeping
    # everything differentiable), I multiply it with zeros.
    mask = torch.tensor([1., 0.], device=ker.device)
    mask = utils.shiftdim(mask, 1-ker.dim())
    ker = ker * mask

    if ker.size()[0] == 1:
        # Pointwise inverse
        ker = ker + (1-mask)
        ker = 1./ker
        ker = ker * mask
    elif ndim == 2:
        # 2x2 block inverse
        det = ker[0, 0, ...]*ker[1, 1, ...] - ker[0, 1, ...]*ker[1, 0, ...]
        det = det[None, None, ...]
        det = det + (1-mask)
        ker = torch.cat((
                torch.cat((ker[None, None, 1, 1, ...],
                           -ker[None, None, 0, 1, ...]), dim=1),
                torch.cat((-ker[None, None, 1, 0, ...],
                           ker[None, None, 0, 0, ...]), dim=1),
            ), dim=0)
        ker = ker / det
    else:
        # 3x3 block inverse
        det = ker[0, 0, ...]*(ker[1, 1, ...]*ker[2, 2, ...] -
                              ker[1, 2, ...]*ker[2, 1, ...]) \
            + ker[0, 1, ...]*(ker[1, 2, ...]*ker[2, 0, ...] -
                              ker[1, 0, ...]*ker[2, 2, ...]) \
            + ker[0, 2, ...]*(ker[1, 0, ...]*ker[2, 1, ...] -
                              ker[1, 1, ...]*ker[2, 0, ...])
        det = det[None, None, ...]
        det = det + (1-mask)

        ker = torch.cat((
                torch.cat((
                    ker[None, None, 1, 1, ...]*ker[None, None, 2, 2, ...] -
                    ker[None, None, 1, 2, ...]*ker[None, None, 2, 1, ...],
                    ker[None, None, 1, 2, ...]*ker[None, None, 2, 0, ...] -
                    ker[None, None, 1, 0, ...]*ker[None, None, 2, 2, ...],
                    ker[None, None, 1, 0, ...]*ker[None, None, 2, 1, ...] -
                    ker[None, None, 1, 1, ...]*ker[None, None, 2, 0, ...]
                ), dim=0),
                torch.cat((
                    ker[None, None, 0, 2, ...]*ker[None, None, 2, 1, ...] -
                    ker[None, None, 0, 1, ...]*ker[None, None, 2, 2, ...],
                    ker[None, None, 0, 0, ...]*ker[None, None, 2, 2, ...] -
                    ker[None, None, 0, 2, ...]*ker[None, None, 2, 0, ...],
                    ker[None, None, 0, 1, ...]*ker[None, None, 2, 0, ...] -
                    ker[None, None, 0, 0, ...]*ker[None, None, 2, 1, ...]
                ), dim=0),
                torch.cat((
                    ker[None, None, 0, 1, ...]*ker[None, None, 1, 2, ...] -
                    ker[None, None, 0, 2, ...]*ker[None, None, 1, 1, ...],
                    ker[None, None, 0, 2, ...]*ker[None, None, 1, 0, ...] -
                    ker[None, None, 0, 0, ...]*ker[None, None, 1, 2, ...],
                    ker[None, None, 0, 0, ...]*ker[None, None, 1, 1, ...] -
                    ker[None, None, 0, 1, ...]*ker[None, None, 1, 0, ...]
                ), dim=0)
            ), dim=1)
        ker = ker / det

    return ker
