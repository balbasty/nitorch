"""Convolution kernels."""

# TODO:
# . Implement Sinc kernel
# . Implement differential energies

import torch

__all__ = ['smooth']


def integrate_poly(l, h, *args):
    """Integrate a polynomial on an interval.

    k = integrate_poly(l, h, a, b, c, ...)
    integrates the polynomial a+b*x+c*x^2+... on [l,h]

    All inputs should be `torch.Tensor`
    """
    # NOTE: operations are not performed inplace (+=, *=) so that autograd
    # can backpropagated.
    zero = torch.zeros(tuple(), dtype=torch.bool)
    k = torch.zeros(l.shape, dtype=l.dtype, device=l.device)
    hh = h
    ll = l
    for i in range(len(args)):
        if torch.any(args[i] != zero):
            k = k + (args[i]/(i+1))*(hh-ll)
        hh = hh * h
        ll = ll * l
    return(k)


def gauss1d(fwhm, basis, x):
    if basis:
        return(gauss1d1(fwhm, x))
    else:
        return(gauss1d0(fwhm, x))


def rect1d(fwhm, basis, x):
    if basis:
        return(rect1d1(fwhm, x))
    else:
        return(rect1d0(fwhm, x))


def triangle1d(fwhm, basis, x):
    if basis:
        return(triangle1d1(fwhm, x))
    else:
        return(triangle1d0(fwhm, x))


def gauss1d0(w, x):
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+0.5).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 1./(sqrttwo*s)
    ker = 0.5*((w1*(x+0.5)).erf() - (w1*(x-0.5)).erf())
    ker = ker.clamp(min=0)
    return(ker, x)


def gauss1d1(w, x):
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
    return(ker, x)


def rect1d0(w, x):
    if x is None:
        lim = torch.floor((w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    ker = torch.max(torch.min(x+0.5, w/2) - torch.max(x-0.5, -w/2), zero)
    ker = ker/w
    return(ker, x)


def rect1d1(w, x):
    if x is None:
        lim = torch.floor((w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=torch.float)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-w/2, -one),   zero)
    neg_upp = torch.max(torch.min(x+w/2,  zero), -one)
    pos_low = torch.min(torch.max(x-w/2,  zero),  one)
    pos_upp = torch.max(torch.min(x+w/2,  one),   zero)
    ker = integrate_poly(neg_low, neg_upp, one,  one) \
        + integrate_poly(pos_low, pos_upp, one, -one)
    ker = ker/w
    return(ker, x)


def triangle1d0(w, x):
    if x is None:
        lim = torch.floor((2*w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=torch.float)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-0.5, -w),     zero)
    neg_upp = torch.max(torch.min(x+0.5,  zero), -w)
    pos_low = torch.min(torch.max(x-0.5,  zero),  w)
    pos_upp = torch.max(torch.min(x+0.5,  w),     zero)
    ker = integrate_poly(neg_low, neg_upp, one,  1/w) \
        + integrate_poly(pos_low, pos_upp, one, -1/w)
    ker = ker/w
    return(ker, x)


def triangle1d1(w, x):
    if x is None:
        lim = torch.floor((2*w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=torch.float)
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
    ker = integrate_poly(neg_neg_low, neg_neg_upp, 1+x/w,  1+x/w-1/w, -1/w) \
        + integrate_poly(neg_pos_low, neg_pos_upp, 1+x/w, -1-x/w-1/w,  1/w) \
        + integrate_poly(pos_neg_low, pos_neg_upp, 1-x/w,  1-x/w+1/w,  1/w) \
        + integrate_poly(pos_pos_low, pos_pos_upp, 1-x/w, -1+x/w+1/w, -1/w)
    ker = ker/w
    return(ker, x)


smooth_switcher = {
    'gauss': gauss1d,
    'rect': rect1d,
    'triangle': triangle1d,
    0: rect1d,
    1: triangle1d,
    }


def smooth(type, fwhm=1, basis=0, x=None, sep=True):
    """Create a smoothing kernel.

    Creates a (separable) smoothing kernel with fixed (i.e., not learned)
    weights. These weights are obtained by analytically convolving a
    smoothing function (e.g., Gaussian) with a basis function that encodes
    the underlying image (e.g., trilinear).
    Note that `smooth` is fully differentiable with respect to `fwhm`.
    If the kernel is evaluated at all integer coordinates from its support,
    its elements are ensured to sum to one.
    The returned kernel is a `torch.Tensor`.

    Args:
        type (str,int): Smoothing function (integrates to one).
            . 0, 'rect': Rectangular function (0th order B-spline)
            . 1, 'tri': Triangular function (1st order B-spline)
            . 'gauss': Gaussian
            . 'sinc': Sinc
        fwhm (array_like,float,optional): Full-width at half-maximum of the
            smoothing function (in voxels), in each dimension.
            Default: 1.
        basis (array_like,int,optional): Image encoding basis (B-spline order)
            Default: 0
        x (tuple,array_like,optional): Coordinates at which to evaluate the
            kernel. If None, evaluate at all integer coordinates from its
            support (truncated support for 'gauss' and 'sinc' kernels).
            Default: None
        sep(boolean): Return separable 1D kernels. If False, the 1D kernels
            are combined to form an N-D kernel.
            Default: True

    Returns:
        If `sep==False` or all input parameters are scalar: a `torch.Tensor`
        Else: a tuple of `torch.Tensor`


    """
    # Convert to tensors
    fwhm = torch.as_tensor(fwhm)
    if not fwhm.is_floating_point():
        fwhm = fwhm.type(torch.float)
    basis = torch.as_tensor(basis)
    return_tuple = True
    if not isinstance(x, tuple):
        return_tuple = not (fwhm.shape == torch.Size([]) and
                            basis.shape == torch.Size([]))
        x = (x,)
    x = tuple(torch.as_tensor(x1).flatten() if x1 is not None else None
              for x1 in x)

    # Ensure all sizes are consistant
    fwhm = fwhm.flatten()
    basis = basis.flatten()
    nker = max(fwhm.numel(), basis.numel(), len(x))
    fwhm = torch.cat((fwhm, fwhm[-1].repeat(max(0, nker-fwhm.numel()))))
    basis = torch.cat((basis, basis[-1].repeat(max(0, nker-basis.numel()))))
    x = x + (x[-1],)*max(0, nker-len(x))

    # Loop over dimensions
    ker = tuple()
    x = list(x)
    for d in range(nker):
        ker1, x[d] = smooth_switcher[type](fwhm[d], basis[d], x[d])
        shape = [1, ] * nker
        shape[d] = ker1.numel()
        ker1 = ker1.reshape(shape)
        ker += (ker1, )

    # Make N-D kernel
    if not sep:
        ker1 = ker
        ker = ker1[0]
        for d in range(1, nker):
            ker = ker * ker1[d]
    elif not return_tuple:
        ker = ker[0]

    return(ker)
