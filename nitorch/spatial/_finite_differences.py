"""Finite-differences operators (gradient, divergence, ...)."""
import itertools

import torch
from nitorch.core import utils, linalg
from nitorch.core.utils import fast_slice_tensor, same_storage, make_vector
from nitorch.core.py import make_list, ensure_list
from ._conv import smooth
import math as pymath
import itertools


__all__ = ['im_divergence', 'im_gradient', 'diff1d', 'diff', 'div1d', 'div',
           'mind', 'rmind', 'sobel', 'frangi', 'sato', 'hessian_eig']


# Converts from nitorch.utils.pad boundary naming to
# nitorch.spatial._grid naming
_bound_converter = {
    'circular': 'circular',
    'reflect': 'reflect',
    'reflect1': 'reflect1',
    'reflect2': 'reflect2',
    'replicate': 'replicate',
    'constant': 'constant',
    'zero': 'constant',
    'dct1': 'reflect',
    'dct2': 'reflect2',
    }


def diff1d(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2', out=None):
    """Finite differences along a dimension.

    Parameters
    ----------
    x : tensor
        Input tensor
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int, default=-1
        Dimension along which to compute finite differences.
    voxel_size : float
        Unit size used in the denominator of the gradient.
    side : {'c', 'f', 'b'}, default='c'
        * 'c': central finite differences
        * 'f': forward finite differences
        * 'b': backward finite differences
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'}, default='dct2'
        Boundary condition.

    Returns
    -------
    diff : tensor
        Tensor of finite differences, with same shape as the input tensor.

    """
    def subto(x, y, out):
        """Smart sub"""
        if (getattr(x, 'requires_grad', False) or
                getattr(y, 'requires_grad', False)):
            return out.copy_(x).sub_(y)
        else:
            return torch.sub(x, y, out=out)

    def addto(x, y, out):
        if (getattr(x, 'requires_grad', False) or
                getattr(y, 'requires_grad', False)):
            return out.copy_(x).add_(y)
        else:
            return torch.add(x, y, out=out)

    # TODO:
    #   - check high order central

    # check options
    bound = bound.lower()
    if bound not in ('dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'):
        raise ValueError('Unknown boundary type {}.'.format(bound))
    side = side.lower()[0]
    if side not in ('f', 'b', 'c'):
        raise ValueError('Unknown side {}.'.format(side))
    order = int(order)
    if order < 0:
        raise ValueError('Order must be nonnegative but got {}.'.format(order))
    elif order == 0:
        return x

    # ensure tensor
    x = torch.as_tensor(x)

    if x.shape[dim] == 1:
        if out is not None:
            return out.view(x.shape).copy_(x)
        else:
            return x.clone()

    shape0 = x.shape
    x = x.transpose(0, dim)
    if out is not None:
        out = out.view(shape0)
        out = out.transpose(0, dim)

    if order == 1:

        if out is None:
            diff = torch.empty_like(x)
        else:
            diff = out

        if side == 'f':  # forward -> diff[i] = x[i+1] - x[i]
            subto(x[1:], x[:-1], out=diff[:-1])
            if bound in ('dct2', 'replicate'):
                # x[end+1] = x[end] => diff[end] = 0
                diff[-1].zero_()
            elif bound == 'dct1':
                # x[end+1] = x[end-1] => diff[end] = -diff[end-1]
                diff[-1] = diff[-2]
                diff[-1].neg_()
            elif bound == 'dst2':
                # x[end+1] = -x[end] => diff[end] = -2*x[end]
                diff[-1] = x[-1]
                diff[-1].mul_(-2)
            elif bound in ('dst1', 'zero'):
                # x[end+1] = 0 => diff[end] = -x[end]
                diff[-1] = x[-1]
                diff[-1].neg_()
            else:
                assert bound == 'dft'
                # x[end+1] = x[0] => diff[end] = x[0] - x[end]
                subto(x[0], x[-1], out=diff[-1])

        elif side == 'b':  # backward -> diff[i] = x[i] - x[i-1]
            subto(x[1:], x[:-1], out=diff[1:])
            if bound in ('dct2', 'replicate'):
                # x[-1] = x[0] => diff[0] = 0
                diff[0].zero_()
            elif bound == 'dct1':
                # x[-1] = x[1] => diff[0] = -diff[1]
                diff[0] = diff[1]
                diff[0].neg_()
            elif bound == 'dst2':
                # x[-1] = -x[0] => diff[0] = 2*x[0]
                diff[0] = x[0]
                diff[0].mul_(2)
            elif bound in ('dst1', 'zero'):
                # x[-1] = 0 => diff[0] = x[0]
                diff[0] = x[0]
            else:
                assert bound == 'dft'
                # x[-1] = x[end] => diff[0] = x[0] - x[end]
                subto(x[0], x[-1], out=diff[0])

        elif side == 'c':  # central -> diff[i] = (x[i+1] - x[i-1])/2
            subto(x[2:], x[:-2], out=diff[1:-1])
            if bound in ('dct2', 'replicate'):
                subto(x[1], x[0], out=diff[0])
                subto(x[-1], x[-2], out=diff[-1])
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                diff[0].zero_()
                diff[-1].zero_()
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                addto(x[1], x[0], out=diff[0])
                addto(x[-1], x[-2], out=diff[-1]).neg_()
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                diff[0] = x[1]
                diff[-1] = x[-2]
                diff[-1].neg_()
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                subto(x[1], x[-1], out=diff[0])
                subto(x[0], x[-2], out=diff[-1])
        if side == 'c':
            if voxel_size != 1:
                diff = diff.div_(voxel_size * 2)
            else:
                diff = diff.div_(2.)
        elif voxel_size != 1:
            diff = diff.div_(voxel_size)

    elif side == 'c':
        # we must deal with central differences differently:
        # -> second order differences are exact but first order
        #    differences are approximated (we should sample between
        #    two voxels so we must interpolate)
        # -> for order > 2, we compose as many second order differences
        #    as possible and then (eventually) deal with the remaining
        #    1st order using the approximate implementation.
        if order == 2:
            fwd = diff1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                         side='f', bound=bound, out=out)
            bwd = diff1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                         side='b', bound=bound)
            diff = fwd.sub_(bwd)
            if voxel_size != 1:
                diff = diff.div_(voxel_size)
        else:
            diff = diff1d(x, order=2, dim=0, voxel_size=voxel_size,
                          side=side, bound=bound)
            diff = diff1d(diff, order=order-2, dim=0, voxel_size=voxel_size,
                          side=side, bound=bound, out=out)

    else:
        diff = diff1d(x, order=1, dim=0, voxel_size=voxel_size,
                      side=side, bound=bound)
        diff = diff1d(diff, order=order-1, dim=0, voxel_size=voxel_size,
                      side=side, bound=bound, out=out)

    diff = diff.transpose(0, dim)
    if out is not None and not utils.same_storage(out, diff):
        out = out.view(diff.shape).copy_(diff)
        diff = out
    return diff


def diff(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2', out=None):
    """Finite differences.

    Parameters
    ----------
    x : tensor
        Input tensor
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int or sequence[int], default=-1
        Dimension along which to compute finite differences.
    voxel_size : float or sequence[float], default=1
        Unit size used in the denominator of the gradient.
    side : {'c', 'f', 'b'}, default='c'
        * 'c': central finite differences
        * 'f': forward finite differences
        * 'b': backward finite differences
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'repeat', 'zero'}, default='dct2'
        Boundary condition.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    diff : tensor
        Tensor of finite differences, with same shape as the input tensor,
        with an additional dimension if any of the input arguments is a list.

    """
    # find number of dimensions
    dim = make_vector(dim, dtype=torch.long)
    voxel_size = make_vector(voxel_size)
    drop_last = dim.dim() == 0 and voxel_size.dim() == 0
    dim = dim.tolist()
    voxel_size = voxel_size.tolist()
    nb_dim = max(len(dim), len(voxel_size))
    dim = ensure_list(dim, nb_dim)
    voxel_size = ensure_list(voxel_size, nb_dim)

    # compute diffs in each dimension
    if out is not None:
        diffs = out.view([*x.shape, nb_dim])
        diffs = utils.fast_movedim(diffs, -1, 0)
    else:
        diffs = x.new_empty([nb_dim, *x.shape])
    # ^ ensures that sliced dim is the least rapidly changing one
    for i, (d, v) in enumerate(zip(dim, voxel_size)):
        diff1d(x, order, d, v, side, bound, out=diffs[i])
    diffs = utils.fast_movedim(diffs, 0, -1)

    # return
    if drop_last:
        diffs = diffs.squeeze(-1)
    return diffs


def div1d(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2', out=None):
    """Divergence along a dimension.

    Notes
    -----
    The divergence if the adjoint to the finite difference.
    This can be checked by:
    ```python
    >>> import torch
    >>> from nitorch.spatial import diff1d, div1d
    >>> u = torch.randn([64, 64, 64], dtype=torch.double)
    >>> v = torch.randn([64, 64, 64], dtype=torch.double)
    >>> Lu = diff1d(u, side=side, bound=bound)
    >>> Kv = div1d(v, side=side, bound=bound)
    >>> assert torch.allclose((v*Lu).sum(), (u*Kv).sum())
    ```

    Parameters
    ----------
    x : tensor
        Input tensor
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int, default=-1
        Dimension along which to compute finite differences.
    voxel_size : float
        Unit size used in the denominator of the gradient.
    side : {'f', 'b'}, default='f'
        * 'f': forward finite differences
        * 'b': backward finite differences
      [ * 'c': central finite differences ] => NotImplemented
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'}, default='dct2'
        Boundary condition.

    Returns
    -------
    div : tensor
        Divergence tensor, with same shape as the input tensor.

    """
    def subto(x, y, out):
        """Smart sub"""
        if ((torch.is_tensor(x) and x.requires_grad) or
                (torch.is_tensor(y) and y.requires_grad)):
            return out.copy_(y).neg_().add_(x)
        else:
            return torch.sub(x, y, out=out)

    def addto(x, y, out):
        """Smart add"""
        if ((torch.is_tensor(x) and x.requires_grad) or
                (torch.is_tensor(y) and y.requires_grad)):
            return out.copy_(y).add_(x)
        else:
            return torch.add(x, y, out=out)

    def div_(x, y):
        """Smart in-place division"""
        # It seems that in-place divisions do not break gradients...
        return x.div_(y)
        # if ((torch.is_tensor(x) and x.requires_grad) or
        #         (torch.is_tensor(y) and y.requires_grad)):
        #     return x / y
        # else:
        #     return x.div_(y)

    # check options
    bound = bound.lower()
    if bound not in ('dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'):
        raise ValueError('Unknown boundary type {}.'.format(bound))
    side = side.lower()[0]
    if side not in ('f', 'b', 'c'):
        raise ValueError('Unknown side {}.'.format(side))
    order = int(order)
    if order < 0:
        raise ValueError('Order must be nonnegative but got {}.'.format(order))
    elif order == 0:
        return x

    # ensure tensor
    x = torch.as_tensor(x)

    if x.shape[dim] == 1:
        if out is not None:
            return out.view(x.shape).copy_(x)
        else:
            return x.clone()

    shape0 = x.shape
    x = x.transpose(0, dim)
    if out is not None:
        out = out.view(shape0)
        out = out.transpose(0, dim)

    if order == 1:

        if out is None:
            div = torch.empty_like(x)
        else:
            div = out

        if side == 'f':
            # forward -> diff[i] = x[i+1] - x[i]
            #            div[i]  = x[i-1] - x[i]
            subto(x[:-1], x[1:], out=div[1:])
            div[0] = x[0]
            div[0].neg_()
            if bound in ('dct2', 'replicate'):
                div[-1] = x[-2]
            elif bound == 'dct1':
                div[-2] += x[-1]
            elif bound == 'dst2':
                div[-1] -= x[-1]
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                subto(x[-1], x[0], out=div[0])

        elif side == 'b':
            # backward -> diff[i] = x[i] - x[i-1]
            #             div[i]  = x[i+1] - x[i]
            subto(x[:-1], x[1:], out=div[:-1])
            div[-1] = x[-1]
            if bound in ('dct2', 'replicate'):
                div[0] = x[1]
                div[0].neg_()
            elif bound == 'dct1':
                div[1] -= x[0]
            elif bound == 'dst2':
                div[0] += x[0]
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                subto(x[-1], x[0], out=div[-1])

        else:
            assert side == 'c'
            # central -> diff[i] = (x[i+1] - x[i-1])/2
            #         -> div[i]  = (x[i-1] - x[i+1])/2
            subto(x[:-2], x[2:], out=div[1:-1])
            if bound in ('dct2', 'replicate'):
                # x[-1]    = x[0]   => diff[0]   = x[1] - x[0]
                # x[end+1] = x[end] => diff[end] = x[end] - x[end-1]
                addto(x[0], x[1], out=div[0]).neg_()
                addto(x[-1], x[-2], out=div[-1])
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                div[0] = x[1]
                div[0].neg_()
                div[1] = x[2]
                div[1].neg_()
                div[-2] = x[-3]
                div[-1] = x[-2]
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                subto(x[0], x[1], out=div[0])
                subto(x[-2], x[-1], out=div[-1])
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                div[0] = x[1]
                div[0].neg_()
                div[-1] = x[-2]
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                subto(x[-1], x[1], out=div[0])
                subto(x[-2], x[0], out=div[-1])

        if side == 'c':
            if voxel_size != 1:
                div = div_(div, voxel_size * 2)
            else:
                div = div_(div, 2.)
        elif voxel_size != 1:
            div = div_(div, voxel_size)

    elif side == 'c':
        # we must deal with central differences differently:
        # -> second order differences are exact but first order
        #    differences are approximated (we should sample between
        #    two voxels so we must interpolate)
        # -> for order > 2, we take the reverse order to what's done
        #    in `diff`: we start with a first-order difference if
        #    the order is odd, and then unroll all remaining second-order
        #    differences.
        if order == 2:
            # exact implementation
            # (I use the forward and backward implementations to save
            #  code, but it could be reimplemented exactly to
            #  save speed)
            fwd = div1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                        side='f', bound=bound, out=out)
            bwd = div1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                        side='b', bound=bound)
            div = fwd.sub_(bwd)
            if voxel_size != 1:
                div = div_(div, voxel_size)
        elif order % 2:
            # odd order -> start with a first order
            div = div1d(x, order=1, dim=0, voxel_size=voxel_size,
                        side=side, bound=bound)
            div = div1d(div, order=order-1, dim=0, voxel_size=voxel_size,
                        side=side, bound=bound, out=out)
        else:
            # even order -> recursive call
            div = div1d(x, order=2, dim=0, voxel_size=voxel_size,
                          side=side, bound=bound)
            div = div1d(div, order=order-2, dim=0, voxel_size=voxel_size,
                          side=side, bound=bound, out=out)

    else:
        div = div1d(x, order=order-1, dim=0, voxel_size=voxel_size,
                    side=side, bound=bound)
        div = div1d(div, order=1, dim=0, voxel_size=voxel_size,
                    side=side, bound=bound, out=out)

    div = div.transpose(0, dim)
    if out is not None and not utils.same_storage(out, div):
        out = out.view(div.shape).copy_(div)
        div = out
    return div


def div(x, order=1, dim=-1, voxel_size=1, side='f', bound='dct2', out=None):
    """Divergence.

    Parameters
    ----------
    x : (*shape, [L]) tensor
        Input tensor
        If `dim` or `voxel_size` is a list, the last dimension of `x`
        must have the same size as their length.
    order : int, default=1
        Finite difference order (1=first derivative, 2=second derivative, ...)
    dim : int or sequence[int], default=-1
        Dimension along which finite differences were computed.
    voxel_size : float or sequence[float], default=1
        Unit size used in the denominator of the gradient.
    side : {'f', 'b'}, default='f'
        * 'f': forward finite differences
        * 'b': backward finite differences
      [ * 'c': central finite differences ] => NotImplemented
    bound : {'dct2', 'dct1', 'dst2', 'dst1', 'dft', 'repeat', 'zero'}, default='dct2'
        Boundary condition.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    div : (*shape) tensor
        Divergence tensor, with same shape as the input tensor, minus the
        (eventual) difference dimension.

    """
    x = torch.as_tensor(x)

    # find number of dimensions
    dim = torch.as_tensor(dim)
    voxel_size = make_vector(voxel_size)
    has_last = (dim.dim() > 0 or voxel_size.dim() > 0)
    dim = dim.tolist()
    voxel_size = voxel_size.tolist()
    nb_dim = max(len(dim), len(voxel_size))
    dim = ensure_list(dim, nb_dim)
    voxel_size = ensure_list(voxel_size, nb_dim)

    if has_last and x.shape[-1] != nb_dim:
        raise ValueError('Last dimension of `x` should be {} but got {}'
                         .format(nb_dim, x.shape[-1]))
    if not has_last:
        x = x[..., None]

    # compute divergence in each dimension
    div = out.view(x.shape[:-1]).zero_() if out is not None else 0
    tmp = torch.zeros_like(x[..., 0])
    for diff, d, v in zip(x.unbind(-1), dim, voxel_size):
        div += div1d(diff, order, d, v, side, bound, out=tmp)

    return div


def mind(x, radius=0, fwhm=1, dim=None, bound='dct2', robust=3, norm='max'):
    """Modality-Independent Neighborhood Descriptor

    Parameters
    ----------
    x : (..., *shape) tensor
        Input tensor
    radius : int,
        Half-width of spatial search
    fwhm : float
        Full-width at half-maximum of Gaussian smoothing
    dim : int, default=x.dim()
        Number of spatial dimensions
    bound : boundary-like
        Boundary condition
    robust : int
        Power of robustness threshold (smaller = more robust, 0 = disabled)
    norm : {'max', 'mean', None}
        Normalization across features

    Returns
    -------
    m : (..., *shape, K) tensor
        MIND feature maps

    References
    ----------
    "MIND: Modality Independent Neighbourhood Descriptor for Multi-Modal
    Deformable Registration"
    M.P. Heinrich et al.
    Medical Image Analysis (2012)
    """
    dim = dim or x.dim()
    fwhm = make_list(fwhm, dim)

    inplace = not x.requires_grad
    square_ = torch.square_ if inplace else torch.square
    div_ = torch.Tensor.div_ if inplace else torch.div
    exp_ = torch.Tensor.exp_ if inplace else torch.exp

    nb_feat = 2*dim if radius == 0 else ((2*radius+1)**dim - 1)
    mnd = x.new_zeros([nb_feat, *x.shape])

    # compute differences
    n_feat = 0
    # first ring neighbors
    for d in range(-dim, 0):
        for side in 'fb':
            diff1d(x, dim=d, side=side, bound=bound, out=mnd[n_feat])
            n_feat += 1
    # other neighbors
    mnd[n_feat:].copy_(x)
    if radius > 0:
        indices = itertools.product(*[range(-radius, radius+1)] * dim)
        for index in indices:
            if sum(map(abs, index)) <= 1:
                # center of first-ring neighbor -> skip
                continue
            mnd[n_feat] -= utils.roll(x, index, range(-dim, 0), bound=bound)
            n_feat += 1
    # square and smooth
    mnd = square_(mnd)
    mnd = smooth(mnd, fwhm=fwhm, dim=dim, bound=bound)

    # compute variance in first-ring neighborhood
    var = mnd[:2*dim].mean(0)
    if robust:
        meanvar = var.mean(list(range(-3, 0)), keepdim=True)
        mn = 10 ** (-robust) * meanvar
        mx = 10 ** robust * meanvar
        var = torch.min(torch.max(var, mn), mx)

    # exponentiation
    mnd = exp_(div_(mnd, var).neg_())

    # normalize
    if norm == 'max':
        mnd /= mnd.max(0).values
    elif norm == 'mean':
        mnd /= mnd.mean(0)

    return utils.movedim(mnd, 0, -1)


def rmind(x, radius=2, fwhm=1, dim=None, bound='dct2', robust=3, norm='max'):
    """Rotationally-Invariant Modality-Independent Neighborhood Descriptor

    Parameters
    ----------
    x : (..., *shape) tensor
        Input tensor
    radius : float,
        Radius of spatial search
    fwhm : float
        Full-width at half-maximum of Gaussian smoothing
    dim : int, default=x.dim()
        Number of spatial dimensions
    bound : boundary-like
        Boundary condition
    robust : int
        Power of robustness threshold (smaller = more robust, 0 = disabled)
    norm : {'max', 'mean', None}
        Normalization across features

    Returns
    -------
    m : (..., *shape, K) tensor
        MIND feature maps

    References
    ----------
    "MIND: Modality Independent Neighbourhood Descriptor for Multi-Modal
    Deformable Registration"
    M.P. Heinrich et al.
    Medical Image Analysis (2012)
    """
    dim = dim or x.dim()
    fwhm = make_list(fwhm, dim)

    inplace = not x.requires_grad
    square_ = torch.square_ if inplace else torch.square
    div_ = torch.Tensor.div_ if inplace else torch.div
    exp_ = torch.Tensor.exp_ if inplace else torch.exp

    iradius = int(pymath.floor(radius))
    square = lambda x: x*x
    sqnorm = lambda x: sum(map(square, x))
    sqdist = set(map(sqnorm, itertools.combinations_with_replacement(range(iradius+1), r=dim)))
    sqdist = list(filter(lambda d: 0 < d <= radius*radius, sqdist))
    indices = itertools.product(*[range(-iradius, iradius+1)] * dim)
    indices = filter(lambda x: 0 < sqnorm(x) <= radius*radius, indices)

    nb_feat = len(sqdist)
    mnd = x.new_zeros([nb_feat, *x.shape])

    # compute differences
    count = [0] * len(sqdist)
    for index in indices:
        sqd1 = sqnorm(index)
        diff = square_(utils.roll(x, index, range(-dim, 0), bound=bound).sub_(x))
        mnd[sqdist.index(sqd1)] += diff
        count[sqdist.index(sqd1)] += 1
    # normalize
    for k, c in enumerate(count):
        mnd[k] /= c
    # smooth
    mnd = smooth(mnd, fwhm=fwhm, dim=dim, bound=bound)

    # compute variance in first-ring neighborhood
    var = mnd[0]
    if robust:
        meanvar = var.mean(list(range(-3, 0)), keepdim=True)
        mn = 10 ** (-robust) * meanvar
        mx = 10 ** robust * meanvar
        var = torch.min(torch.max(var, mn), mx)
    mnd = mnd[1:]

    # exponentiation
    mnd = exp_(div_(mnd, var).neg_())

    # normalize
    if norm == 'max':
        mnd /= mnd.max(0).values
    elif norm == 'mean':
        mnd /= mnd.mean(0)

    return utils.movedim(mnd, 0, -1)


def frangi(x, a=0.5, b=0.5, c=500, white_ridges=False, fwhm=range(1, 8, 2),
           dim=None, bound='replicate', return_scale=False, verbose=False):
    """Frangi (vessel detector) filter.

    Parameters
    ----------
    x : (*batch_shape, *spatial_shape) tensor_like
        Input (batched) tensor.
    a : float or tensor, default=0.5
        First Frangi vesselness constant (deviation from plate-like)
        Only used in 3D.
    b : float or tensor, default=0.5
        Second Frangi vesselness constant (deviation from blob-like)
    c : float or tensor, default=500
        Third Second Frangi vesselness constant (signal to noise)
    white_ridges : bool, default=False
        If True, detect white ridges (black background).
        Else, detect black ridges (white background).
    fwhm : [sequence of] float, default=[1, 3, 5, 7]
        Full width half max of Gaussian filters.
    dim : int
        Length of `spatial_shape`.
    bound : bound_like, default='replicate'
        Boundary condition.
    return_scale : bool, default=False
    verbose : bool, default=False

    Returns
    -------
    vessels : (*batch_shape, *spatial_shape) tensor
        Volume of vesselness.
    scale : (*batch_shape, *spatial_shape) tensor[int], if return_scale
        Index of scale at which each pixel was detected.

    References
    ----------
    ..[1] "Multiscale vessel enhancement filtering"
          Frangi, Niessen, Vincken, Viergever
          MICCAI (1998) https://doi.org/10.1007/BFb0056195

    """
    prm = dict(white_ridges=white_ridges, fwhm=fwhm,
               dim=dim, bound=bound, return_scale=return_scale,
               verbose=verbose)
    if any(map(lambda x: getattr(x, 'requires_grad', False), [x, a, b, c])):
        return frangi_diff(x, a, b, c, **prm)
    else:
        return frangi_nodiff(x, a, b, c, **prm)


def frangi_nodiff(x, a=0.5, b=0.5, c=500, white_ridges=False, fwhm=range(1, 8, 2),
           dim=None, bound='replicate', return_scale=False, verbose=False):
    """Non-differentiable Frangi filter."""
    x = torch.as_tensor(x)
    is_on_cpu = x.device == torch.device('cpu')
    dim = dim or x.dim()
    if not dim in (2, 3):
        raise ValueError('Frangi filter is only implemented in 2D or 3D')

    a = -0.5/(a**2)
    b = -0.5/(b**2)
    c = -0.5/(c**2)

    # allocate buffers
    h = x.new_empty([*x.shape, dim, dim])   # hessian
    v = x.new_empty([*x.shape, dim])        # eigenvalues
    buf1 = torch.empty_like(x)
    buf2 = torch.empty_like(x)
    buf3 = torch.empty_like(x)

    def _frangi(x, f):
        """Frangi filter at one scale. Input must be pre-filtered."""
        x = smooth(x, fwhm=f, dim=dim, bound=bound) if f else x

        # Hessian
        if verbose:
            print('Hessian...')
        for d in range(dim):
            # diagonal elements
            fwd = diff1d(x, order=1, dim=-d-1, side='f', bound=bound, out=buf1)
            bwd = diff1d(x, order=1, dim=-d-1, side='b', bound=bound, out=buf2)
            torch.sub(fwd, bwd, out=h[..., d, d])  # central 2nd order
            # ctr = diff1d(x, order=1, dim=-d-1, side='c', bound=bound, out=buf1)
            for dd in range(d+1, dim):
                # only fill upper part
                hh = h[..., d, dd]
                # diff1d(ctr, order=1, dim=-dd - 1, side='c', bound=bound, out=hh)
                diff1d(fwd, order=1, dim=-dd - 1, side='f', bound=bound, out=hh)
                hh += diff1d(fwd, order=1, dim=-dd - 1, side='b', bound=bound, out=buf3)
                hh += diff1d(bwd, order=1, dim=-dd - 1, side='b', bound=bound, out=buf3)
                hh += diff1d(bwd, order=1, dim=-dd - 1, side='f', bound=bound, out=buf3)
                hh /= 4.

        # Correct for scale
        if f:
            sig2 = (f / 2.355)**2
            h.mul_(sig2)

        # Eigenvalues
        if verbose:
            print('Eigen...')
        if is_on_cpu:
            torch.symeig(h, out=(v, torch.empty([])))
        else:
            v.copy_(linalg.eig_sym_(h))
        # we must order eigenvalues by increasing *magnitude*
        _, perm = v.abs().sort()
        v.copy_(v.gather(-1, perm))
        lam1, lam2, *lam3 = v.unbind(-1)
        lam3 = lam3.pop() if lam3 else None

        if verbose:
            print('Frangi...')

        if dim == 2:
            if white_ridges:
                msk = lam2 > 0
            else:
                msk = lam2 < 0
        else:
            if white_ridges:
                msk = lam2 > 0
                msk.bitwise_or_(lam3 > 0)
            else:
                msk = lam2 < 0
                msk.bitwise_or_(lam3 < 0)

        v.abs_()
        if dim == 2:
            lam2.clamp_min_(1e-10)
            v.square_()
            torch.div(lam1, lam2, out=buf2)                 # < Rb ** 2
            torch.sum(v, dim=-1, out=buf3)                  # < S ** 2

            buf2.mul_(b).exp_()                            # < exp(Rb**2)
            buf3.mul_(c).exp_().neg_().add_(1)             # < 1-exp(S**2)

            buf2.mul_(buf3)

            buf2.masked_fill_(msk, 0)
            return buf2

        elif dim == 3:
            lam3.clamp_min_(1e-10)
            lam2.clamp_min_(1e-10)
            torch.mul(lam2, lam3, out=buf2)                 # < lam2 * lam3
            v.square_()                                     # < lam ** 2
            torch.div(lam2, lam3, out=buf1)                 # < Ra ** 2
            torch.div(lam1, buf2, out=buf2)                 # < Rb ** 2
            torch.sum(v, dim=-1, out=buf3)                  # < S ** 2

            buf1.mul_(a).exp_().neg_().add_(1)             # < 1-exp(Ra**2)
            buf2.mul_(b).exp_()                            # <   exp(Rb**2)
            buf3.mul_(c).exp_().neg_().add_(1)             # < 1-exp(S**2)

            buf1.mul_(buf2).mul_(buf3)

            buf1.masked_fill_(msk, 0)
            return buf1

    v0 = None
    scale = None
    for i, f in enumerate(make_vector(fwhm)):

        if verbose:
            print('fwhm:', f.item())

        v1 = _frangi(x, f)
        v1[~torch.isfinite(v1)] = 0

        # combine scales
        if v0 is None:
            v0 = v1.clone()
            if return_scale:
                scale = torch.zeros_like(v1, dtype=torch.int)
        else:
            if return_scale:
                scale[v1 > v0] = i
            v0 = torch.max(v0, v1, out=v0)

    return (v0, scale) if return_scale else v0


def frangi_diff(x, a=0.5, b=0.5, c=500, white_ridges=False, fwhm=range(1, 8, 2),
                dim=None, bound='replicate', return_scale=False, verbose=False):
    """Differentiable Frangi filter."""
    x = torch.as_tensor(x)
    is_on_cpu = x.device == torch.device('cpu')
    dim = dim or x.dim()
    if dim not in (2, 3):
        raise ValueError('Frangi filter is only implemented in 2D or 3D')

    a = 2*(a**2)
    b = 2*(b**2)
    c = 2*(c**2)

    # allocate buffers

    def _frangi(x, f):
        """Frangi filter at one scale. Input must be pre-filtered."""
        x = smooth(x, fwhm=f, dim=dim, bound=bound) if f else x

        # Hessian
        h = x.new_empty([*x.shape, dim, dim])   # hessian
        if verbose:
            print('Hessian...')
        for d in range(dim):
            # diagonal elements
            fwd = diff1d(x, order=1, dim=-d-1, side='f', bound=bound)
            bwd = diff1d(x, order=1, dim=-d-1, side='b', bound=bound)
            h[..., d, d].copy_(fwd).sub_(bwd)
            for dd in range(d+1, dim):
                # only fill upper part
                hh = h[..., d, dd]
                diff1d(fwd, order=1, dim=-dd - 1, side='f', bound=bound, out=hh)
                hh += diff1d(fwd, order=1, dim=-dd - 1, side='b', bound=bound)
                hh += diff1d(bwd, order=1, dim=-dd - 1, side='b', bound=bound)
                hh += diff1d(bwd, order=1, dim=-dd - 1, side='f', bound=bound)
                hh /= 4.

        # Correct for scale
        if f:
            sig2 = (f / 2.355)**2
            h.mul_(sig2)

        # Eigenvalues
        if verbose:
            print('Eigen...')
        if is_on_cpu:
            eig = torch.symeig(h, eigenvectors=True)[0]
        else:
            eig = linalg.eig_sym_(h)
        _, perm = eig.abs().sort()
        eig = eig.gather(-1, perm)
        lam1, lam2, *lam3 = eig.unbind(-1)
        lam3 = lam3.pop() if lam3 else None

        if verbose:
            print('Frangi...')

        if dim == 2:
            if white_ridges:
                msk = lam2 > 0
            else:
                msk = lam2 < 0
        else:
            if white_ridges:
                msk = lam2 > 0
                msk.bitwise_or_(lam3 > 0)
            else:
                msk = lam2 < 0
                msk.bitwise_or_(lam3 < 0)

        eig[msk, :] = 1
        eig.abs_()
        lam1, lam2, *lam3 = eig.unbind(-1)
        lam3 = lam3.pop() if lam3 else None

        if dim == 2:
            eig = eig.square()
            lam1, lam2 = eig.unbind(-1)
            rb = lam1/lam2                                  # < Rb ** 2
            s = eig.sum(dim=-1)                             # < S ** 2

            rb = rb.div_(-b).exp_()                         # < exp(Rb**2)
            s = s.div_(-c).exp_().neg().add_(1)             # < 1-exp(S**2)

            rb = rb.mul(s)

            rb.masked_fill_(msk, 0)
            return rb

        elif dim == 3:
            rb = lam2*lam3                                  # < lam2 * lam3
            eig = eig.square()                              # < lam ** 2
            lam1, lam2, lam3 = eig.unbind(-1)
            ra = lam2/lam3                                  # < Ra ** 2
            rb = lam1/rb                                    # < Rb ** 2
            s = eig.sum(dim=-1)                             # < S ** 2

            ra = ra.div_(-a).exp_().neg().add_(1)           # < 1-exp(Ra**2)
            rb = rb.div_(-b).exp_()                         # <   exp(Rb**2)
            s = s.div_(-c).exp_().neg().add_(1)             # < 1-exp(S**2)

            ra = ra.mul_(rb).mul_(s)

            ra = ra.masked_fill_(msk, 0)
            return ra

    v0 = None
    scale = None
    for i, f in enumerate(make_vector(fwhm)):

        if verbose:
            print('fwhm:', f.item())

        v1 = _frangi(x, f)
        v1.masked_fill_(torch.isfinite(v1).bitwise_not_(), 0)

        # combine scales
        if v0 is None:
            v0 = v1.clone()
            if return_scale:
                scale = torch.zeros_like(v0, dtype=torch.int)
        else:
            if return_scale:
                scale.masked_fill_(v1 > v0, i)
            v0 = torch.max(v0, v1)

    return (v0, scale) if return_scale else v0


def sato(x, gamma_long=0.5, gamma_ellipse=0.5, alpha=0.25, white_ridges=False,
         fwhm=range(1, 8, 2), dim=None, bound='replicate', return_scale=False,
         verbose=False):
    """Non-differentiable Frangi filter."""
    x = torch.as_tensor(x)
    dim = dim or x.dim()
    if dim not in (2, 3):
        raise ValueError('Frangi filter is only implemented in 2D or 3D')

    def single_scale3d(lam1, lam2, lam3):
        # msk = mask of voxels to *exclude*
        # msk_alpha = mask of voxels in which to use alpha weight instead of 1
        if white_ridges:
            msk = lam2 > 0
            msk.bitwise_or_(lam3 > 0)
            msk_alpha = lam1 < 0
            msk_alpha.bitwise_and_(lam1 > -lam2 / alpha)
        else:
            msk = lam2 < 0
            msk.bitwise_or_(lam3 < 0)
            msk_alpha = lam1 > 0
            msk_alpha.bitwise_and_(lam1 < -lam2 / alpha)

        lam1 = lam1.abs_()
        lam2 = lam2.abs_()
        lam3 = lam3.abs_()

        msk_alpha = torch.where(msk_alpha, float(alpha), 1.).to(lam1)
        # one dimension is long
        # if lam1 << lam2, p -> 1
        # if lam1 == lam2, p -> 0
        w = (1 - msk_alpha * (lam1 / lam2)).pow_(gamma_long)
        # cross section is not an ellipse
        # lam2 == lam3, p -> 1
        # lam2 << lam3, p -> 0
        w *= (lam2 / lam3).pow_(gamma_ellipse)
        # cross section is small
        # lam3 >> 0, p -> inf
        w *= lam3
        w.masked_fill_(msk, 0)
        return w

    def single_scale2d(lam1, lam2):
        # msk = mask of voxels to *exclude*
        # msk_alpha = mask of voxels in which to use alpha weight instead of 1
        if white_ridges:
            msk = lam2 > 0
            msk_alpha = lam1 < 0
            msk_alpha.bitwise_and_(lam1 > -lam2 / alpha)
        else:
            msk = lam2 < 0
            msk_alpha = lam1 > 0
            msk_alpha.bitwise_and_(lam1 < -lam2 / alpha)

        lam1 = lam1.abs_()
        lam2 = lam2.abs_()

        msk_alpha = torch.where(msk_alpha, float(alpha), 1.).to(lam1)
        # one dimension is long
        # if lam1 << lam2, p -> 1
        # if lam1 == lam2, p -> 0
        w = (1 - msk_alpha * (lam1 / lam2)).pow_(gamma_long)
        # cross section is small
        # lam2 >> 0, p -> inf
        w *= lam2
        w.masked_fill_(msk, 0)
        return w

    single_scale = single_scale3d if dim == 3 else single_scale2d

    v0 = None
    scale = None
    for i, f in enumerate(make_vector(fwhm)):

        if verbose:
            print('fwhm:', f.item())
            print('Hessian...')
        lam = hessian_eig(x, f, dim=dim, bound=bound)

        if verbose:
            print('Frangi...')
        v1 = single_scale(*lam.unbind(-1))
        v1.masked_fill_(torch.isfinite(v1).bitwise_not_(), 0)

        # combine scales
        if v0 is None:
            v0 = v1
            if return_scale:
                scale = torch.zeros_like(v1, dtype=torch.int)
        else:
            if return_scale:
                scale[v1 > v0] = i
            v0 = torch.max(v0, v1, out=v0)

    return (v0, scale) if return_scale else v0


def hessian_eig(x, fwhm=1, dim=None, bound='replicate', sort='abs', verbose=False):
    """Sorted eigenvalues of the Hessian .

    Parameters
    ----------
    x : (*batch_shape, *spatial_shape) tensor_like
        Input (batched) tensor.
    fwhm : float, default=1
        Full width half max of the Gaussian filter
    dim : int
        Length of `spatial_shape`.
    bound : bound_like, default='replicate'
        Boundary condition.
    sort : {'val', 'abs', None}, default='abs'
        Sort by value, absolute value or not at all.
    verbose : bool, default=False

    Returns
    -------
    vessels : (*batch_shape, *spatial_shape, dim) tensor
        Sorted eigenvalues of the Hessian.

    """
    x = torch.as_tensor(x)
    is_on_cpu = x.device == torch.device('cpu')
    dim = dim or x.dim()
    if dim not in (2, 3):
        raise ValueError('HessianEig is only implemented in 2D or 3D')

    # Smooth
    x = smooth(x, fwhm=fwhm, dim=dim, bound=bound) if fwhm else x

    # Hessian
    h = x.new_empty([*x.shape, dim, dim])   # hessian
    if verbose:
        print('Hessian...')
    for d in range(dim):
        # diagonal elements
        fwd = diff1d(x, order=1, dim=-d-1, side='f', bound=bound)
        bwd = diff1d(x, order=1, dim=-d-1, side='b', bound=bound)
        h[..., d, d].copy_(fwd).sub_(bwd)
        for dd in range(d+1, dim):
            # only fill upper part
            hh = h[..., d, dd]
            diff1d(fwd, order=1, dim=-dd - 1, side='f', bound=bound, out=hh)
            hh += diff1d(fwd, order=1, dim=-dd - 1, side='b', bound=bound)
            hh += diff1d(bwd, order=1, dim=-dd - 1, side='b', bound=bound)
            hh += diff1d(bwd, order=1, dim=-dd - 1, side='f', bound=bound)
            hh /= 4.

    # Correct for scale
    if fwhm:
        sig2 = (fwhm / 2.355)**2
        h.mul_(sig2)

    # Eigenvalues
    if verbose:
        print('Eigen...')
    if is_on_cpu:
        eig = torch.symeig(h, eigenvectors=True)[0]
    else:
        eig = linalg.eig_sym_(h)
    if sort:
        val = eig
        if sort.lower()[0] == 'a':
            val = val.abs()
        perm = val.sort().indices
        eig = eig.gather(-1, perm)
    return eig


def sobel(x, dim=None, bound='replicate', value=0):
    """Sobel (edge detector) filter.

    This function supports autograd.

    Parameters
    ----------
    x : (*batch_shape, *spatial_shape) tensor_like
        Input (batched) tensor.
    dim : int
        Length of `spatial_shape`.
    bound : bound_like, default='replicate'
        Boundary condition.
    value : number, default=0
        Out-of-bounds value if `bound='constant'`.

    Returns
    -------
    edges : (*batch_shape, *spatial_shape) tensor
        Volume of edges.

    """
    dim = dim or x.dim()
    dims = set(range(x.dim()-dim, x.dim()))
    if not dims:
        return x
    g = 0
    for d in dims:
        g1 = x
        other_dims = set(dims)
        other_dims.discard(dim)
        for dd in other_dims:
            # triangular smoothing
            kernel = [1, 2, 1]
            window = _window1d(g1, dd, [-1, 0, 1], bound, value)
            g1 = _lincomb(window, kernel, dd, ref=g1)
        # central finite differences
        kernel = [-1, 1]
        window = _window1d(g1, d, [-1, 1], bound, value)
        g1 = _lincomb(window, kernel, d, ref=g1)
        g1 = g1.square() if g1.requires_grad else g1.square_()
        g += g1
    g = g.sqrt() if g.requires_grad else g.sqrt_()
    return g


def _lincomb(slices, weights, dim, ref=None):
    """Perform the linear combination of a sequence of chunked tensors.

    Parameters
    ----------
    slices : sequence[sequence[tensor]]
        First level contains elements in the combinations.
        Second level contains chunks.
    weights : sequence[number]
        Linear weights
    dim : int or sequence[int]
        Dimension along which the tensor was chunked.
    ref : tensor, optional
        Tensor whose data we do not want to overwrite.
        If provided, and some slices point to the same underlying data
        as `ref`, the slice is not written into inplace.

    Returns
    -------
    lincomb : tensor

    """
    slices = make_list(slices)
    dims = make_list(dim, len(slices))

    result = None
    for chunks, weight, dim in zip(slices, weights, dims):

        # multiply chunk with weight
        chunks = make_list(chunks)
        if chunks:
            weight = torch.as_tensor(weight, dtype=chunks[0].dtype,
                                     device=chunks[0].device)
        new_chunks = []
        for chunk in chunks:
            if chunk.numel() == 0:
                continue
            if not weight.requires_grad:
                # save computations when possible
                if weight == 0:
                    new_chunks.append(chunk.new_zeros([]).expand(chunk.shape))
                    continue
                if weight == 1:
                    new_chunks.append(chunk)
                    continue
                if weight == -1:
                    if ref is not None and not same_storage(ref, chunk):
                        if any(s == 0 for s in chunk.stride()):
                            chunk = -chunk
                        else:
                            chunk = chunk.neg_()
                    else:
                        chunk = -chunk
                    new_chunks.append(chunk)
                    continue
            if ref is not None and not same_storage(ref, chunk):
                if any(s == 0 for s in chunk.stride()):
                    chunk = chunk * weight
                else:
                    chunk *= weight
            else:
                chunk = chunk * weight
            new_chunks.append(chunk)

        # accumulate
        if result is None:
            if len(new_chunks) == 1:
                result = new_chunks[0]
            else:
                result = torch.cat(new_chunks, dim)
        else:
            offset = 0
            for chunk in new_chunks:
                index = slice(offset, offset+chunk.shape[dim])
                view = fast_slice_tensor(result, index, dim)
                view += chunk
                offset += chunk.shape[dim]

    return result


def _window1d(x, dim, offsets, bound='dct2', value=0):
    """Extract a sliding window from a tensor.

    Views are used to minimize allocations.

    Parameters
    ----------
    x : tensor_like
        Input tensor
    dim : int
        Dimension along which to extract offsets
    offsets : [sequence of] int
        Offsets to extract, with respect to each voxel.
        To extract a centered window of length 3, use `offsets=[-1, 0, 1]`.
    bound : bound_like, default='dct2'
        Boundary conditions
    value : number, default=0
        Filling value if `bound='constant'`

    Returns
    -------
    win : [tuple of] tuple[tensor]
        If a sequence of offsets was provided, the first level
        corresponds to offsets.
        The second levels are tensors that could be concatenated along
        `dim` to generate the input tensor shifted by `offset`. However,
        to avoid unnecessary allocations, a list of (eventually empty)
        chunks is returned instead of the full shifted tensor.
        Some (hopefully most) of these tensors can be views into the
        input tensor.

    """
    return_list = isinstance(offsets, (list, tuple))
    offsets = make_list(offsets)
    return_list = return_list or len(offsets) > 1
    x = torch.as_tensor(x)
    backend = dict(dtype=x.dtype, device=x.device)
    length = x.shape[dim]

    # sanity check
    for i in offsets:
        nb_pre = max(0, -i)
        nb_post = max(0, i)
        if nb_pre > x.shape[dim] or nb_post > x.shape[dim]:
            raise ValueError('Offset cannot be farther than one length away.')

    slices = []
    for i in offsets:
        nb_pre = max(0, -i)
        nb_post = max(0, i)
        central = fast_slice_tensor(x, slice(nb_post or None, -nb_pre or None), dim)
        if bound == 'dct2':
            pre = fast_slice_tensor(x, slice(None, nb_pre), dim)
            pre = torch.flip(pre, [dim])
            post = fast_slice_tensor(x, slice(length-nb_post, None), dim)
            post = torch.flip(post, [dim])
            slices.append(tuple([pre, central, post]))
        elif bound == 'dct1':
            pre = fast_slice_tensor(x, slice(1, nb_pre+1), dim)
            pre = torch.flip(pre, [dim])
            post = fast_slice_tensor(x, slice(length-nb_post-1, -1), dim)
            post = torch.flip(post, [dim])
            slices.append(tuple([pre, central, post]))
        elif bound == 'dst2':
            pre = fast_slice_tensor(x, slice(None, nb_pre), dim)
            pre = -torch.flip(pre, [dim])
            post = fast_slice_tensor(x, slice(-nb_post, None), dim)
            post = -torch.flip(post, [dim])
            slices.append(tuple([pre, central, post]))
        elif bound == 'dst1':
            pre = fast_slice_tensor(x, slice(None, nb_pre-1), dim)
            pre = -torch.flip(pre, [dim])
            post = fast_slice_tensor(x, slice(length-nb_post+1, None), dim)
            post = -torch.flip(post, [dim])
            shape1 = list(x.shape)
            shape1[dim] = 1
            zero = torch.zeros([], **backend).expand(shape1)
            slices.append(tuple([pre, zero, central, zero, post]))
        elif bound == 'dft':
            pre = fast_slice_tensor(x, slice(length-nb_pre, None), dim)
            post = fast_slice_tensor(x, slice(None, nb_post), dim)
            slices.append(tuple([pre, central, post]))
        elif bound == 'replicate':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = fast_slice_tensor(x, slice(None, 1), dim).expand(shape_pre)
            post = fast_slice_tensor(x, slice(-1, None), dim).expand(shape_post)
            slices.append(tuple([pre, central, post]))
        elif bound == 'zero':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = torch.zeros([], **backend).expand(shape_pre)
            post = torch.zeros([], **backend).expand(shape_post)
            slices.append(tuple([pre, central, post]))
        elif bound == 'constant':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = torch.full([], value, **backend).expand(shape_pre)
            post = torch.full([], value, **backend).expand(shape_post)
            slices.append(tuple([pre, central, post]))

    slices = tuple(slices)
    if not return_list:
        slices = slices[0]
    return slices


def im_divergence(dat, vx=None, which='forward', bound='constant'):
    """ Computes the divergence of 2D or 3D data.

    Args:
        dat (torch.tensor()): A 3D|4D tensor (2, X, Y) | (3, X, Y, Z).
        vx (tuple(float), optional): Voxel size. Defaults to (1, 1, 1).
        which (string, optional): Gradient type:
            . 'forward': Forward difference (next - centre)
            . 'backward': Backward difference (centre - previous)
            . 'central': Central difference ((next - previous)/2)
            Defaults to 'forward'.
        bound (string, optional): Boundary conditions:
                . 'circular' -> FFT
                . 'reflect' or 'reflect1' -> DCT-I
                . 'reflect2' -> DCT-II
                . 'replicate' -> replicates border values
                . 'constant zero'
            Defaults to 'constant zero'

    Returns:
        div (torch.tensor()): Divergence (X, Y) | (X, Y, Z).

    """
    if vx is None:
        vx = (1,) * 3
    if not isinstance(vx, torch.Tensor):
        vx = torch.tensor(vx, dtype=dat.dtype, device=dat.device)
    half = torch.tensor(0.5, dtype=dat.dtype, device=dat.device)
    ndim = len(dat.shape) - 1
    bound = _bound_converter[bound]

    if which == 'forward':
        # Pad + reflected forward difference
        if ndim == 2:  # 2D data
            x = utils.pad(dat[0, ...], (1, 0, 0, 0), mode=bound)
            x = x[:-1, :] - x[1:, :]
            y = utils.pad(dat[1, ...], (0, 0, 1, 0), mode=bound)
            y = y[:, :-1] - y[:, 1:]
        else:  # 3D data
            x = utils.pad(dat[0, ...], (1, 0, 0, 0, 0, 0), mode=bound)
            x = x[:-1, :, :] - x[1:, :, :]
            y = utils.pad(dat[1, ...], (0, 0, 1, 0, 0, 0), mode=bound)
            y = y[:, :-1, :] - y[:, 1:, :]
            z = utils.pad(dat[2, ...], (0, 0, 0, 0, 1, 0), mode=bound)
            z = z[:, :, :-1] - z[:, :, 1:]
    elif which == 'backward':
        # Pad + reflected backward difference
        if ndim == 2:  # 2D data
            x = utils.pad(dat[0, ...], (0, 1, 0, 0), mode=bound)
            x = x[:-1, :] - x[1:, :]
            y = utils.pad(dat[1, ...], (0, 0, 0, 1), mode=bound)
            y = y[:, :-1] - y[:, 1:]
        else:  # 3D data
            x = utils.pad(dat[0, ...], (0, 1, 0, 0, 0, 0), mode=bound)
            x = x[:-1, :, :] - x[1:, :, :]
            y = utils.pad(dat[1, ...], (0, 0, 0, 1, 0, 0), mode=bound)
            y = y[:, :-1, :] - y[:, 1:, :]
            z = utils.pad(dat[2, ...], (0, 0, 0, 0, 0, 1), mode=bound)
            z = z[:, :, :-1] - z[:, :, 1:]
    elif which == 'central':
        # Pad + reflected central difference
        if ndim == 2:  # 2D data
            x = utils.pad(dat[0, ...], (1, 1, 0, 0), mode=bound)
            x = half * (x[:-2, :] - x[2:, :])
            y = utils.pad(dat[1, ...], (0, 0, 1, 1), mode=bound)
            y = half * (y[:, :-2] - y[:, 2:])
        else:  # 3D data
            x = utils.pad(dat[0, ...], (1, 1, 0, 0, 0, 0), mode=bound)
            x = half * (x[:-2, :, :] - x[2:, :, :])
            y = utils.pad(dat[1, ...], (0, 0, 1, 1, 0, 0), mode=bound)
            y = half * (y[:, :-2, :] - y[:, 2:, :])
            z = utils.pad(dat[2, ...], (0, 0, 0, 0, 1, 1), mode=bound)
            z = half * (z[:, :, :-2] - z[:, :, 2:])
    else:
        raise ValueError('Undefined divergence')
    if ndim == 2:  # 2D data
        return x / vx[0] + y / vx[1]
    else:  # 3D data
        return x / vx[0] + y / vx[1] + z / vx[2]


def im_gradient(dat, vx=None, which='forward', bound='constant'):
    """ Computes the gradient of 2D or 3D data.

    Args:
        dat (torch.tensor()): A 2D|3D tensor (X, Y) | (X, Y, Z).
        vx (tuple(float), optional): Voxel size. Defaults to (1, 1, 1).
        which (string, optional): Gradient type:
            . 'forward': Forward difference (next - centre)
            . 'backward': Backward difference (centre - previous)
            . 'central': Central difference ((next - previous)/2)
            Defaults to 'forward'.
        bound (string, optional): Boundary conditions:
                . 'circular' -> FFT
                . 'reflect' or 'reflect1' -> DCT-I
                . 'reflect2' -> DCT-II
                . 'replicate' -> replicates border values
                . 'constant zero'
            Defaults to 'constant zero'

    Returns:
          grad (torch.tensor()): Gradient (2, X, Y) | (3, X, Y, Z).

    """
    if vx is None:
        vx = (1,) * 3
    if not isinstance(vx, torch.Tensor):
        vx = torch.tensor(vx, dtype=dat.dtype, device=dat.device)
    half = torch.tensor(0.5, dtype=dat.dtype, device=dat.device)
    ndim = len(dat.shape)
    bound = _bound_converter[bound]

    if which == 'forward':
        # Pad + forward difference
        if ndim == 2:  # 2D data
            dat = utils.pad(dat, (0, 1, 0, 1), mode=bound)
            gx = -dat[:-1, :-1] + dat[1:, :-1]
            gy = -dat[:-1, :-1] + dat[:-1, 1:]
        else:  # 3D data
            dat = utils.pad(dat, (0, 1, 0, 1, 0, 1), mode=bound)
            gx = -dat[:-1, :-1, :-1] + dat[1:, :-1, :-1]
            gy = -dat[:-1, :-1, :-1] + dat[:-1, 1:, :-1]
            gz = -dat[:-1, :-1, :-1] + dat[:-1, :-1, 1:]
    elif which == 'backward':
        # Pad + backward difference
        if ndim == 2:  # 2D data
            dat = utils.pad(dat, (1, 0, 1, 0), mode=bound)
            gx = -dat[:-1, 1:] + dat[1:, 1:]
            gy = -dat[1:, :-1] + dat[1:, 1:]
        else:  # 3D data
            dat = utils.pad(dat, (1, 0, 1, 0, 1, 0), mode=bound)
            gx = -dat[:-1, 1:, 1:] + dat[1:, 1:, 1:]
            gy = -dat[1:, :-1, 1:] + dat[1:, 1:, 1:]
            gz = -dat[1:, 1:, :-1] + dat[1:, 1:, 1:]
    elif which == 'central':
        # Pad + central difference
        if ndim == 2:  # 2D data
            dat = utils.pad(dat, (1, 1, 1, 1), mode=bound)
            gx = half * (-dat[:-2, 1:-1] + dat[2:, 1:-1])
            gy = half * (-dat[1:-1, :-2] + dat[1:-1, 2:])
        else:  # 3D data
            dat = utils.pad(dat, (1, 1, 1, 1, 1, 1), mode=bound)
            gx = half * (-dat[:-2, 1:-1, 1:-1] + dat[2:, 1:-1, 1:-1])
            gy = half * (-dat[1:-1, :-2, 1:-1] + dat[1:-1, 2:, 1:-1])
            gz = half * (-dat[1:-1, 1:-1, :-2] + dat[1:-1, 1:-1, 2:])
    else:
        raise ValueError('Undefined gradient')
    if ndim == 2:  # 2D data
        return torch.stack((gx / vx[0], gy / vx[1]), dim=0)
    else:  # 3D data
        return torch.stack((gx / vx[0], gy / vx[1], gz / vx[2]), dim=0)


def _check_adjoint_grad_div(which='central', vx=None, dtype=torch.float64,
                           ndim=3, dim=64, device='cpu', bound='constant'):
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
    if not isinstance(vx, torch.Tensor):
        vx = torch.tensor(vx, dtype=dtype, device=device)
    if isinstance(dim, int):
        dim = (dim,) * 3

    torch.manual_seed(0)
    # Check adjointness of..
    if which == 'forward' or which == 'backward' or which == 'central':
        # ..various gradient operators
        if ndim == 2:
            u = torch.rand(dim[0], dim[1], dtype=dtype, device=device)
            v = torch.rand(2, dim[0], dim[1], dtype=dtype, device=device)
        else:
            u = torch.rand(dim[0], dim[1], dim[2], dtype=dtype, device=device)
            v = torch.rand(3, dim[0], dim[1], dim[2], dtype=dtype, device=device)
        gradu = im_gradient(u, vx=vx, which=which, bound=bound)
        divv = im_divergence(v, vx=vx, which=which, bound=bound)
        val = torch.sum(gradu*v, dtype=torch.float64) - torch.sum(divv*u, dtype=torch.float64)
    # Print okay? (close to zero)
    print('val={}'.format(val))
