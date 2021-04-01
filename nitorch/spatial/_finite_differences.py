"""Finite-differences operators (gradient, divergence, ...)."""

import torch
from nitorch.core import utils
from nitorch.core.utils import expand, slice_tensor, same_storage, make_vector
from nitorch.core.py import make_list
from ._conv import smooth


__all__ = ['im_divergence', 'im_gradient', 'diff1d', 'diff', 'div1d', 'div',
           'sobel']


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

    # build "light" zero using zero strides
    edge_shape = list(x.shape)
    edge_shape[dim] = 1

    if order == 1:
        diff = torch.empty_like(x) if out is None else out.reshape(x.shape)

        if side == 'f':  # forward -> diff[i] = x[i+1] - x[i]
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            subto(post, pre, out=slice_tensor(diff, slice(None, -1), dim))
            if bound in ('dct2', 'replicate'):
                # x[end+1] = x[end] => diff[end] = 0
                slice_tensor(diff, -1, dim).zero_()
            elif bound == 'dct1':
                # x[end+1] = x[end-1] => diff[end] = -diff[end-1]
                last = slice_tensor(diff, -2, dim)
                slice_tensor(diff, -1, dim).copy_(last).neg_()
            elif bound == 'dst2':
                # x[end+1] = -x[end] => diff[end] = -2*x[end]
                last = 2*slice_tensor(x, -1, dim)
                slice_tensor(diff, -1, dim).copy_(last).neg_()
            elif bound in ('dst1', 'zero'):
                # x[end+1] = 0 => diff[end] = -x[end]
                last = slice_tensor(x, -1, dim)
                slice_tensor(diff, -1, dim).copy_(last).neg_()
            else:
                assert bound == 'dft'
                # x[end+1] = x[0] => diff[end] = x[0] - x[end]
                subto(slice_tensor(x, 0, dim), slice_tensor(x, -1, dim),
                          out=slice_tensor(diff, -1, dim))

        elif side == 'b':  # backward -> diff[i] = x[i] - x[i-1]
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            subto(post, pre, out=slice_tensor(diff, slice(1, None), dim))
            if bound in ('dct2', 'replicate'):
                # x[-1] = x[0] => diff[0] = 0
                slice_tensor(diff, 0, dim).zero_()
            elif bound == 'dct1':
                # x[-1] = x[1] => diff[0] = -diff[1]
                first = slice_tensor(diff, 1, dim)
                slice_tensor(diff, 0, dim).copy_(first).neg_()
            elif bound == 'dst2':
                # x[-1] = -x[0] => diff[0] = 2*x[0]
                first = 2*slice_tensor(x, 0, dim)
                slice_tensor(diff, 0, dim).copy_(first)
            elif bound in ('dst1', 'zero'):
                # x[-1] = 0 => diff[0] = x[0]
                first = slice_tensor(x, 0, dim)
                slice_tensor(diff, 0, dim).copy_(first)
            else:
                assert bound == 'dft'
                # x[-1] = x[end] => diff[0] = x[0] - x[end]
                subto(slice_tensor(x, 0, dim), slice_tensor(x, -1, dim),
                      out=slice_tensor(diff, 0, dim))

        elif side == 'c':  # central -> diff[i] = (x[i+1] - x[i-1])/2
            pre = slice_tensor(x, slice(None, -2), dim)
            post = slice_tensor(x, slice(2, None), dim)
            subto(post, pre, out=slice_tensor(diff, slice(1, -1), dim))
            if bound in ('dct2', 'replicate'):
                subto(slice_tensor(x, 1, dim), slice_tensor(x, 0, dim),
                      out=slice_tensor(diff, 0, dim))
                subto(slice_tensor(x, -1, dim), slice_tensor(x, -2, dim),
                      out=slice_tensor(diff, -1, dim))
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                slice_tensor(diff, 0, dim).zero_()
                slice_tensor(diff, -1, dim).zero_()
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                addto(slice_tensor(x, 1, dim), slice_tensor(x, 0, dim),
                      out=slice_tensor(diff, 0, dim))
                addto(slice_tensor(x, -1, dim), slice_tensor(x, -2, dim),
                      out=slice_tensor(diff, -1, dim)).neg_()
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                first = slice_tensor(x, 1, dim)
                slice_tensor(diff, 0, dim).copy_(first)
                last = slice_tensor(x, -2, dim)
                slice_tensor(diff, -1, dim).copy_(last).neg_()
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                subto(slice_tensor(x, 1, dim), slice_tensor(x, -1, dim),
                      out=slice_tensor(diff, 0, dim))
                subto(slice_tensor(x, 0, dim), slice_tensor(x, -2, dim),
                      out=slice_tensor(diff, -1, dim))
        if side == 'c':
            if voxel_size != 1:
                diff = div_(diff, voxel_size * 2)
            else:
                diff = div_(diff, 2.)
        elif voxel_size != 1:
            diff = div_(diff, voxel_size)

    elif side == 'c':
        # we must deal with central differences differently:
        # -> second order differences are exact but first order
        #    differences are approximated (we should sample between
        #    two voxels so we must interpolate)
        # -> for order > 2, we compose as many second order differences
        #    as possible and then (eventually) deal with the remaining
        #    1st order using the approximate implementation.
        if order == 2:
            fwd = diff1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                         side='f', bound=bound, out=out)
            bwd = diff1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                         side='b', bound=bound)
            diff = fwd.sub_(bwd)
            if voxel_size != 1:
                diff = div_(diff, voxel_size)
        else:
            diff = diff1d(x, order=2, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound)
            diff = diff1d(diff, order=order-2, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound, out=out)

    else:
        diff = diff1d(x, order=1, dim=dim, voxel_size=voxel_size,
                      side=side, bound=bound)
        diff = diff1d(diff, order=order-1, dim=dim, voxel_size=voxel_size,
                      side=side, bound=bound, out=out)

    if out is not None and not utils.same_storage(out, diff):
        out = out.reshape(diff.shape).copy_(diff)
        diff = out
    return diff


def diff(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2'):
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

    Returns
    -------
    diff : tensor
        Tensor of finite differences, with same shape as the input tensor,
        with an additional dimension if any of the input arguments is a list.

    """
    # find number of dimensions
    dim = torch.as_tensor(dim)
    voxel_size = make_vector(voxel_size)
    drop_last = dim.dim() == 0 and voxel_size.dim() == 0
    dim = make_list(dim.tolist())
    voxel_size = make_list(voxel_size.tolist())
    nb_dim = max(len(dim), len(voxel_size))
    dim = make_list(dim, nb_dim)
    voxel_size = make_list(voxel_size, nb_dim)

    # compute diffs in each dimension
    diffs = x.new_empty([nb_dim, *x.shape])
    diffs = utils.movedim(diffs, 0, -1)
    # ^ ensures that sliced dim is the least rapidly changing one
    for i, (d, v) in enumerate(zip(dim, voxel_size)):
        diff1d(x, order, d, v, side, bound, out=diffs[..., i])

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

    if order == 1:

        div = torch.empty_like(x) if out is None else out.reshape(x.shape)

        if side == 'f':
            # forward -> diff[i] = x[i+1] - x[i]
            #            div[i]  = x[i-1] - x[i]
            first = slice_tensor(x, 0, dim)
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            subto(pre, post, out=slice_tensor(div, slice(1, None), dim))
            slice_tensor(div, 0, dim).copy_(first).neg_()
            if bound in ('dct2', 'replicate'):
                ante = slice_tensor(x, -2, dim)
                slice_tensor(div, -1, dim).copy_(ante)
            elif bound == 'dct1':
                last = slice_tensor(x, -1, dim)
                slice_tensor(div, -2, dim).add_(last)
            elif bound == 'dst2':
                last = slice_tensor(x, -1, dim)
                slice_tensor(div, -1, dim).sub_(last)
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                last = slice_tensor(x, -1, dim)
                first = slice_tensor(x, 0, dim)
                subto(last, first, out=slice_tensor(div, 0, dim))

        elif side == 'b':
            # backward -> diff[i] = x[i] - x[i-1]
            #             div[i]  = x[i+1] - x[i]
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            subto(pre, post, out=slice_tensor(div, slice(None, -1), dim))
            slice_tensor(div, -1, dim).copy_(slice_tensor(x, -1, dim))
            if bound in ('dct2', 'replicate'):
                second = slice_tensor(x, 1, dim)
                slice_tensor(div, 0, dim).copy_(second).neg_()
            elif bound == 'dct1':
                first = slice_tensor(x, 0, dim)
                slice_tensor(div, 1, dim).sub_(first)
            elif bound == 'dst2':
                first = slice_tensor(x, 0, dim)
                slice_tensor(div, 0, dim).add_(first)
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                last = slice_tensor(x, -1, dim)
                first = slice_tensor(x, 0, dim)
                subto(last, first, out=slice_tensor(div, -1, dim))

        else:
            assert side == 'c'
            # central -> diff[i] = (x[i+1] - x[i-1])/2
            #         -> div[i]  = (x[i-1] - x[i+1])/2
            pre = slice_tensor(x, slice(None, -2), dim)
            post = slice_tensor(x, slice(2, None), dim)
            subto(pre, post, out=slice_tensor(div, slice(1, -1), dim))
            if bound in ('dct2', 'replicate'):
                # x[-1]    = x[0]   => diff[0]   = x[1] - x[0]
                # x[end+1] = x[end] => diff[end] = x[end] - x[end-1]
                first = slice_tensor(x, 0, dim)
                second = slice_tensor(x, 1, dim)
                last = slice_tensor(x, -1, dim)
                ante = slice_tensor(x, -2, dim)
                addto(first, second, out=slice_tensor(div, 0, dim)).neg_()
                addto(last, ante, out=slice_tensor(div, -1, dim))
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                first = slice_tensor(x, 1, dim)
                second = slice_tensor(x, 2, dim)
                last = slice_tensor(x, -2, dim)
                secondlast = slice_tensor(x, -3, dim)
                slice_tensor(div, 0, dim).copy_(first).neg_()
                slice_tensor(div, 1, dim).copy_(second).neg_()
                slice_tensor(div, -2, dim).copy_(secondlast)
                slice_tensor(div, -1, dim).copy_(last)
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                first = slice_tensor(x, 0, dim)
                second = slice_tensor(x, 1, dim)
                secondlast = slice_tensor(x, -2, dim)
                last = slice_tensor(x, -1, dim)
                subto(first, second, out=slice_tensor(div, 0, dim))
                subto(secondlast, last, out=slice_tensor(div, -1, dim))
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                second = slice_tensor(x, 1, dim)
                secondlast = slice_tensor(x, -2, dim)
                slice_tensor(div, 0, dim).copy_(second).neg_()
                slice_tensor(div, -1, dim).copy_(secondlast)
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                first = slice_tensor(x, 0, dim)
                second = slice_tensor(x, 1, dim)
                last = slice_tensor(x, -1, dim)
                secondlast = slice_tensor(x, -2, dim)
                subto(last, second, out=slice_tensor(div, 0, dim))
                subto(secondlast, first, out=slice_tensor(div, -1, dim))

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
            fwd = div1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                         side='f', bound=bound, out=out)
            bwd = div1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                         side='b', bound=bound)
            div = fwd.sub_(bwd)
            if voxel_size != 1:
                div = div_(div, voxel_size)
        elif order % 2:
            # odd order -> start with a first order
            div = div1d(x, order=1, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound)
            div = div1d(div, order=order-1, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound, out=out)
        else:
            # even order -> recursive call
            div = div1d(x, order=2, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound)
            div = div1d(div, order=order-2, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound, out=out)

    else:
        div = div1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                    side=side, bound=bound)
        div = div1d(div, order=1, dim=dim, voxel_size=voxel_size,
                    side=side, bound=bound, out=out)

    if out is not None and not utils.same_storage(out, div):
        out = out.reshape(div.shape).copy_(div)
        div = out
    return div


def div(x, order=1, dim=-1, voxel_size=1, side='f', bound='dct2'):
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
    dim = make_list(dim.tolist())
    voxel_size = make_list(voxel_size.tolist())
    nb_dim = max(len(dim), len(voxel_size))
    dim = make_list(dim, nb_dim)
    voxel_size = make_list(voxel_size, nb_dim)

    if has_last and x.shape[-1] != nb_dim:
        raise ValueError('Last dimension of `x` should be {} but got {}'
                         .format(nb_dim, x.shape[-1]))
    if not has_last:
        x = x[..., None]

    # compute divergence in each dimension
    div = 0.
    for diff, d, v in zip(x.unbind(-1), dim, voxel_size):
        div += div1d(diff, order, d, v, side, bound)

    return div


def frangi(x, a=0.5, b=0.5, c=500, inv_contrast=False, fwhm=range(1, 8, 2),
           dim=None, bound='replicate', return_scale=False, verbose=False):
    """Frangi (vessel detector) filter.

    Notes
    -----
    .. This function does not support autograd through x.

    Parameters
    ----------
    x : (*batch_shape, *spatial_shape) tensor_like
        Input (batched) tensor.
    a : float, default=0.5
        First Frangi vesselness constant (deviation from line)
        Only used in 3D.
    b : float, default=0.5
        Second Frangi vesselness constant (deviation from blob-like)
    c : float, default=500
        Third Second Frangi vesselness constant (signal to noise)
    inv_contrast : bool, default=False
        If True, detect white ridges (black background).
        Else, detect bloack ridges (white background).
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

    """
    x = torch.as_tensor(x)
    dim = dim or x.dim()
    if not dim in (2, 3):
        raise ValueError('Frangi filter is only implemented in 2D or 3D')

    a = 2*(a**2)
    b = 2*(b**2)
    c = 2*(c**2)

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
            for dd in range(d+1, dim):
                hh = h[..., d, d]
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
        torch.symeig(h, out=(v, torch.empty([])))

        if verbose:
            print('Frangi...')

        if dim == 2:
            msk = v[..., 0] < 0
            if inv_contrast:
                msk.bitwise_not_()
        else:
            if inv_contrast:
                msk = v[..., 1] > 0
                msk.bitwise_or_(v[..., 2] > 0)
            else:
                msk = v[..., 1] < 0
                msk.bitwise_or_(v[..., 2] < 0)

        v.abs_()

        if dim == 2:
            torch.div(v[..., 1], v[..., 0], out=buf2)       # < Rb
            buf2.square_()
            v.square_()
            torch.sum(v, dim=-1, out=buf3)                  # < S

            buf2.div_(-b).exp_()                            # < expRb
            buf3.div_(-c).exp_().neg_().add_(1)             # < expS

            buf2.mul_(buf3)

            buf2[msk] = 0
            return buf2

        elif dim == 3:
            torch.div(v[..., 1], v[..., 2], out=buf1)       # < Ra
            torch.mul(v[..., 1], v[..., 2], out=buf2)
            buf2.sqrt_()
            torch.div(v[..., 0], buf2, out=buf2)            # < Rb
            v.square_()
            torch.sum(v, dim=-1, out=buf3)                  # < S

            buf1.square_().div_(-a).exp_().neg_().add_(1)   # < expRa
            buf2.square_().div_(-b).exp_()                  # < expRb
            buf3.div_(-c).exp_().neg_().add_(1)             # < expS

            buf1.mul_(buf2).mul_(buf3)

            buf1[msk] = 0
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
                scale = torch.zeros_like(v, dtype=torch.int)
        else:
            if return_scale:
                scale[v1 > v0] = i
            v0 = torch.max(v0, v1, out=v0)

    return (v0, scale) if return_scale else v0


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
                view = slice_tensor(result, index, dim)
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
        central = slice_tensor(x, slice(nb_post or None, -nb_pre or None), dim)
        if bound == 'dct2':
            pre = slice_tensor(x, slice(None, nb_pre), dim)
            pre = torch.flip(pre, [dim])
            post = slice_tensor(x, slice(length-nb_post, None), dim)
            post = torch.flip(post, [dim])
            slices.append(tuple([pre, central, post]))
        elif bound == 'dct1':
            pre = slice_tensor(x, slice(1, nb_pre+1), dim)
            pre = torch.flip(pre, [dim])
            post = slice_tensor(x, slice(length-nb_post-1, -1), dim)
            post = torch.flip(post, [dim])
            slices.append(tuple([pre, central, post]))
        elif bound == 'dst2':
            pre = slice_tensor(x, slice(None, nb_pre), dim)
            pre = -torch.flip(pre, [dim])
            post = slice_tensor(x, slice(-nb_post, None), dim)
            post = -torch.flip(post, [dim])
            slices.append(tuple([pre, central, post]))
        elif bound == 'dst1':
            pre = slice_tensor(x, slice(None, nb_pre-1), dim)
            pre = -torch.flip(pre, [dim])
            post = slice_tensor(x, slice(length-nb_post+1, None), dim)
            post = -torch.flip(post, [dim])
            shape1 = list(x.shape)
            shape1[dim] = 1
            zero = torch.zeros([], **backend).expand(shape1)
            slices.append(tuple([pre, zero, central, zero, post]))
        elif bound == 'dft':
            pre = slice_tensor(x, slice(length-nb_pre, None), dim)
            post = slice_tensor(x, slice(None, nb_post), dim)
            slices.append(tuple([pre, central, post]))
        elif bound == 'replicate':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = slice_tensor(x, slice(None, 1), dim).expand(shape_pre)
            post = slice_tensor(x, slice(-1, None), dim).expand(shape_post)
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
