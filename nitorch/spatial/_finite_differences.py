"""Finite-differences operators (gradient, divergence, ...)."""

import torch
from ..core import utils
from ..core.utils import expand, slice_tensor, same_storage, make_vector
from ..core.pyutils import make_list


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


def diff1d(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2'):
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
    dtype = x.dtype
    device = x.device

    # build "light" zero using zero strides
    edge_shape = list(x.shape)
    edge_shape[dim] = 1
    zero = torch.zeros(1, dtype=dtype, device=device)
    zero = expand(zero, edge_shape)

    if order == 1:
        if side == 'f':  # forward -> diff[i] = x[i+1] - x[i]
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            diff = post - pre
            if bound in ('dct2', 'replicate'):
                # x[end+1] = x[end] => diff[end] = 0
                diff = torch.cat((diff, zero), dim)
            elif bound == 'dct1':
                # x[end+1] = x[end-1] => diff[end] = -diff[end-1]
                last = -slice_tensor(diff, -1, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((diff, last), dim)
            elif bound == 'dst2':
                # x[end+1] = -x[end] => diff[end] = -2*x[end]
                last = -2*slice_tensor(x, -1, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((diff, last), dim)
            elif bound in ('dst1', 'zero'):
                # x[end+1] = 0 => diff[end] = -x[end]
                last = -slice_tensor(x, -1, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((diff, last), dim)
            else:
                assert bound == 'dft'
                # x[end+1] = x[0] => diff[end] = x[0] - x[end]
                last = slice_tensor(x, 0, dim) - slice_tensor(x, -1, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((diff, last), dim)

        elif side == 'b':  # backward -> diff[i] = x[i] - x[i-1]
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            diff = post - pre
            if bound in ('dct2', 'replicate'):
                # x[-1] = x[0] => diff[0] = 0
                diff = torch.cat((zero, diff), dim)
            elif bound == 'dct1':
                # x[-1] = x[1] => diff[0] = -diff[1]
                first = -slice_tensor(diff, 0, dim)
                first = torch.unsqueeze(first, dim)
                diff = torch.cat((first, diff), dim)
            elif bound == 'dst2':
                # x[-1] = -x[0] => diff[0] = 2*x[0]
                first = 2*slice_tensor(x, 0, dim)
                first = torch.unsqueeze(first, dim)
                diff = torch.cat((first, diff), dim)
            elif bound in ('dst1', 'zero'):
                # x[-1] = 0 => diff[0] = x[0]
                first = slice_tensor(x, 0, dim)
                first = torch.unsqueeze(first, dim)
                diff = torch.cat((first, diff), dim)
            else:
                assert bound == 'dft'
                # x[-1] = x[end] => diff[0] = x[0] - x[end]
                first = slice_tensor(x, 0, dim) - slice_tensor(x, -1, dim)
                first = torch.unsqueeze(first, dim)
                diff = torch.cat((first, diff), dim)

        elif side == 'c':  # central -> diff[i] = (x[i+1] - x[i-1])/2
            pre = slice_tensor(x, slice(None, -2), dim)
            post = slice_tensor(x, slice(2, None), dim)
            diff = post - pre
            if bound in ('dct2', 'replicate'):
                # x[-1]    = x[0]   => diff[0]   = x[1] - x[0]
                # x[end+1] = x[end] => diff[end] = x[end] - x[end-1]
                first = slice_tensor(x, 1, dim) - slice_tensor(x, 0, dim)
                first = torch.unsqueeze(first, dim)
                last = slice_tensor(x, -1, dim) - slice_tensor(x, -2, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((first, diff, last), dim)
            elif bound == 'dct1':
                # x[-1]    = x[1]     => diff[0]   = 0
                # x[end+1] = x[end-1] => diff[end] = 0
                diff = torch.cat((zero, diff, zero), dim)
            elif bound == 'dst2':
                # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
                # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
                first = slice_tensor(x, 1, dim) + slice_tensor(x, 0, dim)
                first = torch.unsqueeze(first, dim)
                last = - slice_tensor(x, -1, dim) - slice_tensor(x, -2, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((first, diff, last), dim)
            elif bound in ('dst1', 'zero'):
                # x[-1]    = 0 => diff[0]   = x[1]
                # x[end+1] = 0 => diff[end] = -x[end-1]
                first = slice_tensor(x, 1, dim)
                first = torch.unsqueeze(first, dim)
                last = -slice_tensor(x, -2, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((first, diff, last), dim)
            else:
                assert bound == 'dft'
                # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
                # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
                first = slice_tensor(x, 1, dim) - slice_tensor(x, -1, dim)
                first = torch.unsqueeze(first, dim)
                last = slice_tensor(x, 0, dim) - slice_tensor(x, -1, dim)
                last = torch.unsqueeze(last, dim)
                diff = torch.cat((first, diff, last), dim)
            diff = diff / 2.

        if voxel_size != 1:
            diff = diff / voxel_size

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
                         side='f', bound=bound)
            bwd = diff1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                         side='b', bound=bound)
            diff = (fwd - bwd) / voxel_size
        else:
            diff = diff1d(x, order=2, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound)
            diff = diff1d(diff, order=order-2, dim=dim, voxel_size=voxel_size,
                          side=side, bound=bound)

    else:
        diff = diff1d(x, order=1, dim=dim, voxel_size=voxel_size,
                      side=side, bound=bound)
        diff = diff1d(diff, order=order-1, dim=dim, voxel_size=voxel_size,
                      side=side, bound=bound)

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
    diffs = []
    for d, v in zip(dim, voxel_size):
        diffs.append(diff1d(x, order, d, v, side, bound))

    # return
    if drop_last:
        diffs = diffs[0]
    else:
        diffs = torch.stack(diffs, dim=-1)
    return diffs


def div1d(x, order=1, dim=-1, voxel_size=1, side='c', bound='dct2'):
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

    # check options
    bound = bound.lower()
    if bound not in ('dct2', 'dct1', 'dst2', 'dst1', 'dft', 'replicate', 'zero'):
        raise ValueError('Unknown boundary type {}.'.format(bound))
    side = side.lower()[0]
    if side not in ('f', 'b'):
        raise ValueError('Unknown side {}.'.format(side))
    order = int(order)
    if order < 0:
        raise ValueError('Order must be nonnegative but got {}.'.format(order))
    elif order == 0:
        return x

    # ensure tensor
    x = torch.as_tensor(x)

    if order == 1:
        if side == 'f':
            # forward -> diff[i] = x[i+1] - x[i]
            #            div[i]  = x[i-1] - x[i]
            first = -slice_tensor(x, 0, dim).unsqueeze(dim)
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            div = pre - post
            div = torch.cat((first, div), dim)
            if bound in ('dct2', 'replicate'):
                div = slice_tensor(div, slice(None, -1), dim)
                ante = slice_tensor(x, -2, dim).unsqueeze(dim)
                div = torch.cat((div, ante), dim)
            elif bound == 'dct1':
                div_last = slice_tensor(div, -1, dim).unsqueeze(dim)
                div_ante = slice_tensor(div, -2, dim).unsqueeze(dim)
                div = slice_tensor(div, slice(None, -2), dim)
                last = slice_tensor(x, -1, dim).unsqueeze(dim)
                div_ante = div_ante + last
                div = torch.cat((div, div_ante, div_last), dim)
            elif bound == 'dst2':
                div_last = slice_tensor(div, -1, dim).unsqueeze(dim)
                div = slice_tensor(div, slice(None, -1), dim)
                last = slice_tensor(x, -1, dim).unsqueeze(dim)
                div_last = div_last - last
                div = torch.cat((div, div_last), dim)
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                first = slice_tensor(x, -1, dim) - slice_tensor(x, 0, dim)
                first = torch.unsqueeze(first, dim)
                div = slice_tensor(div, slice(1, None), dim)
                div = torch.cat((first, div), dim)

        elif side == 'b':
            # backward -> diff[i] = x[i] - x[i-1]
            #             div[i]  = x[i+1] - x[i]
            last = slice_tensor(x, -1, dim).unsqueeze(dim)
            pre = slice_tensor(x, slice(None, -1), dim)
            post = slice_tensor(x, slice(1, None), dim)
            div = pre - post
            div = torch.cat((div, last), dim)
            if bound in ('dct2', 'replicate'):
                div = slice_tensor(div, slice(1, None), dim)
                second = -slice_tensor(x, 1, dim).unsqueeze(dim)
                div = torch.cat((second, div), dim)
            elif bound == 'dct1':
                div_first = slice_tensor(div, 0, dim).unsqueeze(dim)
                div_second = slice_tensor(div, 1, dim).unsqueeze(dim)
                div = slice_tensor(div, slice(2, None), dim)
                first = slice_tensor(x, 0, dim).unsqueeze(dim)
                div_second = div_second - first
                div = torch.cat((div_first, div_second, div), dim)
            elif bound == 'dst2':
                div_first = slice_tensor(div, 0, dim).unsqueeze(dim)
                div = slice_tensor(div, slice(1, None), dim)
                first = slice_tensor(x, 0, dim).unsqueeze(dim)
                div_first = div_first + first
                div = torch.cat((div_first, div), dim)
            elif bound in ('dst1', 'zero'):
                pass
            else:
                assert bound == 'dft'
                last = slice_tensor(x, -1, dim) - slice_tensor(x, 0, dim)
                last = torch.unsqueeze(last, dim)
                div = slice_tensor(div, slice(None, -1), dim)
                div = torch.cat((div, last), dim)

        # elif side == 'c':
        #     # central -> diff[i] = (x[i+1] - x[i-1])/2
        #     #         -> div[i]  = (x[i-1] - x[i+1])/2
        #     pre = slice_tensor(x, slice(None, -2), dim)
        #     post = slice_tensor(x, slice(2, None), dim)
        #     div = pre - post
        #     if bound in ('dct2', 'replicate'):
        #         # x[-1]    = x[0]   => diff[0]   = x[1] - x[0]
        #         # x[end+1] = x[end] => diff[end] = x[end] - x[end-1]
        #         first = slice_tensor(x, 1, dim) - slice_tensor(x, 0, dim)
        #         first = torch.unsqueeze(first, dim)
        #         last = slice_tensor(x, -1, dim) - slice_tensor(x, -2, dim)
        #         last = torch.unsqueeze(last, dim)
        #         diff = torch.cat((first, diff, last), dim)
        #     elif bound == 'dct1':
        #         # x[-1]    = x[1]     => diff[0]   = 0
        #         # x[end+1] = x[end-1] => diff[end] = 0
        #         diff = torch.cat((zero, diff, zero), dim)
        #     elif bound == 'dst2':
        #         # x[-1]    = -x[0]   => diff[0]   = x[1] + x[0]
        #         # x[end+1] = -x[end] => diff[end] = -(x[end] + x[end-1])
        #         first = slice_tensor(x, 1, dim) + slice_tensor(x, 0, dim)
        #         first = torch.unsqueeze(first, dim)
        #         last = - slice_tensor(x, -1, dim) - slice_tensor(x, -2, dim)
        #         last = torch.unsqueeze(last, dim)
        #         diff = torch.cat((first, diff, last), dim)
        #     elif bound in ('dst1', 'zero'):
        #         # x[-1]    = 0 => diff[0]   = x[1]
        #         # x[end+1] = 0 => diff[end] = -x[end-1]
        #         first = slice_tensor(x, 1, dim)
        #         first = torch.unsqueeze(first, dim)
        #         last = -slice_tensor(x, -2, dim)
        #         last = torch.unsqueeze(last, dim)
        #         diff = torch.cat((first, diff, last), dim)
        #     else:
        #         assert bound == 'dft'
        #         # x[-1]    = x[end] => diff[0]   = x[1] - x[end]
        #         # x[end+1] = x[0]   => diff[end] = x[0] - x[end-1]
        #         first = slice_tensor(x, 1, dim) - slice_tensor(x, -1, dim)
        #         first = torch.unsqueeze(first, dim)
        #         last = slice_tensor(x, 0, dim) - slice_tensor(x, -1, dim)
        #         last = torch.unsqueeze(last, dim)
        #         diff = torch.cat((first, diff, last), dim)

        if voxel_size != 1:
            div = div / voxel_size

    else:
        div = div1d(x, order=order-1, dim=dim, voxel_size=voxel_size,
                    side=side, bound=bound)
        div = div1d(div, order=1, dim=dim, voxel_size=voxel_size,
                    side=side, bound=bound)

    return div


def div(x, order=1, dim=-1, voxel_size=1, side='f', bound='dct2', value=0):
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
        div = div + div1d(diff, order, d, v, side, bound)

    return div


def sobel(x, dim=None, bound='replicate', value=0):
    """Sobel (edge detector) filter.

    This function supports autograd.

    Parameters
    ----------
    x : (*batch_shape, *spatial_shape) tensor_like
        Input (batched) tensor.
    dim : int
        Length of `spatial_shape`.
    bound : bound_like, default='dct2'
        Boundary condition.
    value : number, defualt=0
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
    dim : int
        Dimension along which the tensor was chunked.

    Returns
    -------
    lincomb : tensor

    """

    result = None
    for chunks, weight in zip(slices, weights):

        # multiply chunk with weight
        new_chunks = []
        for chunk in chunks:
            if chunk.numel() == 0:
                continue
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
    offsets : sequence[int]
        Offsets to extract, with respect to each voxel.
        To extract a centered window of length 3, use `offsets=[-1, 0, 1]`.
    bound : bound_like, default='dct2'
        Boundary conditions
    value : number, default=0
        Filling value if `bound='constant'`

    Returns
    -------
    win : tuple[tuple[tensor]]
        The first level corresponds to offsets.
        The second levels are tensors that could be concatenated along
        `dim` to generate the input tensor shifted by `offset`. However,
        to avoid unnecessary allocations, a list of (eventually empty)
        chunks is returned instead of the full shifted tensor.
        Some of these tensors can be views into the input tensor.

    """
    offsets = list(offsets)
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
            slices.append([pre, central, post])
        elif bound == 'dct1':
            pre = slice_tensor(x, slice(1, nb_pre+1), dim)
            pre = torch.flip(pre, [dim])
            post = slice_tensor(x, slice(length-nb_post-1, -1), dim)
            post = torch.flip(post, [dim])
            slices.append([pre, central, post])
        elif bound == 'dst2':
            pre = slice_tensor(x, slice(None, nb_pre), dim)
            pre = -torch.flip(pre, [dim])
            post = slice_tensor(x, slice(-nb_post, None), dim)
            post = -torch.flip(post, [dim])
            slices.append([pre, central, post])
        elif bound == 'dst1':
            pre = slice_tensor(x, slice(None, nb_pre-1), dim)
            pre = -torch.flip(pre, [dim])
            post = slice_tensor(x, slice(length-nb_post+1, None), dim)
            post = -torch.flip(post, [dim])
            shape1 = list(x.shape)
            shape1[dim] = 1
            zero = torch.zeros([], **backend).expand(shape1)
            slices.append([pre, zero, central, zero, post])
        elif bound == 'dft':
            pre = slice_tensor(x, slice(length-nb_pre, None), dim)
            post = slice_tensor(x, slice(None, nb_post), dim)
            slices.append([pre, central, post])
        elif bound == 'replicate':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = slice_tensor(x, slice(None, 1), dim).expand(shape_pre)
            post = slice_tensor(x, slice(-1, None), dim).expand(shape_post)
            slices.append([pre, central, post])
        elif bound == 'zero':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = torch.zeros([], **backend).expand(shape_pre)
            post = torch.zeros([], **backend).expand(shape_post)
            slices.append([pre, central, post])
        elif bound == 'constant':
            shape_pre = list(x.shape)
            shape_pre[dim] = nb_pre
            shape_post = list(x.shape)
            shape_post[dim] = nb_post
            pre = torch.full([], value, **backend).expand(shape_pre)
            post = torch.full([], value, **backend).expand(shape_post)
            slices.append([pre, central, post])
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
