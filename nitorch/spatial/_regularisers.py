import torch
from nitorch import core
from nitorch.core.utils import movedim, make_vector, unsqueeze, fast_movedim
from nitorch.core.py import ensure_list
from nitorch.core.linalg import sym_matvec, sym_solve
from ._finite_differences import diff, div, diff1d, div1d
from ._spconv import spconv
import itertools


def absolute(field, weights=None):
    """Precision matrix for the Absolute energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : tensor

    """
    field = torch.as_tensor(field)
    if weights is not None:
        backend = dict(dtype=field.dtype, device=field.device)
        weights = torch.as_tensor(weights, **backend)
        return field * weights
    else:
        return field


def absolute_grid(grid, voxel_size=1, weights=None):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = grid * voxel_size.square()
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)
        grid = grid * weights[..., None]
    return grid


def membrane(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Membrane energy

    Note
    ----
    .. This is exactly equivalent to SPM's membrane energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial) tensor

    """
    if weights is None:
        return _membrane_l2(field, voxel_size, bound, dim)

    def mul_(x, y):
        """Smart in-place multiplication"""
        if ((torch.is_tensor(x) and x.requires_grad) or
                (torch.is_tensor(y) and y.requires_grad)):
            return x * y
        else:
            return x.mul_(y)

    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)
    dims = list(range(field.dim()-dim, field.dim()))
    fieldf = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    weights = torch.as_tensor(weights, **backend)
    fieldf = mul_(fieldf, weights[..., None])
    fieldb = diff(field, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    fieldb = mul_(fieldb, weights[..., None])
    dims = list(range(fieldb.dim() - 1 - dim, fieldb.dim() - 1))
    fieldb = div(fieldb, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    dims = list(range(fieldf.dim()-1-dim, fieldf.dim()-1))
    field = div(fieldf, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    del fieldf
    field += fieldb
    field *= 0.5
    return field


def _membrane_l2(field, voxel_size=1, bound='dct2', dim=None):
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)

    dims = list(range(field.dim()-dim, field.dim()))
    fieldf = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    dims = list(range(fieldf.dim()-1-dim, fieldf.dim()-1))
    field = div(fieldf, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    return field


def membrane_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    if (voxel_size != 1).any():
        grid = grid * voxel_size.square()
    grid = fast_movedim(grid, -1, -(dim + 1))
    grid = membrane(grid, weights=weights, voxel_size=voxel_size,
                    bound=bound, dim=dim)
    grid = fast_movedim(grid, -(dim + 1), -1)
    return grid


def bending(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Bending energy

    Note
    ----
    .. This is exactly equivalent to SPM's bending energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial) tensor

    """
    if weights is None:
        return _bending_l2(field, voxel_size, bound, dim)

    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)
    else:
        voxel_size = core.py.ensure_list(voxel_size, dim)
    bound = ensure_list(bound, dim)
    dims = list(range(field.dim()-dim, field.dim()))
    if weights is not None:
        backend = dict(dtype=field.dtype, device=field.device)
        weights = torch.as_tensor(weights, **backend)

    mom = 0
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj)
                    dj = dj * weights
                    dj = div1d(dj, **optj)
                    dj = div1d(dj, **opti)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj.mul_(2)
                    mom += dj
    mom.div_(4.)
    return mom


def _bending_l2(field, voxel_size=1, bound='dct2', dim=None):
    """Precision matrix for the Bending energy

    Note
    ----
    .. This is exactly equivalent to SPM's bending energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()

    Returns
    -------
    field : (..., *spatial) tensor

    """
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)
    else:
        voxel_size = core.py.ensure_list(voxel_size, dim)
    bound = ensure_list(bound, dim)
    dims = list(range(field.dim()-dim, field.dim()))

    # allocate buffers
    if not field.requires_grad:
        bufi = torch.empty_like(field)
        bufij = torch.empty_like(field)
        bufjj = torch.empty_like(field)
    else:
        bufi = bufij = bufjj = None

    mom = 0
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti, out=bufi)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj, out=bufij)
                    dj = div1d(dj, **optj, out=bufjj)
                    dj = div1d(dj, **opti, out=bufij)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj.mul_(2)
                    mom += dj
    mom.div_(4.)
    return mom


def bending_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    if (voxel_size != 1).any():
        grid = grid * voxel_size.square()
    grid = fast_movedim(grid, -1, -(dim + 1))
    grid = bending(grid, weights=weights, voxel_size=voxel_size,
                   bound=bound, dim=dim)
    grid = fast_movedim(grid, -(dim + 1), -1)
    return grid


def lame_shear(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Shear component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the second Lame constant (the shear modulus).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the Frobenius norm of the symmetric part
       of the Jacobian (shears on the off-diagonal and zooms on the
       diagonal).
    .. Formaly: `<f, Lf> = \int \sum_i (df_i/dx_i)^2
                                + \sum_{j > i} (df_j/dx_i + df_i/dx_j)^2 dx

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    bound = ensure_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)

    mom = torch.zeros_like(grid)
    for i in range(dim):
        # symmetric part
        x_i = grid[..., i]
        for j in range(i, dim):
            for side_i in ('f', 'b'):
                opt_ij = dict(dim=dims[j], side=side_i, bound=bound[j],
                              voxel_size=voxel_size[j])
                diff_ij = diff1d(x_i, **opt_ij).mul_(voxel_size[i])
                if i == j:
                    # diagonal elements
                    diff_ij_w = diff_ij if weights is None else diff_ij * weights
                    mom[..., i].add_(div1d(diff_ij_w, **opt_ij), alpha=0.5)
                else:
                    # off diagonal elements
                    x_j = grid[..., j]
                    for side_j in ('f', 'b'):
                        opt_ji = dict(dim=dims[i], side=side_j, bound=bound[i],
                                      voxel_size=voxel_size[i])
                        diff_ji = diff1d(x_j, **opt_ji).mul_(voxel_size[j])
                        diff_ji = diff_ji.add_(diff_ij).mul_(0.5)
                        if weights is not None:
                            diff_ji = diff_ji * weights
                        mom[..., j].add_(div1d(diff_ji, **opt_ji), alpha=0.25)
                        mom[..., i].add_(div1d(diff_ji, **opt_ij), alpha=0.25)
                    del x_j
        del x_i
    del grid

    mom *= 2 * voxel_size  # JA added an additional factor 2 to the kernel
    return mom


def lame_div(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Divergence component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the first Lame constant (the divergence).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the square of the trace of the Jacobian
       (i.e., volume changes)
    .. Formaly: `<f, Lf> = \int (\sum_i df_i/dx_i)^2 dx

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1 (actually unused)
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    bound = ensure_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)

    # precompute gradients
    grad = [dict(f={}, b={}) for _ in range(dim)]
    opt = [dict(f={}, b={}) for _ in range(dim)]
    for i in range(dim):
        x_i = grid[..., i]
        for side in ('f', 'b'):
            opt_i = dict(dim=dims[i], side=side, bound=bound[i])
            grad[i][side] = diff1d(x_i, **opt_i)
            opt[i][side] = opt_i

    # compute divergence
    mom = torch.zeros_like(grid)
    all_sides = list(itertools.product(['f', 'b'], repeat=dim))
    for sides in all_sides:
        div = 0
        for i, side in enumerate(sides):
            div += grad[i][side]
        if weights is not None:
            div = div * weights
        for i, side in enumerate(sides):
            mom[..., i] += div1d(div, **(opt[i][side]))

    mom /= float(2 ** dim)  # weight sides combinations
    return mom


# aliases to avoid shadowing
_absolute = absolute
_membrane = membrane
_bending = bending


def absolute_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Absolute energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's absolute energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1 (unused)

    Returns
    -------
    kernel : (1,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()

    kernel = torch.sparse_coo_tensor(
        torch.zeros([dim, 1], dtype=torch.long, device=device),
        torch.ones([1], dtype=dtype, device=device),
        [1] * dim)
    return kernel


def absolute_grid_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Absolute energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's absolute energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (1,)*dim sparse tensor

    """
    kernel = absolute_kernel(dim, voxel_size, dtype=dtype, device=device)
    voxel_size = core.utils.make_vector(voxel_size, dim,
                                        **core.utils.backend(kernel))
    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def membrane_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Membrane energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's membrane energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (3,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = [2 * vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [-vx[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    return kernel


def membrane_grid_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Membrane energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's membrane energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, [3,]*dim) sparse tensor

    """
    kernel = membrane_kernel(dim, voxel_size, dtype=dtype, device=device)
    voxel_size = core.utils.make_vector(voxel_size, dim,
                                        **core.utils.backend(kernel))
    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def bending_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Bending energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's bending energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (5,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()
    vx2 = vx.square()
    cvx = torch.combinations(vx, r=2).prod(dim=-1)

    # build sparse kernel
    kernel = [6 * vx2.sum() + 8 * cvx.sum()]
    center_index = [2] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross 1st order
        kernel += [-4*vx[d]*vx.sum()] * 2
        index = list(center_index)
        index[d] = 1
        indices.append(index)
        index = list(center_index)
        index[d] = 3
        indices.append(index)
        # cross 2nd order
        kernel += [vx2[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 4
        indices.append(index)
        for dd in range(d+1, dim):
            # off
            kernel += [2 * vx[d] * vx[dd]] * 4
            index = list(center_index)
            index[d] = 1
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 1
            index[dd] = 3
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 3
            indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [5] * dim)

    return kernel


def bending_grid_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Bending energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's bending energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, dim, [5,]*dim) sparse tensor

    """
    kernel = bending_kernel(dim, voxel_size, dtype=dtype, device=device)
    voxel_size = core.utils.make_vector(voxel_size, dim,
                                        **core.utils.backend(kernel))
    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def lame_shear_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's LE energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = []
    center_index = [1] * dim
    indices = []
    for d in range(dim):  # input channel
        kernel += [2 + 2*vx.sum()/vx[d]]
        index = [d, d, *center_index]
        indices.append(index)
        for dd in range(dim):  # cross
            if dd == d:
                kernel += [-2] * 2
            else:
                kernel += [-vx[dd]/vx[d]] * 2
            index = [d, d, *center_index]
            index[2 + dd] = 0
            indices.append(index)
            index = [d, d, *center_index]
            index[2 + dd] = 2
            indices.append(index)
        for dd in range(d+1, dim):  # output channel
            kernel += [-0.25] * 4
            index = [d, dd, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 0
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 0
            indices.append(index)
            index = [d, dd, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 2
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 2
            indices.append(index)
            kernel += [0.25] * 4
            index = [d, dd, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 2
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 2
            indices.append(index)
            index = [d, dd, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 0
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 0
            indices.append(index)

    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [dim, dim] + [3] * dim)

    return kernel


def lame_div_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's LE energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1 (actually unused)

    Returns
    -------
    kernel : (dim, dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()

    # build sparse kernel
    kernel = []
    center_index = [1] * dim
    indices = []
    for d in range(dim):  # input channel
        kernel += [2]
        index = [d, d, *center_index]
        indices.append(index)
        kernel += [-1] * 2
        index = [d, d, *center_index]
        index[2 + d] = 0
        indices.append(index)
        index = [d, d, *center_index]
        index[2 + d] = 2
        indices.append(index)
        for dd in range(d+1, dim):  # output channel
            for d1 in range(dim):   # interation 1
                for d2 in range(d + 1, dim):  # interation 2
                    kernel += [-0.25] * 4
                    index = [d, dd, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [d, dd, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 2
                    indices.append(index)
                    kernel += [0.25] * 4
                    index = [d, dd, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [d, dd, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 0
                    indices.append(index)

    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [dim, dim] + [3] * dim)

    return kernel


def sum_kernels(dim, *kernels):
    """Sum sparse kernels of different shapes

    Parameters
    ----------
    dim : int
        Number of spatial dimensions
    *kernels : sparse tensor
        Kernels to sum

    Returns
    -------
    kernel : sparse tensor
        Sum of input tensors

    """
    # compute maximum shape
    spatial = [0] * dim
    device = None
    dtype = None
    for k in kernels:
        device = k.device
        dtype = k.dtype
        kspatial = k.shape[-dim:]
        spatial = [max(s, ks) for s, ks in zip(spatial, kspatial)]
    has_matrix = any(kernel.dim() == dim+2 for kernel in kernels)
    has_diag = any(kernel.dim() == dim+1 for kernel in kernels)

    # prepare output
    out_shape = [dim, dim] if has_matrix else [dim] if has_diag else []
    out_shape += spatial
    out = torch.sparse_coo_tensor(
        torch.zeros([len(out_shape), 0], dtype=torch.long, device=device),
        torch.zeros([0], dtype=dtype, device=device),
        out_shape)

    # sum kernels
    for kernel in kernels:
        offset = [(s - ks)//2 for s, ks in zip(spatial, kernel.shape[-dim:])]
        if any(offset):
            new_shape = [*kernel.shape[:-dim], *spatial]
            offset = torch.as_tensor(offset, **core.utils.backend(kernel._indices()))
            indices = kernel._indices()
            indices[-dim:] += offset[:, None]
            kernel = torch.sparse_coo_tensor(
                indices, kernel._values(), new_shape)
        if has_matrix:
            if kernel.dim() == dim:
                for d in range(len(out)):
                    pad_indices = torch.full([], d, dtype=torch.long, device=kernel.device)
                    pad_indices = pad_indices.expand([2, kernel._indices().shape[-1]])
                    indices = torch.cat([pad_indices, kernel._indices()], 0)
                    new_kernel = torch.sparse_coo_tensor(
                        indices, kernel._values(), out_shape)
                    out += new_kernel
            elif kernel.dim() == dim + 1:
                for d in range(len(out)):
                    pad_indices = torch.full([], d, dtype=torch.long, device=kernel.device)
                    pad_indices = pad_indices.expand([2, kernel[d]._indices().shape[-1]])
                    indices = torch.cat([pad_indices, kernel[d]._indices()], 0)
                    new_kernel = torch.sparse_coo_tensor(
                        indices, kernel[d]._values(), out_shape)
                    out += new_kernel
            else:
                out += kernel
        else:
            out += kernel

    out = out.coalesce()
    return out


def regulariser_grid(v, absolute=0, membrane=0, bending=0, lame=0,
                     factor=1, voxel_size=1, bound='dft', weights=None,
                     kernel=False):
    """Precision matrix for a mixture of energies for a deformation grid.

    Parameters
    ----------
    v : (..., *spatial, dim) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1
    bound : str, default='dft'
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    v = torch.as_tensor(v)
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    if torch.is_tensor(kernel) or kernel:
        if not torch.is_tensor(kernel):
            kernel = regulariser_grid_kernel(dim, absolute, membrane, bending,
                                             lame, factor, voxel_size, **backend)
        v = core.utils.fast_movedim(v, -1, -dim-1)
        v = spconv(v, kernel, bound=bound, dim=dim)
        v = core.utils.fast_movedim(v, -dim-1, -1)
        return v

    voxel_size = make_vector(voxel_size, dim, **backend)
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = make_vector(lame, 2, **backend)
    lame = [l*factor for l in lame]
    fdopt = dict(bound=bound, voxel_size=voxel_size)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = ensure_list(wl, 2)

    y = 0
    if absolute:
        y += absolute_grid(v, weights=wa, voxel_size=voxel_size) * absolute
    if membrane:
        y += membrane_grid(v, weights=wm, **fdopt) * membrane
    if bending:
        y += bending_grid(v, weights=wb, **fdopt) * bending
    if lame[0]:
        y += lame_div(v, weights=wl[0], **fdopt) * lame[0]
    if lame[1]:
        y += lame_shear(v, weights=wl[1], **fdopt) * lame[1]

    if y is 0:
        y = torch.zeros_like(v)
    return y


def regulariser_grid_kernel(dim, absolute=0, membrane=0, bending=0, lame=0,
                            factor=1, voxel_size=1, dtype=None, device=None):
    """Precision kernel for a mixture of energies for a deformation grid.

    Parameters
    ----------
    dim : int
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1

    Returns
    -------
    kernel : ([dim, [dim,]] *spatial) sparse tensor

    """
    lame_div, lame_shear = core.py.make_list(lame, 2)
    backend = dict(dtype=dtype, device=device)

    kernels = []
    if absolute:
        kernels.append(absolute_grid_kernel(dim, voxel_size, **backend) * absolute)
    if membrane:
        kernels.append(membrane_grid_kernel(dim, voxel_size, **backend) * membrane)
    if bending:
        kernels.append(bending_grid_kernel(dim, voxel_size, **backend) * bending)
    if lame_div:
        kernels.append(lame_div_kernel(dim, voxel_size, **backend) * lame_div)
    if lame_shear:
        kernels.append(lame_shear_kernel(dim, voxel_size, **backend) * lame_shear)
    kernel = sum_kernels(dim, *kernels)
    kernel *= factor
    return kernel


def regulariser(x, absolute=0, membrane=0, bending=0, factor=1,
                voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for a mixture of energies.

    Parameters
    ----------
    x : (..., K, *spatial) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : (sequence of) float, default=1
    voxel_size : (sequence of) float, default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lx : (..., K, *spatial) tensor

    """
    if not (absolute or membrane or bending):
        return torch.zeros_like(x)

    backend = dict(dtype=x.dtype, device=x.device)
    dim = dim or x.dim() - 1
    nb_prm = x.shape[-dim-1]

    voxel_size = make_vector(voxel_size, dim, **backend)
    factor = make_vector(factor, nb_prm, **backend)
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
    else:
        wa = wm = wb = weights
    # wa = core.utils.unsqueeze(wa, 0, max(0, dim+1-wa.dim()))
    # wm = core.utils.unsqueeze(wm, 0, max(0, dim+1-wa.dim()))
    # wb = core.utils.unsqueeze(wb, 0, max(0, dim+1-wa.dim()))

    y = torch.zeros_like(x)
    if absolute:
        y.add_(_absolute(x, weights=wa), alpha=absolute)
    if membrane:
        y.add_(_membrane(x, weights=wm, **fdopt), alpha=membrane)
    if bending:
        y.add_(_bending(x, weights=wb, **fdopt), alpha=bending)

    pad_spatial = (Ellipsis,) + (None,) * nb_prm
    return y.mul_(factor[pad_spatial])


def regulariser_kernel(dim, absolute=0, membrane=0, bending=0,
                       factor=1, voxel_size=1, dtype=None, device=None):
    """Precision kernel for a mixture of energies for a deformation grid.

    Parameters
    ----------
    dim : int
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1

    Returns
    -------
    kernel : ([dim, [dim,]] *spatial) sparse tensor

    """
    backend = dict(dtype=dtype, device=device)

    kernels = []
    if absolute:
        kernels.append(absolute_kernel(dim, voxel_size, **backend))
    if membrane:
        kernels.append(membrane_kernel(dim, voxel_size, **backend))
    if bending:
        kernels.append(bending_kernel(dim, voxel_size, **backend))
    kernel = sum_kernels(dim, *kernels)
    kernel *= factor
    return kernel


def quadnesterov(A, b, x=None, precond=None, lr=0.5, momentum=0.9, max_iter=None,
                 tolerance=1e-5, inplace=True, verbose=False, stop='E',
                 sum_dtype=torch.double):
    """Nesterov accelerated gradient for quadratic problems."""
    if x is None:
        x = torch.zeros_like(b)
    elif not inplace:
        x = x.clone()
    max_iter = max_iter or len(b) * 10

    # Create functor if A is a tensor
    if isinstance(A, torch.Tensor):
        A_tensor = A
        A = lambda x: A_tensor.mm(x)

    # Create functor if D is a tensor
    if isinstance(precond, torch.Tensor):
        D_tensor = precond
        precond = lambda x: x * D_tensor
    precond = precond or (lambda x: x)

    r = b - A(x)

    if tolerance or verbose:
        if stop == 'residual':
            stop = 'e'
        elif stop == 'norm':
            stop = 'a'
        stop = stop[0].lower()
        if stop == 'e':
            obj0 = r.square().sum(dtype=sum_dtype).sqrt()
        else:
            obj0 = A(x).sub_(2 * b).mul_(x)
            obj0 = 0.5 * torch.sum(obj0, dtype=sum_dtype)
        if verbose:
            s = '{:' + str(len(str(max_iter + 1))) + '} | {} = {:12.6g}'
            print(s.format(0, stop, obj0))
        obj = torch.zeros(max_iter + 1, dtype=sum_dtype, device=obj0.device)
        obj[0] = obj0

    delta = torch.zeros_like(x)
    for n_iter in range(max_iter):

        prev_momentum = momentum or n_iter / (n_iter + 3)
        cur_momentum = momentum or (n_iter + 1) / (n_iter + 4)
        r = precond(r)

        delta.mul_(prev_momentum)
        delta.add_(r, alpha=lr)
        x.add_(delta, alpha=cur_momentum).add_(r, alpha=lr)
        r = b - A(x)

        # Check convergence
        if tolerance or verbose:
            if stop == 'e':
                obj1 = r.square().sum(dtype=sum_dtype).sqrt()
            else:
                obj1 = A(x).sub_(2 * b).mul_(x)
                obj1 = 0.5 * torch.sum(obj1, dtype=sum_dtype)
            obj[n_iter] = obj1
            gain = core.optim.get_gain(obj[:n_iter + 1], monotonicity='decreasing')
            if verbose:
                width = str(len(str(max_iter + 1)))
                s = '{:' + width + '} | {} = {:12.6g} | gain = {:12.6g}'
                print(s.format(n_iter, stop, obj[n_iter], gain))
            if gain.abs() < tolerance:
                break

    return x


def solve_field_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                    factor=1, voxel_size=1, bound='dct2', dim=None,
                    optim='cg', max_iter=16, stop='e', verbose=False, weights=None,
                    precond=None):
    """Solve a positive-definite linear system of the form (H + L)x = g

    Parameters
    ----------
    hessian : (..., 1 or K or K*(K+1)//2, *spatial) tensor
    gradient : (..., K, *spatial) tensor
    absolute : (sequence of) float, default=0
    membrane : (sequence of) float, default=0
    bending : (sequence of) float, default=0
    factor : (sequence of) float, default=1
    voxel_size : (sequence of) float, default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.

    """
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = dim or gradient.dim() - 1
    nb_prm = gradient.shape[-dim-1]

    voxel_size = make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-dim-1] in (1, gradient.shape[-dim-1])

    factor = make_vector(factor, nb_prm, **backend)
    pad_spatial = (Ellipsis,) + (None,) * nb_prm
    factor = factor[pad_spatial]
    no_reg = not (membrane or bending)

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
    else:
        wa = wm = wb = weights
    has_weights = (wa is not None or wm is not None or wb is not None)

    def regulariser(x):
        y = torch.zeros_like(x)
        if absolute:
            y.add_(_absolute(x, weights=wa), alpha=absolute)
        if membrane:
            y.add_(_membrane(x, weights=wm, **fdopt), alpha=membrane)
        if bending:
            y.add_(_bending(x, weights=wb, **fdopt), alpha=bending)
        return y.mul_(factor)

    # diagonal of the regulariser
    if has_weights:
        smo = torch.zeros_like(gradient)
    else:
        smo = gradient.new_zeros([nb_prm] + [1]*dim)
    if absolute:
        smo.add_(absolute_diag(weights=wa), alpha=absolute)
    if membrane:
        smo.add_(membrane_diag(weights=wm, **fdopt), alpha=membrane)
    if bending:
        smo.add_(bending_diag(weights=wb, **fdopt), alpha=bending)
    smo = smo * factor

    if is_diag:
        hessian_smo = hessian + smo
    else:
        hessian_smo = hessian.clone()
        core.utils.slice_tensor(hessian_smo, slice(nb_prm), -dim-1).add_(smo)

    def s2h(s):
        # do not slice if hessian_smo is constant across space
        if s is Ellipsis:
            return s
        if all(sz == 1 for sz in hessian_smo.shape[-dim:]):
            s = list(s)
            s[-dim:] = [slice(None)] * dim
            s = tuple(s)
        return s

    def matvec(h, x):
        h = h.transpose(-dim-1, -1)
        x = x.transpose(-dim-1, -1)
        x = sym_matvec(h, x)
        x = x.transpose(-dim-1, -1)
        return x

    def solve(h, x):
        h = h.transpose(-dim-1, -1)
        x = x.transpose(-dim-1, -1)
        x = sym_solve(h, x)
        x = x.transpose(-dim-1, -1)
        return x

    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: matvec(hessian, x) + regulariser(x)))
    if precond is None:
        precond = ((lambda x, s=Ellipsis: x[s] / hessian_smo[s2h(s)]) if is_diag else
                   (lambda x, s=Ellipsis: solve(hessian_smo[s2h(s)], x[s])))
    elif precond is False:
        precond = lambda x: x

    if no_reg:
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else 'checkerboard')
        optim = getattr(core.optim, optim)
        result = optim(forward, gradient, precond=precond, stop=stop, **prm)
    return result


def solve_grid_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                   lame=0, factor=1, voxel_size=1, bound='dft', weights=None,
                   optim='cg', max_iter=16, stop='e', verbose=False, precond=None):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Parameters
    ----------
    hessian : (..., *spatial, 1 or D or D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    """
    hessian, gradient = core.utils.to_max_backend(hessian, gradient)
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)

    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = [l*factor for l in make_vector(lame, 2, **backend)]
    no_reg = not (membrane or bending or any(lame))

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = ensure_list(wl, 2)
    has_weights = (wa is not None or wm is not None or wb is not None or
                   wl[0] is not None or wl[1] is not None)

    def regulariser(v):
        y = torch.zeros_like(v)
        if absolute:
            y.add_(absolute_grid(v, weights=wa, voxel_size=voxel_size), alpha=absolute)
        if membrane:
            y.add_(membrane_grid(v, weights=wm, **fdopt), alpha=membrane)
        if bending:
            y.add_(bending_grid(v, weights=wb, **fdopt), alpha=bending)
        if lame[0]:
            y.add_(lame_div(v, weights=wl[0], **fdopt), alpha=lame[0])
        if lame[1]:
            y.add_(lame_shear(v, weights=wl[1], **fdopt), alpha=lame[1])
        return y

    # diagonal of the regulariser
    vx2 = voxel_size.square()
    ivx2 = vx2.reciprocal()
    smo = torch.zeros_like(gradient) if has_weights else 0
    if absolute:
        if wa is not None:
            smo.add_(wa, alpha=absolute * vx2)
        else:
            smo += absolute * vx2
    if membrane:
        if wm is not None:
            smo.add_(wm, alpha=2 * membrane * ivx2.sum() * vx2)
        else:
            smo += 2 * membrane * ivx2.sum() * vx2
    if bending:
        val = torch.combinations(ivx2, r=2).prod(dim=-1).sum()
        if wb is not None:
            smo.add_(wb, alpha=(8 * val + 6 * ivx2.square().sum()) * vx2)
        else:
            smo += bending * (8 * val + 6 * ivx2.square().sum()) * vx2
    if lame[0]:
        if wl[0] is not None:
            smo.add_(wl[0], alpha=2 * lame[0])
        else:
            smo += 2 * lame[0]
    if lame[1]:
        if wl[1] is not None:
            smo.add_(wl[1], alpha=2 * lame[1] * (1 + ivx2.sum() / ivx2))
        else:
            smo += 2 * lame[1] * (1 + ivx2.sum() / ivx2)

    if smo.shape[-1] > hessian.shape[-1]:
        hessian_smo = hessian + smo
    else:
        hessian_smo = hessian.clone()
        hessian_smo[..., :dim] += smo
    is_diag = hessian_smo.shape[-1] in (1, gradient.shape[-1])

    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))
    if precond is None:
        precond = ((lambda x, s=Ellipsis: x[s] / hessian_smo[s]) if is_diag else
                   (lambda x, s=Ellipsis: sym_solve(hessian_smo[s], x[s])))
    elif precond is False:
        precond = lambda x: x

    if no_reg:
        # no spatial regularisation: we can use a closed-form
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else
                             2 if any(lame) else
                             'checkerboard')
        optim = (quadnesterov if optim.startswith('nesterov') else
                 getattr(core.optim, optim))
        # prm['verbose'] = True
        result = optim(forward, gradient, precond=precond, stop=stop, **prm)
    return result


def solve_kernel_grid_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                   lame=0, factor=1, voxel_size=1, bound='dft',
                   optim='relax', max_iter=16, stop='e', verbose=False, precond=None):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Parameters
    ----------
    hessian : (..., *spatial, 1 or D or D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    """
    hessian, gradient = core.utils.to_max_backend(hessian, gradient)
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    lame = core.py.ensure_list(lame)
    no_reg = not (membrane or bending or any(lame))
    if not no_reg:
        ker = regulariser_grid_kernel(dim, absolute=absolute, membrane=membrane,
                                      bending=bending, lame=lame, factor=factor,
                                      voxel_size=voxel_size, **backend)

    # pre-multiply by factor
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = [l*factor for l in make_vector(lame, 2, **backend)]

    # regulariser
    fdopt = dict(bound=bound)
    def regulariser(v, s=Ellipsis):
        if s and s is not Ellipsis:
            if s[0] is Ellipsis:
                s = s[1:]
            start = [sl.start for sl in s[:dim]]
            stop = [sl.stop for sl in s[:dim]]
            step = [sl.step for sl in s[:dim]]
        else:
            start = step = stop = None
        if no_reg:
            v = absolute_grid(v[s], voxel_size=voxel_size)
        else:
            v = core.utils.fast_movedim(v, -1, -dim-1)
            v = spconv(v, ker, start=start, stop=stop, step=step, dim=dim, **fdopt)
            v = core.utils.fast_movedim(v, -dim-1, -1)
        return v

    # diagonal of the regulariser
    vx2 = voxel_size.square()
    ivx2 = vx2.reciprocal()
    smo = 0
    if absolute:
        smo += absolute * vx2
    if membrane:
        smo += 2 * membrane * ivx2.sum() * vx2
    if bending:
        val = torch.combinations(ivx2, r=2).prod(dim=-1).sum()
        smo += bending * (8 * val + 6 * ivx2.square().sum()) * vx2
    if lame[0]:
        smo += 2 * lame[0]
    if lame[1]:
        smo += 2 * lame[1] * (1 + ivx2.sum() / ivx2)
    hessian_smo = hessian.clone()
    hessian_smo[..., :dim] += smo

    forward = ((lambda x, s=Ellipsis: (x[s] * hessian[s]).add_(regulariser(x, s))) if is_diag else
               (lambda x, s=Ellipsis: sym_matvec(hessian[s], x[s]).add_(regulariser(x, s))))
    if precond is None:
        precond = ((lambda x, s=Ellipsis: x[s] / hessian_smo[s]) if is_diag else
                   (lambda x, s=Ellipsis: sym_solve(hessian_smo[s], x[s])))
    elif precond is False:
        precond = lambda x, s=Ellipsis: x[s]

    if no_reg:
        # no spatial regularisation: we can use a closed-form
        result = precond(gradient)
    else:
        prm = dict(max_iter=max_iter, verbose=verbose)
        if optim == 'relax':
            prm['scheme'] = (3 if bending else
                             2 if any(lame) else
                             'checkerboard')
            prm['mode'] = 2
        optim = (quadnesterov if optim.startswith('nesterov') else
                 getattr(core.optim, optim))
        prm['verbose'] = False
        result = optim(forward, gradient, precond=precond, stop=stop, **prm)
    return result


# ---- WORK IN PROGRESS ----


def absolute_diag(weights=None):
    """Diagonal of the membrane regulariser.

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    if weights is None:
        return 1
    else:
        return weights


def membrane_weights(field, factor=1, voxel_size=1, bound='dct2',
                     dim=None, joint=True, return_sum=False, eps=1e-5):
    """Update the (L1) weights of the membrane energy.

    Parameters
    ----------
    field : (..., K, *spatial) tensor
        Field
    factor : float or (K,) sequence[float], default=1
        Regularisation factor
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions
    joint : bool, default=False
        Joint norm across channels.
    return_sum : bool, default=False

    Returns
    -------
    weight : (..., 1 or K, *spatial) tensor
        Weights for the reweighted least squares scheme
    sum : () tensor, if `return_sum`
        Sum of weights (before inversion)
    """
    field = torch.as_tensor(field)
    backend = core.utils.backend(field)
    dim = dim or field.dim() - 1
    nb_prm = field.shape[-dim-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    factor = make_vector(factor, nb_prm, **backend)
    factor = core.utils.unsqueeze(factor, -1, dim + 1)
    if joint:
        factor = factor * nb_prm
    dims = list(range(field.dim()-dim, field.dim()))
    fieldb = diff(field, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    field = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    field.square_().mul_(factor)
    field += fieldb.square_().mul_(factor)
    field /= 2.
    dims = [-1] + ([-dim-2] if joint else [])
    field = field.sum(dim=dims, keepdims=True)[..., 0].sqrt_()
    if return_sum:
        ll = field.sum()
        return field.clamp_min_(eps).reciprocal_(), ll
    else:
        return field.clamp_min_(eps).reciprocal_()


def bending_weights(field, lam=1, voxel_size=1, bound='dct2',
                     dim=None, joint=True, return_sum=False):
    """Update the (L1) weights of the membrane energy.

    Parameters
    ----------
    field : (..., K, *spatial) tensor
        Field
    lam : [(K,) sequence of] float, default=1
        Regularisation factor
    voxel_size : [(dim,) sequence of] float, default=1
        Voxel size
    bound : [(dim,) sequence of] str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions
    joint : bool, default=False
        Joint norm across channels.
    return_sum : bool, default=False

    Returns
    -------
    weight : (..., 1 or K, *spatial) tensor
        Weights for the reweighted least squares scheme
    """
    field = torch.as_tensor(field)
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim() - 1
    field = unsqueeze(field, 0, max(0, dim+1-field.dim()))
    nb_prm = field.shape[-dim-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    bound = ensure_list(bound, dim)
    lam = make_vector(lam, nb_prm, **backend)
    lam = core.utils.unsqueeze(lam, -1, dim)
    if joint:
        lam = lam * nb_prm
    dims = list(range(field.dim()-dim, field.dim()))

    diffs = []
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj).square_()
                    # normalize by number of sides
                    # + off diagonal -> x2  (upper + lower element)
                    dj /= (2. if i != j else 4.)
                    diffs.append(dj)
    field = sum(diffs).mul_(lam)
    if joint:
        field = field.sum(dim=-dim-1, keepdims=True)
    field = field.sqrt_()
    if return_sum:
        ll = field.sum()
        return field.clamp_min_(1e-5).reciprocal_(), ll
    else:
        return field.clamp_min_(1e-5).reciprocal_()


def membrane_diag(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membrane regulariser.

    If no weight map is provided, the diagonal of the membrane regulariser
    is a scaled identity with scale `2 * alpha`, where
    `alpha = vx.reciprocal().square().sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a first order "diamond" shape:
                                    b0
                                b1  a   b1
                                    b0

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = core.utils.make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = core.utils.make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
        # move spatial dimensions to the front
        spdim = list(range(weights.dim() - dim, weights.dim()))
        weights = movedim(weights, spdim, list(range(dim)))
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()

    if weights is None:
        return 2 * vx.sum()

    from ._finite_differences import _window1d, _lincomb
    values = [[weights]]
    dims = [None] + [d for d in range(dim) for _ in range(2)]
    kernel = [2 * vx.sum()]
    for d in range(dim):
        values.extend(_window1d(weights, d, [-1, 1], bound=bound))
        kernel += [vx[d], vx[d]]
    weights = _lincomb(values, kernel, dims, ref=weights)

    # send spatial dimensions to the back
    weights = movedim(weights, list(range(dim)), spdim)
    return weights


def membrane_diag_alt(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membane regulariser.

    This is an alternate implementation of `membrane_diag` that uses the
    sparse convolution code, which relies on linear combinations of views
    into the input tensor. It seems to be a fair bit slower than the
    implementation that relies on moving windows (maybe because it
    does a bit more indexing?).
    For higher orders, we don't have a choice and must use the sparse
    conv code.

    If no weight map is provided, the diagonal of the membrane regulariser
    is a scaled identity with scale `2 * alpha`, where
    `alpha = vx.reciprocal().square().sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a first order "diamond" shape:
                                    b0
                                b1  a   b1
                                    b0
    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()

    if weights is None:
        return 2 * vx.sum()

    # build sparse kernel
    kernel = [2 * vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [vx[d], vx[d]]
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=backend['device'])
    kernel = torch.as_tensor(kernel, **backend)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    weights = spconv(weights, kernel, bound=bound, dim=dim)
    return weights


def bending_diag(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the bending regulariser.

    If no weight map is provided, the diagonal of the bending regulariser
    is a scaled identity with scale `6 * alpha + 8 * beta`, where
        - `alpha = vx.reciprocal().square().sum()`
        - `beta = vx.combinations(r=2).prod(-1).sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a second order "diamond" shape:
                                    c1
                                d   b0  d
                            c1  b1  a   b1  c1
                                d   b0  d
                                    c0

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()
    cvx = torch.combinations(vx, r=2).prod(dim=-1)
    ivx = torch.combinations(torch.arange(dim), r=2)

    if weights is None:
        return 6 * vx.sum() + 8 * cvx.sum()

    # build sparse kernel
    kernel = [10 * vx.sum() + 8 * cvx.sum()]
    center_index = [2] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        subcvx = sum(cvx1 for cvx1, ivx1 in zip(cvx, ivx) if d in ivx1)
        w1 = 6 * vx[d].square() + 4 * subcvx
        w2 = vx[d].square()
        kernel += [w1, w1, w2, w2]
        index = list(center_index)
        index[d] = 1
        indices.append(index)
        index = list(center_index)
        index[d] = 3
        indices.append(index)
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 4
        indices.append(index)
        # corners
        for dd in range(d+1, dim):
            subcvx = [cvx1 for cvx1, ivx1 in zip(cvx, ivx)
                      if ivx1[0] == d and ivx1[1] == dd][0]
            kernel += [subcvx] * 4
            index = list(center_index)
            index[d] = 1
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 1
            index[dd] = 3
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 3
            indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=backend['device'])
    kernel = torch.as_tensor(kernel, **backend)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [5] * dim)

    weights = spconv(weights, kernel, bound=bound, dim=dim)
    return weights


def membrane_diag_old(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membrane regulariser.

    This is an old implementation that hapeens to be slower than
    a simpler/more generic one.

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = core.utils.make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = core.utils.make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
        # move spatial dimensions to the front
        spdim = list(range(weights.dim() - dim, weights.dim()))
        weights = movedim(weights, spdim, list(range(dim)))
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()

    if weights is None:
        return 2 * vx.sum()

    w = weights  # alias for readability
    out = (2 * vx.sum()) * w

    # define out-of-bound coordinate on the left and right
    # and the eventual value transformer depending on the
    # boundary condition
    f = (1 if bound == 'dct1' else -1 if bound == 'dft' else 0)
    l = (-2 if bound == 'dct1' else 0 if bound == 'dft' else -1)
    m = ((lambda w: -w) if bound == 'dst2' else
         (lambda w: 0) if bound in ('dst1', 'zero') else
         (lambda w: w))

    if dim == 3:
        # center
        out[1:-1, 1:-1, 1:-1] += (
                (w[:-2, 1:-1, 1:-1] + w[2:, 1:-1, 1:-1]) * vx[0] +
                (w[1:-1, :-2, 1:-1] + w[1:-1, 2:, 1:-1]) * vx[1] +
                (w[1:-1, 1:-1, :-2] + w[1:-1, 1:-1, 2:]) * vx[2])
        # sides
        out[0, 1:-1, 1:-1] += (
                (m(w[f, 1:-1, 1:-1]) + w[1, 1:-1, 1:-1]) * vx[0] +
                (w[0, :-2, 1:-1] + w[0, 2:, 1:-1]) * vx[1] +
                (w[0, 1:-1, :-2] + w[0, 1:-1, 2:]) * vx[2])
        out[-1, 1:-1, 1:-1] += (
                (w[-2, 1:-1, 1:-1] + m(w[l, 1:-1, 1:-1])) * vx[0] +
                (w[-1, :-2, 1:-1] + w[-1, 2:, 1:-1]) * vx[1] +
                (w[-1, 1:-1, :-2] + w[-1, 1:-1, 2:]) * vx[2])
        out[1:-1, 0, 1:-1] += (
                (w[:-2, 0, 1:-1] + w[2:, 0, 1:-1]) * vx[0] +
                (m(w[1:-1, f, 1:-1]) + w[1:-1, 1, 1:-1]) * vx[1] +
                (w[1:-1, 0, :-2] + w[1:-1, 0, 2:]) * vx[2])
        out[1:-1, -1, 1:-1] += (
                (w[:-2, -1, 1:-1] + w[2:, -1, 1:-1]) * vx[0] +
                (w[1:-1, -2, 1:-1] + m(w[1:-1, l, 1:-1])) * vx[1] +
                (w[1:-1, -1, :-2] + w[1:-1, -1, 2:]) * vx[2])
        out[1:-1, 1:-1, 0] += (
                (w[:-2, 1:-1, 0] + w[2:, 1:-1, 0]) * vx[0] +
                (w[1:-1, :-2, 0] + w[1:-1, 2:, 0]) * vx[1] +
                (m(w[1:-1, 1:-1, f]) + w[1:-1, 1:-1, 1]) * vx[2])
        out[1:-1, 1:-1, -1] += (
                (w[:-2, 1:-1, -1] + w[2:, 1:-1, -1]) * vx[0] +
                (w[1:-1, :-2, -1] + w[1:-1, 2:, -1]) * vx[1] +
                (w[1:-1, 1:-1, -2] + m(w[1:-1, 1:-1, l])) * vx[2])
        # edges
        out[0, 0, 1:-1] += (
                (m(w[f, 0, 1:-1]) + w[1, 0, 1:-1]) * vx[0] +
                (m(w[0, f, 1:-1]) + w[0, 1, 1:-1]) * vx[1] +
                (w[0, 0, :-2] + w[0, 0, 2:]) * vx[2])
        out[0, -1, 1:-1] += (
                (m(w[f, -1, 1:-1]) + w[1, -1, 1:-1]) * vx[0] +
                (w[0, -2, 1:-1] + m(w[0, l, 1:-1])) * vx[1] +
                (w[0, -1, :-2] + w[0, -1, 2:]) * vx[2])
        out[-1, 0, 1:-1] += (
                (w[-1, 0, 1:-1] + m(w[l, 0, 1:-1])) * vx[0] +
                (m(w[-1, f, 1:-1]) + w[-1, 1, 1:-1]) * vx[1] +
                (w[-1, 0, :-2] + w[-1, 0, 2:]) * vx[2])
        out[-1, -1, 1:-1] += (
                (w[-2, -1, 1:-1] + m(w[l, -1, 1:-1])) * vx[0] +
                (w[-1, -2, 1:-1] + m(w[-1, l, 1:-1])) * vx[1] +
                (w[-1, -1, :-2] + w[-1, -1, 2:]) * vx[2])
        out[0, 1:-1, 0] += (
                (m(w[f, 1:-1, 0]) + w[1, 1:-1, 0]) * vx[0] +
                (w[0, :-2, 0] + w[0, 2:, 0]) * vx[1] +
                (m(w[0, 1:-1, f]) + w[0, 1:-1, 1]) * vx[2])
        out[0, 1:-1, -1] += (
                (m(w[f, 1:-1, -1]) + w[1, 1:-1, -1]) * vx[0] +
                (w[0, :-2, -1] + w[0, 2:, -1]) * vx[1] +
                (w[0, 1:-1, -2] + m(w[0, 1:-1, l])) * vx[2])
        out[-1, 1:-1, 0] += (
                (w[-2, 1:-1, 0] + w[l, 1:-1, 0]) * vx[0] +
                (w[-1, :-2, 0] + w[-1, 2:, 0]) * vx[1] +
                (w[-1, 1:-1, f] + w[-1, 1:-1, -1]) * vx[2])
        out[-1, 1:-1, -1] += (
                (w[-2, 1:-1, -1] + m(w[l, 1:-1, -1])) * vx[0] +
                (w[-1, :-2, -1] + w[-1, 2:, -1]) * vx[1] +
                (w[-1, 1:-1, -2] + m(w[-1, 1:-1, l])) * vx[2])
        out[1:-1, 0, 0] += (
                (w[:-2, 0, 0] + w[2:, 0, 0]) * vx[0] +
                (w[1:-1, f, 0] + w[1:-1, 1, 0]) * vx[1] +
                (w[1:-1, 0, f] + w[1:-1, 0, 1]) * vx[2])
        out[1:-1, 0, -1] += (
                (w[:-2, 0, -1] + w[2:, 0, -1]) * vx[0] +
                (w[1:-1, f, -1] + w[1:-1, 1, -1]) * vx[1] +
                (w[1:-1, 0, -2] + w[1:-1, 0, l]) * vx[2])
        out[1:-1, -1, 0] += (
                (w[:-2, 0, 0] + w[2:, -1, 0]) * vx[0] +
                (w[1:-1, -2, 0] + w[1:-1, l, 0]) * vx[1] +
                (w[1:-1, 0, f] + w[1:-1, -1, 1]) * vx[2])
        out[1:-1, -1, -1] += (
                (w[:-2, -1, -1] + w[2:, -1, -1]) * vx[0] +
                (w[1:-1, -2, -1] + m(w[1:-1, l, -1])) * vx[1] +
                (w[1:-1, -1, -2] + m(w[1:-1, -1, l])) * vx[2])

        # corners
        out[0, 0, 0] += ((m(w[f, 0, 0]) + w[1, 0, 0]) * vx[0] +
                         (m(w[0, f, 0]) + w[0, 1, 0]) * vx[1] +
                         (m(w[0, 0, f]) + w[0, 0, 1]) * vx[2])
        out[0, 0, -1] += ((m(w[f, 0, -1]) + w[1, 0, -1]) * vx[0] +
                          (m(w[0, f, -1]) + w[0, 1, -1]) * vx[1] +
                          (w[0, 0, -2] + m(w[0, 0, l])) * vx[2])
        out[0, -1, 0] += ((m(w[f, -1, 0]) + w[1, -1, 0]) * vx[0] +
                          (w[0, -2, 0] + m(w[0, l, 0])) * vx[1] +
                          (m(w[0, -1, f]) + w[0, -1, 1]) * vx[2])
        out[0, -1, -1] += ((m(w[f, -1, -1]) + w[1, -1, -1]) * vx[0] +
                           (w[0, -2, -1] + m(w[0, l, -1])) * vx[1] +
                           (w[0, -1, -2] + m(w[0, -1, l])) * vx[2])
        out[-1, 0, 0] += ((w[-2, 0, 0] + w[l, 0, 0]) * vx[0] +
                          (m(w[-1, f, 0]) + w[-1, 1, 0]) * vx[1] +
                          (m(w[-1, 0, f]) + w[-1, 0, 1]) * vx[2])
        out[-1, 0, -1] += ((w[-2, 0, -1] + m(w[l, 0, -1])) * vx[0] +
                           (m(w[-1, f, -1]) + w[-1, 1, -1]) * vx[1] +
                           (w[-1, 0, -2] + m(w[-1, 0, l])) * vx[2])
        out[-1, -1, 0] += ((w[-2, -1, 0] + m(w[l, -1, 0])) * vx[0] +
                           (w[-1, -2, 0] + m(w[-1, l, 0])) * vx[1] +
                           (m(w[-1, -1, f]) + w[-1, -1, 1]) * vx[2])
        out[-1, -1, -1] += ((w[-2, -1, -1] + m(w[l, -1, -1])) * vx[0] +
                            (w[-1, -2, -1] + m(w[-1, l, -1])) * vx[1] +
                            (w[-1, -1, -2] + m(w[-1, -1, l])) * vx[2])
    elif dim == 2:
        # center
        out[1:-1, 1:-1] += ((w[:-2, 1:-1] + w[2:, 1:-1]) * vx[0] +
                            (w[1:-1, :-2] + w[1:-1, 2:]) * vx[1])
        # edges
        out[0, 1:-1] += ((m(w[f, 1:-1]) + w[1, 1:-1]) * vx[0] +
                         (w[0, :-2] + w[0, 2:]) * vx[1])
        out[-1, 1:-1] += ((w[-2, 1:-1] + m(w[l, 1:-1])) * vx[0] +
                          (w[-1, :-2] + w[-1, 2:]) * vx[1])
        out[1:-1, 0] += ((w[:-2, 0] + w[2:, 0]) * vx[0] +
                         (m(w[1:-1, f]) + w[1:-1, 1]) * vx[1])
        out[1:-1, -1] += ((w[:-2, -1] + w[2:, -1]) * vx[0] +
                          (w[1:-1, -2] + m(w[1:-1, l])) * vx[1])
        # corners
        out[0, 0] += ((m(w[f, 0]) + w[1, 0]) * vx[0] +
                      (m(w[0, f]) + w[0, 1]) * vx[1])
        out[0, -1] += ((m(w[f, -1]) + w[1, -1]) * vx[0] +
                       (w[0, -2] + m(w[0, l])) * vx[1])
        out[-1, 0] += ((w[-1, 0] + m(w[l, 0])) * vx[0] +
                       (m(w[-1, f]) + w[-1, 1]) * vx[1])
        out[-1, -1] += ((w[-2, -1] + m(w[l, -1])) * vx[0] +
                        (w[-1, -2] + w[-1, l]) * vx[1])
    elif dim == 1:
        # center
        out[1:-1] += (w[:-2] + w[2:]) * vx[0]
        # corners
        out[0] += (m(w[f]) + w[1]) * vx[0]
        out[-1] += (w[-2] + m(w[l])) * vx[0]
    else:
        raise NotImplementedError

    # send spatial dimensions to the back
    out = movedim(out, list(range(dim)), spdim)
    return out
