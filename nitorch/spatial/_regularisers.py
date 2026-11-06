"""
This file implements spatial regularizers for vector and displacement fields.

All functions exist in a "field" (by default) for vector fields or "grid"
for displacement fields.

- absolute(x) / absolute_grid(v)
    Squared norm of absolute values
- membrane(x) / membrane_grid(v)
    Squared norm of first order derivatives
- bending(x) / bending_grid(v)
    Squared norm of second order derivatives
- lame_div(v) [displacement only]
    Squared norm of the trace of the Jacobian matrix
- lame_shear(v) [displacement only]
    Squared norm of the off-diagonal elements of the Jacobian matrix
- regulariser(x) / regulariser_grid(v)
    Combination of all regularisers.
All these functions accept non-stationary weight maps.

Another set of functions generate an equivalent convolution kernel
- regulariser_kernel / regulariser_grid_kernel
    Return a sparse convolution kernel
- regulariser_grid_diag
    Return the diagonal of the regulariser

A set of functions return the diagonal of each regulariser (when
though of in matrix form). They are only implemented in the "field" case.
- absolute_diag
- membrane_diag
- bending_diag

Finally, some functions compute updated weight maps in the RLS context:
- membrane_weights
- bending_weights
These functions are not yet implemented for the other regularisers (WIP).
"""

import torch
from nitorch import core, compiled_backend
from nitorch.core.utils import movedim, make_vector, unsqueeze, fast_movedim
from nitorch.core.py import ensure_list
from ._finite_differences import diff, div, diff1d, div1d
from ._spconv import spconv
from nitorch.core import bounds as _bounds
import itertools
if compiled_backend == 'C':
    import nitorch._C.solve as c_solvers
else:
    c_solvers = None


# ----------------------------------------------------------------------
#   Boundary conditions
# ----------------------------------------------------------------------
# Every energy is a sum of rows centred on the voxels of the field of
# view (no ghost rows, no row weights), built from finite differences
# (`diff1d`) of the field extended according to its boundary condition:
#
#     A = E' D' F' W F D E
#
# where `E` extends the field, `D` are finite differences, `F` keeps the
# rows centred on the voxels and `W` are the (optional) voxel weights,
# applied once per row. `diff1d` implements `F D E` and `div1d` its
# exact adjoint.
#
# Displacement grids may also use 'sliding' boundary conditions, which
# reflect the vector field geometrically: along a sliding axis `d`, the
# component `d` (normal to the border) is antisymmetric ('dst2') and the
# other components are symmetric ('dct2'), so that each component has
# its own boundary conditions (`core.bounds.for_component`).
#
# A stencil (convolution, `spconv`) applied to the extended field and
# read at the voxels sums all the rows of the extended lattice within
# its radius, instead of the rows centred on the voxels. Near the borders
# it differs from `A` by `B = A - S` (folds of the taps of the kept rows
# that lie outside of the field of view, minus the rows centred outside
# of it), which `correct_stencil` adds.


def _no_sliding(bound):
    """Sliding boundary conditions only exist for displacement grids"""
    if _bounds.has_sliding(bound):
        raise ValueError("Sliding boundary conditions only apply to "
                         "displacement grids (`*_grid` functions)")


def _normalise_bound(bound, dim):
    """Boundary condition of each axis, in nitorch's convention"""
    return [_bounds.to_nitorch(b) for b in ensure_list(bound, dim)]


def _component_bounds(bound, dim, ncomp=None):
    """Boundary conditions of each axis, for each component of a grid"""
    bound = _normalise_bound(bound, dim)
    return [_bounds.for_component(bound, k) for k in range(ncomp or dim)]


# TEMPORARY: the compiled solvers (relaxation, preconditioner) do not
# implement the border correction of definition F yet (see
# `correct_stencil`). The compiled regularisers do.
_C_EXACT = {
    'membrane': ('dft', 'dct2', 'dst2', 'replicate', 'sliding'),
    'bending': ('dft', 'dct2'),
    'lame': ('dft', 'dst1', 'zero', 'sliding'),
}


def _c_supports(weights=None, bending=0, lame=0, bound=None):
    """Whether the compiled solvers implement this problem: they do not
    implement per-penalty weight maps (dict), nor weighted (RLS)
    bending or linear-elastic energies (the TorchScript solvers do)."""
    if bound is not None:
        bound = [_bounds.to_nitorch(b) for b in ensure_list(bound)]
        if not all(b in _C_EXACT['membrane'] for b in bound):
            return False
        if any(ensure_list(bending)) and \
                not all(b in _C_EXACT['bending'] for b in bound):
            return False
        if any(ensure_list(lame)) and \
                not all(b in _C_EXACT['lame'] for b in bound):
            return False
    if weights is None:
        return True
    if isinstance(weights, dict):
        return False
    return not (any(core.py.ensure_list(bending)) or any(core.py.ensure_list(lame)))


def _voxel_size_list(voxel_size, dim):
    if torch.is_tensor(voxel_size):
        voxel_size = voxel_size.reshape(-1).tolist()
    return [float(v) for v in core.py.ensure_list(voxel_size, dim)]


def _inplace_ok(*tensors):
    """Whether in-place operations are allowed (no backpropagation)"""
    return not torch.is_grad_enabled() or not any(
        getattr(t, 'requires_grad', False) for t in tensors)


def _mul_(x, y):
    """Smart in-place multiplication"""
    if torch.is_tensor(y) and y.requires_grad:
        return x * y
    else:
        return x.mul_(y)


def _div_(x, y):
    """Smart in-place division"""
    if torch.is_tensor(y) and y.requires_grad:
        return x / y
    else:
        return x.div_(y)


def _weight(g, w, inplace):
    """g * w, in place if allowed (and if it does not broadcast g)"""
    if w is None:
        return g
    if inplace and torch.broadcast_shapes(g.shape, w.shape) == g.shape:
        return g.mul_(w)
    return g * w


# ----------------------------------------------------------------------
#   Energies
# ----------------------------------------------------------------------

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
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = grid * voxel_size.square()
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)
        if weights.requires_grad:
            grid = grid * weights[..., None]
        else:
            grid *= weights[..., None]
    return grid


def _membrane_(field, vx, bound, dim, weights=None, out=None):
    """Membrane of a (..., *spatial) field (voxel sizes and boundary
    conditions: one per axis)"""
    dims = list(range(field.dim() - dim, field.dim()))
    inplace = _inplace_ok(field, weights)
    buf1 = torch.empty_like(field) if inplace else None
    buf2 = torch.empty_like(field) if inplace else None
    mom = out.zero_() if (out is not None and inplace) else 0
    for i in range(dim):
        for side in 'fb':
            opt = dict(dim=dims[i], bound=bound[i], voxel_size=vx[i], side=side)
            g = diff1d(field, **opt, out=buf1)
            g = _weight(g, weights, inplace)
            mom += div1d(g, **opt, out=buf2)
    return mom * 0.5 if not inplace else mom.mul_(0.5)


def _bending_(field, vx, bound, dim, weights=None, out=None):
    """Bending of a (..., *spatial) field (voxel sizes and boundary
    conditions: one per axis). Central second differences on the
    diagonal of the Hessian, the four one-sided compositions of first
    differences (averaged) off its diagonal."""
    dims = list(range(field.dim() - dim, field.dim()))
    inplace = _inplace_ok(field, weights)
    bufi = torch.empty_like(field) if inplace else None
    bufij = torch.empty_like(field) if inplace else None
    bufjj = torch.empty_like(field) if inplace else None
    mom = out.zero_() if (out is not None and inplace) else 0
    for i in range(dim):
        opt = dict(dim=dims[i], bound=bound[i], voxel_size=vx[i],
                   order=2, side='c')
        d = diff1d(field, **opt, out=bufi)
        d = _weight(d, weights, inplace)
        mom += div1d(d, **opt, out=bufij)
    for i in range(dim):
        for side_i in 'fb':
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=vx[i])
            di = diff1d(field, **opti, out=bufi)
            for j in range(i + 1, dim):
                for side_j in 'fb':
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=vx[j])
                    dj = diff1d(di, **optj, out=bufij)
                    dj = _weight(dj, weights, inplace)
                    dj = div1d(dj, **optj, out=bufjj)
                    dj = div1d(dj, **opti, out=bufij)
                    # off diagonal: x2 (upper + lower), /4 (sides)
                    if inplace:
                        mom.add_(dj, alpha=0.5)
                    else:
                        mom = mom + dj * 0.5
    return mom


def _grid_energy(fn, grid, voxel_size, bound, weights):
    """Per-component energy of a (..., *spatial, dim) grid: component k
    has its own boundary conditions and is scaled by vx[k]**2"""
    dim = grid.shape[-1]
    vx = _voxel_size_list(voxel_size, dim)
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=grid.dtype, device=grid.device)
    grid = fast_movedim(grid, -1, -dim-1)
    inplace = _inplace_ok(grid, weights)
    if not _bounds.has_sliding(bound):
        mom = fn(grid, vx, _normalise_bound(bound, dim), dim,
                 None if weights is None else weights.unsqueeze(-dim-1))
        scale = torch.as_tensor([v * v for v in vx], dtype=mom.dtype,
                                device=mom.device)
        scale = scale.reshape([dim] + [1] * dim)
        mom = mom.mul_(scale) if inplace else mom * scale
    else:
        cbound = _component_bounds(bound, dim)
        mom = torch.empty_like(grid) if inplace else []
        for k in range(dim):
            out = mom.select(-dim-1, k) if inplace else None
            y = fn(grid.select(-dim-1, k), vx, cbound[k], dim, weights, out=out)
            if inplace:
                if y.data_ptr() != out.data_ptr():
                    out.copy_(y)
                out.mul_(vx[k] ** 2)
            else:
                mom.append(y * vx[k] ** 2)
        if not inplace:
            mom = torch.stack(mom, -dim-1)
    return fast_movedim(mom, -dim-1, -1)


def membrane(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Membrane energy

    Note
    ----
    .. This is exactly equivalent to SPM's membrane energy
    .. Rows are centred on the voxels of the field of view
       (see "Boundary conditions" in `nitorch.spatial._regularisers`).

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
    _no_sliding(bound)
    dim = dim or field.dim()
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=field.dtype, device=field.device)
    return _membrane_(field, _voxel_size_list(voxel_size, dim),
                      _normalise_bound(bound, dim), dim, weights)


def membrane_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : [sequence of] str, default='dft'
        Boundary condition of each axis. 'sliding' reflects the grid
        geometrically: along a sliding axis `d`, the component `d` is
        antisymmetric ('dst2') and the others are symmetric ('dct2').
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    return _grid_energy(_membrane_, grid, voxel_size, bound, weights)


def bending(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Bending energy

    Note
    ----
    .. This is exactly equivalent to SPM's bending energy
    .. It is the squared Frobenius norm of the Hessian, with central
       second differences on its diagonal and one-sided first
       differences (averaged over the four sides) off its diagonal.
    .. Rows are centred on the voxels of the field of view
       (see "Boundary conditions" in `nitorch.spatial._regularisers`).

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
    _no_sliding(bound)
    dim = dim or field.dim()
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=field.dtype, device=field.device)
    return _bending_(field, _voxel_size_list(voxel_size, dim),
                     _normalise_bound(bound, dim), dim, weights)


def bending_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : [sequence of] str, default='dft'
        Boundary condition of each axis. 'sliding' reflects the grid
        geometrically: along a sliding axis `d`, the component `d` is
        antisymmetric ('dst2') and the others are symmetric ('dct2').
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    return _grid_energy(_bending_, grid, voxel_size, bound, weights)


def lame_shear(grid, voxel_size=1, bound='dft', weights=None):
    r"""Precision matrix for the Shear component of the Linear-Elastic energy.

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
    .. Each component is extended with its own boundary conditions
       (they differ along 'sliding' axes), and rows are centred on the
       voxels of the field of view.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : [sequence of] str, default='dft'
        Boundary condition of each axis. 'sliding' reflects the grid
        geometrically: along a sliding axis `d`, the component `d` is
        antisymmetric ('dst2') and the others are symmetric ('dct2').
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    dim = grid.shape[-1]
    vx = _voxel_size_list(voxel_size, dim)
    cb = _component_bounds(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=grid.dtype, device=grid.device)
    inplace = _inplace_ok(grid, weights)

    if inplace:
        buf1 = torch.empty_like(grid[..., 0])
        buf2 = torch.empty_like(buf1)
        buf3 = torch.empty_like(buf1)
    else:
        buf1 = buf2 = buf3 = None

    def add(mom, k, y, alpha):
        if inplace:
            mom[..., k].add_(y, alpha=alpha)
        else:
            mom[k] = mom[k] + y * alpha

    mom = torch.zeros_like(grid) if inplace else [0] * dim
    for i in range(dim):
        x_i = grid[..., i]
        for j in range(i, dim):
            for side_i in 'fb':
                # d_j x_i (in voxels)
                opt_ij = dict(dim=dims[j], side=side_i, bound=cb[i][j],
                              voxel_size=vx[j])
                diff_ij = diff1d(x_i, **opt_ij, out=buf1)
                diff_ij = diff_ij.mul_(vx[i]) if inplace else diff_ij * vx[i]
                if i == j:
                    # diagonal elements
                    diff_ij = _weight(diff_ij, weights, inplace)
                    add(mom, i, div1d(diff_ij, **opt_ij, out=buf2), 0.5)
                    continue
                # off diagonal elements: symmetric strain
                # (d_i x_j + d_j x_i) / 2, averaged over the four sides
                x_j = grid[..., j]
                for side_j in 'fb':
                    opt_ji = dict(dim=dims[i], side=side_j, bound=cb[j][i],
                                  voxel_size=vx[i])
                    diff_ji = diff1d(x_j, **opt_ji, out=buf2)
                    if inplace:
                        diff_ji.mul_(vx[j]).add_(diff_ij).mul_(0.5)
                    else:
                        diff_ji = (diff_ji * vx[j] + diff_ij) * 0.5
                    diff_ji = _weight(diff_ji, weights, inplace)
                    add(mom, j, div1d(diff_ji, **opt_ji, out=buf3), 0.25)
                    add(mom, i, div1d(diff_ji, **opt_ij, out=buf3), 0.25)

    # JA added an additional factor 2 to the kernel
    scale = torch.as_tensor([2 * v for v in vx], dtype=grid.dtype,
                            device=grid.device)
    if inplace:
        return mom.mul_(scale)
    return torch.stack([m * s for m, s in zip(mom, scale)], -1)


def lame_div(grid, voxel_size=1, bound='dft', weights=None):
    r"""Precision matrix for the Divergence component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the first Lame constant (the divergence).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the square of the trace of the Jacobian
       (i.e., volume changes)
    .. Formally: `<f, Lf> = \int [(\sum_i df_i/dx_i)^2 - \sum_i (df_i/dx_i)^2] dx
    .. Each component is extended with its own boundary conditions
       (they differ along 'sliding' axes), and rows are centred on the
       voxels of the field of view.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1 (actually unused)
    bound : [sequence of] str, default='dft'
        Boundary condition of each axis. 'sliding' reflects the grid
        geometrically: along a sliding axis `d`, the component `d` is
        antisymmetric ('dst2') and the others are symmetric ('dct2').
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    dim = grid.shape[-1]
    cb = _component_bounds(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=grid.dtype, device=grid.device)
    inplace = _inplace_ok(grid, weights)

    # precompute gradients
    opt = [{side: dict(dim=dims[i], side=side, bound=cb[i][i])
            for side in 'fb'} for i in range(dim)]
    grad = [{side: diff1d(grid[..., i], **opt[i][side]) for side in 'fb'}
            for i in range(dim)]

    if inplace:
        buf1 = torch.empty_like(grid[..., 0])
        buf2 = torch.empty_like(grid[..., 0])
    else:
        buf1 = buf2 = None

    # compute divergence
    mom = torch.zeros_like(grid) if inplace else [0] * dim
    for sides in itertools.product('fb', repeat=dim):
        if inplace:
            div = buf1.copy_(grad[0][sides[0]])
            for i in range(1, dim):
                div += grad[i][sides[i]]
        else:
            div = sum(grad[i][side] for i, side in enumerate(sides))
        div = _weight(div, weights, inplace)
        for i, side in enumerate(sides):
            y = div1d(div, **opt[i][side], out=buf2)
            if inplace:
                mom[..., i] += y
            else:
                mom[i] = mom[i] + y

    if inplace:
        return mom.div_(2 ** dim)   # weight sides combinations
    return torch.stack(mom, -1) / (2 ** dim)




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
            # mixed derivative d/dx_d d/dx_dd
            # (this used to loop over other pairs of axes, which cancel
            # out in 2D but not in 3D)
            for d1 in (d,):
                for d2 in (dd,):
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
    bound : [sequence of] str, default='dft'
        Boundary condition of each axis. 'sliding' reflects the grid
        geometrically: along a sliding axis `d`, the component `d` is
        antisymmetric ('dst2') and the others are symmetric ('dct2').
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.
    kernel : bool or sparse tensor, default=False
        Use a convolution (`spconv`) and its border correction
        (`correct_stencil`). Unweighted only.

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    if not (absolute or membrane or bending or any(ensure_list(lame))):
        return torch.zeros_like(v)

    if (c_solvers and not v.requires_grad
            and _c_supports(weights, bending, lame)):
        return c_solvers.c_regulariser_grid(
            v, weights, None,
            absolute, membrane, bending, lame, factor,
            voxel_size=voxel_size, bound=bound)

    if torch.is_tensor(kernel) or kernel:
        if not torch.is_tensor(kernel):
            kernel = regulariser_grid_kernel(dim, absolute, membrane, bending,
                                             lame, factor, voxel_size, **backend)
        v = core.utils.fast_movedim(v, -1, -dim-1)
        y = spconv(v, kernel, bound=bound, dim=dim)
        y = correct_stencil(v, y, bound, dim, voxel_size, membrane, bending,
                            lame, factor, grid=True)
        return core.utils.fast_movedim(y, -dim-1, -1)

    voxel_size = make_vector(voxel_size, dim, **backend)
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = make_vector(lame, 2, dtype=torch.double).tolist()  # no float32 rounding
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

    y = torch.zeros_like(v)
    if absolute:
        y.add_(absolute_grid(v, weights=wa, voxel_size=voxel_size), alpha=absolute)
    if membrane:
        y.add_(membrane_grid(v, weights=wm, **fdopt), alpha=membrane)
    if bending:
        y.add_(bending_grid(v, weights=wb, **fdopt), alpha=bending)
    if lame[0]:
        y.add_(lame_shear(v, weights=wl[0], **fdopt), alpha=lame[0])
    if lame[1]:
        y.add_(lame_div(v, weights=wl[1], **fdopt), alpha=lame[1])

    return y


def correct_stencil(x, y, bound, dim, voxel_size=1, membrane=0, bending=0,
                    lame=0, factor=1, grid=False, subset=None):
    """Add the border correction of a stencil (convolution) regulariser.

    A stencil `S` (`spconv` with a regulariser kernel) applied to the
    bound-extended field and read at the voxels sums all the rows of the
    extended lattice within its radius. The operator `A` of the energy
    only has the rows centred on the voxels (see "Boundary conditions"):
    near the borders, `A = S + B`, where `B` folds the taps of the kept
    rows that lie outside of the field of view back onto their source
    voxel (with the sign of the extension), and removes the rows centred
    outside of the field of view. This adds `B x` to `y = S x`.

    Parameters
    ----------
    x : (..., C, *spatial) tensor
        Input field (or grid, with its components along `C`)
    y : (..., C, *subspatial) tensor
        Output of the stencil at the voxels `x[subset]` (updated in place)
    bound : [sequence of] str
        Boundary condition of each axis
    dim : int
        Number of spatial dimensions
    voxel_size : [sequence of] float, default=1
    membrane, bending : [sequence of] float, default=0
        Factors of the (unweighted) energies, per channel for fields.
        The absolute energy needs no correction.
    lame : (float, float), default=0
        Factors of the linear-elastic energy (grids only).
    factor : [sequence of] float, default=1
        Global factor (per channel for fields).
    grid : bool, default=False
        Whether `x` is a displacement grid: component `k` then has its
        own boundary conditions (`core.bounds.for_component`) and its
        energy is scaled by `voxel_size[k]**2`.
    subset : list[slice], optional
        Voxels at which `y` was computed (one slice per spatial axis)

    Returns
    -------
    y : (..., C, *subspatial) tensor

    """
    shape = x.shape[-dim:]
    nc = x.shape[-dim-1]
    vx = _voxel_size_list(voxel_size, dim)
    if grid:
        cbound = _component_bounds(bound, dim, nc)
        f = float(factor)
        mu, lam = [float(l) * f for l in core.py.make_list(lame, 2)]
        membrane, bending = float(membrane) * f, float(bending) * f
        scale = [v * v for v in vx]
        mem = [[membrane * scale[c] / vx[a] ** 2
                + mu * (2 if a == c else scale[c] / vx[a] ** 2)
                + lam * (a == c) for a in range(dim)] for c in range(nc)]
        ben = [bending * scale[c] for c in range(nc)]
    else:
        _no_sliding(bound)
        cbound = [_normalise_bound(bound, dim)] * nc

        def per_channel(v):
            if torch.is_tensor(v):
                v = v.reshape(-1).tolist()
            return [float(vv) for vv in core.py.make_list(v, nc)]
        f = per_channel(factor)
        membrane = [m * fc for m, fc in zip(per_channel(membrane), f)]
        ben = [b * fc for b, fc in zip(per_channel(bending), f)]
        mem = [[membrane[c] / vx[a] ** 2 for a in range(dim)]
               for c in range(nc)]
        mu = lam = 0

    # voxels of the output
    subset = list(subset or [])
    subset += [slice(None)] * (dim - len(subset))
    positions = [{p: k for k, p in enumerate(range(n)[sl])}
                 for n, sl in zip(shape, subset)]

    def sel(t, fixed):
        # select the planes `fixed` ({axis: index}) of a (..., *spatial)
        # tensor (the highest axes first, so that the others keep their
        # position)
        nb = t.dim() - dim
        for d in sorted(fixed, reverse=True):
            t = t.select(nb + d, fixed[d])
        return t

    def source(c, fixed):
        # source voxels and sign of the extended samples `fixed`
        src, sign = {}, 1
        for d, j in fixed.items():
            p, s = getattr(_bounds, cbound[c][d])(j, shape[d])
            if not s:
                return None, 0
            src[d], sign = p, sign * s
        return src, sign

    def sample(t, c, fixed):
        # extended samples `fixed` of the component `c` (None if zero)
        src, sign = source(c, fixed)
        if src is None:
            return None
        t = sel(t, src)
        return t if sign > 0 else -t

    def add(c, fixed, val):
        # y[c, fixed] += val, at the voxels of the output
        idx, vidx = [], []
        for d in range(dim):
            if d in fixed:
                if fixed[d] not in positions[d]:
                    return
                idx.append(positions[d][fixed[d]])
            else:
                idx.append(slice(None))
                vidx.append(subset[d])
        yc = y.select(-dim-1, c)
        yc[(Ellipsis, *idx)].add_(val[(Ellipsis, *vidx)])

    def fold(c, fixed, val):
        # y[source(fixed)] += sign * val
        src, sign = source(c, fixed)
        if src is not None:
            add(c, src, val if sign > 0 else -val)

    def lincomb(*terms):
        out = None
        for a, t in terms:
            if t is not None:
                out = t * a if out is None else out.add_(t, alpha=a)
        return out

    def sides(n):
        # (border voxel, ghost sample, inward voxel, second ghost)
        return ((0, -1, 1, -2), (n - 1, n, n - 2, n + 1))

    def neumann(t, axis):
        # 1D Laplacian along `axis` (of the spatial axes of t), with the
        # missing neighbours dropped
        opt = dict(dim=t.dim() - dim + 1 + axis, bound='dct2', side='f')
        return div1d(diff1d(t, **opt), **opt)

    for c in range(nc):
        xc = x.select(-dim-1, c)
        cb = cbound[c]
        for a in range(dim):
            n = shape[a]
            # membrane (and squared linear-elastic terms)
            k = mem[c][a]
            if k and cb[a] not in ('dft', 'dct2', 'dst2', 'replicate'):
                for j0, jg, _, _ in sides(n):
                    d = lincomb((1, sel(xc, {a: j0})),
                                (-1, sample(xc, c, {a: jg})))
                    d.mul_(-0.5 * k)
                    add(c, {a: j0}, d)
                    fold(c, {a: jg}, d)
            # bending: central second differences
            k = ben[c] / vx[a] ** 4
            if k and cb[a] not in ('dft', 'dct2', 'dst2', 'dst1'):
                for j0, jg, j1, jg2 in sides(n):
                    xg = sample(xc, c, {a: jg})
                    x0 = sel(xc, {a: j0})
                    ghost = lincomb((1, sample(xc, c, {a: jg2})), (-2, xg),
                                    (1, x0))
                    add(c, {a: j0}, ghost.mul_(-k))
                    kept = lincomb((1, xg), (-2, x0),
                                   (1, sample(xc, c, {a: j1})))
                    fold(c, {a: jg}, kept.mul_(k))
            # bending: cross terms of the pairs (a, b)
            for b in range(dim):
                k = ben[c] / (vx[a] * vx[b]) ** 2 if b != a else 0
                if not k:
                    continue
                if cb[a] not in ('dft', 'dct2', 'dst2', 'replicate'):
                    # edges of axis a
                    for j0, jg, _, _ in sides(n):
                        d = lincomb((1, sel(xc, {a: j0})),
                                    (-1, sample(xc, c, {a: jg})))
                        d = neumann(d, b - (b > a)).mul_(-k)
                        add(c, {a: j0}, d)
                        fold(c, {a: jg}, d)
                # (the corners vanish if an axis is dct2 or replicate, and
                # for dst2 x dst2 and dft x dft)
                if b < a or any(bb in ('dct2', 'replicate')
                                for bb in (cb[a], cb[b])) or \
                        cb[a] == cb[b] and cb[a] in ('dst2', 'dft'):
                    continue
                # corners (once per pair)
                for ja, ga, _, _ in sides(n):
                    for jb, gb, _, _ in sides(shape[b]):
                        g = lincomb(
                            (1, sel(xc, {a: ja, b: jb})),
                            (-1, sample(xc, c, {a: ja, b: gb})),
                            (-1, sample(xc, c, {a: ga, b: jb})),
                            (1, sample(xc, c, {a: ga, b: gb})))
                        g.mul_(k)
                        add(c, {a: ja, b: jb}, g * -1.5)
                        fold(c, {a: ja, b: gb}, g * -0.5)
                        fold(c, {a: ga, b: jb}, g * -0.5)
                        fold(c, {a: ga, b: gb}, g * 0.5)
        # linear-elastic cross terms: E_c' C_a' F'F C_b E_j u_j
        for j in range(nc if grid else 0):
            if j == c:
                continue
            for kappa, a, b in ((mu, j, c), (lam, c, j)):
                if not kappa:
                    continue
                bi, bj = cbound[c][a], cbound[j][a]
                if (bi == bj == 'dft' or {bi, bj} <= {'dst1', 'zero'}
                        or {bi, bj} == {'dct2', 'dst2'}):
                    continue
                uj = x.select(-dim-1, j)
                n = shape[a]

                def central(fixed):
                    # C_b u_j at the (extended) plane `fixed` of axis a
                    t = sample(uj, j, fixed)
                    if t is None:
                        return None
                    axis = t.dim() - dim + 1 + b - (b > a)
                    return diff1d(t, order=1, side='c', dim=axis,
                                  bound=cbound[j][b])

                for (j0, jg, _, _), sgn in zip(sides(n), (-1, 1)):
                    ghost = central({a: jg})
                    if ghost is not None:
                        add(c, {a: j0}, ghost.mul_(0.5 * sgn * kappa))
                    fold(c, {a: jg}, central({a: j0}).mul_(0.5 * sgn * kappa))
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
    lame_shear, lame_div = core.py.make_list(lame, 2)
    backend = dict(dtype=dtype, device=device)

    kernels = []
    if absolute:
        kernels.append(absolute_grid_kernel(dim, voxel_size, **backend) * absolute)
    if membrane:
        kernels.append(membrane_grid_kernel(dim, voxel_size, **backend) * membrane)
    if bending:
        kernels.append(bending_grid_kernel(dim, voxel_size, **backend) * bending)
    if lame_shear:
        kernels.append(lame_shear_kernel(dim, voxel_size, **backend) * lame_shear)
    if lame_div:
        kernels.append(lame_div_kernel(dim, voxel_size, **backend) * lame_div)
    kernel = sum_kernels(dim, *kernels)
    kernel *= factor
    return kernel


def regulariser_grid_diag(dim, absolute=0, membrane=0, bending=0, lame=0,
                          factor=1, voxel_size=1, bound='dft', weights=None,
                          dtype=None, device=None):
    """Diagonal of the precision matrix of a deformation grid
    (i.e., of `regulariser_grid` in matrix form).

    Parameters
    ----------
    dim : int
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
    diag : (dim,) or (..., *spatial, dim) tensor
        Exact diagonal (in the interior of the field of view).
        Without weights, the diagonal is constant across space.

    """
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = ensure_list(wl, 2)
    for w in (wa, wm, wb, *wl):
        if torch.is_tensor(w):
            dtype = dtype or w.dtype
            device = device or w.device
    backend = dict(dtype=dtype or torch.get_default_dtype(), device=device)
    wa, wm, wb, *wl = [torch.as_tensor(w, **backend) if w is not None else w
                       for w in (wa, wm, wb, *wl)]

    vx = make_vector(voxel_size, dim, **backend)
    vx2 = vx.square()
    ivx2 = vx2.reciprocal()
    lame = make_vector(lame, 2, dtype=torch.double).tolist()  # no float32 rounding
    opt = dict(voxel_size=vx, bound=bound, dim=dim)

    def mul(w, x):
        # (..., *spatial) weight x (dim,) vector
        return x if w is None else w[..., None] * x

    out = torch.zeros([dim], **backend)
    if absolute:
        out = out + absolute * mul(wa, vx2)
    if membrane:
        out = out + membrane * mul(membrane_diag(weights=wm, **opt), vx2)
    if bending:
        out = out + bending * mul(bending_diag(weights=wb, **opt), vx2)

    def m1d(w):
        # diagonal of the 1D membrane along each dimension
        # M_j(w) = w + (w[-1] + w[+1]) / 2  along j
        if w is None:
            return [2] * dim
        return [_membrane_diag1d(w, d, dim, bound) for d in range(dim)]

    if lame[0]:
        # diag of the shear: 2 * M_i(w) + sum_{j != i} (vx_i / vx_j)^2 M_j(w)
        m = m1d(wl[0])
        shear = [2 * m[i] + sum(m[j] * (vx2[i] * ivx2[j])
                                for j in range(dim) if j != i)
                 for i in range(dim)]
        out = out + lame[0] * _stack_last(shear, **backend)
    if lame[1]:
        # diag of the divergence: M_i(w)
        out = out + lame[1] * _stack_last(m1d(wl[1]), **backend)
    return out * factor


def _stack_last(values, **backend):
    """Stack a list of scalars and/or tensors along a new last dimension"""
    values = [torch.as_tensor(v, **backend) for v in values]
    shape = core.utils.expanded_shape(*[v.shape for v in values])
    return torch.stack([v.expand(shape) for v in values], -1)


def _membrane_diag1d(weights, d, dim, bound='dft'):
    """Diagonal of a weighted membrane along a single dimension
    (with unit voxel size): `w + (w[-1] + w[+1]) / 2` along `d`"""
    from ._finite_differences import _window1d, _lincomb
    wbound = _bounds.even(_normalise_bound(bound, dim))[d]
    d = weights.dim() - dim + d
    values = [[weights]]
    values.extend(_window1d(weights, d, [-1, 1], bound=wbound))
    dims = [None, d, d]
    return _lincomb(values, [1, 0.5, 0.5], dims, ref=weights)


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
    _no_sliding(bound)
    if not (absolute or membrane or bending):
        return torch.zeros_like(x)

    dim = dim or (x.dim() - 1)
    if (c_solvers and not x.requires_grad
            and _c_supports(weights, bending)):
        return c_solvers.c_regulariser(
            x, weights, None, dim,
            absolute, membrane, bending, factor,
            voxel_size=voxel_size, bound=bound)

    backend = dict(dtype=x.dtype, device=x.device)
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

    absolute = core.py.make_list(absolute, nb_prm) if absolute else [0] * nb_prm
    membrane = core.py.make_list(membrane, nb_prm) if membrane else [0] * nb_prm
    bending = core.py.make_list(bending, nb_prm) if bending else [0] * nb_prm

    # (iterate over the channel dimension, not over the first one, which
    # is a batch dimension when there is one)
    def per_channel(w):
        if w is None:
            return [None] * nb_prm
        return torch.as_tensor(w, **backend).expand(x.shape).unbind(-dim-1)
    wa, wm, wb = per_channel(wa), per_channel(wm), per_channel(wb)

    inplace = _inplace_ok(x, *wa, *wm, *wb)
    y = torch.zeros_like(x) if inplace else [0] * nb_prm
    for k, x1 in enumerate(x.unbind(-dim-1)):
        y1 = y.select(-dim-1, k) if inplace else 0
        for alpha, fn, w1 in ((absolute[k], _absolute, wa[k]),
                              (membrane[k], _membrane, wm[k]),
                              (bending[k], _bending, wb[k])):
            if not alpha:
                continue
            opt = {} if fn is _absolute else fdopt
            z = fn(x1, weights=w1, **opt)
            if inplace:
                y1.add_(z, alpha=alpha)
            else:
                y1 = y1 + z * alpha
        if not inplace:
            y[k] = y1 if torch.is_tensor(y1) else torch.zeros_like(x1)
    if not inplace:
        y = torch.stack(y, -dim-1)

    pad_spatial = (Ellipsis,) + (None,) * dim
    if inplace:
        return y.mul_(factor[pad_spatial])
    return y * factor[pad_spatial]


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
        kernels.append(absolute_kernel(dim, voxel_size, **backend) * absolute)
    if membrane:
        kernels.append(membrane_kernel(dim, voxel_size, **backend) * membrane)
    if bending:
        kernels.append(bending_kernel(dim, voxel_size, **backend) * bending)
    kernel = sum_kernels(dim, *kernels)
    kernel *= factor
    return kernel


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
                     dim=None, joint=True, return_sum=False, eps=None):
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
    backend = core.utils.backend(field)
    dim = dim or field.dim() - 1
    nb_prm = field.shape[-dim-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    factor = make_vector(factor, nb_prm, **backend)
    factor = core.utils.unsqueeze(factor, -1, dim+1)
    # if joint:
    #     factor = factor * nb_prm
    dims = list(range(field.dim()-dim, field.dim()))
    fieldb = diff(field, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    field = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    field.square_().add_(fieldb.square_())
    del fieldb
    field.mul_(factor/2)
    dims = [-1] + ([-dim-2] if joint else [])
    field = field.sum(dim=dims, keepdims=True)[..., 0].sqrt_()
    ll = field.sum().double() if return_sum else None
    if eps is None:
        eps = core.constants.eps(field.dtype)
    field = field.clamp_min_(eps).reciprocal_()
    return (field, ll) if return_sum else field


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
                                    b1
                                b0  a   b0
                                    b1
    with `a = alpha` and `b0 = ivx2[0] / 2`, where `ivx2 = vx.reciprocal().square()`
    (the kernel sums to the unweighted diagonal).

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
    kernel = [vx.sum()]
    for d in range(dim):
        values.extend(_window1d(weights, d, [-1, 1],
                                bound=_bounds.even(_normalise_bound(bound, dim))[d]))
        kernel += [vx[d] / 2, vx[d] / 2]
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
                                    b1
                                b0  a   b0
                                    b1
    with `a = alpha` and `b0 = ivx2[0] / 2`, where `ivx2 = vx.reciprocal().square()`
    (the kernel sums to the unweighted diagonal).
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
    kernel = [vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [vx[d] / 2, vx[d] / 2]
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=backend['device'])
    kernel = torch.as_tensor(kernel, **backend)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    weights = spconv(weights, kernel, bound=_bounds.even(_normalise_bound(bound, dim)), dim=dim)
    return weights


def bending_diag(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the bending regulariser.

    If no weight map is provided, the diagonal of the bending regulariser
    is a scaled identity with scale `6 * alpha + 8 * beta`, where
        - `ivx2 = vx.reciprocal().square()`
        - `alpha = ivx2.square().sum()`
        - `beta = ivx2.combinations(r=2).prod(-1).sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a 3x3 shape:
                                d   b1  d
                                b0  a   b0
                                d   b1  d
    with
        - `a  = 4 * alpha + 2 * beta`
        - `b0 = ivx2[0]**2 + ivx2[0]*ivx2[1]`
        - `d  = 0.5 * ivx2[0]*ivx2[1]`
    (the kernel sums to the unweighted diagonal). The same-axis terms are
    exact second differences `(x[k+1] - 2 x[k] + x[k-1]) / vx^2`, whose
    squared coefficients are (1, 4, 1); the cross terms average the four
    one-sided compositions.

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
        return 6 * vx.square().sum() + 8 * cvx.sum()

    # build sparse kernel
    kernel = [4 * vx.square().sum() + 2 * cvx.sum()]
    center_index = [2] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        subcvx = sum(cvx1 for cvx1, ivx1 in zip(cvx, ivx) if d in ivx1)
        w1 = vx[d].square() + subcvx
        kernel += [w1, w1]
        index = list(center_index)
        index[d] = 1
        indices.append(index)
        index = list(center_index)
        index[d] = 3
        indices.append(index)
        # corners
        for dd in range(d+1, dim):
            subcvx = [cvx1 for cvx1, ivx1 in zip(cvx, ivx)
                      if ivx1[0] == d and ivx1[1] == dd][0]
            kernel += [0.5 * subcvx] * 4
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

    weights = spconv(weights, kernel, bound=_bounds.even(_normalise_bound(bound, dim)), dim=dim)
    return weights


def membrane_diag_old(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membrane regulariser.

    This is an old implementation that happens to be slower than
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
    out = vx.sum() * w
    vx = vx / 2  # weight of the neighbours

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
