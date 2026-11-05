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
import collections
if compiled_backend == 'C':
    import nitorch._C.solve as c_solvers
else:
    c_solvers = None


def _c_supports(weights=None, bending=0, lame=0, bound=None):
    """Whether the compiled solvers implement this problem.

    They do not implement per-penalty weight maps (dict), nor weighted
    (RLS) bending or linear-elastic energies. Their linear-elastic
    stencil is only exact for dft, dst1 and zero boundaries (at the
    borders of the other ones, its cross-component terms are not those
    of "Exact operators" in this module). The TorchScript implementations are
    used instead.
    """
    if any(core.py.ensure_list(lame)) and bound is not None:
        from nitorch._C.grid import bound_to_nitorch
        names = bound_to_nitorch(core.py.make_list(bound), as_type='str')
        if any(b not in ('dft', 'dst1', 'zero') for b in names):
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


# ----------------------------------------------------------------------
#   Exact operators
# ----------------------------------------------------------------------
# Every regulariser is a quadratic form whose precision operator is
#
#     A = sum_t  c_t  D_t'  diag(omega * w)  D_t
#
# - each row of `D_t` is a linear combination of finite differences
#   (`diff1d`) of the components of the field, extended according to
#   their boundary condition; `D_t'` is their adjoint (`div1d`);
# - `w` are optional voxel weights (RLS), extended with the even
#   counterpart of the boundary condition (`core.bounds.even`);
# - `omega` are row weights, which make the energy that of one period of
#   the extended field divided by the number of copies of the field that
#   it contains (periodic, symmetric and antisymmetric boundaries), the
#   energy over the whole space ('zero'), or the energy of the rows
#   centred in the field of view ('replicate').
#
# Rows are centred on the voxels, and finite differences that live
# between voxels ('f', 'b') are averaged over both sides. That average
# equals the first difference on the half-voxel lattice ('h'), whose rows
# lie between voxels and have the average weight of their two
# neighbouring rows: it needs one finite difference instead of two.
# The weights of the rows, per axis, are:
#
#     dft, dct2, dst2, replicate   [0,   1, 1, ..., 1, 1,   0]
#     dct1                         [0, 1/2, 1, ..., 1, 1/2, 0]
#     dst1                         [1/2, 1, 1, ..., 1, 1, 1/2]
#     zero                         [1,   1, 1, ..., 1, 1,   1]
#
# where the first and last rows are ghost rows, centred one voxel
# outside of the field of view. `dct1` mirrors the field about its
# border voxels, which appear once per period while the other voxels
# appear twice; `dst1` mirrors it about the ghost voxels, which are zero
# and appear once per period. Along axes whose ghost rows have a weight
# (dst1, zero), the field is padded (`core.utils.pad`) and its rows are
# computed on the padded array, whose padding is then added back onto
# the voxels it copies (`core.utils.pad(..., adjoint=True)`).
#
# Applying the stencil `L = D'D` to the extended field and reading it at
# the voxels (as convolution-based implementations do) is equivalent
# for 'dft', 'dct2', 'dst2', 'dst1' and 'zero', but not for 'dct1' and
# 'replicate' (see `_correct_stencil`).

# Weight of the ghost rows of each boundary condition
_GHOST_WEIGHT = {'dst1': 0.5, 'zero': 1.}

# Finite difference (order, side) of each kind of row ('h' is the
# forward difference, read on the half-voxel lattice)
_KINDS = {'f': (1, 'f'), 'b': (1, 'b'), 'c': (2, 'c'), 'h': (1, 'f')}

# Boundary conditions whose half-voxel rows are all forward differences
# (the row before the first voxel is zero, or that after the last one)
_HALF_UNPADDED = ('dft', 'dct2', 'replicate')


def _normalise_bound(bound, dim):
    """Boundary condition of each axis, in nitorch's convention"""
    return [_bounds.to_nitorch(b) for b in ensure_list(bound, dim)]


def _row_weights(n, bound, margin):
    """Row weights of an axis of length `n`, on the rows `[-margin, n+margin)`"""
    w = [1.] * n
    if bound == 'dct1' and n > 1:
        w[0] = w[-1] = 0.5
    if margin:
        g = _GHOST_WEIGHT.get(bound, 0.)
        w = [0.] * (margin - 1) + [g] + w + [g] + [0.] * (margin - 1)
    return w


def _rows(x, kinds, bound, dim):
    """Rows `kinds` (one of 'i', 'f', 'b', 'c' per axis) of a
    (..., *spatial) field (finite differences)"""
    for d, kind in enumerate(kinds):
        if kind != 'i':
            order, side = _KINDS[kind]
            x = diff1d(x, order=order, side=side, dim=x.dim() - dim + d,
                       bound=bound[d])
    return x


def _rows_t(r, kinds, bound, dim):
    """Adjoint of `_rows`"""
    for d in reversed(range(dim)):
        kind = kinds[d]
        if kind != 'i':
            order, side = _KINDS[kind]
            r = div1d(r, order=order, side=side, dim=r.dim() - dim + d,
                      bound=bound[d])
    return r


def _quadratic(xs, terms, bound, dim, weights=None):
    """Apply the operator `sum_t c_t D_t' diag(omega w) D_t`.

    Parameters
    ----------
    xs : list[(..., *spatial) tensor]
        Components of the field
    terms : list[(float, list[(int, str, float)])]
        Each term is `(c_t, parts)`, and each part is
        `(component, kinds, scale)`, where `kinds` has one character
        per spatial axis among 'i', 'f', 'b', 'c', 'h'. All parts of a
        term have the same 'h' axes.
    bound : [list of] bound_like
    dim : int
    weights : (..., *spatial) tensor, optional
        Voxel weights

    Returns
    -------
    ys : list[(..., *spatial) tensor]
    """
    shape = xs[0].shape[-dim:]
    backend = dict(dtype=xs[0].dtype, device=xs[0].device)
    bound = _normalise_bound(bound, dim)

    # Axes with weighted ghost rows are padded, and their rows computed on
    # the padded array (whose own boundary condition is then irrelevant:
    # the rows that would use it have no weight). So are the half-voxel
    # axes whose first half row (before the first voxel) is not a
    # forward difference of the field.
    half = [any(k[d] == 'h' for _, parts in terms for _, k, _ in parts)
            for d in range(dim)]
    margin = [2 if b in _GHOST_WEIGHT or (h and b not in _HALF_UNPADDED)
              else 0 for b, h in zip(bound, half)]
    rbound = ['zero' if m else b for b, m in zip(bound, margin)]
    padded = any(margin)

    # row weights of each axis (None if all ones), and the weights of its
    # half-voxel rows (the average of their two neighbouring rows)
    omega1d, half1d = [], []
    for d, (n, b, m) in enumerate(zip(shape, bound, margin)):
        w = _row_weights(n, b, m)
        if m:
            h = [(w0 + w1) / 2 for w0, w1 in zip(w, w[1:] + [0.])]
        else:
            # dft wraps around; with dct2/replicate the last half row is
            # zero (so that its weight does not matter)
            h = w
        omega1d.append(None if all(wi == 1 for wi in w) else w)
        half1d.append(None if all(wi == 1 for wi in h) else h)
    if weights is not None:
        weights = torch.as_tensor(weights, **backend)
        if padded:
            weights = core.utils.pad(weights, margin, side='both',
                                     mode=_bounds.even(bound))

    if padded:
        xs = [core.utils.pad(x, margin, side='both', mode=bound) for x in xs]

    # Along a half-voxel axis, the weight of a row is the average of the
    # weights of its two neighbouring (voxel) rows: it is computed on the
    # row weights of the axis, or on the voxel weights (for which dft
    # wraps around, and the last half row of dct2/replicate is zero).
    half_omega = {}

    def term_weights(kinds):
        key = tuple(d for d in range(dim) if kinds[d] == 'h')
        if key not in half_omega:
            # without voxel weights, the row weights are a product of 1D
            # weights, which are averaged before the product
            w = None
            for d in range(dim):
                wd = half1d[d] if (d in key and weights is None) else omega1d[d]
                if wd is None:
                    continue
                wd = torch.as_tensor(wd, **backend)
                wd = wd.reshape([-1] + [1] * (dim - d - 1))
                w = wd if w is None else w * wd
            if weights is not None:
                w = weights if w is None else w * weights
                for d in key:
                    ax = w.dim() - dim + d
                    if w.shape[ax] == 1:
                        continue
                    rb = 'dft' if margin[d] == 0 and bound[d] == 'dft' else 'zero'
                    w = (w + core.utils.roll(w, -1, ax, bound=rb)) * 0.5
            half_omega[key] = w
        return half_omega[key]

    # Rows that are used by several terms are computed once, and their
    # adjoint is applied once to the sum of their weighted rows.
    shared = collections.Counter((c, k) for _, parts in terms
                                 for c, k, _ in parts)
    shared = {key for key, count in shared.items() if count > 1}
    cache, acc = {}, {}
    ys = [None] * len(xs)

    def iadd(a, b, alpha=1):
        # a += alpha * b, in place if it does not change the shape of `a`
        if a.shape == torch.broadcast_shapes(a.shape, b.shape):
            return a.add_(b, alpha=alpha)
        return a + b * alpha if alpha != 1 else a + b

    def imul(a, b):
        if a.shape == torch.broadcast_shapes(a.shape, b.shape):
            return a.mul_(b)
        return a * b

    for coef, parts in terms:
        row = None
        for c, kinds, scale in parts:
            key = (c, kinds)
            if key in shared:
                if key not in cache:
                    cache[key] = _rows(xs[c], kinds, rbound, dim)
                r = cache[key]
            else:
                r = _rows(xs[c], kinds, rbound, dim)
            if row is None:
                # (a new tensor: `r` may be cached, or be the field itself)
                row = r.mul(scale)
            else:
                row = iadd(row, r, scale)
        w = term_weights(parts[0][1])
        if w is not None:
            row = imul(row, w)
        for i, (c, kinds, scale) in enumerate(parts):
            key = (c, kinds)
            alpha = coef * scale
            if key in shared:
                if key not in acc:
                    acc[key] = row.mul(alpha)
                else:
                    acc[key] = iadd(acc[key], row, alpha)
            else:
                # the row may only be modified by its last part
                last = i == len(parts) - 1
                if not last:
                    r = row.mul(alpha)
                else:
                    r = row.mul_(alpha) if alpha != 1 else row
                y = _rows_t(r, kinds, rbound, dim)
                ys[c] = y if ys[c] is None else iadd(ys[c], y)
    for (c, kinds), r in acc.items():
        y = _rows_t(r, kinds, rbound, dim)
        ys[c] = y if ys[c] is None else iadd(ys[c], y)

    out = []
    for y, x in zip(ys, xs):
        if y is None:
            y = torch.zeros_like(x)
        if padded:
            y = core.utils.pad(y, margin, side='both', mode=bound, adjoint=True)
        out.append(y)
    return out


def _field_op(field, terms, bound, dim, weights=None):
    """Apply a quadratic regulariser to a (..., *spatial) field"""
    return _quadratic([field], terms, bound, dim, weights)[0]


def _grid_op(grid, terms, bound, weights=None):
    """Apply a quadratic regulariser to a (..., *spatial, dim) grid"""
    dim = grid.shape[-1]
    ys = _quadratic(list(grid.unbind(-1)), terms, bound, dim, weights)
    return torch.stack(ys, -1)


# ----------------------------------------------------------------------
#   Terms of each energy
#   (voxel_size: list of floats; the energies match the conventions of
#   `nitorch.spatial` and of the compiled regularisers)
# ----------------------------------------------------------------------

def _kinds(dim, **at):
    kinds = ['i'] * dim
    for d, k in at.items():
        kinds[int(d[1:])] = k
    return ''.join(kinds)


def _terms_absolute(dim, comp=0, scale=1.):
    return [(scale, [(comp, 'i' * dim, 1.)])]


def _terms_membrane(dim, voxel_size, comp=0, scale=1.):
    """sum_i |d_i x|^2 / h_i^2, averaged over both sides ('h')"""
    terms = []
    for i in range(dim):
        kinds = _kinds(dim, **{f'a{i}': 'h'})
        terms.append((scale / voxel_size[i] ** 2, [(comp, kinds, 1.)]))
    return terms


def _terms_bending(dim, voxel_size, comp=0, scale=1.):
    """sum_ij |d_i d_j x|^2 / (h_i h_j)^2 (Frobenius norm of the Hessian),
    with central second differences on the diagonal and one-sided first
    differences, averaged over the four sides ('h'), off the diagonal"""
    terms = []
    for i in range(dim):
        kinds = _kinds(dim, **{f'a{i}': 'c'})
        terms.append((scale / voxel_size[i] ** 4, [(comp, kinds, 1.)]))
    for i in range(dim):
        for j in range(i + 1, dim):
            coef = 2 * scale / (voxel_size[i] * voxel_size[j]) ** 2
            kinds = _kinds(dim, **{f'a{i}': 'h', f'a{j}': 'h'})
            terms.append((coef, [(comp, kinds, 1.)]))
    return terms


def _terms_lame_shear(dim, voxel_size):
    """Symmetric part of the Jacobian of a displacement grid (in voxels)"""
    vx = voxel_size
    terms = []
    for i in range(dim):
        for side in 'fb':
            kinds = _kinds(dim, **{f'a{i}': side})
            terms.append((1., [(i, kinds, 1.)]))
        for j in range(i + 1, dim):
            for si in 'fb':      # d_j x_i
                for sj in 'fb':  # d_i x_j
                    terms.append((0.25, [
                        (i, _kinds(dim, **{f'a{j}': si}), vx[i] / vx[j]),
                        (j, _kinds(dim, **{f'a{i}': sj}), vx[j] / vx[i]),
                    ]))
    return terms


def _terms_lame_div(dim):
    """Trace of the Jacobian of a displacement grid (in voxels)"""
    terms = []
    for sides in itertools.product('fb', repeat=dim):
        parts = [(i, _kinds(dim, **{f'a{i}': s}), 1.)
                 for i, s in enumerate(sides)]
        terms.append((1. / 2 ** dim, parts))
    return terms


def _per_component(dim, terms_fn, voxel_size):
    """Terms of a per-component energy of a grid (in voxels), scaled by
    the squared voxel size of each component"""
    terms = []
    for k in range(dim):
        for coef, parts in terms_fn():
            terms.append((coef * voxel_size[k] ** 2,
                          [(k, kinds, scale) for _, kinds, scale in parts]))
    return terms


def _kernel_supports(lame, bound, dim):
    """Whether a convolution (stencil) implements the linear-elastic
    energy exactly with these boundary conditions (see `_c_supports`)"""
    if not any(core.py.ensure_list(lame)):
        return True
    return all(b in ('dft', 'dst1', 'zero')
               for b in _normalise_bound(bound, dim))


def _needs_correction(bound, dim, bending=0):
    """Whether a stencil (convolution) implementation must be corrected
    (`_correct_stencil`) to be exact with these boundary conditions"""
    bound = _normalise_bound(bound, dim)
    return 'dct1' in bound or (bool(bending) and 'replicate' in bound)


def _correct_stencil(x, y, bound, coef=None, subset=None):
    """Make the output of a stencil (convolution) regulariser exact.

    Applying the stencil `L = D'D` to the bound-extended field and reading
    it at the voxels equals the exact operator `D' diag(omega) D` for
    'dft', 'dct2', 'dst2', 'dst1' and 'zero', but not for:

    - 'replicate': the central second differences of the bending energy
      fold their ghost sample back onto the border voxel, which the
      stencil does not, so `coef * (x[inward] - x[border])` must be
      added on the border voxels;
    - 'dct1': the stencil equals `W^{-1} A`, where `W` is the product of
      the row weights of the dct1 axes (1/2 on their border voxels), so
      its output must be multiplied by `W`.

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input field
    y : (..., *subspatial) tensor
        Output of the stencil (regularisation term only, without any
        Hessian of the data term) at the voxels `x[subset]`
    bound : list[str]
        Boundary condition of each spatial axis
    coef : list[float or tensor], optional
        Bending coefficient (factor / h**4) of each axis, possibly a
        tensor that broadcasts against the batch dimensions of `y`.
    subset : list[slice], optional
        Voxels at which `y` was computed (one slice per spatial axis)

    Returns
    -------
    y : (..., *subspatial) tensor
    """
    dim = len(bound)
    shape = x.shape[-dim:]
    subset = list(subset or [slice(None)] * dim)
    subset += [slice(None)] * (dim - len(subset))
    positions = [list(range(n))[sl] for n, sl in zip(shape, subset)]
    nbatch = y.dim() - dim

    if coef is not None:
        delta = None
        for d, b in enumerate(bound):
            n = shape[d]
            if b != 'replicate' or n < 2:
                continue
            for border, inward in ((0, 1), (n - 1, n - 2)):
                if border not in positions[d]:
                    continue
                k = positions[d].index(border)
                index_b = list(subset)
                index_i = list(subset)
                index_b[d], index_i[d] = border, inward
                xb = x[(Ellipsis, *index_b)]
                xi = x[(Ellipsis, *index_i)]
                c = coef[d]
                if torch.is_tensor(c):
                    c = c.reshape(list(c.shape) + [1] * (dim - 1))
                if delta is None:
                    delta = torch.zeros_like(y)
                delta.select(nbatch + d, k).add_((xi - xb) * c)
        if delta is not None:
            y = y + delta

    weights = None
    for d, b in enumerate(bound):
        n = shape[d]
        if b != 'dct1' or n < 2:
            continue
        w = [0.5 if p in (0, n - 1) else 1. for p in positions[d]]
        w = torch.as_tensor(w, dtype=y.dtype, device=y.device)
        w = w.reshape([-1] + [1] * (dim - d - 1))
        weights = w if weights is None else weights * w
    if weights is not None:
        y = y * weights
    return y


def _correct_grid_stencil(v, y, bending, voxel_size, bound, dim,
                          subset=None):
    """Exact borders for the output `y` (at `v[subset]`) of a stencil
    applied to the grid `v`, both laid out as (..., dim, *spatial).
    `bending` is the (global) bending factor."""
    if not _needs_correction(bound, dim, bending):
        return y
    vx = _voxel_size_list(voxel_size, dim)
    coef = None
    if bending:
        # component k of a grid energy is scaled by vx[k]**2
        vx2 = torch.as_tensor([h * h for h in vx], dtype=y.dtype,
                              device=y.device)
        coef = [vx2 * (bending / vx[d] ** 4) for d in range(dim)]
    return _correct_stencil(v, y, _normalise_bound(bound, dim), coef, subset)


def _correct_field_stencil(x, y, bending, voxel_size, bound, dim,
                           subset=None):
    """Exact borders for the output `y` (at `x[subset]`) of a stencil
    applied to the field `x`, both laid out as (..., K, *spatial).
    `bending` is the bending factor of each channel (times `factor`)."""
    if not _needs_correction(bound, dim, bending):
        return y
    vx = _voxel_size_list(voxel_size, dim)
    coef = None
    if torch.is_tensor(bending) or any(core.py.ensure_list(bending)):
        bending = torch.as_tensor(bending, dtype=y.dtype, device=y.device)
        coef = [bending / vx[d] ** 4 for d in range(dim)]
    return _correct_stencil(x, y, _normalise_bound(bound, dim), coef, subset)


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


def absolute(field, weights=None, bound='dct2', dim=None):
    """Precision matrix for the Absolute energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    weights : (..., *spatial) tensor, optional
    bound : str, default='dct2'
        Only matters for 'dct1', whose border voxels have weight 1/2
        (see "Exact operators" in `nitorch.spatial._regularisers`).
    dim : int, default=field.dim()

    Returns
    -------
    field : tensor

    """
    dim = dim or field.dim()
    if 'dct1' in _normalise_bound(bound, dim):
        return _field_op(field, _terms_absolute(dim), bound, dim, weights)
    if weights is not None:
        backend = dict(dtype=field.dtype, device=field.device)
        weights = torch.as_tensor(weights, **backend)
        return field * weights
    else:
        return field


def absolute_grid(grid, voxel_size=1, weights=None, bound='dft'):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    weights : (..., *spatial) tensor, optional
    bound : str, default='dft'
        Only matters for 'dct1', whose border voxels have weight 1/2
        (see "Exact operators" in `nitorch.spatial._regularisers`).

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    if 'dct1' in _normalise_bound(bound, dim):
        vx = _voxel_size_list(voxel_size, dim)
        terms = _per_component(dim, lambda: _terms_absolute(dim), vx)
        return _grid_op(grid, terms, bound, weights)
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


def membrane(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Membrane energy

    Note
    ----
    .. This is exactly equivalent to SPM's membrane energy
    .. The boundary conditions are handled exactly
       (see "Exact operators" in `nitorch.spatial._regularisers`).

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
    dim = dim or field.dim()
    vx = _voxel_size_list(voxel_size, dim)
    return _field_op(field, _terms_membrane(dim, vx), bound, dim, weights)


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
    dim = grid.shape[-1]
    vx = _voxel_size_list(voxel_size, dim)
    terms = _per_component(dim, lambda: _terms_membrane(dim, vx), vx)
    return _grid_op(grid, terms, bound, weights)


def bending(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Bending energy

    Note
    ----
    .. This is exactly equivalent to SPM's bending energy
    .. It is the squared Frobenius norm of the Hessian, with central
       second differences on its diagonal and one-sided first
       differences (averaged over the four sides) off its diagonal.
    .. The boundary conditions are handled exactly
       (see "Exact operators" in `nitorch.spatial._regularisers`).

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
    dim = dim or field.dim()
    vx = _voxel_size_list(voxel_size, dim)
    return _field_op(field, _terms_bending(dim, vx), bound, dim, weights)


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
    dim = grid.shape[-1]
    vx = _voxel_size_list(voxel_size, dim)
    terms = _per_component(dim, lambda: _terms_bending(dim, vx), vx)
    return _grid_op(grid, terms, bound, weights)


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
    .. All components share the boundary condition of each axis, and
       the energy is that of the rows centred on the voxels (and ghost
       voxels), with the row weights of
       "Exact operators" in `nitorch.spatial._regularisers`.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    dim = grid.shape[-1]
    vx = _voxel_size_list(voxel_size, dim)
    return _grid_op(grid, _terms_lame_shear(dim, vx), bound, weights)


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
    .. All components share the boundary condition of each axis, and
       the energy is that of the rows centred on the voxels (and ghost
       voxels), with the row weights of
       "Exact operators" in `nitorch.spatial._regularisers`.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1 (actually unused)
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    dim = grid.shape[-1]
    return _grid_op(grid, _terms_lame_div(dim), bound, weights)


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
    bound : str, default='dft'
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    if not (absolute or membrane or bending or any(ensure_list(lame))):
        return torch.zeros_like(v)

    if (c_solvers and not v.requires_grad
            and _c_supports(weights, bending, lame, bound)):
        return c_solvers.c_regulariser_grid(
            v, weights, None,
            absolute, membrane, bending, lame, factor,
            voxel_size=voxel_size, bound=bound)

    if (torch.is_tensor(kernel) or kernel) and _kernel_supports(lame, bound, dim):
        if not torch.is_tensor(kernel):
            kernel = regulariser_grid_kernel(dim, absolute, membrane, bending,
                                             lame, factor, voxel_size, **backend)
        v = core.utils.fast_movedim(v, -1, -dim-1)
        y = spconv(v, kernel, bound=bound, dim=dim)
        y = _correct_grid_stencil(v, y, bending * factor, voxel_size,
                                  bound, dim)
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

    # All energies that share their weights are applied in a single pass
    # (one extension and one fold of the grid).
    vx = _voxel_size_list(voxel_size, dim)
    groups = {}

    def add(w, factor, terms):
        group = groups.setdefault(id(w), [w, []])
        group[1].extend((c * factor, p) for c, p in terms)

    if absolute:
        add(wa, absolute, _per_component(dim, lambda: _terms_absolute(dim), vx))
    if membrane:
        add(wm, membrane, _per_component(dim, lambda: _terms_membrane(dim, vx), vx))
    if bending:
        add(wb, bending, _per_component(dim, lambda: _terms_bending(dim, vx), vx))
    if lame[0]:
        add(wl[0], lame[0], _terms_lame_shear(dim, vx))
    if lame[1]:
        add(wl[1], lame[1], _terms_lame_div(dim))

    y = None
    for w, terms in groups.values():
        yw = _grid_op(v, terms, bound, w)
        y = yw if y is None else y + yw
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
    if not (absolute or membrane or bending):
        return torch.zeros_like(x)

    dim = dim or (x.dim() - 1)
    if (c_solvers and not x.requires_grad
            and _c_supports(weights, bending, bound=bound)):
        return c_solvers.c_regulariser(
            x, weights, None, dim,
            absolute, membrane, bending, factor,
            voxel_size=voxel_size, bound=bound)

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

    absolute = core.py.make_list(absolute, nb_prm) if absolute else []
    membrane = core.py.make_list(membrane, nb_prm) if membrane else []
    bending = core.py.make_list(bending, nb_prm) if bending else []

    # Iterate over the channel dimension (-dim-1), not the first one (which
    # is a batch dimension when there is one), and accumulate out of
    # place (in-place updates of views of `y` broke autograd).
    def per_channel(w):
        if w is None:
            return [None] * nb_prm
        return torch.as_tensor(w, **backend).expand(x.shape).unbind(-dim-1)

    # All energies of a channel that share their weights are applied in a
    # single pass (one extension and one fold of the field).
    vx = _voxel_size_list(voxel_size, dim)
    xs = x.unbind(-dim-1)
    keys = (id(wa), id(wm), id(wb))
    wa, wm, wb = per_channel(wa), per_channel(wm), per_channel(wb)
    ys = []
    for k in range(nb_prm):
        groups = {}

        def add(key, w, factor, terms):
            group = groups.setdefault(key, [w, []])
            group[1].extend((c * factor, p) for c, p in terms)

        if absolute and absolute[k]:
            add(keys[0], wa[k], absolute[k], _terms_absolute(dim))
        if membrane and membrane[k]:
            add(keys[1], wm[k], membrane[k], _terms_membrane(dim, vx))
        if bending and bending[k]:
            add(keys[2], wb[k], bending[k], _terms_bending(dim, vx))
        yk = None
        for w, terms in groups.values():
            yw = _field_op(xs[k], terms, bound, dim, w)
            yk = yw if yk is None else yk + yw
        ys.append(yk if yk is not None else torch.zeros_like(xs[k]))
    y = torch.stack(ys, -dim-1)

    pad_spatial = (Ellipsis,) + (None,) * dim
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
