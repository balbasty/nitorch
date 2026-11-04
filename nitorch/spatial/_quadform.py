"""Quadratic regularisers on a lattice with boundary conditions.

Every regulariser is a quadratic form whose precision operator is

    A = sum_t  c_t  D_t'  diag(omega * w)  D_t

Each row of `D_t` is a linear combination of finite differences of the
components of the field, evaluated on its bound-extended version:

    D_t x = sum_p  s_p  K_p  E  x[comp_p]

- `E` extends the field according to its boundary condition: each of its
  rows copies a voxel, flips its sign, or is zero.
- `K_p` is a tensor product of 1D stencils, one per spatial axis:
  'i' (identity), 'f' / 'b' (forward / backward first difference) or
  'c' (central second difference).
- `w` are optional voxel weights (RLS), evaluated at the rows.
- `omega` are row weights, one per row of the lattice, chosen so that the
  energy is that of one period of the extended field divided by the
  number of copies of the field that it contains (for the periodic,
  symmetric and antisymmetric boundaries), the energy over the whole
  space ('zero'), or the energy of the rows centred in the field of
  view ('replicate').

All rows are centred on the voxels of the lattice, including one ghost
voxel on each side of each axis. Finite differences that live between
voxels ('f' and 'b') are always averaged over both sides, so that the
energy at a row is symmetric about any mirror. The row weights of an
axis are then:

    dft, dct2, dst2, replicate   [0,   1, 1, ..., 1, 1,   0]
    dct1                         [0, 1/2, 1, ..., 1, 1/2, 0]
    dst1                         [1/2, 1, 1, ..., 1, 1, 1/2]
    zero                         [1,   1, 1, ..., 1, 1,   1]

`dct1` mirrors the field about its border voxels, which therefore appear
once per period while the other voxels appear twice (trapezoidal rule);
`dst1` mirrors it about the ghost voxels, which are zero and appear once
per period.

The operator is A = E' K' diag(omega w) K E: the transpose of the
extension (`fold`) adds the extended values back onto the voxels they
copy. Applying the stencil `L = K'K` to the extended field and reading it
at the voxels (as convolution-based implementations do) is equivalent
for 'dft', 'dct2', 'dst2', 'dst1' and 'zero', but not for 'dct1' and
'replicate' (see `correct_stencil`).
"""
import torch
from nitorch.core import py

__all__ = ['Lattice', 'quadform', 'needs_correction', 'correct_stencil',
           'weight_bound']


_BOUNDS = ('dft', 'dct2', 'dst2', 'dct1', 'dst1', 'zero', 'replicate')


def _normalise_bound(bound):
    from nitorch._C.grid import bound_to_nitorch
    return bound_to_nitorch(bound, as_type='str')


def _extend_index(j, n, bound):
    """Source voxel and sign of the extended sample `j` (sign 0: zero)"""
    if bound == 'dft':
        return j % n, 1
    if bound == 'replicate':
        return min(max(j, 0), n - 1), 1
    if bound == 'zero':
        return (j, 1) if 0 <= j < n else (0, 0)
    if bound in ('dct2', 'dst2'):
        period = 2 * n
        k, s = j % period, 1
        if k >= n:
            k, s = period - 1 - k, (-1 if bound == 'dst2' else 1)
        return k, s
    if bound == 'dct1':
        if n == 1:
            return 0, 1
        period = 2 * n - 2
        k = j % period
        return (period - k if k >= n else k), 1
    if bound == 'dst1':
        period = 2 * n + 2
        k = (j + 1) % period          # k=0 <-> j=-1, k=n+1 <-> j=n (zeros)
        if k == 0 or k == n + 1:
            return 0, 0
        if k <= n:
            return k - 1, 1
        return period - k - 1, -1
    raise ValueError(f'Unknown bound {bound}')


def _weight_bound(bound):
    """Boundary condition of the weights (always even) of a field"""
    return {'dst2': 'dct2', 'dst1': 'replicate', 'zero': 'replicate'}.get(
        bound, bound)


def weight_bound(bound, dim):
    """Boundary conditions of voxel weights (RLS), which are always even:
    'dst2' -> 'dct2', and 'replicate' where the field is zero outside
    ('dst1', 'zero')"""
    return [_weight_bound(b) for b in Lattice.normalise_bound(bound, dim)]


def _row_weights(n, bound):
    """Row weights on the lattice [-1, n] (one ghost row on each side)"""
    w = [0.] + [1.] * n + [0.]
    if bound == 'dct1' and n > 1:
        w[1] = w[n] = 0.5
    elif bound == 'dst1':
        w[0] = w[-1] = 0.5
    elif bound == 'zero':
        w[0] = w[-1] = 1.
    return w


class Lattice:
    """Extension, fold and row weights of a field with boundary conditions

    Arrays are laid out as `(..., *spatial)`. Extended arrays have a
    margin of `margin` voxels on each side of each axis (two for
    `quadform`); row arrays have a margin of one voxel (the ghost rows).
    """

    @staticmethod
    def normalise_bound(bound, dim):
        """Boundary condition of each axis, in nitorch's convention"""
        return _normalise_bound(py.ensure_list(bound, dim))

    def __init__(self, shape, bound='dct2', device=None, margin=2):
        shape = list(shape)
        bound = _normalise_bound(py.ensure_list(bound, len(shape)))
        self.shape = shape
        self.bound = bound
        self.dim = len(shape)
        self.margin = margin = py.ensure_list(margin, self.dim)
        self.index, self.sign, self.windex, self.omega = [], [], [], []
        for n, b, m in zip(shape, bound, margin):
            idx, sgn = zip(*[_extend_index(j, n, b) for j in range(-m, n + m)])
            self.index.append(torch.as_tensor(idx, dtype=torch.long, device=device))
            self.sign.append(torch.as_tensor(sgn, dtype=torch.int8, device=device))
            wb = _weight_bound(b)
            widx = [_extend_index(j, n, wb)[0] for j in range(-1, n + 1)]
            self.windex.append(torch.as_tensor(widx, dtype=torch.long, device=device))
            self.omega.append(_row_weights(n, b))

    def _axis(self, x, d):
        return x.dim() - self.dim + d

    def extend(self, x):
        """(..., *spatial) -> (..., *[n + 2 * margin])"""
        for d in range(self.dim):
            ax = self._axis(x, d)
            sign = self.sign[d].to(x.dtype)
            sign = sign.reshape([-1] + [1] * (x.dim() - ax - 1))
            x = x.index_select(ax, self.index[d]) * sign
        return x

    def fold(self, y):
        """(..., *[n + 2 * margin]) -> (..., *spatial): transpose of `extend`"""
        for d in range(self.dim):
            ax = self._axis(y, d)
            sign = self.sign[d].to(y.dtype)
            sign = sign.reshape([-1] + [1] * (y.dim() - ax - 1))
            shape = list(y.shape)
            shape[ax] = self.shape[d]
            y = y.new_zeros(shape).index_add(ax, self.index[d], y * sign)
        return y

    def row_weights(self, dtype=None, device=None):
        """Product of the row weights of each axis: (*[n + 2]) tensor"""
        out = None
        for d in range(self.dim):
            w = torch.as_tensor(self.omega[d], dtype=dtype, device=device)
            w = w.reshape([-1] + [1] * (self.dim - d - 1))
            out = w if out is None else out * w
        return out

    def is_uniform(self):
        """True if all row weights inside the field of view are one and
        all ghost row weights are zero"""
        return all(b not in ('dct1', 'dst1', 'zero') for b in self.bound)

    def extend_weights(self, w):
        """Voxel weights (..., *spatial) at the rows (..., *[n + 2])"""
        for d in range(self.dim):
            ax = self._axis(w, d)
            w = w.index_select(ax, self.windex[d])
        return w


def _slab(x, ax, start, length):
    return x.narrow(ax, start, length)


def _stencil(x, ax, kind):
    """Apply a 1D stencil along `ax`: size n + 4 -> rows, size n + 2"""
    n = x.shape[ax] - 4
    lo, mid, hi = _slab(x, ax, 0, n + 2), _slab(x, ax, 1, n + 2), _slab(x, ax, 2, n + 2)
    if kind == 'i':
        return mid
    if kind == 'f':
        return hi - mid
    if kind == 'b':
        return mid - lo
    if kind == 'c':
        return lo - 2 * mid + hi
    raise ValueError(kind)


def _pad(x, ax, before, after):
    shape = list(x.shape)
    parts = []
    if before:
        shape[ax] = before
        parts.append(x.new_zeros(shape))
    parts.append(x)
    if after:
        shape[ax] = after
        parts.append(x.new_zeros(shape))
    return torch.cat(parts, ax)


def _stencil_t(r, ax, kind):
    """Transpose of `_stencil`: rows, size n + 2 -> size n + 4"""
    if kind == 'i':
        return _pad(r, ax, 1, 1)
    if kind == 'f':
        return _pad(r, ax, 2, 0) - _pad(r, ax, 1, 1)
    if kind == 'b':
        return _pad(r, ax, 1, 1) - _pad(r, ax, 0, 2)
    if kind == 'c':
        return _pad(r, ax, 0, 2) - 2 * _pad(r, ax, 1, 1) + _pad(r, ax, 2, 0)
    raise ValueError(kind)


def _apply(x, kinds, dim):
    for d, k in enumerate(kinds):
        x = _stencil(x, x.dim() - dim + d, k)
    return x


def _apply_t(r, kinds, dim):
    for d, k in enumerate(kinds):
        r = _stencil_t(r, r.dim() - dim + d, k)
    return r


def quadform(xs, terms, lattice, weights=None):
    """Apply the operator `sum_t c_t D_t' diag(omega w) D_t`.

    Parameters
    ----------
    xs : list[(..., *spatial) tensor]
        Components of the field
    terms : list[(float, list[(int, str, float)])]
        Each term is `(c_t, parts)`, and each part is
        `(component, kinds, scale)`, where `kinds` has one character
        per spatial axis among 'i', 'f', 'b', 'c'.
    lattice : Lattice
    weights : (..., *spatial) tensor, optional
        Voxel weights

    Returns
    -------
    ys : list[(..., *spatial) tensor]
    """
    dim = lattice.dim
    if any(m != 2 for m in lattice.margin):
        raise ValueError('quadform needs a lattice with a margin of 2')
    x0 = xs[0]
    omega = lattice.row_weights(dtype=x0.dtype, device=x0.device)
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=x0.dtype, device=x0.device)
        omega = omega * lattice.extend_weights(weights)
    exts = [lattice.extend(x) for x in xs]
    outs = [None] * len(xs)
    for coef, parts in terms:
        row = 0
        for comp, kinds, scale in parts:
            row = row + _apply(exts[comp], kinds, dim) * scale
        row = row * omega
        for comp, kinds, scale in parts:
            y = _apply_t(row, kinds, dim) * (coef * scale)
            outs[comp] = y if outs[comp] is None else outs[comp] + y
    return [lattice.fold(o) if o is not None else torch.zeros_like(x)
            for o, x in zip(outs, xs)]


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


def terms_absolute(dim, comp=0, scale=1.):
    return [(scale, [(comp, 'i' * dim, 1.)])]


def terms_membrane(dim, voxel_size, comp=0, scale=1.):
    """sum_i |d_i x|^2 / h_i^2, averaged over both sides"""
    terms = []
    for i in range(dim):
        for side in 'fb':
            kinds = _kinds(dim, **{f'a{i}': side})
            terms.append((0.5 * scale / voxel_size[i] ** 2, [(comp, kinds, 1.)]))
    return terms


def terms_bending(dim, voxel_size, comp=0, scale=1.):
    """sum_ij |d_i d_j x|^2 / (h_i h_j)^2 (Frobenius norm of the Hessian),
    with central second differences on the diagonal and one-sided first
    differences, averaged over the four sides, off the diagonal"""
    terms = []
    for i in range(dim):
        kinds = _kinds(dim, **{f'a{i}': 'c'})
        terms.append((scale / voxel_size[i] ** 4, [(comp, kinds, 1.)]))
    for i in range(dim):
        for j in range(i + 1, dim):
            coef = 2 * scale / (voxel_size[i] * voxel_size[j]) ** 2 / 4
            for si in 'fb':
                for sj in 'fb':
                    kinds = _kinds(dim, **{f'a{i}': si, f'a{j}': sj})
                    terms.append((coef, [(comp, kinds, 1.)]))
    return terms


def terms_lame_shear(dim, voxel_size):
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


def terms_lame_div(dim):
    """Trace of the Jacobian of a displacement grid (in voxels)"""
    terms = []
    for sides in _product('fb', dim):
        parts = [(i, _kinds(dim, **{f'a{i}': s}), 1.) for i, s in enumerate(sides)]
        terms.append((1. / 2 ** dim, parts))
    return terms


def _product(values, n):
    import itertools
    return itertools.product(values, repeat=n)


# ----------------------------------------------------------------------
#   Correction of stencil-based (convolution) implementations
# ----------------------------------------------------------------------

def needs_correction(bound, dim, bending=0):
    """Whether a stencil (convolution) implementation must be corrected
    (`correct_stencil`) to be exact with these boundary conditions"""
    bound = Lattice.normalise_bound(bound, dim)
    return 'dct1' in bound or (bool(bending) and 'replicate' in bound)


def correct_stencil(x, y, lattice, coef=None, subset=None):
    """Make the output of a stencil (convolution) regulariser exact.

    Applying the stencil `L = K'K` to the bound-extended field and reading
    it at the voxels equals the exact operator `E' K' diag(omega) K E`
    for 'dft', 'dct2', 'dst2', 'dst1' and 'zero', but not for:

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
    lattice : Lattice
    coef : list[float or tensor], optional
        Bending coefficient (factor / h**4) of each axis, possibly a
        tensor that broadcasts against the batch dimensions of `y`.
    subset : list[slice], optional
        Voxels at which `y` was computed (one slice per spatial axis)

    Returns
    -------
    y : (..., *subspatial) tensor
    """
    dim = lattice.dim
    subset = list(subset or [slice(None)] * dim)
    subset += [slice(None)] * (dim - len(subset))
    positions = [list(range(n))[sl] for n, sl in zip(lattice.shape, subset)]
    nbatch = y.dim() - dim

    if coef is not None:
        delta = None
        for d, b in enumerate(lattice.bound):
            n = lattice.shape[d]
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
    for d, b in enumerate(lattice.bound):
        n = lattice.shape[d]
        if b != 'dct1' or n < 2:
            continue
        w = [0.5 if p in (0, n - 1) else 1. for p in positions[d]]
        w = torch.as_tensor(w, dtype=y.dtype, device=y.device)
        w = w.reshape([-1] + [1] * (dim - d - 1))
        weights = w if weights is None else weights * w
    if weights is not None:
        y = y * weights
    return y
