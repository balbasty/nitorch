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
        self.mpos, self.msrc, self.msign = [], [], []
        for n, b, m in zip(shape, bound, margin):
            idx, sgn = zip(*[_extend_index(j, n, b) for j in range(-m, n + m)])
            self.index.append(torch.as_tensor(idx, dtype=torch.long, device=device))
            self.sign.append(torch.as_tensor(sgn, dtype=torch.int8, device=device))
            # margins only: position in the extended array, source voxel
            # and sign
            pos = list(range(m)) + list(range(n + m, n + 2 * m))
            self.mpos.append(torch.as_tensor(pos, dtype=torch.long, device=device))
            self.msrc.append(torch.as_tensor([idx[p] for p in pos], dtype=torch.long, device=device))
            self.msign.append(torch.as_tensor([sgn[p] for p in pos], dtype=torch.int8, device=device))
            wb = _weight_bound(b)
            widx = [_extend_index(j, n, wb)[0] for j in range(-1, n + 1)]
            self.windex.append(torch.as_tensor(widx, dtype=torch.long, device=device))
            self.omega.append(_row_weights(n, b))

    def _axis(self, x, d):
        return x.dim() - self.dim + d

    def _sign(self, d, ndim, ax, dtype):
        sign = self.msign[d].to(dtype)
        return sign.reshape([-1] + [1] * (ndim - ax - 1))

    def extend(self, x):
        """(..., *spatial) -> (..., *[n + 2 * margin])

        The field is copied once into the interior of the extended array,
        whose margins are then filled axis by axis (the margins of an axis
        are copied from the full extent of the previous axes, so that the
        corners are filled too)."""
        dim = self.dim
        ext_shape = [n + 2 * m for n, m in zip(self.shape, self.margin)]
        ext = x.new_empty([*x.shape[:-dim], *ext_shape])
        interior = ext
        for d in range(dim):
            interior = interior.narrow(self._axis(ext, d), self.margin[d], self.shape[d])
        interior.copy_(x)
        for d in range(dim):
            if not self.margin[d]:
                continue
            ax = self._axis(ext, d)
            src = ext.index_select(ax, self.msrc[d] + self.margin[d])
            src = src * self._sign(d, ext.dim(), ax, ext.dtype)
            ext.index_copy_(ax, self.mpos[d], src)
        return ext

    def fold(self, y):
        """(..., *[n + 2 * margin]) -> (..., *spatial): transpose of `extend`

        The margins of each axis are added (with their signs) onto the
        voxels they copy, in reverse order of `extend`. `y` is modified
        in place."""
        dim = self.dim
        for d in reversed(range(dim)):
            ax = self._axis(y, d)
            m = self.margin[d]
            if m:
                src = y.index_select(ax, self.mpos[d])
                src = src * self._sign(d, y.dim(), ax, y.dtype)
                y.narrow(ax, m, self.shape[d]).index_add_(ax, self.msrc[d], src)
            y = y.narrow(ax, m, self.shape[d])
        return y.contiguous()

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


# Taps (offset, coefficient) of each 1D stencil, at row `r`:
# 'f': x[r+1] - x[r], 'b': x[r] - x[r-1], 'c': x[r-1] - 2 x[r] + x[r+1]
# 'h' is the first difference on the half-voxel lattice (rows n + 1/2,
# from first - 1/2 to last + 1/2): averaging 'f' and 'b' over both sides
# equals 'h' with the row weights averaged over the two neighbouring rows.
_TAPS = {
    'i': ((0, 1.),),
    'f': ((0, -1.), (1, 1.)),
    'b': ((-1, -1.), (0, 1.)),
    'c': ((-1, 1.), (0, -2.), (1, 1.)),
    'h': ((0, -1.), (1, 1.)),
}


def _row_range(n, bound):
    """Rows of an axis with a nonzero weight, and their weights
    (None if they are all one): (first row, number of rows, weights)"""
    omega = _row_weights(n, bound)          # rows [-1, n]
    first = 0 if omega[0] == 0 else -1
    last = n - 1 if omega[-1] == 0 else n
    omega = omega[first + 1:last + 2]
    if all(w == 1 for w in omega):
        omega = None
    return first, last - first + 1, omega


def _rows(ext, kinds, ranges, margin=2):
    """Apply a separable stencil to an extended array, at the rows
    `ranges` (one (first, count) per axis)"""
    dim = len(kinds)
    for d, (kind, (first, count)) in enumerate(zip(kinds, ranges)):
        ax = ext.dim() - dim + d
        taps = _TAPS[kind]
        if kind == 'i':
            ext = ext.narrow(ax, first + margin, count)
            continue
        out = None
        for off, c in taps:
            sl = ext.narrow(ax, first + margin + off, count)
            if out is None:
                out = sl * c if c != 1 else sl.clone()
            elif c == 1:
                out = out.add_(sl)
            elif c == -1:
                out = out.sub_(sl)
            else:
                out = out.add_(sl, alpha=c)
        ext = out
    return ext


def _add_rows_t(acc, row, kinds, ranges, alpha, margin=2):
    """acc += alpha * (transpose of the separable stencil)(row), where
    `acc` is an extended array (in place)"""
    import itertools
    dim = len(kinds)
    taps = [_TAPS[k] for k in kinds]
    for combo in itertools.product(*taps):
        view = acc
        c = alpha
        for d, ((off, cd), (first, count)) in enumerate(zip(combo, ranges)):
            view = view.narrow(acc.dim() - dim + d, first + margin + off, count)
            c = c * cd
        view.add_(row, alpha=c)


def _half_ranges(kinds, ranges):
    """Row ranges of a term: one more (half) row along its 'h' axes"""
    return [(first - 1, count + 1) if k == 'h' else (first, count)
            for k, (first, count) in zip(kinds, ranges)]


def _half_weights(omega, kinds, ranges, dim, **backend):
    """Row weights of a term: along its 'h' axes, the weight of a half
    row is the average of the weights of its two neighbouring rows (zero
    outside of the rows)"""
    if 'h' not in kinds:
        return omega
    if omega is None:
        omega = torch.ones([1] * dim, **backend)
    for d, k in enumerate(kinds):
        if k != 'h':
            continue
        ax = omega.dim() - dim + d
        if omega.shape[ax] == 1:
            omega = omega.expand([*omega.shape[:ax], ranges[d][1],
                                  *omega.shape[ax+1:]])
        zero = omega.new_zeros([*omega.shape[:ax], 1, *omega.shape[ax+1:]])
        omega = torch.cat([zero, omega], ax) + torch.cat([omega, zero], ax)
        omega = omega * 0.5
    return omega


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

    Notes
    -----
    Rows are only computed where their weight is nonzero (the field of
    view, plus the ghost rows of 'zero' and 'dst1'), and the transposed
    stencils are accumulated in place into one extended buffer per
    component, which is then folded.
    """
    dim = lattice.dim
    if any(m != 2 for m in lattice.margin):
        raise ValueError('quadform needs a lattice with a margin of 2')
    x0 = xs[0]
    backend = dict(dtype=x0.dtype, device=x0.device)

    # rows with a nonzero weight, and the weights that are not one
    ranges, omega = [], None
    for d, (n, b) in enumerate(zip(lattice.shape, lattice.bound)):
        first, count, w = _row_range(n, b)
        ranges.append((first, count))
        if w is not None:
            w = torch.as_tensor(w, **backend).reshape([-1] + [1] * (dim - d - 1))
            omega = w if omega is None else omega * w
    if weights is not None:
        weights = torch.as_tensor(weights, **backend)
        weights = lattice.extend_weights(weights)    # rows [-1, n]
        for d, (first, count) in enumerate(ranges):
            weights = weights.narrow(weights.dim() - dim + d, first + 1, count)
        omega = weights if omega is None else omega * weights

    exts = [lattice.extend(x) for x in xs]
    accs = [None] * len(xs)
    half_omega = {}
    for coef, parts in terms:
        # all parts of a term share their rows: their 'h' axes must match
        halves = ''.join('h' if 'h' in [p[1][d] for p in parts] else '.'
                         for d in range(dim))
        if any(('h' in [p[1][d] for p in parts]) and
               any(p[1][d] != 'h' for p in parts) for d in range(dim)):
            raise ValueError("All parts of a term must have the same 'h' axes")
        term_ranges = _half_ranges(halves, ranges)
        if halves not in half_omega:
            half_omega[halves] = _half_weights(omega, halves, ranges, dim,
                                               **backend)
        term_omega = half_omega[halves]
        row = None
        for comp, kinds, scale in parts:
            r = _rows(exts[comp], kinds, term_ranges)
            if row is None:
                row = r * scale
            else:
                row = row.add_(r, alpha=scale)
        if term_omega is not None:
            row = row * term_omega
        for comp, kinds, scale in parts:
            if accs[comp] is None:
                shape = torch.broadcast_shapes(exts[comp].shape[:-dim],
                                               row.shape[:-dim])
                accs[comp] = exts[comp].new_zeros(
                    [*shape, *exts[comp].shape[-dim:]])
            _add_rows_t(accs[comp], row, kinds, term_ranges, coef * scale)
    return [lattice.fold(a) if a is not None else torch.zeros_like(x)
            for a, x in zip(accs, xs)]


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
    """sum_i |d_i x|^2 / h_i^2, averaged over both sides ('h')"""
    terms = []
    for i in range(dim):
        kinds = _kinds(dim, **{f'a{i}': 'h'})
        terms.append((scale / voxel_size[i] ** 2, [(comp, kinds, 1.)]))
    return terms


def terms_bending(dim, voxel_size, comp=0, scale=1.):
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
