"""Dense reference for the regularisers of `nitorch.spatial`.

Energies are lists of terms `(c_t, parts)`, each part being
`(component, kinds, scale)` with one kind per axis among 'i' (identity),
'f'/'b' (forward/backward first difference) and 'c' (central second
difference).

Definition ("F")
----------------
Each energy is a sum of rows centred on the voxels of the field of view,
built from the components of the field extended according to their
boundary condition:

    A = E' D' F' diag(w) F D E

- `E` extends each component on a lattice that is larger than the field
  of view (by `PAD` samples on each side of each axis);
- `D` are the finite differences on that lattice;
- `F` keeps the rows centred on the voxels of the field of view (no
  ghost rows, no row weights);
- `w` are optional voxel weights, applied once per row.

A stencil (convolution) implementation applies `L = D' diag(w~) D` to
the extended field and reads it at the voxels:

    S = R D' diag(w~) D E

where `R` (= `F`) selects the voxels of the field of view and `w~` is
the even extension of the weights (`even_bound`). `S` sums all the rows
of the lattice within the radius of the stencil, so that it differs from
`A` near the borders: `B = A - S` is the border correction
(`border_correction`), a sum of folds (taps of the kept rows that lie
outside of the field of view, folded back onto their source voxel) and
ghosts (rows centred outside of the field of view, which `S` includes).

With 'sliding' boundary conditions (displacement grids), each component
has its own extension: along a sliding axis `d`, the component `d` is
extended antisymmetrically ('dst2') and the others symmetrically
('dct2').

The reference is independent from the implementation (no padding, no
finite difference routine), and only meant for small fields.
"""
import itertools
import torch

dt = torch.double
PAD = 4


# ----------------------------------------------------------------------
#   Extensions
# ----------------------------------------------------------------------

def extend_index(j, n, bound):
    """Source voxel and sign of the extended sample `j` (sign 0: zero)"""
    if bound == 'dft':
        return j % n, 1
    if bound == 'replicate':
        return min(max(j, 0), n - 1), 1
    if bound == 'zero':
        return (j, 1) if 0 <= j < n else (0, 0)
    if bound in ('dct2', 'dst2'):
        k = j % (2 * n)
        if k >= n:
            return 2 * n - 1 - k, (-1 if bound == 'dst2' else 1)
        return k, 1
    if bound == 'dct1':
        if n == 1:
            return 0, 1
        k = j % (2 * n - 2)
        return (2 * n - 2 - k if k >= n else k), 1
    if bound == 'dst1':
        k = (j + 1) % (2 * n + 2)     # k = 0 <-> j = -1 (a zero)
        if k == 0 or k == n + 1:
            return 0, 0
        if k <= n:
            return k - 1, 1
        return 2 * n + 1 - k, -1
    raise ValueError(bound)


def even_bound(bound):
    """Boundary condition of the weights (never negated; where the field
    is zero, they take the value of the nearest voxel)"""
    return {'dst2': 'dct2', 'sliding': 'dct2', 'dst1': 'replicate',
            'zero': 'replicate'}.get(bound, bound)


def component_bounds(bound, dim, ncomp):
    """Boundary condition of each axis, for each component.

    Along a 'sliding' axis `d`, the component `d` is antisymmetric
    ('dst2') and the other ones symmetric ('dct2').
    """
    if isinstance(bound, str):
        bound = [bound] * dim
    bound = list(bound)
    if 'sliding' in bound and ncomp != dim:
        raise ValueError('sliding needs one component per axis')
    return [[('dst2' if d == k else 'dct2') if b == 'sliding' else b
             for d, b in enumerate(bound)] for k in range(ncomp)]


def extension(n, bound):
    """(n + 2 * PAD, n) extension of an axis"""
    E = torch.zeros(n + 2 * PAD, n, dtype=dt)
    for p, j in enumerate(range(-PAD, n + PAD)):
        k, s = extend_index(j, n, bound)
        if s:
            E[p, k] = s
    return E


def difference(L, kind):
    """(L, L) finite difference on a lattice of length L (zero past it)"""
    I = torch.eye(L, dtype=dt)
    S = torch.diag(torch.ones(L - 1, dtype=dt), 1)   # (S x)[k] = x[k + 1]
    return dict(i=I, f=S - I, b=I - S.T, c=S - 2 * I + S.T)[kind]


def select(n):
    """(n, n + 2 * PAD) selection of the voxels of the field of view"""
    R = torch.zeros(n, n + 2 * PAD, dtype=dt)
    R[torch.arange(n), torch.arange(n) + PAD] = 1
    return R


def kron(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


# ----------------------------------------------------------------------
#   Operators
# ----------------------------------------------------------------------

def operators(shape, bound, terms, ncomp=1, weights=None):
    """Dense operator `A` (definition F) and stencil `S` of `terms`.

    Returns two (ncomp * prod(shape), ncomp * prod(shape)) matrices
    acting on the components stacked along the first dimension.
    """
    dim = len(shape)
    N = int(torch.Size(shape).numel())
    cb = component_bounds(bound, dim, ncomp)
    Es = [[extension(n, b) for n, b in zip(shape, cbk)] for cbk in cb]
    Rs = [select(n) for n in shape]
    Ls = [n + 2 * PAD for n in shape]
    w = (torch.ones(N, dtype=dt) if weights is None else
         torch.as_tensor(weights, dtype=dt).flatten())
    # even extension of the weights, on the lattice
    W = kron([extension(n, even_bound(b)) for n, b in zip(shape, cb[0])])
    wl = W @ w
    A = torch.zeros(ncomp * N, ncomp * N, dtype=dt)
    S = torch.zeros_like(A)
    for coef, parts in terms:
        DE = torch.zeros(int(torch.Size(Ls).numel()), ncomp * N, dtype=dt)
        RDt = torch.zeros(ncomp * N, DE.shape[0], dtype=dt)
        for comp, kinds, scale in parts:
            Ds = [difference(L, k) for L, k in zip(Ls, kinds)]
            DE[:, comp * N:(comp + 1) * N] += scale * kron(
                [D @ E for D, E in zip(Ds, Es[comp])])
            RDt[comp * N:(comp + 1) * N, :] += scale * kron(
                [R @ D.T for R, D in zip(Rs, Ds)])
        FDE = kron(Rs) @ DE
        A += coef * FDE.T @ (w[:, None] * FDE)
        S += coef * RDt @ (wl[:, None] * DE)
    return A, S


def operator(shape, bound, terms, ncomp=1, weights=None):
    """Dense operator `A` (definition F) of `terms`"""
    return operators(shape, bound, terms, ncomp, weights)[0]


def reflect(v, axis):
    """Mirror image of a displacement grid (..., *spatial, dim) about a
    plane normal to `axis`: the field is flipped along `axis` and its
    component `axis` changes sign."""
    dim = v.shape[-1]
    v = v.flip(v.dim() - dim - 1 + axis).clone()
    v[..., axis].neg_()
    return v


# ----------------------------------------------------------------------
#   Energies (voxel sizes `vx`; grids are in voxels, and the energy of
#   their component k is scaled by vx[k]**2)
# ----------------------------------------------------------------------

def _kinds(dim, **at):
    kinds = ['i'] * dim
    for d, k in at.items():
        kinds[int(d[1:])] = k
    return ''.join(kinds)


def terms_absolute(dim):
    return [(1., [(0, 'i' * dim, 1.)])]


def terms_membrane(dim, vx):
    """sum_i |d_i x|^2 / h_i^2, averaged over both sides"""
    return [(0.5 / vx[i] ** 2, [(0, _kinds(dim, **{f'a{i}': s}), 1.)])
            for i in range(dim) for s in 'fb']


def terms_bending(dim, vx):
    """Frobenius norm of the Hessian: central second differences on its
    diagonal, one-sided first differences (averaged over the four sides)
    off its diagonal"""
    terms = [(1. / vx[i] ** 4, [(0, _kinds(dim, **{f'a{i}': 'c'}), 1.)])
             for i in range(dim)]
    for i in range(dim):
        for j in range(i + 1, dim):
            for si in 'fb':
                for sj in 'fb':
                    kinds = _kinds(dim, **{f'a{i}': si, f'a{j}': sj})
                    terms.append((0.5 / (vx[i] * vx[j]) ** 2, [(0, kinds, 1.)]))
    return terms


def terms_lame_shear(dim, vx):
    """Symmetric part of the Jacobian of a displacement grid"""
    terms = []
    for i in range(dim):
        for s in 'fb':
            terms.append((1., [(i, _kinds(dim, **{f'a{i}': s}), 1.)]))
        for j in range(i + 1, dim):
            for si in 'fb':
                for sj in 'fb':
                    terms.append((0.25, [
                        (i, _kinds(dim, **{f'a{j}': si}), vx[i] / vx[j]),
                        (j, _kinds(dim, **{f'a{i}': sj}), vx[j] / vx[i])]))
    return terms


def terms_lame_div(dim):
    """Trace of the Jacobian of a displacement grid"""
    return [(1. / 2 ** dim, [(i, _kinds(dim, **{f'a{i}': s}), 1.)
                             for i, s in enumerate(sides)])
            for sides in itertools.product('fb', repeat=dim)]


def per_component(dim, terms, vx):
    """Terms of a per-component energy of a grid"""
    return [(c * vx[k] ** 2, [(k, kinds, s) for _, kinds, s in parts])
            for k in range(dim) for c, parts in terms]


# ----------------------------------------------------------------------
#   Closed forms of the border correction B = A - S
#   (left border of the first axis; the right border by flipping)
# ----------------------------------------------------------------------

def _sample(x, j, b):
    """Extended sample j of x along its first axis"""
    k, s = extend_index(j, x.shape[0], b)
    return s * x[k]


def _scatter(out, j, b, val):
    """out[source of j] += sign * val (fold of the extended sample j)"""
    k, s = extend_index(j, out.shape[0], b)
    if s:
        out[k] += s * val


def _membrane_left(x, b, w=None):
    """Membrane (unit coefficient): ghost row -1 and fold of row 0"""
    out = torch.zeros_like(x)
    w = torch.ones_like(x) if w is None else w
    xm1 = _sample(x, -1, b)
    wm1 = _sample(w, -1, even_bound(b))
    out[0] -= 0.5 * wm1 * (x[0] - xm1)
    _scatter(out, -1, b, -0.5 * w[0] * (x[0] - xm1))
    return out


def _bending_diag_left(x, b):
    """Central second differences: ghost row -1 and fold of row 0"""
    out = torch.zeros_like(x)
    xm1, xm2 = _sample(x, -1, b), _sample(x, -2, b)
    x1 = x[1] if x.shape[0] > 1 else _sample(x, 1, b)
    out[0] -= xm2 - 2 * xm1 + x[0]
    _scatter(out, -1, b, xm1 - 2 * x[0] + x1)
    return out


def _neumann_laplacian(y):
    """L y along the first axis, with the missing neighbours dropped"""
    out = torch.zeros_like(y)
    if y.shape[0] > 1:
        d = y[1:] - y[:-1]
        out[:-1] -= d
        out[1:] += d
    return out


def _bending_cross_left(x, b_a, b_b, corners=True):
    """Cross term of the pair (a, b) = (first, second axis) of x: edge of
    axis a, and (if `corners`) its corners along axis b"""
    out = torch.zeros_like(x)
    d = x[0] - _sample(x, -1, b_a)
    Ld = _neumann_laplacian(d)
    out[0] -= Ld
    _scatter(out, -1, b_a, -Ld)
    for side in ((0, 1) if corners else ()):
        xb = x if side == 0 else x.flip(1)
        xa = _sample(xb, -1, b_a)
        g = xb[0, 0] - _sample(xb[0], -1, b_b) - xa[0] + _sample(xa, -1, b_b)
        corner = torch.zeros_like(xb)
        corner[0, 0] -= 1.5 * g
        tmp = torch.zeros_like(xb[0])
        _scatter(tmp, -1, b_b, -0.5 * g)
        corner[0] += tmp
        tmp2 = torch.zeros_like(xb[0])
        tmp2[0] = -0.5 * g
        _scatter(corner, -1, b_a, tmp2)
        tmp2 = torch.zeros_like(xb[0])
        _scatter(tmp2, -1, b_b, 0.5 * g)
        _scatter(corner, -1, b_a, tmp2)
        out += corner if side == 0 else corner.flip(1)
    return out


def _central(x, axis, b):
    """Central first difference along `axis` (unit voxel)"""
    x = x.movedim(axis, 0)
    out = torch.zeros_like(x)
    for k in range(x.shape[0]):
        out[k] = 0.5 * (_sample(x, k + 1, b) - _sample(x, k - 1, b))
    return out.movedim(0, axis)


def _lame_cross_left(u_j, bound_i, bound_j, a, b):
    """Term E_i' C_a' F'F C_b E_j u_j of the output component i: ghost
    row -1 and fold of row 0 of axis a"""
    g = _central(u_j, b, bound_j[b]).movedim(a, 0)
    out = torch.zeros_like(g)
    out[0] -= 0.5 * _sample(g, -1, bound_j[a])
    _scatter(out, -1, bound_i[a], -0.5 * g[0])
    return out.movedim(0, a)


def border_correction(x, bound, energy, vx, weights=None):
    """Closed form of `B x = (A - S) x` for a (ncomp, *shape) field.

    energy : {'absolute', 'membrane', 'bending', 'shear', 'div'}
        Unit coefficient. With ncomp > 1, 'membrane' and 'bending' are
        the energies of a grid (component k scaled by vx[k]**2).
    weights : (*shape) tensor, optional (membrane only)
    """
    ncomp, shape = x.shape[0], x.shape[1:]
    dim = len(shape)
    cb = component_bounds(bound, dim, ncomp)
    out = torch.zeros_like(x)
    if energy == 'absolute':
        return out

    def membrane(xc, axis, b, w=None):
        xa = xc.movedim(axis, 0)
        wa = None if w is None else w.movedim(axis, 0)
        left = _membrane_left(xa, b, wa)
        right = _membrane_left(xa.flip(0), b,
                               None if wa is None else wa.flip(0)).flip(0)
        return (left + right).movedim(0, axis)

    def lame_cross(i, j, a, b):
        left = _lame_cross_left(x[j], cb[i], cb[j], a, b)
        right = -_lame_cross_left(x[j].flip(a), cb[i], cb[j], a, b).flip(a)
        return left + right

    if energy == 'membrane':
        for k in range(ncomp):
            scale = vx[k] ** 2 if ncomp > 1 else 1.
            for a in range(dim):
                out[k] += scale / vx[a] ** 2 * membrane(x[k], a, cb[k][a], weights)
    elif energy == 'bending':
        for k in range(ncomp):
            scale = vx[k] ** 2 if ncomp > 1 else 1.
            for a in range(dim):
                xa = x[k].movedim(a, 0)
                left = _bending_diag_left(xa, cb[k][a])
                right = _bending_diag_left(xa.flip(0), cb[k][a]).flip(0)
                out[k] += scale / vx[a] ** 4 * (left + right).movedim(0, a)
                for b in range(dim):
                    if b == a:
                        continue
                    coef = scale / (vx[a] * vx[b]) ** 2
                    xab = x[k].movedim((a, b), (0, 1))
                    opt = dict(b_a=cb[k][a], b_b=cb[k][b], corners=a < b)
                    left = _bending_cross_left(xab, **opt)
                    right = _bending_cross_left(xab.flip(0), **opt).flip(0)
                    out[k] += coef * (left + right).movedim((0, 1), (a, b))
    elif energy in ('shear', 'div'):
        shear = energy == 'shear'
        for i in range(dim):
            if shear:
                out[i] += 2 * membrane(x[i], i, cb[i][i])
                for j in range(dim):
                    if j != i:
                        out[i] += (vx[i] / vx[j]) ** 2 * membrane(x[i], j, cb[i][j])
            else:
                out[i] += membrane(x[i], i, cb[i][i])
            for j in range(dim):
                if j != i:
                    # shear: d_j u_i d_i u_j ; div: d_i u_i d_j u_j
                    out[i] += lame_cross(i, j, *((j, i) if shear else (i, j)))
    else:
        raise ValueError(energy)
    return out

