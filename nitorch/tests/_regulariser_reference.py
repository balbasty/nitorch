"""Dense reference for the regularisers of `nitorch.spatial`.

Energies are lists of terms `(c_t, parts)`, each part being
`(component, kinds, scale)` with one kind per axis among 'i' (identity),
'f'/'b' (forward/backward first difference) and 'c' (central second
difference). Their operator `sum_t c_t D_t' diag(w) D_t` is built
densely, in two independent ways.

`operator` builds it from its definition, on one explicit period of the
extended field:

    A = (1/M) E' K' diag(w) K E

- `E` maps the field to one full period of its extension (periodic,
  symmetric and antisymmetric boundaries), to a zero-padded domain
  ('zero') or to a domain padded by replication ('replicate');
- `K` are circular finite differences on that domain;
- `M` is the number of copies of the field in a period; for 'zero' all
  rows are kept, and for 'replicate' only those centred on a voxel.

`lattice_operator` builds it from the rows of the lattice: one row per
voxel plus one ghost row on each side of each axis, with the row weights

    dft, dct2, dst2, replicate   [0,   1, 1, ..., 1, 1,   0]
    dct1                         [0, 1/2, 1, ..., 1, 1/2, 0]
    dst1                         [1/2, 1, 1, ..., 1, 1, 1/2]
    zero                         [1,   1, 1, ..., 1, 1,   1]

Both agree for the energies of a single component. The linear-elastic
energy of a displacement grid only has a periodic counterpart with
'dft', 'zero' and 'replicate': with the mirrors, every component is
extended with the same parity, which is not the mirror image of a
displacement field, and the energy is defined by the rows of the lattice.

With 'sliding' boundary conditions (displacement grids), each component
has its own extension: along a sliding axis `d`, the component `d` is
extended antisymmetrically ('dst2') and the others symmetrically
('dct2'), so that `E` is block-diagonal across components. All
components share the period, the rows and their weights.

Both are independent from the implementation (no padding, no finite
difference routine), and only meant for small fields.
"""
import itertools
import torch

dt = torch.double
PAD = 4


def axis(n, bound):
    """Extension (P, n), row weights (P,), and even extension (P, n)"""
    if bound == 'dft':
        idx = [(j, 1) for j in range(n)]
        widx = list(range(n))
    elif bound in ('dct2', 'dst2'):
        s = -1 if bound == 'dst2' else 1
        idx = [(j, 1) for j in range(n)] + [(n - 1 - j, s) for j in range(n)]
        widx = list(range(n)) + [n - 1 - j for j in range(n)]
    elif bound == 'dct1':
        idx = [(j, 1) for j in range(n)] + [(n - 2 - j, 1) for j in range(n - 2)]
        widx = [j for j, _ in idx]
    elif bound == 'dst1':
        # 0, v0..v_{n-1}, 0, -v_{n-1}..-v0
        idx = [(0, 0)] + [(j, 1) for j in range(n)] + [(0, 0)] \
            + [(n - 1 - j, -1) for j in range(n)]
        # even weights about the zeros (the weight of a zero is that of
        # its neighbour)
        widx = [0] + list(range(n)) + [n - 1] + [n - 1 - j for j in range(n)]
    elif bound == 'zero':
        idx = [(0, 0)] * PAD + [(j, 1) for j in range(n)] + [(0, 0)] * PAD
        widx = [0] * PAD + list(range(n)) + [n - 1] * PAD
    elif bound == 'replicate':
        idx = [(0, 1)] * PAD + [(j, 1) for j in range(n)] + [(n - 1, 1)] * PAD
        widx = [0] * PAD + list(range(n)) + [n - 1] * PAD
    P = len(idx)
    E = torch.zeros(P, n, dtype=dt)
    W = torch.zeros(P, n, dtype=dt)
    for p, ((j, s), wj) in enumerate(zip(idx, widx)):
        if s:
            E[p, j] = s
        W[p, wj] = 1
    if bound in ('zero',):
        rows, m = torch.ones(P, dtype=dt), 1
    elif bound == 'replicate':
        rows, m = torch.zeros(P, dtype=dt), 1
        rows[PAD:PAD + n] = 1
    else:
        rows = torch.ones(P, dtype=dt)
        m = 1 if bound == 'dft' or (bound == 'dct1' and n == 1) else 2
    return E, rows / m, W


def stencil(P, kind):
    I = torch.eye(P, dtype=dt)
    S = torch.roll(I, 1, 1)          # (S x)[r] = x[r + 1]
    St = torch.roll(I, -1, 1)        # (St x)[r] = x[r - 1]
    return dict(i=I, f=S - I, b=I - St, c=St - 2 * I + S)[kind]


def kron(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def expand_sides(terms):
    """Replace the half-lattice differences ('h') of the terms by their
    definition: the average of the forward and backward differences"""
    out = []
    for coef, parts in terms:
        axes = sorted({d for _, kinds, _ in parts
                       for d, k in enumerate(kinds) if k == 'h'})
        if not axes:
            out.append((coef, parts))
            continue
        for sides in itertools.product('fb', repeat=len(axes)):
            new_parts = []
            for comp, kinds, scale in parts:
                kinds = list(kinds)
                for d, side in zip(axes, sides):
                    kinds[d] = side
                new_parts.append((comp, ''.join(kinds), scale))
            out.append((coef / 2 ** len(axes), new_parts))
    return out


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


def _assemble(axes, shape, terms, ncomp, weights, stencil_fn):
    """Operator from the extension, row weights and weight extension of
    each axis (one list of axes per component)"""
    Es = [kron([a[0] for a in caxes]) for caxes in axes]
    rows = kron([a[1][:, None] for a in axes[0]]).flatten()
    Wx = kron([a[2] for a in axes[0]])
    for caxes in axes[1:]:
        # the components share their rows and row weights
        assert torch.equal(kron([a[1][:, None] for a in caxes]).flatten(), rows)
        assert torch.equal(kron([a[2] for a in caxes]), Wx)
    if weights is not None:
        rows = rows * (Wx @ weights.flatten().to(dt))
    terms = expand_sides(terms)
    N = Es[0].shape[1]
    A = torch.zeros(ncomp * N, ncomp * N, dtype=dt)
    for coef, parts in terms:
        D = torch.zeros(rows.shape[0], ncomp * N, dtype=dt)
        for comp, kinds, scale in parts:
            K = stencil_fn(kinds)
            D[:, comp * N:(comp + 1) * N] += scale * (K @ Es[comp])
        A += coef * D.T @ (rows[:, None] * D)
    return A


def operator(shape, bound, terms, ncomp=1, weights=None):
    """Dense operator of `terms`, from one period of the extended field

    Returns a (ncomp * prod(shape), ncomp * prod(shape)) matrix acting on
    the components stacked along the first dimension.
    """
    dim = len(shape)
    axes = [[axis(n, b) for n, b in zip(shape, cbound)]
            for cbound in component_bounds(bound, dim, ncomp)]
    Ps = [a[0].shape[0] for a in axes[0]]
    return _assemble(axes, shape, terms, ncomp, weights,
                     lambda kinds: kron([stencil(P, k) for P, k in zip(Ps, kinds)]))


# ----------------------------------------------------------------------
#   Rows of the lattice
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
    return {'dst2': 'dct2', 'dst1': 'replicate', 'zero': 'replicate'}.get(
        bound, bound)


def lattice_axis(n, bound):
    """Extension (n + 4, n) on the samples [-2, n + 1], row weights on
    the rows [-1, n], and even extension of the weights (n + 2, n)"""
    E = torch.zeros(n + 4, n, dtype=dt)
    for p, j in enumerate(range(-2, n + 2)):
        k, s = extend_index(j, n, bound)
        if s:
            E[p, k] = s
    omega = [0.] + [1.] * n + [0.]
    if bound == 'dct1' and n > 1:
        omega[1] = omega[n] = 0.5
    elif bound == 'dst1':
        omega[0] = omega[-1] = 0.5
    elif bound == 'zero':
        omega[0] = omega[-1] = 1.
    W = torch.zeros(n + 2, n, dtype=dt)
    for p, j in enumerate(range(-1, n + 1)):
        W[p, extend_index(j, n, even_bound(bound))[0]] = 1
    return E, torch.as_tensor(omega, dtype=dt), W


def lattice_stencil(n, kind):
    """(n + 2, n + 4): kind at the rows [-1, n], from the samples [-2, n+1]"""
    K = torch.zeros(n + 2, n + 4, dtype=dt)
    taps = dict(i=((0, 1),), f=((0, -1), (1, 1)), b=((-1, -1), (0, 1)),
                c=((-1, 1), (0, -2), (1, 1)))[kind]
    for r in range(n + 2):
        for off, c in taps:
            K[r, r + 1 + off] = c
    return K


def lattice_operator(shape, bound, terms, ncomp=1, weights=None):
    """Dense operator of `terms`, from the rows of the lattice"""
    dim = len(shape)
    axes = [[lattice_axis(n, b) for n, b in zip(shape, cbound)]
            for cbound in component_bounds(bound, dim, ncomp)]
    return _assemble(axes, shape, terms, ncomp, weights,
                     lambda kinds: kron([lattice_stencil(n, k)
                                         for n, k in zip(shape, kinds)]))


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
