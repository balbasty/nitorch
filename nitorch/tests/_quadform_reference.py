"""Dense reference for the regularisers of `nitorch.spatial._quadform`.

The operator of a quadratic energy is built from its definition, on one
explicit period of the extended field:

    A = (1/M) E' K' diag(w) K E

- `E` maps the field to one full period of its extension (periodic,
  symmetric and antisymmetric boundaries), to a zero-padded domain
  ('zero') or to a domain padded by replication ('replicate');
- `K` are circular finite differences on that domain;
- `M` is the number of copies of the field in a period; for 'zero' all
  rows are kept, and for 'replicate' only those centred on a voxel.

It is independent from the implementation in `_quadform` (no ghost rows,
no row weights, no fold), and only meant for small fields.
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
        m = 1 if bound == 'dft' else 2
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


def operator(shape, bound, terms, ncomp=1, weights=None):
    """Dense operator of `terms` (see `_quadform.quadform`)

    Returns a (ncomp * prod(shape), ncomp * prod(shape)) matrix acting on
    the components stacked along the first dimension.
    """
    dim = len(shape)
    if isinstance(bound, str):
        bound = [bound] * dim
    axes = [axis(n, b) for n, b in zip(shape, bound)]
    E = kron([a[0] for a in axes])
    rows = kron([a[1][:, None] for a in axes]).flatten()
    if weights is not None:
        Wx = kron([a[2] for a in axes])
        rows = rows * (Wx @ weights.flatten().to(dt))
    Ps = [a[0].shape[0] for a in axes]
    N = E.shape[1]
    A = torch.zeros(ncomp * N, ncomp * N, dtype=dt)
    for coef, parts in terms:
        D = torch.zeros(E.shape[0], ncomp * N, dtype=dt)
        for comp, kinds, scale in parts:
            K = kron([stencil(P, k) for P, k in zip(Ps, kinds)])
            D[:, comp * N:(comp + 1) * N] += scale * (K @ E)
        A += coef * D.T @ (rows[:, None] * D)
    return A
