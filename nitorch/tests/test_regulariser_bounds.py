"""Exact boundary conditions of the regularisers.

The operator of each regulariser is `A = E' K' diag(omega w) K E`
(see `nitorch.spatial._quadform`). It is compared to a dense reference
built from its definition on one explicit period of the extended field
(`_quadform_reference`), for every boundary condition, on the backend in
use (TorchScript or compiled): operators, convolutions and solvers.
"""
import torch
import pytest
from nitorch import spatial
from nitorch.spatial import _quadform as qf
from nitorch.tests import _quadform_reference as ref

dt = torch.double
BOUNDS = ['dft', 'dct2', 'dst2', 'dct1', 'dst1', 'zero', 'replicate']
MIRRORS = ('dct2', 'dst2', 'dct1', 'dst1')
SHAPES = [(6, 5), (5, 4, 4)]
VX = [1.0, 1.3, 0.8]


def field_terms(dim, vx, a=0, m=0, b=0):
    terms = []
    for coef, fn in ((a, qf.terms_absolute(dim)),
                     (m, qf.terms_membrane(dim, vx)),
                     (b, qf.terms_bending(dim, vx))):
        if coef:
            terms += [(c * coef, p) for c, p in fn]
    return terms


def grid_terms(dim, vx, a=0, m=0, b=0, shear=0, div=0):
    terms = []
    for k in range(dim):
        for coef, part in field_terms(dim, vx, a, m, b):
            terms.append((coef * vx[k] ** 2,
                          [(k, kinds, s) for _, kinds, s in part]))
    if shear:
        terms += [(c * shear, p) for c, p in qf.terms_lame_shear(dim, vx)]
    if div:
        terms += [(c * div, p) for c, p in qf.terms_lame_div(dim)]
    return terms


def grid_target(v, bound, terms, weights=None):
    """Exact operator applied to a grid. With mirror boundaries, the
    linear-elastic energy has no periodic counterpart (its components
    are all extended evenly, which is not the reflection of a
    displacement field): it is defined by the rows of the lattice."""
    shape, dim = v.shape[:-1], v.shape[-1]
    lattice = qf.Lattice(shape, bound)
    ys = qf.quadform(list(v.unbind(-1)), terms, lattice, weights)
    return torch.stack(ys, -1)


def grid_matrix(shape, bound, terms, weights=None):
    dim = len(shape)
    n = dim * torch.Size(shape).numel()
    eye = torch.eye(n, dtype=dt).reshape(n, dim, *shape).movedim(1, -1)
    return torch.stack([grid_target(e, bound, terms, weights)
                        .movedim(-1, 0).flatten() for e in eye], 1)


FIELD_CASES = dict(absolute=dict(a=1), membrane=dict(m=1), bending=dict(b=1),
                   all=dict(a=0.3, m=0.7, b=1.1))
GRID_CASES = dict(absolute=dict(a=1), membrane=dict(m=1), bending=dict(b=1),
                  shear=dict(shear=1), div=dict(div=1),
                  all=dict(a=0.3, m=0.7, b=1.1, shear=0.4, div=0.9))


def _field_kwargs(case):
    return dict(absolute=case.get('a', 0), membrane=case.get('m', 0),
                bending=case.get('b', 0))


def _grid_kwargs(case):
    return dict(absolute=case.get('a', 0), membrane=case.get('m', 0),
                bending=case.get('b', 0),
                lame=(case.get('shear', 0), case.get('div', 0)))


# ----------------------------------------------------------------------
#   The TorchScript engine is the definition
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(5,), (5, 4), (4, 3, 3)])
@pytest.mark.parametrize('bound', BOUNDS + ['mixed'])
@pytest.mark.parametrize('weighted', [False, True])
def test_engine_is_the_period_energy(shape, bound, weighted):
    """The row weights make the energy of the rows of the lattice that of
    one period of the extended field (fields; grids without coupling)"""
    dim = len(shape)
    if bound == 'mixed':
        bound = ['dct1', 'replicate', 'dst1'][:dim]
    weights = torch.rand(shape, dtype=dt) + 0.5 if weighted else None
    lattice = qf.Lattice(shape, bound)
    n = torch.Size(shape).numel()
    for name, case in FIELD_CASES.items():
        terms = field_terms(dim, VX, **case)
        A = ref.operator(shape, bound, terms, 1, weights)
        eye = torch.eye(n, dtype=dt).reshape(n, *shape)
        B = torch.stack([qf.quadform([e], terms, lattice, weights)[0].flatten()
                         for e in eye], 1)
        assert torch.allclose(A, B, atol=1e-10), name


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 3)])
@pytest.mark.parametrize('bound', ['dft', 'zero', 'replicate'])
def test_engine_lame_is_the_period_energy(shape, bound):
    """Without mirrors, the linear-elastic energy is also that of the
    extended field"""
    dim = len(shape)
    for name in ('shear', 'div'):
        terms = grid_terms(dim, VX, **GRID_CASES[name])
        A = ref.operator(shape, bound, terms, dim)
        B = grid_matrix(shape, bound, terms)
        assert torch.allclose(A, B, atol=1e-10), name


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 3)])
@pytest.mark.parametrize('bound', BOUNDS)
def test_operators_symmetric_psd(shape, bound):
    dim = len(shape)
    weights = torch.rand(shape, dtype=dt) + 0.5
    for name, case in GRID_CASES.items():
        A = grid_matrix(shape, bound, grid_terms(dim, VX, **case), weights)
        assert torch.allclose(A, A.T, atol=1e-10), name
        assert torch.linalg.eigvalsh(A).min() > -1e-10, name


# ----------------------------------------------------------------------
#   Public operators (TorchScript or compiled backend)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('weighted', [False, True])
def test_regulariser(shape, bound, weighted):
    dim = len(shape)
    vx = VX[:dim]
    torch.manual_seed(0)
    x = torch.randn([2, *shape], dtype=dt)
    weights = torch.rand([1, *shape], dtype=dt) + 0.5 if weighted else None
    for name, case in FIELD_CASES.items():
        A = ref.operator(shape, bound, field_terms(dim, vx, **case), 1,
                         None if weights is None else weights[0])
        y = spatial.regulariser(x, **_field_kwargs(case), voxel_size=vx,
                                bound=bound, dim=dim, weights=weights)
        target = (x.reshape(2, -1) @ A.T).reshape(x.shape)
        assert torch.allclose(y, target, atol=1e-9), name


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('kernel', [False, True])
def test_regulariser_grid(shape, bound, weighted, kernel):
    if weighted and kernel:
        pytest.skip('no weighted kernels')
    dim = len(shape)
    vx = VX[:dim]
    torch.manual_seed(0)
    v = torch.randn([*shape, dim], dtype=dt)
    weights = torch.rand(shape, dtype=dt) + 0.5 if weighted else None
    for name, case in GRID_CASES.items():
        terms = grid_terms(dim, vx, **case)
        target = grid_target(v, bound, terms, weights)
        y = spatial.regulariser_grid(v, **_grid_kwargs(case), voxel_size=vx,
                                     bound=bound, weights=weights,
                                     kernel=kernel)
        assert torch.allclose(y, target, atol=1e-9), name
        if bound not in MIRRORS or not (case.get('shear') or case.get('div')):
            A = ref.operator(shape, bound, terms, dim, weights)
            vflat = v.movedim(-1, 0).flatten()
            assert torch.allclose(target, (A @ vflat).reshape(dim, *shape)
                                  .movedim(0, -1), atol=1e-9), name


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('name', ['membrane', 'bending'])
def test_spconv_is_the_stencil_on_the_extension(bound, name):
    shape = (6, 5)
    torch.manual_seed(0)
    x = torch.randn([1, *shape], dtype=dt)
    ker = spatial.regulariser_kernel(2, **{name: 1}, dtype=dt).to_dense()
    lattice = qf.Lattice(shape, bound, margin=2)
    ext = lattice.extend(x[0])
    target = torch.zeros(shape, dtype=dt)
    c = [k // 2 for k in ker.shape]
    for i in range(ker.shape[0]):
        for j in range(ker.shape[1]):
            if ker[i, j]:
                target += ker[i, j] * ext[2 + i - c[0]:2 + i - c[0] + shape[0],
                                          2 + j - c[1]:2 + j - c[1] + shape[1]]
    y = spatial.spconv(x, ker.to_sparse(), bound=bound, dim=2)[0]
    assert torch.allclose(y, target, atol=1e-10)


# ----------------------------------------------------------------------
#   Solvers solve the exact system
# ----------------------------------------------------------------------

def _residual(A, h, x, g):
    return ((A @ x.flatten()) + h.flatten() * x.flatten() - g.flatten()).abs().max()


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('optim', ['cg', 'relax', 'kernel', 'fmg'])
@pytest.mark.parametrize('name', ['membrane', 'bending'])
def test_solve_field(bound, optim, name):
    shape, dim = (8, 7), 2
    vx = VX[:dim]
    torch.manual_seed(0)
    g = torch.randn([1, *shape], dtype=dt)
    h = torch.rand([1, *shape], dtype=dt) + 1
    prm = dict(absolute=1e-3, **{name: 0.5})
    A = ref.operator(shape, bound, field_terms(dim, vx, a=1e-3, **{name[0]: 0.5}))
    opt = dict(voxel_size=vx, bound=bound, dim=dim, **prm)
    if optim == 'fmg':
        x = spatial.solve_field_fmg(h, g, nb_cycles=50, nb_iter=4, **opt)
    elif optim == 'kernel':
        x = spatial.solve_field_kernel(h, g, optim='relax', max_iter=2000,
                                       tolerance=0, **opt)
    else:
        x = spatial.solve_field(h, g, optim=optim, max_iter=2000,
                                tolerance=0, **opt)
    assert _residual(A, h, x, g) < 1e-6


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('optim', ['cg', 'relax', 'kernel', 'fmg'])
@pytest.mark.parametrize('name', ['membrane', 'bending', 'lame'])
def test_solve_grid(bound, optim, name):
    shape, dim = (8, 7), 2
    vx = VX[:dim]
    torch.manual_seed(0)
    g = torch.randn([*shape, dim], dtype=dt)
    h = torch.rand([*shape, dim], dtype=dt) + 1
    # full symmetric Hessian (zero off-diagonal): the TS multigrid needs one
    hfull = torch.cat([h, h.new_zeros([*shape, dim * (dim - 1) // 2])], -1)
    case = dict(a=1e-3, m=0.5) if name == 'membrane' else \
        dict(a=1e-3, b=0.5) if name == 'bending' else \
        dict(a=1e-3, m=0.1, shear=0.5, div=0.3)
    A = grid_matrix(shape, bound, grid_terms(dim, vx, **case))
    opt = dict(voxel_size=vx, bound=bound, **_grid_kwargs(case))
    if optim == 'fmg':
        x = spatial.solve_grid_fmg(hfull, g, nb_cycles=50, nb_iter=4, **opt)
    elif optim == 'kernel':
        x = spatial.solve_grid_kernel(h, g, optim='relax', max_iter=2000,
                                      tolerance=0, **opt)
    else:
        x = spatial.solve_grid(h, g, optim=optim, max_iter=2000,
                               tolerance=0, **opt)
    xf, gf, hf = x.movedim(-1, 0), g.movedim(-1, 0), h.movedim(-1, 0)
    assert _residual(A, hf, xf, gf) < 1e-6
