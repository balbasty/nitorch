"""Exact boundary conditions of the regularisers.

Every energy is a sum of rows centred on the voxels of the field of view,
built from finite differences of the bound-extended field ("definition
F", see `_regulariser_reference`). The implementations are compared to
its dense operator, for every boundary condition, on the backend in use
(TorchScript or compiled): operators, convolutions (with their border
correction) and solvers.

Displacement grids also have 'sliding' boundary conditions (a geometric
reflection: the component normal to the border is antisymmetric, the
others symmetric).
"""
import torch
import pytest
from nitorch import spatial
from nitorch.core import utils
from nitorch.tests import _regulariser_reference as ref
from nitorch.spatial._regularisers import c_solvers

dt = torch.double
BOUNDS = ['dft', 'dct2', 'dst2', 'dct1', 'dst1', 'zero', 'replicate']
GRID_BOUNDS = BOUNDS + ['sliding']
SHAPES = [(6, 5), (5, 4, 4)]
VX = [1.0, 1.3, 0.8]


def field_terms(dim, vx, a=0, m=0, b=0):
    terms = []
    for coef, fn in ((a, ref.terms_absolute(dim)),
                     (m, ref.terms_membrane(dim, vx)),
                     (b, ref.terms_bending(dim, vx))):
        if coef:
            terms += [(c * coef, p) for c, p in fn]
    return terms


def grid_terms(dim, vx, a=0, m=0, b=0, shear=0, div=0):
    terms = ref.per_component(dim, field_terms(dim, vx, a, m, b), vx)
    if shear:
        terms += [(c * shear, p) for c, p in ref.terms_lame_shear(dim, vx)]
    if div:
        terms += [(c * div, p) for c, p in ref.terms_lame_div(dim)]
    return terms


def grid_matrix(shape, bound, terms, weights=None):
    """Exact operator of a grid energy"""
    return ref.operator(shape, bound, terms, len(shape), weights)


def grid_target(v, bound, terms, weights=None):
    """Exact operator applied to a (*shape, dim) grid"""
    shape, dim = v.shape[:-1], v.shape[-1]
    A = grid_matrix(shape, bound, terms, weights)
    return (A @ v.movedim(-1, 0).flatten()).reshape(dim, *shape).movedim(0, -1)


def apply_matrix(fn, shape, ncomp=None):
    """Dense matrix of a linear function of a (*shape[, ncomp]) tensor"""
    n = torch.Size(shape).numel() * (ncomp or 1)
    eye = torch.eye(n, dtype=dt)
    if ncomp:
        eye = eye.reshape(n, ncomp, *shape).movedim(1, -1)
        return torch.stack([fn(e).movedim(-1, 0).flatten() for e in eye], 1)
    eye = eye.reshape(n, *shape)
    return torch.stack([fn(e).flatten() for e in eye], 1)


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


def _mixed(bound, dim):
    if bound == 'mixed':
        return ['dct1', 'replicate', 'dst1'][:dim]
    if bound == 'mixed-sliding':
        return ['sliding', 'dft', 'zero'][:dim]
    return bound


# ----------------------------------------------------------------------
#   The reference (definition F)
# ----------------------------------------------------------------------

DEF_SHAPES = [(5,), (2,), (1,), (5, 4), (2, 3), (1, 4), (4, 3, 3), (2, 2, 3)]


def _def_bounds(dim, grid):
    bounds = list(BOUNDS)
    if dim > 1:
        bounds += [['dct1', 'replicate', 'dst1'][:dim],
                   ['dst2', 'dct1', 'dct2'][:dim]]
        if grid:
            bounds += ['sliding', ['sliding', 'dft', 'zero'][:dim]]
    return bounds


def _def_cases(dim, vx):
    """(name, terms, ncomp) of every energy, field and grid"""
    cases = [('absolute', ref.terms_absolute(dim), 1),
             ('membrane', ref.terms_membrane(dim, vx), 1),
             ('bending', ref.terms_bending(dim, vx), 1)]
    if dim > 1:
        cases += [
            ('membrane', ref.per_component(dim, ref.terms_membrane(dim, vx), vx), dim),
            ('bending', ref.per_component(dim, ref.terms_bending(dim, vx), vx), dim),
            ('shear', ref.terms_lame_shear(dim, vx), dim),
            ('div', ref.terms_lame_div(dim), dim)]
    return cases


def _is_sliding(bound):
    return 'sliding' in ([bound] if isinstance(bound, str) else bound)


@pytest.mark.parametrize('shape', DEF_SHAPES)
@pytest.mark.parametrize('weighted', [False, True])
def test_definition_symmetric_psd(shape, weighted):
    """Every energy of definition F is symmetric positive semi-definite,
    and the stencil is exact with periodic boundaries"""
    dim = len(shape)
    vx = VX[:dim]
    torch.manual_seed(0)
    weights = torch.rand(shape, dtype=dt) + 0.5 if weighted else None
    for bound in _def_bounds(dim, grid=True):
        for name, terms, ncomp in _def_cases(dim, vx):
            if _is_sliding(bound) and ncomp == 1:
                continue
            A, S = ref.operators(shape, bound, terms, ncomp, weights)
            assert torch.allclose(A, A.T, atol=1e-10), (bound, name)
            assert torch.linalg.eigvalsh(A).min() > -1e-10, (bound, name)
            if bound == 'dft':
                assert torch.allclose(A, S, atol=1e-10), (bound, name)


@pytest.mark.parametrize('shape', DEF_SHAPES)
def test_stencil_plus_correction_is_the_operator(shape):
    """A = S + B, with the closed forms of the border correction B
    (`ref.border_correction`)"""
    dim = len(shape)
    vx = VX[:dim]
    torch.manual_seed(0)
    for bound in _def_bounds(dim, grid=True):
        for weighted in (False, True):
            weights = torch.rand(shape, dtype=dt) + 0.5 if weighted else None
            for name, terms, ncomp in _def_cases(dim, vx):
                if _is_sliding(bound) and ncomp == 1:
                    continue
                if weighted and name not in ('absolute', 'membrane'):
                    continue
                A, S = ref.operators(shape, bound, terms, ncomp, weights)
                x = torch.randn([ncomp, *shape], dtype=dt)
                Bx = ((A - S) @ x.flatten()).reshape(x.shape)
                target = ref.border_correction(x, bound, name, vx, weights)
                assert torch.allclose(Bx, target, atol=1e-10), \
                    (bound, name, ncomp, weighted)


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 3)])
def test_sliding_is_a_reflection(shape):
    """The sliding extension along an axis is the field followed by its
    mirror image (as a displacement field)"""
    dim = len(shape)
    torch.manual_seed(0)
    v = torch.randn([*shape, dim], dtype=dt)
    for d in range(dim):
        bound = ['dft'] * dim
        bound[d] = 'sliding'
        ext = []
        for k, cbound in enumerate(ref.component_bounds(bound, dim, dim)):
            # samples [0, 2n) along d, [0, n) along the other axes
            Es = []
            for a, (n, b) in enumerate(zip(shape, cbound)):
                E = torch.zeros(n * (1 + (a == d)), n, dtype=dt)
                for j in range(E.shape[0]):
                    i, s = ref.extend_index(j, n, b)
                    E[j, i] = s
                Es.append(E)
            E = ref.kron(Es)
            shape_ext = list(shape)
            shape_ext[d] *= 2
            ext.append((E @ v[..., k].flatten()).reshape(shape_ext))
        ext = torch.stack(ext, -1)
        first, second = ext.split(shape[d], d)
        assert torch.equal(first, v)
        assert torch.equal(second, ref.reflect(v, d))


# ----------------------------------------------------------------------
#   TorchScript energies
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(5,), (5, 4), (4, 3, 3), (1, 4), (2, 3)])
@pytest.mark.parametrize('bound', BOUNDS + ['mixed'])
@pytest.mark.parametrize('weighted', [False, True])
def test_field_energies(shape, bound, weighted):
    dim = len(shape)
    bound = _mixed(bound, dim)
    weights = torch.rand(shape, dtype=dt) + 0.5 if weighted else None
    vx = VX[:dim]
    opt = dict(bound=bound, dim=dim, weights=weights)
    # (the absolute energy w * x has no boundary condition)
    for name, fn in (('absolute', lambda e: spatial.absolute(e, weights=weights)),
                     ('membrane', lambda e: spatial.membrane(e, voxel_size=vx, **opt)),
                     ('bending', lambda e: spatial.bending(e, voxel_size=vx, **opt))):
        A = ref.operator(shape, bound, field_terms(dim, vx, **{name[0]: 1}),
                         1, weights)
        assert torch.allclose(apply_matrix(fn, shape), A, atol=1e-10), name


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 3), (1, 4)])
@pytest.mark.parametrize('bound', GRID_BOUNDS + ['mixed', 'mixed-sliding'])
@pytest.mark.parametrize('weighted', [False, True])
def test_grid_energies(shape, bound, weighted):
    dim = len(shape)
    bound = _mixed(bound, dim)
    weights = torch.rand(shape, dtype=dt) + 0.5 if weighted else None
    vx = VX[:dim]
    opt = dict(bound=bound, weights=weights, voxel_size=vx)
    fns = dict(absolute=(spatial.absolute_grid, dict(a=1)),
               membrane=(spatial.membrane_grid, dict(m=1)),
               bending=(spatial.bending_grid, dict(b=1)),
               shear=(spatial.lame_shear, dict(shear=1)),
               div=(spatial.lame_div, dict(div=1)))
    for name, (fn, case) in fns.items():
        A = grid_matrix(shape, bound, grid_terms(dim, vx, **case), weights)
        kw = dict(opt)
        if name == 'absolute':
            # (the absolute energy w * x has no boundary condition)
            kw.pop('bound')
        B = apply_matrix(lambda e: fn(e, **kw), shape, dim)
        assert torch.allclose(A, B, atol=1e-10), name


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 3)])
@pytest.mark.parametrize('bound', GRID_BOUNDS)
def test_operators_symmetric_psd(shape, bound):
    dim = len(shape)
    weights = torch.rand(shape, dtype=dt) + 0.5
    for name, case in GRID_CASES.items():
        # (the compiled backend is not used with autograd)
        fn = lambda e: spatial.regulariser_grid(
            e.requires_grad_(), **_grid_kwargs(case), voxel_size=VX[:dim],
            bound=bound, weights=weights).detach()
        A = apply_matrix(fn, shape, dim)
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
@pytest.mark.parametrize('bound', GRID_BOUNDS)
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


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('backend', ['ts', 'kernel', 'C'])
@pytest.mark.parametrize('bound', ['sliding', 'mixed-sliding'])
def test_regulariser_grid_sliding_backends(shape, backend, bound):
    """Every grid energy with sliding boundaries is the dense operator,
    on each backend: TorchScript (explicit operator), sparse convolution
    (with its border correction) and compiled"""
    dim = len(shape)
    vx = VX[:dim]
    bound = _mixed(bound, dim)
    torch.manual_seed(0)
    v = torch.randn([*shape, dim], dtype=dt)
    vflat = v.movedim(-1, 0).flatten()
    for name, case in GRID_CASES.items():
        A = ref.operator(shape, bound, grid_terms(dim, vx, **case), dim)
        target = (A @ vflat).reshape(dim, *shape).movedim(0, -1)
        opt = dict(**_grid_kwargs(case), voxel_size=vx, bound=bound)
        if backend == 'C':
            if not c_solvers:
                pytest.skip('compiled backend not available')
            y = c_solvers.c_regulariser_grid(
                v, absolute=opt['absolute'], membrane=opt['membrane'],
                bending=opt['bending'], lame=opt['lame'],
                voxel_size=vx, bound=bound)
        elif backend == 'kernel':
            # (the compiled backend is not used with autograd)
            vg = v.clone().requires_grad_()
            y = spatial.regulariser_grid(vg, **opt, kernel=True).detach()
        else:
            # (the compiled backend is not used with autograd)
            vg = v.clone().requires_grad_()
            y = spatial.regulariser_grid(vg, **opt).detach()
        assert torch.allclose(y, target, atol=1e-9), name


# ----------------------------------------------------------------------
#   Compiled kernels (border pass): every boundary condition
# ----------------------------------------------------------------------

# (axes of 1 to 4 voxels: the interior box can be empty, and the border
# slabs overlap the folds of the extension)
C_SHAPES = [(6, 5), (5, 4, 4), (1, 4), (2, 3), (3, 4), (4, 2, 3), (3, 3, 1)]
C_BOUNDS = BOUNDS + ['mixed', 'mixed-dft']
needs_c = pytest.mark.skipif(not c_solvers, reason='compiled backend not available')


def _c_bound(bound, shape):
    """Boundary conditions of the compiled tests. The compiled `dst1`
    extends an axis of one voxel as `replicate` (`bounds_common.h`),
    which is not an antisymmetric extension: not tested."""
    dim = len(shape)
    if bound == 'mixed-dft':
        bound = ['dft', 'zero', 'dct1'][:dim]
    bound = _mixed(bound, dim)
    blist = [bound] * dim if isinstance(bound, str) else list(bound)
    if any(b == 'dst1' and n == 1 for b, n in zip(blist, shape)):
        return None
    return bound


@needs_c
@pytest.mark.parametrize('shape', C_SHAPES)
@pytest.mark.parametrize('bound', C_BOUNDS)
def test_c_regulariser(shape, bound):
    """The compiled field regulariser is the operator of definition F"""
    bound = _c_bound(bound, shape)
    if bound is None:
        pytest.skip('dst1 on an axis of one voxel')
    dim = len(shape)
    vx = VX[:dim]
    torch.manual_seed(0)
    x = torch.randn([1, 2, *shape], dtype=dt)
    w = torch.rand([1, 1, *shape], dtype=dt) + 0.5
    cases = dict(FIELD_CASES, rls=dict(a=0.3, m=0.7))
    for name, case in cases.items():
        weights = w if name == 'rls' else None
        A = ref.operator(shape, bound, field_terms(dim, vx, **case), 1,
                         None if weights is None else weights[0, 0])
        y = c_solvers.c_regulariser(x, weights, dim=dim, **_field_kwargs(case),
                                    voxel_size=vx, bound=bound)
        target = (x.reshape(2, -1) @ A.T).reshape(x.shape)
        assert torch.allclose(y, target, atol=1e-9), name


@needs_c
@pytest.mark.parametrize('shape', [s for s in C_SHAPES if len(s) > 1])
@pytest.mark.parametrize('bound', C_BOUNDS + ['sliding', 'mixed-sliding'])
def test_c_regulariser_grid(shape, bound):
    """The compiled grid regulariser is the operator of definition F"""
    bound = _c_bound(bound, shape)
    if bound is None:
        pytest.skip('dst1 on an axis of one voxel')
    dim = len(shape)
    vx = VX[:dim]
    torch.manual_seed(0)
    v = torch.randn([*shape, dim], dtype=dt)
    w = torch.rand(shape, dtype=dt) + 0.5
    cases = dict(GRID_CASES, rls=dict(a=0.3, m=0.7))
    for name, case in cases.items():
        weights = w if name == 'rls' else None
        target = grid_target(v, bound, grid_terms(dim, vx, **case), weights)
        y = c_solvers.c_regulariser_grid(v, weights, **_grid_kwargs(case),
                                         voxel_size=vx, bound=bound)
        assert torch.allclose(y, target, atol=1e-9), name


def test_sliding_needs_a_grid():
    x = torch.randn([2, 6, 5], dtype=dt)
    with pytest.raises(ValueError):
        spatial.regulariser(x, membrane=1, bound='sliding', dim=2)
    with pytest.raises(ValueError):
        spatial.membrane(x, bound='sliding', dim=2)
    with pytest.raises(ValueError):
        spatial.solve_field(torch.ones_like(x), x, membrane=1, dim=2,
                            bound='sliding')
    with pytest.raises(ValueError):
        ker = spatial.regulariser_kernel(2, membrane=1, dtype=dt)
        spatial.spconv(x, ker, bound='sliding', dim=2)
    with pytest.raises(ValueError):
        utils.pad(x, [1, 1], mode='sliding', side='both')


@pytest.mark.parametrize('bound', GRID_BOUNDS)
@pytest.mark.parametrize('name', ['membrane', 'bending', 'shear', 'div'])
def test_spconv_is_the_stencil(bound, name):
    """spconv is a pure convolution of the extended field (no border
    correction): it is the stencil `S` of the reference, and
    `correct_stencil` adds `A - S`"""
    shape, dim = (6, 5), 2
    vx = VX[:dim]
    torch.manual_seed(0)
    grid = name in ('shear', 'div') or bound == 'sliding'
    ncomp = dim if grid else 1
    x = torch.randn([ncomp, *shape], dtype=dt)
    if grid:
        case = GRID_CASES['membrane' if name == 'membrane' else
                          'bending' if name == 'bending' else name]
        terms = grid_terms(dim, vx, **case)
        kwargs = _grid_kwargs(case)
        del kwargs['absolute']
        ker = spatial.regulariser_grid_kernel(dim, **kwargs, voxel_size=vx,
                                              dtype=dt)
    else:
        terms = field_terms(dim, vx, **{name[0]: 1})
        kwargs = {name: 1}
        ker = spatial.regulariser_kernel(dim, **kwargs, voxel_size=vx,
                                         dtype=dt)
    A, S = ref.operators(shape, bound, terms, ncomp)
    y = spatial.spconv(x, ker, bound=bound, dim=dim)
    assert torch.allclose(y.flatten(), S @ x.flatten(), atol=1e-10)
    y = spatial.correct_stencil(x, y, bound, dim, vx, grid=grid, **kwargs)
    assert torch.allclose(y.flatten(), A @ x.flatten(), atol=1e-10)


# ----------------------------------------------------------------------
#   Solvers solve the exact system
# ----------------------------------------------------------------------

# The bounds that need a correction of the stencil (dct1, dst1, zero,
# replicate) and a control (dft) run by default; all bounds run with
# --runslow.
SOLVER_BOUNDS = [b if b in ('dft', 'dct1', 'dst1', 'zero', 'replicate')
                 else pytest.param(b, marks=pytest.mark.slow) for b in BOUNDS]
# sliding boundaries (grids only) have their own stencil signs: always run
GRID_SOLVER_BOUNDS = SOLVER_BOUNDS + ['sliding']
SOLVER_ITER = dict(cg=50, relax=200, kernel=200)


def _residual(A, h, x, g):
    return ((A @ x.flatten()) + h.flatten() * x.flatten() - g.flatten()).abs().max()


@pytest.mark.parametrize('bound', SOLVER_BOUNDS)
@pytest.mark.parametrize('optim', ['cg', 'relax', 'kernel', 'fmg'])
@pytest.mark.parametrize('name', ['membrane', 'bending'])
def test_solve_field(bound, optim, name):
    shape, dim = (6, 5), 2
    vx = VX[:dim]
    torch.manual_seed(0)
    g = torch.randn([1, *shape], dtype=dt)
    h = torch.rand([1, *shape], dtype=dt) + 1
    prm = dict(absolute=1e-3, **{name: 0.5})
    A = ref.operator(shape, bound, field_terms(dim, vx, a=1e-3, **{name[0]: 0.5}))
    opt = dict(voxel_size=vx, bound=bound, dim=dim, **prm)
    if optim == 'fmg':
        x = spatial.solve_field_fmg(h, g, nb_cycles=8, nb_iter=4, **opt)
    elif optim == 'kernel':
        x = spatial.solve_field_kernel(h, g, optim='relax', tolerance=0,
                                       max_iter=SOLVER_ITER[optim], **opt)
    else:
        x = spatial.solve_field(h, g, optim=optim, tolerance=0,
                                max_iter=SOLVER_ITER[optim], **opt)
    assert _residual(A, h, x, g) < 1e-6


@pytest.mark.parametrize('bound', GRID_SOLVER_BOUNDS)
@pytest.mark.parametrize('optim', ['cg', 'relax', 'kernel', 'fmg'])
@pytest.mark.parametrize('name', ['membrane', 'bending', 'lame'])
def test_solve_grid(bound, optim, name):
    shape, dim = (6, 5), 2
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
        x = spatial.solve_grid_fmg(hfull, g, nb_cycles=8, nb_iter=4, **opt)
    elif optim == 'kernel':
        x = spatial.solve_grid_kernel(h, g, optim='relax', tolerance=0,
                                      max_iter=SOLVER_ITER[optim], **opt)
    else:
        x = spatial.solve_grid(h, g, optim=optim, tolerance=0,
                               max_iter=SOLVER_ITER[optim], **opt)
    xf, gf, hf = x.movedim(-1, 0), g.movedim(-1, 0), h.movedim(-1, 0)
    assert _residual(A, hf, xf, gf) < 1e-6
