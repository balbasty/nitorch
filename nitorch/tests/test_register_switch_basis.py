"""Tests for the affine Lie bases of the registration objects and for
`LogAffine._switch_basis` / `LogAffine2d._switch_basis` / `_optim_grad`.

Every basis is orthonormal for the Frobenius inner product of the
homogeneous matrices (3D, 2D, and 2D embedded in each plane of the 3D
space). `_switch_basis(x, src, dst)` is `M x` with the Gram matrix
`M = B_dst B_src^T` of the generators, so:
- if `dst` contains `src`, the log-matrix (hence the matrix) is kept;
- otherwise the result is the orthogonal (Frobenius) projection of the
  log-matrix onto the span of the `dst` generators.
When only a sub-basis is optimised, `_optim_grad` applies `M^T`, so
that `<optim_grad(g), d> == <g, switch(d)>`.
"""
import itertools
import pytest
import torch
from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.objects import LogAffine, LogAffine2d

BASES = ['translation', 'rotation', 'rigid', 'similitude', 'affine']
CONTAINS = {  # dst -> set of src it can represent exactly
    'translation': {'translation'},
    'rotation': {'rotation'},
    'rigid': {'translation', 'rotation', 'rigid'},
    'similitude': {'translation', 'rotation', 'rigid', 'similitude'},
    'affine': set(BASES),
}
# (dim, plane): 3D, 2D, and 2D embedded in the plane orthogonal to `plane`
SPACES = [(3, None), (2, None), (3, 0), (3, 1), (3, 2)]
SPACE_IDS = ['3d', '2d', 'plane0', 'plane1', 'plane2']


def _embed_2d(basis2d, plane):
    """Embed (F, 3, 3) 2D generators in the 3D plane orthogonal to `plane`"""
    axes = [d for d in range(3) if d != plane] + [3]
    out = basis2d.new_zeros([len(basis2d), 4, 4])
    for a, i in enumerate(axes):
        for b, j in enumerate(axes):
            out[:, i, j] = basis2d[:, a, b]
    return out


def _generators(name, dim, plane):
    """Reference generators (built independently of `objects`)"""
    if plane is None:
        return spatial.affine_basis(name, dim, dtype=torch.double)
    return _embed_2d(spatial.affine_basis(name, 2, dtype=torch.double), plane)


def _logaffine(name, dim, plane, dat=None, optim=None):
    if plane is None:
        return LogAffine(dat, name, dim=dim, optim=optim, dtype=torch.double)
    return LogAffine2d(dat, name, plane=plane, optim=optim, dtype=torch.double)


def _switch(x, src, dst, dim, plane):
    if plane is None:
        return LogAffine._switch_basis(x, src, dst, dim)
    return LogAffine2d._switch_basis(x, src, dst, plane)


def _logm(x, basis):
    return (x[:, None, None] * basis).sum(0)


def _nb_prm(name, dim, plane):
    return spatial.affine_basis_size(name, dim if plane is None else 2)


@pytest.mark.parametrize('dim,plane', SPACES, ids=SPACE_IDS)
@pytest.mark.parametrize('name', BASES)
def test_basis_orthonormal(name, dim, plane):
    """B B^T = I (Frobenius inner product of the homogeneous matrices),
    for the bases the registration uses and for the reference ones"""
    basis = _logaffine(name, dim, plane).basis
    assert basis.shape == (_nb_prm(name, dim, plane), dim + 1, dim + 1)
    for b in (basis, _generators(name, dim, plane)):
        b = b.reshape([len(b), -1])
        eye = torch.eye(len(b), dtype=torch.double)
        assert torch.allclose(b @ b.T, eye, atol=1e-12, rtol=0)


@pytest.mark.parametrize('plane', [0, 1, 2])
@pytest.mark.parametrize('name', BASES)
def test_logaffine2d_basis(name, plane):
    """`LogAffine2d` generators are the 2D generators embedded in the plane
    (including the similitude scale, which is diag(1, 1, 0)/sqrt(2))"""
    basis = LogAffine2d(name, plane=plane, dtype=torch.double).basis
    expected = _generators(name, 3, plane)
    assert basis.shape == expected.shape
    assert torch.allclose(basis, expected, atol=1e-15, rtol=0)
    if name == 'similitude':
        scale = torch.zeros([4, 4], dtype=torch.double)
        scale.diagonal()[:3] = 2 ** -0.5
        scale[plane, plane] = 0
        assert torch.allclose(basis[-1], scale, atol=1e-15, rtol=0)


@pytest.mark.parametrize('dim,plane', SPACES, ids=SPACE_IDS)
@pytest.mark.parametrize('src,dst', list(itertools.product(BASES, BASES)))
def test_switch_basis(src, dst, dim, plane):
    torch.manual_seed(1234)
    bsrc, bdst = _generators(src, dim, plane), _generators(dst, dim, plane)
    x = torch.randn(len(bsrc), dtype=torch.double) * 0.3
    y = _switch(x, src, dst, dim, plane)
    assert y.shape == (len(bdst),)
    log_src = _logm(x, bsrc)
    log_dst = _logm(y, bdst)
    if src in CONTAINS[dst]:
        assert torch.allclose(log_dst, log_src, atol=1e-12, rtol=0)
        assert torch.allclose(torch.matrix_exp(log_dst),
                              torch.matrix_exp(log_src), atol=1e-12, rtol=0)
    else:
        mat = bdst.reshape(len(bdst), -1).T
        proj = torch.linalg.lstsq(mat, log_src.reshape(-1, 1)).solution[:, 0]
        assert torch.allclose(y, proj, atol=1e-12, rtol=0)


@pytest.mark.parametrize('dim,plane', SPACES, ids=SPACE_IDS)
@pytest.mark.parametrize('name', BASES)
def test_switch_basis_same(name, dim, plane):
    """Switching to the same basis copies the parameters exactly"""
    x = torch.randn(_nb_prm(name, dim, plane), dtype=torch.double)
    y = _switch(x, name, name, dim, plane)
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()


@pytest.mark.parametrize('plane', [0, 1, 2])
@pytest.mark.parametrize('src,dst', list(itertools.product(BASES, BASES)))
def test_switch_basis_2d_copy(src, dst, plane):
    """The embedded 2D switch matches the generic one called with dim=2"""
    torch.manual_seed(1234)
    x = torch.randn(spatial.affine_basis_size(src, 2), dtype=torch.double)
    assert torch.allclose(LogAffine2d._switch_basis(x, src, dst, plane),
                          LogAffine._switch_basis(x, src, dst, 2),
                          atol=1e-15, rtol=0)


OPTIM_PAIRS = [(o, b) for b in BASES for o in BASES
               if o != b and o in CONTAINS[b]]


@pytest.mark.parametrize('dim,plane', SPACES, ids=SPACE_IDS)
@pytest.mark.parametrize('optim,basis', OPTIM_PAIRS)
def test_switch_basis_matches_optim_grad(optim, basis, dim, plane):
    """`AffineModel.Parameters` applies steps with `_switch_basis(step,
    optim, basis)` while gradients come from `_optim_grad`: the two must
    be adjoint, <optim_grad(g), d> = <g, switch(d)> (for matrix-valued
    gradients g)."""
    torch.manual_seed(1234)
    aff = _logaffine(basis, dim, plane, optim=optim)
    g = torch.randn([_nb_prm(basis, dim, plane), dim + 1, dim + 1],
                    dtype=torch.double)
    d = torch.randn(_nb_prm(optim, dim, plane), dtype=torch.double)
    gopt = aff._optim_grad(g)
    assert gopt.shape == (len(d), dim + 1, dim + 1)
    lhs = torch.einsum('kij,k->ij', gopt, d)
    rhs = torch.einsum('kij,k->ij', g, _switch(d, optim, basis, dim, plane))
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=0)


@pytest.mark.parametrize('dim,plane', [(3, None), (3, 1)], ids=['3d', 'plane1'])
@pytest.mark.parametrize('optim,basis', OPTIM_PAIRS)
def test_optim_grad_chain_rule(optim, basis, dim, plane):
    """`exp(grad=True)` with a sub-basis returns the derivative of the
    matrix wrt the optimised parameters p, where q = switch(p)"""
    torch.manual_seed(1234)
    p = torch.randn(_nb_prm(optim, dim, plane), dtype=torch.double) * 0.3
    q = _switch(p, optim, basis, dim, plane)
    aff = _logaffine(basis, dim, plane, dat=q.clone(), optim=optim)
    _, grad = aff.exp(grad=True)
    bas = _generators(basis, dim, plane)
    jac = torch.autograd.functional.jacobian(
        lambda p: torch.matrix_exp(_logm(_switch(p, optim, basis, dim, plane), bas)), p)
    assert torch.allclose(grad, jac.movedim(-1, 0), atol=1e-10, rtol=0)


@pytest.mark.parametrize('optim,basis', [('rigid', 'translation'),
                                         ('similitude', 'rigid'),
                                         ('affine', 'similitude')])
def test_optim_grad_not_contained(optim, basis):
    aff = LogAffine(basis, optim=optim, dtype=torch.double)
    g = torch.zeros([spatial.affine_basis_size(basis), 4, 4], dtype=torch.double)
    with pytest.raises(ValueError):
        aff._optim_grad(g)


@pytest.mark.parametrize('dim,plane', SPACES, ids=SPACE_IDS)
def test_switch_basis_similitude_keeps_matrix(dim, plane):
    """similitude -> affine keeps the matrix, and back is a round-trip"""
    torch.manual_seed(1234)
    x = torch.randn(_nb_prm('similitude', dim, plane), dtype=torch.double) * 0.3
    sim = _logaffine('similitude', dim, plane, dat=x)
    aff = sim.switch_basis('affine')
    assert type(aff) is type(sim) and aff.dat.shape == (_nb_prm('affine', dim, plane),)
    assert torch.allclose(aff.exp(), sim.exp(), atol=1e-12, rtol=0)
    back = aff.switch_basis('similitude')
    assert torch.allclose(back.dat, x, atol=1e-12, rtol=0)
    assert torch.allclose(back.exp(), sim.exp(), atol=1e-12, rtol=0)


def test_affine_model_switch_basis():
    """`AffineModel.switch_basis` keeps the transform when the new basis
    contains the old one"""
    torch.manual_seed(1234)
    x = torch.randn(7, dtype=torch.double) * 0.3
    model = objects.AffineModel('similitude', dat=x.clone())
    new = model.switch_basis('affine')
    assert new.basis_name == 'affine' and len(new.dat.dat) == 12
    assert torch.allclose(new.exp(), model.exp(), atol=1e-12, rtol=0)


# --- affine model parameters -----------------------------------------


def test_affine2d_model_dat():
    """`Affine2dModel(..., dat=x)` keeps `x` (it used to land in `optim`)"""
    x = torch.as_tensor([1., -2., 0.3], dtype=torch.double)
    model = objects.Affine2dModel('rigid', plane=2,
                                  ref_affine=torch.eye(4, dtype=torch.double),
                                  dat=x)
    assert model.optim is None
    assert model.dat is not None and torch.equal(model.dat.dat, x)
    assert model.plane == 2


@pytest.mark.parametrize('value', [2., [2., 3., 4.]])
def test_affine_parameters_mul(value):
    """With a parameter subset, `mul`/`mul_` scale the optimised parameters
    and leave the others unchanged"""
    x = torch.as_tensor([1., -2., 0.5, 0.1, -0.2, 0.3], dtype=torch.double)
    model = objects.AffineModel('rigid', optim='translation')
    model.set_dat(x.clone(), dim=3)
    expected = x.clone()
    expected[:3] *= torch.as_tensor(value, dtype=torch.double)
    params = model.parameters()
    assert torch.allclose(params.mul(value), expected)
    assert torch.equal(model.dat.dat, x)
    params.mul_(value)
    assert torch.allclose(model.dat.dat, expected)


def test_affine_parameters_mul_similitude():
    """A factor on the similitude scale applies to every zoom"""
    torch.manual_seed(1234)
    x = torch.randn(12, dtype=torch.double)
    model = objects.AffineModel('affine', optim='similitude')
    model.set_dat(x.clone(), dim=3)
    factor = torch.arange(1., 8., dtype=torch.double)
    expected = x.clone()
    expected[:6] *= factor[:6]
    expected[6:9] *= factor[6]
    assert torch.allclose(model.parameters().mul(factor), expected)


@pytest.mark.parametrize('plane', [0, 1, 2])
def test_affine2d_parameters_add(plane):
    """`Affine2dModel` parameters apply sub-basis steps with the 2D switch"""
    torch.manual_seed(1234)
    model = objects.Affine2dModel('affine', plane=plane,
                                  ref_affine=torch.eye(4, dtype=torch.double))
    model.set_dat(dtype=torch.double)
    model.optim = 'similitude'
    params = model.parameters()
    assert len(params) == 4
    step = torch.randn(4, dtype=torch.double)
    params.add_(step)
    expected = LogAffine2d._switch_basis(step, 'similitude', 'affine', plane)
    assert torch.allclose(model.dat.dat, expected, atol=1e-15, rtol=0)
    sim = LogAffine2d(step, 'similitude', plane=plane)
    assert torch.allclose(model.dat.exp(), sim.exp(), atol=1e-12, rtol=0)
