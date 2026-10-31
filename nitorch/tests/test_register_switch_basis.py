"""Tests for `LogAffine._switch_basis` / `LogAffine2d._switch_basis`.

`_switch_basis(x, src, dst)` maps Lie parameters from one affine basis
to another. Its parameter layout is `spatial.affine_basis(name, dim)`
(translations, rotations, zooms, shears), except that the similitude
scale `s` stands for zooms `(s, ..., s)`: its generator is the sum of
the affine zoom generators, which is what `LogAffine._optim_grad`
assumes when it sums the zoom gradients.

With that generator set:
- if `dst` contains `src`, the log-matrix (hence the matrix) is kept;
- otherwise the result is the orthogonal (Frobenius) projection of the
  log-matrix onto the span of the `dst` generators.
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


def _generators(name, dim):
    basis = spatial.affine_basis(name, dim, dtype=torch.double)
    if name == 'similitude':
        scale = torch.zeros([dim + 1, dim + 1], dtype=torch.double)
        scale.diagonal()[:dim] = 1
        basis = torch.cat([basis[:-1], scale[None]])
    return basis


def _switch(x, src, dst, dim):
    if dim == 2:
        return LogAffine2d._switch_basis(x, src, dst)
    return LogAffine._switch_basis(x, src, dst, dim)


def _logm(x, basis):
    return (x[:, None, None] * basis).sum(0)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('src,dst', list(itertools.product(BASES, BASES)))
def test_switch_basis(src, dst, dim):
    torch.manual_seed(1234)
    bsrc, bdst = _generators(src, dim), _generators(dst, dim)
    x = torch.randn(len(bsrc), dtype=torch.double) * 0.3
    y = _switch(x, src, dst, dim)
    assert y.shape == (len(bdst),)
    log_src = _logm(x, bsrc)
    log_dst = _logm(y, bdst)
    if src in CONTAINS[dst]:
        assert torch.allclose(torch.matrix_exp(log_dst),
                              torch.matrix_exp(log_src), atol=1e-10, rtol=0)
    else:
        mat = bdst.reshape(len(bdst), -1).T
        proj = torch.linalg.lstsq(mat, log_src.reshape(-1, 1)).solution[:, 0]
        assert torch.allclose(y, proj, atol=1e-10, rtol=0)


@pytest.mark.parametrize('src,dst', list(itertools.product(BASES, BASES)))
def test_switch_basis_2d_copy(src, dst):
    """The 2D copy matches the generic one called with dim=2"""
    torch.manual_seed(1234)
    x = torch.randn(spatial.affine_basis_size(src, 2), dtype=torch.double)
    assert torch.equal(LogAffine2d._switch_basis(x, src, dst),
                       LogAffine._switch_basis(x, src, dst, 2))


OPTIM_PAIRS = [(o, b) for b in BASES for o in BASES
               if o != b and o in CONTAINS[b]]


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('optim,basis', OPTIM_PAIRS)
def test_switch_basis_matches_optim_grad(optim, basis, dim):
    """`AffineModel.Parameters` applies steps with `_switch_basis(step,
    optim, basis)` while gradients come from `_optim_grad`: the two must
    be transposes of each other, <optim_grad(g), d> = <g, switch(d)>."""
    torch.manual_seed(1234)
    if dim == 2:
        aff = LogAffine2d(basis, optim=optim, dtype=torch.double)
    else:
        aff = LogAffine(basis, optim=optim, dim=dim, dtype=torch.double)
    g = torch.randn(spatial.affine_basis_size(basis, dim), dtype=torch.double)
    d = torch.randn(spatial.affine_basis_size(optim, dim), dtype=torch.double)
    gopt = aff._optim_grad(g[:, None, None])[:, 0, 0]
    assert gopt.shape == d.shape
    lhs = gopt.dot(d)
    rhs = g.dot(_switch(d, optim, basis, dim))
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=0)


@pytest.mark.xfail(strict=True, reason='similitude basis scale generator is '
                   'I/sqrt(dim), but _switch_basis treats s as zooms '
                   '(s, s, s); see module docstring')
def test_switch_basis_similitude_keeps_matrix():
    torch.manual_seed(1234)
    sim = LogAffine(torch.randn(7, dtype=torch.double) * 0.3, 'similitude')
    aff = sim.switch_basis('affine')
    assert torch.allclose(aff.exp(), sim.exp(), atol=1e-10, rtol=0)


# --- 2D bases and affine model parameters ----------------------------


def _embed_2d(basis2d, plane):
    """Embed (F, 3, 3) 2D generators in the 3D plane orthogonal to `plane`"""
    axes = [d for d in range(3) if d != plane] + [3]
    out = basis2d.new_zeros([len(basis2d), 4, 4])
    for a, i in enumerate(axes):
        for b, j in enumerate(axes):
            out[:, i, j] = basis2d[:, a, b]
    return out


@pytest.mark.parametrize('plane', [0, 1, 2])
@pytest.mark.parametrize('name', BASES)
def test_logaffine2d_basis(name, plane):
    """`LogAffine2d` generators are the 2D generators embedded in the plane
    (except the similitude scale, which is diag(1, 1, 0)/sqrt(3))"""
    basis = LogAffine2d(name, plane=plane, dtype=torch.double).basis
    expected = _embed_2d(spatial.affine_basis(name, 2, dtype=torch.double),
                         plane)
    assert basis.shape == expected.shape
    if name == 'similitude':
        scale = torch.zeros([4, 4], dtype=torch.double)
        scale.diagonal()[:3] = 3 ** -0.5
        scale[plane, plane] = 0
        assert torch.allclose(basis[-1], scale)
        basis, expected = basis[:-1], expected[:-1]
    assert torch.allclose(basis, expected)


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
