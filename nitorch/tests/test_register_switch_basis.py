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
