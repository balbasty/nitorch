"""Closed-form linear algebra for fields of small general matrices
(`nitorch.core.linalg.small_*`), shared by `exp_backward` (SVF) and the
Greens kernels of geodesic shooting."""
import torch
import pytest
from nitorch.core import linalg
from nitorch.spatial import _shoot

dt = torch.double


def _mats(dim, batch=(50,), seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn([*batch, dim, dim], dtype=dt, generator=g)
    m = m + 3 * torch.eye(dim, dtype=dt)
    v = torch.randn([*batch, dim], dtype=dt, generator=g)
    return m, v


@pytest.mark.parametrize('dim', [1, 2, 3, 4])
@pytest.mark.parametrize('batch', [(50,), (4, 3, 2)])
@pytest.mark.parametrize('transposed', [False, True])   # non-contiguous
def test_small_linalg(dim, batch, transposed):
    m, v = _mats(dim, batch)
    if transposed:
        m = m.transpose(-1, -2)
    tol = dict(rtol=1e-10, atol=1e-12)
    assert torch.allclose(linalg.small_det(m), m.det(), **tol)
    assert torch.allclose(linalg.small_inv(m), m.inverse(), **tol)
    adj = linalg.small_adj(m)
    eye = torch.eye(dim, dtype=dt).expand(m.shape)
    assert torch.allclose(adj @ m, m.det()[..., None, None] * eye, **tol)
    assert torch.allclose(linalg.small_solve(m, v),
                          torch.linalg.solve(m, v), **tol)
    assert torch.allclose(linalg.small_matvec(m, v),
                          (m @ v[..., None])[..., 0], **tol)


def test_small_linalg_does_not_modify_inputs():
    m, v = _mats(3)
    m0, v0 = m.clone(), v.clone()
    for fn in (linalg.small_det, linalg.small_adj, linalg.small_inv):
        fn(m)
    linalg.small_solve(m, v)
    linalg.small_matvec(m, v)
    assert torch.equal(m, m0) and torch.equal(v, v0)


@pytest.mark.parametrize('dim', [2, 3])
def test_shoot_inv(dim):
    # regularised inverse used for the Greens kernels of geodesic shooting
    m, _ = _mats(dim, (6, 5))
    assert torch.allclose(_shoot._inv(m), m.inverse(), rtol=1e-8, atol=1e-10)
