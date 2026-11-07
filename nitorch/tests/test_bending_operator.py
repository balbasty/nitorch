"""Bending energy operator at the borders, and the field regulariser.

The same-axis terms of the bending operator used to compose two first
differences, applying the boundary condition of the field to its first
derivative (whose symmetry differs), which made the operator wrong at
the borders.
"""
import math
import torch
import pytest
from nitorch import spatial

dt = torch.double
BOUNDS = ['dct2', 'dct1', 'dst2', 'dst1', 'dft', 'zero', 'replicate']


def _matrix(fn, shape):
    n = math.prod(shape)
    eye = torch.eye(n, dtype=dt).reshape(n, 1, *shape)
    return torch.stack([fn(e).flatten() for e in eye])


def _dct2(n):
    k = torch.arange(n, dtype=dt)[:, None]
    i = torch.arange(n, dtype=dt)[None]
    c = torch.cos(math.pi * k * (i + 0.5) / n)
    c[0] /= math.sqrt(2)
    return c * math.sqrt(2 / n)


@pytest.mark.parametrize('bound', ['dct2', 'dft'])
@pytest.mark.parametrize('shape', [(7, 6), (4, 5, 3)])
def test_bending_is_membrane_squared(bound, shape):
    # unit voxels: the bending energy is the squared Laplacian
    dim = len(shape)
    lb = _matrix(lambda e: spatial.regulariser(e, dim=dim, bending=1., bound=bound), shape)
    lm = _matrix(lambda e: spatial.regulariser(e, dim=dim, membrane=1., bound=bound), shape)
    assert torch.allclose(lb, lm @ lm, atol=1e-10)


def test_bending_dct2_is_diagonal_in_dct_basis():
    shape = (7, 6)
    lb = _matrix(lambda e: spatial.regulariser(
        e, dim=2, bending=1., voxel_size=[0.8, 1.3], bound='dct2'), shape)
    c = torch.kron(_dct2(shape[0]), _dct2(shape[1]))
    d = c @ lb @ c.T
    assert (d - torch.diag(torch.diag(d))).abs().max() < 1e-10


@pytest.mark.parametrize('bound', BOUNDS)
def test_bending_is_symmetric_psd(bound):
    shape = (6, 5)
    lb = _matrix(lambda e: spatial.regulariser(
        e, dim=2, bending=1., voxel_size=[0.8, 1.3], bound=bound), shape)
    assert torch.allclose(lb, lb.T, atol=1e-10)
    assert torch.linalg.eigvalsh(lb).min() > -1e-10


@pytest.mark.parametrize('bound', ['dct2', 'dft'])
def test_bending_grid_matches_field(bound):
    g = torch.Generator().manual_seed(0)
    v = torch.randn([6, 5, 4, 3], dtype=dt, generator=g)
    vg = spatial.regulariser_grid(v, bending=1., bound=bound)
    vf = spatial.regulariser(v.movedim(-1, 0), dim=3, bending=1., bound=bound)
    assert torch.allclose(vg, vf.movedim(0, -1))


def test_regulariser_batch_and_autograd():
    # (..., K, *spatial): per-channel penalties along K, batch dimensions
    # in front; autograd works
    g = torch.Generator().manual_seed(0)
    kw = dict(absolute=[0.1, 0.2], membrane=[0.3, 0.05], bending=[0.2, 0.4],
              voxel_size=[0.8, 1.3], bound='dct2', dim=2)
    x = torch.randn([3, 2, 6, 5], dtype=dt, generator=g)
    w = torch.rand([6, 5], dtype=dt, generator=g) + 0.5
    for weights in (None, w):
        yb = spatial.regulariser(x, weights=weights, **kw)
        yi = torch.stack([spatial.regulariser(xi, weights=weights, **kw)
                          for xi in x])
        assert torch.allclose(yb, yi)
    xg = x.clone().requires_grad_()
    y = spatial.regulariser(xg, **kw)
    y.square().sum().backward()
    assert torch.allclose(y.detach(), spatial.regulariser(x, **kw))
    assert torch.isfinite(xg.grad).all()
