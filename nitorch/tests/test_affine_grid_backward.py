"""`affine_grid_backward`: chunked matmul implementation (#158)."""
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import utils as regutils

dt = torch.double


def reference(grad, hess=None, grid=None):
    """Dense reference: g[i, j] = sum_n grad[n, i] X[n, j] and
    H[i, j, k, l] = sum_n hess[n, ik] X[n, j] X[n, l], with X = [grid, 1]"""
    dim = grad.shape[-1]
    shape = grad.shape[-dim-1:-1]
    batch = grad.shape[:-dim-1]
    if grid is None:
        grid = spatial.identity_grid(shape, dtype=grad.dtype)
    x = grid.reshape([-1, dim])
    x = torch.cat([x, torch.ones_like(x[:, :1])], -1)
    grad = grad.reshape([*batch, -1, dim])
    g = torch.einsum('...ni,nj->...ij', grad, x)
    if hess is None:
        return g
    hess = hess.reshape([*batch, -1, dim*(dim+1)//2])
    full = hess.new_zeros([*hess.shape[:-1], dim, dim])
    c = 0
    for i in range(dim):
        full[..., i, i] = hess[..., c]
        c += 1
    for i in range(dim):
        for k in range(i+1, dim):
            full[..., i, k] = full[..., k, i] = hess[..., c]
            c += 1
    h = torch.einsum('...nik,nj,nl->...ijkl', full, x, x)
    return g, h


CASES = [(1, [11], []), (2, [7, 5], []), (3, [5, 4, 3], []), (3, [4, 3, 3], [2])]


@pytest.mark.parametrize('dim,shape,batch', CASES)
@pytest.mark.parametrize('chunk', [7, 2 ** 18])   # several chunks / one
@pytest.mark.parametrize('with_grid', [False, True])
def test_affine_grid_backward(dim, shape, batch, chunk, with_grid,
                              monkeypatch):
    monkeypatch.setattr(regutils, 'AFFINE_GRID_BACKWARD_CHUNK', chunk)
    g = torch.Generator().manual_seed(0)
    grad = torch.randn([*batch, *shape, dim], dtype=dt, generator=g)
    hess = torch.randn([*batch, *shape, dim*(dim+1)//2], dtype=dt,
                       generator=g)
    grid = (3 * torch.randn([*shape, dim], dtype=dt, generator=g)
            if with_grid else None)
    g0, h0 = reference(grad, hess, grid)
    g1, h1 = regutils.affine_grid_backward(grad, hess, grid=grid)
    assert g1.shape == (*batch, dim, dim+1)
    assert h1.shape == (*batch, dim, dim+1, dim, dim+1)
    assert torch.allclose(g1, g0, rtol=1e-12, atol=1e-12)
    assert torch.allclose(h1, h0, rtol=1e-12, atol=1e-12)
    assert torch.equal(regutils.affine_grid_backward(grad, grid=grid), g1)
    g2, h2 = regutils.affine_grid_backward(grad, None, grid=grid)
    assert torch.equal(g2, g1) and h2 is None


def test_identity_chunk():
    shape = [4, 3, 5]
    ref = spatial.identity_grid(shape, dtype=dt).reshape([-1, 3])
    for start, stop in [(0, 60), (7, 23), (59, 60)]:
        x = regutils._identity_chunk(shape, start, stop, dtype=dt)
        assert torch.equal(x, ref[start:stop])
