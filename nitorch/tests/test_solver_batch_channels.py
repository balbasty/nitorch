"""Solver and resampling invariants that the compiled (C) backend used to
break; backend-agnostic, so they also guard the TorchScript backend.

- batches: each item of a batch is solved as if it were alone (the C grid
  regulariser used the input's batch stride for the Hessian, so items > 0
  read the wrong Hessian; FMG with CG then returned NaN for them);
- two-channel fields with a diagonal Hessian (CC == C == 2): the C relax
  dispatched them to the ESTATICS layout and read out of bounds;
- weighted (RLS) bending / linear-elastic energies, which the C solvers do
  not implement (the TorchScript ones are used instead of raising);
- resampling with a grid of another dtype than the image, or with fewer
  output spatial dimensions than the grid has components.
"""
import torch
import pytest
from nitorch import spatial
from nitorch.spatial._solvers import (
    solve_grid_fmg, solve_grid_sym, solve_field_fmg, solve_field_sym)
from nitorch.spatial._regularisers import regulariser_grid, regulariser

dt = torch.double


def _grid_system(batch, shape=(12, 10), seed=0):
    g = torch.Generator().manual_seed(seed)
    dim = len(shape)
    h = torch.zeros([*shape, dim * (dim + 1) // 2], dtype=dt)
    h[..., :dim] = torch.rand([*shape, dim], dtype=dt, generator=g) + 0.5
    grad = torch.randn([*shape, dim], dtype=dt, generator=g)
    return (h.expand([batch, *h.shape]).contiguous(),
            grad.expand([batch, *grad.shape]).contiguous(), h, grad)


@pytest.mark.parametrize('shape', [(12, 10), (8, 7, 6)])
@pytest.mark.parametrize('optim', ['cg', 'relax'])
def test_grid_fmg_batch_items_are_independent(shape, optim):
    hb, gb, h, g = _grid_system(3, shape)
    xb = solve_grid_fmg(hb, gb, membrane=0.1, optim=optim)
    x = solve_grid_fmg(h, g, membrane=0.1, optim=optim)
    assert torch.isfinite(xb).all()
    for i in range(3):
        assert torch.allclose(xb[i], x, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('shape', [(12, 10), (8, 7, 6)])
def test_grid_regulariser_batch_items_are_independent(shape):
    hb, gb, h, g = _grid_system(3, shape)
    rb = regulariser_grid(gb, membrane=0.1, bending=0.1)
    r = regulariser_grid(g, membrane=0.1, bending=0.1)
    for i in range(3):
        assert torch.allclose(rb[i], r)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('solver', ['fmg', 'sym'])
def test_two_channel_field_diagonal_hessian(dim, solver):
    # CC == C == 2: same solution as the equivalent scaled identity
    # (CC == 1) when both channels have the same Hessian
    shape = (12, 10) if dim == 2 else (8, 7, 6)
    g = torch.Generator().manual_seed(0)
    grad = torch.randn([2, *shape], dtype=dt, generator=g)
    h1 = torch.full([1, *shape], 0.05, dtype=dt)
    h2 = h1.expand([2, *shape]).contiguous()
    fn = solve_field_fmg if solver == 'fmg' else solve_field_sym
    kw = dict(optim='relax') if solver == 'fmg' else dict(optim='relax', max_iter=8)
    x1 = fn(h1, grad, dim=dim, membrane=0.1, **kw)
    x2 = fn(h2, grad, dim=dim, membrane=0.1, **kw)
    assert torch.isfinite(x2).all()
    assert torch.allclose(x1, x2, rtol=1e-6, atol=1e-8)


def test_weighted_bending_and_lame():
    # weighted (RLS) bending / linear-elastic energies: implemented
    # (by the TorchScript solvers when the compiled ones do not)
    shape = (8, 7)
    g = torch.Generator().manual_seed(0)
    v = torch.randn([*shape, 2], dtype=dt, generator=g)
    w = torch.rand(shape, dtype=dt, generator=g) + 0.5
    for kw in [dict(bending=0.1), dict(lame=(0.05, 0.2))]:
        r = regulariser_grid(v, weights=w, **kw)
        assert torch.isfinite(r).all()
        # weights of ones = unweighted
        assert torch.allclose(regulariser_grid(v, weights=torch.ones_like(w), **kw),
                              regulariser_grid(v, **kw), atol=1e-8)
    x = torch.randn([2, *shape], dtype=dt, generator=g)
    r = regulariser(x, dim=2, bending=0.1, weights=w)
    assert torch.isfinite(r).all()


def test_pull_mixed_dtypes_keeps_image_dtype():
    x = torch.rand([1, 1, 5, 6, 7])
    grid = spatial.identity_grid([5, 6, 7], dtype=dt)[None] + 0.25
    out = spatial.grid_pull(x, grid)
    assert out.dtype == x.dtype
    ref = spatial.grid_pull(x.double(), grid)
    assert torch.allclose(out.double(), ref, atol=1e-5)
    out = spatial.grid_pull(x.double(), grid.float())
    assert out.dtype == dt


@pytest.mark.parametrize('grid_shape,out_shape', [
    ([1, 1, 3], [1, 1, 1, 1, 1]),
    ([4, 3], [1, 1, 1, 1, 4]),
    ([2, 4, 3], [1, 1, 1, 2, 4]),
])
def test_pull_grid_with_fewer_spatial_dims(grid_shape, out_shape):
    # output spatial dimensions are left-padded with singletons
    x = torch.ones([1, 1, 5, 5, 5], dtype=dt)
    grid = torch.full(grid_shape, 2., dtype=dt)
    out = spatial.grid_pull(x, grid)
    assert list(out.shape) == out_shape
    assert torch.allclose(out, torch.ones_like(out))
