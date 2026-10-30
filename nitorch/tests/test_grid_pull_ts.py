import pytest
import torch

from nitorch.spatial import grid_pull


@pytest.mark.parametrize('order', [2, 3, 5])
def test_grid_pull_1d_spline(order):
    # a (non-prefiltered) spline reproduces linear functions in the interior
    x = torch.arange(1., 11., dtype=torch.double)[None, None]
    coords = torch.tensor([3., 4.25, 5.5, 6.75], dtype=torch.double)
    out = grid_pull(x, coords[None, :, None], interpolation=order,
                    bound='dct2', extrapolate=True)
    assert torch.allclose(out[0, 0], coords + 1)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_grid_pull_edge_tolerance(dim):
    # points (very slightly) outside of the field of view are kept, as in
    # the compiled backend
    x = torch.ones([1, 1] + [5] * dim, dtype=torch.double)
    grid = torch.full([1, 1, dim], 2., dtype=torch.double)
    grid[..., 0] = -0.01
    out = grid_pull(x, grid, interpolation=1, bound='dct2', extrapolate=False)
    assert out.item() == 1
    grid[..., 0] = -0.1
    out = grid_pull(x, grid, interpolation=1, bound='dct2', extrapolate=False)
    assert out.item() == 0
