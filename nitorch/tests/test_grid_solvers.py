"""Options of the displacement-grid solvers used by `nitorch register`
(`@@optim gn -g 0`, `@@optim gn -q relax`) that used to crash."""
import pytest
import torch
from nitorch.core.optim import relax_slicers
from nitorch.core.linalg import sym_matvec
from nitorch.spatial import solve_grid, solve_grid_fmg, regulariser_grid


@pytest.mark.parametrize('shape', [(5, 4, 2), (7, 3, 1), (6, 6, 6), (2,)])
@pytest.mark.parametrize('scheme', ['checkerboard', 2, 3])
def test_relax_slicers_partition(shape, scheme):
    """Slicers are non-empty and cover each voxel exactly once, including
    along dimensions shorter than the bandwidth."""
    count = torch.zeros(shape + (1,), dtype=torch.long)
    for group in relax_slicers(shape, scheme):
        for slicer in group:
            sub = count[slicer]
            assert sub.numel() > 0
            count[slicer] += 1
    assert (count == 1).all()


@pytest.mark.parametrize('shape', [(12, 10, 2), (12, 10, 4)])
@pytest.mark.parametrize('solver, optim', [
    (solve_grid, 'cg'),        # no FMG: used to fail on stop='e'
    (solve_grid, 'relax'),
    (solve_grid_fmg, 'relax'), # bending: bandwidth 3 > shortest dim
])
def test_grid_solvers_run(shape, solver, optim):
    g = torch.Generator().manual_seed(0)
    dim = len(shape)
    grad = torch.randn(shape + (dim,), dtype=torch.double, generator=g)
    # full symmetric Hessian (as in registration): diagonal + off-diagonal
    nh = dim * (dim + 1) // 2
    hess = 0.1 * torch.rand(shape + (nh,), dtype=torch.double, generator=g)
    hess[..., :dim] += 1
    prm = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2))
    it = dict(nb_iter=4) if solver is solve_grid_fmg else dict(max_iter=8)
    x = solver(hess, grad, optim=optim, **it, **prm)
    assert torch.isfinite(x).all()
    # the residual decreases w.r.t. the initial guess (x = 0)
    res = sym_matvec(hess, x) + regulariser_grid(x, **prm) - grad
    assert res.norm() < grad.norm()
