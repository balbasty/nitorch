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


@pytest.mark.parametrize('bound', ['dft', 'dct2', 'dct1', 'replicate', 'zero',
                                   'dst2', 'dst1'])
@pytest.mark.parametrize('side', ['f', 'b', 'c'])
@pytest.mark.parametrize('order', [1, 2])
def test_finite_differences_singleton_dim(bound, side, order):
    """Finite differences along a dimension of size one used to return the
    input itself (not a derivative, not scaled by the voxel size)."""
    from nitorch.spatial import diff1d, div1d
    g = torch.Generator().manual_seed(0)
    x = torch.randn([4, 1], dtype=torch.double, generator=g)
    y = torch.randn([4, 1], dtype=torch.double, generator=g)
    d = diff1d(x, order, -1, 2., side, bound)
    if bound in ('dft', 'dct2', 'dct1', 'replicate'):
        # all neighbours equal the voxel itself
        assert torch.allclose(d, torch.zeros_like(d))
    # same result as the generic implementation on a padded line, with
    # explicit (boundary-implied) neighbours
    lhs = (d * y).sum()
    rhs = (x * div1d(y, order, -1, 2., side, bound)).sum()
    assert torch.allclose(lhs, rhs) or torch.allclose(lhs, -rhs)


def test_membrane_singleton_dim_has_no_energy_along_it():
    from nitorch.spatial import membrane_grid
    x = torch.zeros([4, 4, 1, 3], dtype=torch.double)
    x[1, 1, 0, 0] = 1
    vx = [32., 32., 8.]
    y = membrane_grid(x, voxel_size=vx, bound='dft')
    # 2 / vx^2 along x and y, scaled by vx[0]^2: no contribution along z
    assert torch.allclose(y[1, 1, 0, 0], torch.as_tensor(4.).double())


@pytest.mark.parametrize('shape', [(32, 32, 2), (32, 32, 1)])
def test_fmg_relax_stable_on_anisotropic_coarse_grids(shape):
    """Coarse multigrid levels of thin grids (e.g., 4x4x1 with anisotropic
    voxels) made the relaxation diverge (to NaN in single precision),
    because of the singleton-dimension finite differences."""
    g = torch.Generator().manual_seed(0)
    grad = torch.randn(shape + (3,), generator=g) * 1e-3
    hess = torch.zeros(shape + (6,))
    prm = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2),
               voxel_size=[4.] * 3)
    x = solve_grid_fmg(hess, grad, optim='relax', **prm)
    xc = solve_grid_fmg(hess, grad, optim='cg', **prm)
    assert torch.isfinite(x).all()
    assert x.abs().max() < 10 * xc.abs().max()
