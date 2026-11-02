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


@pytest.mark.parametrize('shape', [(32, 32, 1), (8, 8, 2), (5, 1), (3, 4)])
def test_greens_small_and_singleton_dims(shape):
    """`greens` (used by geodesic shooting and as a preconditioner) used
    to fail on dimensions smaller than its 5^D stencil patch."""
    from nitorch.spatial import greens
    k = greens(shape, absolute=1e-4, membrane=1e-3, bending=0.2,
               lame=(0.05, 0.2), dtype=torch.double)
    assert k.shape == tuple(shape) + (len(shape), len(shape))
    assert torch.isfinite(k).all()


@pytest.mark.parametrize('shape', [(16, 16, 16), (24, 24), (32, 32, 1)])
def test_grid_inv_runs_and_inverts(shape):
    """`grid_inv` (used to write the outputs of `smalldef` registrations)
    passed a single-channel Hessian to the multigrid solver, which failed."""
    from nitorch.spatial import grid_inv, identity_grid, grid_pull
    g = torch.Generator().manual_seed(0)
    dim = len(shape)
    d = torch.randn([1, dim, *[max(s // 4, 1) for s in shape]],
                    dtype=torch.double, generator=g)
    mode = 'trilinear' if dim == 3 else 'bilinear'
    d = torch.nn.functional.interpolate(d, size=shape, mode=mode,
                                        align_corners=True)[0]
    d = d.movedim(0, -1) * 0.5
    d[..., [i for i, s in enumerate(shape) if s == 1]] = 0
    inv = grid_inv(d, type='disp')
    assert inv.shape == d.shape and torch.isfinite(inv).all()
    grid = identity_grid(shape, dtype=d.dtype) + d
    comp = d + grid_pull(inv.movedim(-1, 0)[None], grid[None], bound='dft',
                         extrapolate=True)[0].movedim(0, -1)
    inner = tuple(slice(3, -3) if s > 6 else slice(None) for s in shape)
    assert comp[inner].abs().mean() < 0.1 * d[inner].abs().mean()


@pytest.mark.parametrize('shape', [(8, 8), (6, 7, 5)])
@pytest.mark.parametrize('bound', ['dft', 'dct2'])
def test_lame_shear_self_adjoint(shape, bound):
    """The (unweighted) shear operator used `d_i x_j + d_j x_i / 2` instead
    of the symmetric strain, so it was not self-adjoint (and differed
    from the weighted implementation)."""
    from nitorch.spatial._regularisers import lame_shear
    g = torch.Generator().manual_seed(0)
    dim = len(shape)
    vx = [1.5, 1., 2.][:dim]
    a = torch.randn(shape + (dim,), dtype=torch.double, generator=g)
    b = torch.randn(shape + (dim,), dtype=torch.double, generator=g)
    La = lame_shear(a, voxel_size=vx, bound=bound)
    Lb = lame_shear(b, voxel_size=vx, bound=bound)
    assert torch.allclose((La * b).sum(), (a * Lb).sum())
    w = torch.ones(shape, dtype=torch.double)
    assert torch.allclose(La, lame_shear(a, voxel_size=vx, bound=bound, weights=w))


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('name', ['shear', 'div'])
def test_lame_kernels_match_operators(dim, name):
    """The 3D `lame_div_kernel` included mixed derivatives along the wrong
    pairs of axes."""
    from nitorch.spatial import _regularisers as R
    fn = getattr(R, f'lame_{name}')
    kfn = getattr(R, f'lame_{name}_kernel')
    vx = [1.5, 1., 2.][:dim]
    shape = (9,) * dim
    K = kfn(dim, voxel_size=vx, dtype=torch.double).to_dense()
    resp = torch.zeros((dim, dim) + (3,) * dim, dtype=torch.double)
    center = (4,) * dim
    for d in range(dim):
        x = torch.zeros(shape + (dim,), dtype=torch.double)
        x[center + (d,)] = 1
        y = fn(x, voxel_size=vx, bound='dft')
        resp[:, d] = y[(slice(3, 6),) * dim].movedim(-1, 0)
    assert torch.allclose(K, resp)


@pytest.mark.parametrize('shape', [(32, 32), (16, 16, 16), (32, 32, 1)])
def test_greens_inverts_regulariser(shape):
    """In 2D, the Green's function did not invert the regulariser when both
    Lame terms were used (non-symmetric shear + transposed 2x2 inverse)."""
    from nitorch.spatial import greens, greens_apply, regulariser_grid
    g = torch.Generator().manual_seed(0)
    dim = len(shape)
    prm = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2))
    v = torch.randn(shape + (dim,), dtype=torch.double, generator=g)
    k = greens(shape, **prm, voxel_size=[1.5] * dim, dtype=torch.double)
    m = regulariser_grid(v, **prm, voxel_size=[1.5] * dim, bound='dft')
    vv = greens_apply(m, k, voxel_size=[1.5] * dim)
    assert torch.allclose(vv, v, rtol=1e-4, atol=1e-6 * v.abs().max())


# ----------------------------------------------------------------------
#   Diagonal of the regularisers (used by the relaxation smoother)
# ----------------------------------------------------------------------


def _probe_diag(op, shape, voxels):
    """Diagonal of a linear operator on (*shape, dim) grids, probed with
    unit impulses at a few voxels"""
    dim = len(shape)
    out = torch.zeros([len(voxels), dim], dtype=torch.double)
    for n, idx in enumerate(voxels):
        for d in range(dim):
            e = torch.zeros(shape + (dim,), dtype=torch.double)
            e[tuple(idx) + (d,)] = 1
            out[n, d] = op(e)[tuple(idx) + (d,)]
    return out


@pytest.mark.parametrize('vx', [[1., 1., 1.], [4., 4., 4.], [64., 64., 8.],
                                [2., 3., 0.5]])
@pytest.mark.parametrize('name', ['membrane', 'bending'])
def test_diag_matches_grid_operator_impulse(vx, name):
    """`bending_diag` used `6 * sum(1/vx^2)` instead of `6 * sum(1/vx^4)`,
    so it did not match `bending_grid` unless vx == 1 (e.g., 19.5 instead
    of 2.625 for vx = 4), which over-damped (or, for some anisotropic
    voxels, destabilised) the relaxation smoother."""
    from nitorch.spatial import _regularisers as R
    grid_op = getattr(R, f'{name}_grid')
    diag_op = getattr(R, f'{name}_diag')
    shape = (9, 9, 9)
    true = _probe_diag(lambda e: grid_op(e, voxel_size=vx, bound='dft'),
                       shape, [(4, 4, 4)])[0]
    vx2 = torch.as_tensor(vx, dtype=torch.double).square()
    diag = diag_op(voxel_size=vx, bound='dft', dim=3) * vx2
    assert torch.allclose(diag, true)


@pytest.mark.parametrize('shape, vx', [((7, 7, 7), [2., 3., 0.5]),
                                       ((8, 7), [4., 1.5])])
@pytest.mark.parametrize('bound', ['dft', 'dct2'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('prm', [dict(absolute=0.1), dict(membrane=0.1),
                                 dict(bending=0.3), dict(lame=(0.05, 0)),
                                 dict(lame=(0, 0.2))])
def test_regulariser_grid_diag(shape, vx, bound, weighted, prm):
    """`regulariser_grid_diag` (used by the grid solvers) is the exact
    diagonal of `regulariser_grid` in the interior of the field of view,
    and never underestimates it near the edges (which would make the
    relaxation diverge). The weighted (RLS) diagonals were wrong (and
    crashed the grid solvers)."""
    from nitorch.spatial import regulariser_grid, regulariser_grid_diag
    import itertools
    dim = len(shape)
    g = torch.Generator().manual_seed(0)
    w = (0.5 + torch.rand(shape, dtype=torch.double, generator=g)
         if weighted else None)
    prm = dict(prm, voxel_size=vx, bound=bound, factor=1.5)
    if dim == 3:
        voxels = [(i, i, i) for i in range(shape[0])] + [(0, 3, 6), (3, 0, 1)]
    else:
        voxels = list(itertools.product(*[range(s) for s in shape]))
    true = _probe_diag(lambda e: regulariser_grid(e, weights=w, **prm),
                       shape, voxels)
    diag = regulariser_grid_diag(dim, weights=w, **prm,
                                 dtype=torch.double)
    if weighted:
        diag = torch.stack([diag[idx] for idx in voxels])
    else:
        diag = diag.expand(true.shape)
    interior = [all(2 <= i < s - 2 for i, s in zip(idx, shape))
                for idx in voxels]
    assert any(interior)
    assert torch.allclose(diag[interior], true[interior])
    assert (diag >= true * (1 - 1e-12)).all()


def test_diag_does_not_modify_weights():
    """`_lincomb` accumulated in-place into its first input when its
    weight was one, overwriting the input weight map."""
    from nitorch.spatial import regulariser_grid_diag, sobel
    g = torch.Generator().manual_seed(0)
    w = torch.rand([6, 6, 6], dtype=torch.double, generator=g)
    w0 = w.clone()
    regulariser_grid_diag(3, membrane=1, bending=1, lame=(1, 1), weights=w,
                          voxel_size=3 ** 0.5)  # => membrane center == 1
    assert torch.equal(w, w0)
    sobel(w)
    assert torch.equal(w, w0)


# ----------------------------------------------------------------------
#   Sparse convolutions and regulariser kernels
# ----------------------------------------------------------------------


def _dense_circular_conv(x, kernel, dim):
    """y[i] = sum_k K[k] x[i + k - center] with circular boundaries"""
    import itertools
    ks = kernel.shape[-dim:]
    y = 0
    for k in itertools.product(*[range(s) for s in ks]):
        shift = [s // 2 - kk for kk, s in zip(k, ks)]
        xs = torch.roll(x, shift, list(range(-dim, 0)))
        w = kernel[(Ellipsis, *k)]
        if w.dim() == 0:
            y = y + w * xs
        elif w.dim() == 1:
            y = y + w.reshape([-1] + [1] * dim) * xs
        else:
            y = y + torch.einsum('io,i...->o...', w, xs)
    return y


@pytest.mark.parametrize('kshape, xshape', [
    ((5, 5, 5), (7, 7, 7)),          # scalar kernel
    ((3, 3, 3), (7, 7, 7)),          # scalar kernel
    ((2, 3, 3, 3), (2, 7, 7, 7)),    # one kernel per channel
    ((3, 3, 3, 3, 3), (3, 7, 7, 7)), # channel-mixing kernel
])
def test_spconv_matches_dense(kshape, xshape):
    """`spconv` dropped the off-center weights of kernels without
    channel dimensions (and crashed on per-channel kernels)."""
    from nitorch.spatial import spconv
    g = torch.Generator().manual_seed(0)
    kernel = torch.randn(kshape, dtype=torch.double, generator=g)
    x = torch.randn(xshape, dtype=torch.double, generator=g)
    y = spconv(x, kernel.to_sparse(), bound='dft', dim=3)
    assert torch.allclose(y, _dense_circular_conv(x, kernel, 3))


@pytest.mark.parametrize('prm', [dict(absolute=0.01), dict(membrane=0.1),
                                 dict(bending=0.3), dict(lame=(0.05, 0)),
                                 dict(lame=(0, 0.2)),
                                 dict(membrane=0.1, lame=(0.05, 0.2))])
def test_regulariser_grid_kernel_matches_operator(prm):
    """`regulariser_grid` swapped the two Lame constants (and the
    sparse convolution of grid kernels was wrong)."""
    from nitorch.spatial import (spconv, regulariser_grid,
                                 regulariser_grid_kernel)
    g = torch.Generator().manual_seed(0)
    vx = [2., 3., 0.5]
    v = torch.randn([7, 7, 7, 3], dtype=torch.double, generator=g)
    kernel = regulariser_grid_kernel(3, voxel_size=vx, **prm,
                                     dtype=torch.double)
    y = spconv(v.movedim(-1, 0), kernel, bound='dft', dim=3).movedim(0, -1)
    assert torch.allclose(y, regulariser_grid(v, voxel_size=vx, **prm))


@pytest.mark.parametrize('prm', [dict(absolute=0.01), dict(membrane=0.1),
                                 dict(bending=0.3)])
def test_regulariser_kernel_matches_operator(prm):
    """`regulariser_kernel` ignored the regularisation weights."""
    from nitorch.spatial import spconv, regulariser, regulariser_kernel
    g = torch.Generator().manual_seed(0)
    vx = [2., 3., 0.5]
    x = torch.randn([2, 7, 7, 7], dtype=torch.double, generator=g)
    kernel = regulariser_kernel(3, voxel_size=vx, **prm, dtype=torch.double)
    y = spconv(x, kernel, bound='dft', dim=3)
    assert torch.allclose(y, regulariser(x, voxel_size=vx, bound='dft', **prm))


# ----------------------------------------------------------------------
#   Convergence of the multigrid relaxation
# ----------------------------------------------------------------------


@pytest.mark.parametrize('shape, weighted', [((8, 8, 8), False),
                                             ((16, 16), False),
                                             ((16, 16), True)])
def test_fmg_relax_converges(shape, weighted):
    """With voxel size 4 (and on coarse multigrid levels, whatever the
    voxel size), the relaxation smoother used an overestimated diagonal
    for the bending energy. FMG+relax was ~20-40% away from the solution
    (vs ~3-7% for FMG+CG), and the weighted solver crashed."""
    g = torch.Generator().manual_seed(0)
    dim = len(shape)
    grad = torch.randn(shape + (dim,), dtype=torch.double, generator=g)
    hess = torch.zeros(shape + (dim * (dim + 1) // 2,), dtype=torch.double)
    hess[..., :dim] = 1e-2 + 0.1 * torch.rand(shape + (dim,),
                                              dtype=torch.double, generator=g)
    w = (0.5 + torch.rand(shape, dtype=torch.double, generator=g)
         if weighted else None)
    prm = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2),
               voxel_size=[4.] * dim, weights=w)
    ref = solve_grid(hess, grad, optim='cg', max_iter=500, tolerance=0, **prm)
    x = solve_grid_fmg(hess, grad, optim='relax', **prm)
    err = (x - ref).norm() / ref.norm()
    assert err < 0.06


@pytest.mark.parametrize('weighted', [False, True])
def test_field_fmg_relax_converges(weighted):
    """Same as above for vector fields (`bending_diag` and the weighted
    `membrane_diag` were wrong): FMG+relax was 17-30% away from the
    solution."""
    from nitorch.spatial import solve_field, solve_field_fmg
    g = torch.Generator().manual_seed(0)
    shape = (2, 24, 24)
    grad = torch.randn(shape, dtype=torch.double, generator=g)
    hess = 1e-2 + 0.1 * torch.rand(shape, dtype=torch.double, generator=g)
    w = (0.5 + torch.rand(shape[1:], dtype=torch.double, generator=g)
         if weighted else None)
    prm = dict(absolute=1e-4, membrane=1e-3, bending=0.2,
               voxel_size=[4., 4.], weights=w)
    ref = solve_field(hess, grad, optim='cg', max_iter=1000, tolerance=0,
                      **prm)
    x = solve_field_fmg(hess, grad, optim='relax', **prm)
    assert (x - ref).norm() / ref.norm() < 1e-3
