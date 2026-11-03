"""Regression tests for #171: `denoise -x relax` and `solve_field(optim='relax')`.

1. `tv.denoise(optim='relax')` used to force CG for the first 9 IRLS
   iterations, so relax was (almost) never used with the CLI default of
   10 iterations.
2. `solve_field(optim='relax')` did not pass the spatial `dim` (nor the
   channel-last layout) to `core.optim.relax`, so for a (K, X, Y, Z)
   problem the red-black checkerboard was defined over (K, X, Y)
   and neighbours along Z had the same colour.
"""
import pytest
import torch
from nitorch import spatial
from nitorch.core import optim as core_optim
from nitorch.spatial import _solvers
from nitorch.tools.denoising import tv


pytestmark = pytest.mark.skipif(
    _solvers.c_solvers is not None,
    reason='compiled backend bypasses the python relax solver')


def _phantom(n=12, K=2, sd=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    lin = torch.linspace(-1, 1, n, dtype=torch.double)
    z, y, x = torch.meshgrid(lin, lin, lin, indexing='ij')
    t0 = (x**2 + y**2 + z**2 < 0.6).double()
    t1 = ((x.abs() < 0.5) & (y.abs() < 0.3)).double() * 0.7 \
        + 0.3 * (z > 0).double()
    gt = torch.stack([t0 + 0.5 * t1, t1])[:K]
    noisy = gt + sd * torch.randn(gt.shape, generator=g, dtype=torch.double)
    return gt, noisy, sd


def _spy_solve_field(monkeypatch):
    calls = []
    orig = spatial.solve_field

    def spy(*args, **kwargs):
        calls.append(kwargs.get('optim', 'cg'))
        return orig(*args, **kwargs)

    monkeypatch.setattr(spatial, 'solve_field', spy)
    return calls


def test_denoise_relax_used_from_first_iteration(monkeypatch):
    _, img, sd = _phantom(n=8)
    calls = _spy_solve_field(monkeypatch)
    tv.denoise(img, lam=20, sigma=sd, max_iter=3, sub_iter=4,
               optim='relax', tol=0)
    assert calls == ['relax'] * 3


def test_denoise_relax_matches_cg():
    gt, img, sd = _phantom()
    kw = dict(lam=20, sigma=sd, max_iter=10, sub_iter=32, tol=0,
              sub_tol=0)
    out_cg = tv.denoise(img, optim='cg', **kw)
    out_rx = tv.denoise(img, optim='relax', **kw)
    rel = (out_rx - out_cg).norm() / out_cg.norm()
    assert rel < 1e-2, rel.item()
    # and denoising actually helped
    assert (out_rx - gt).norm() < (img - gt).norm()


def test_solve_field_relax_checkerboard_spatial(monkeypatch):
    """The colouring used by relax for (K, X, Y, Z) must alternate along
    every spatial axis (X, Y and Z), and must not split channels."""
    K, shape = 2, (5, 6, 7)
    seen = []
    orig = core_optim.relax_slicers

    def spy(shp, scheme='checkerboard'):
        seen.append(tuple(shp))
        return orig(shp, scheme)

    monkeypatch.setattr(core_optim, 'relax_slicers', spy)
    g = torch.randn(K, *shape, dtype=torch.double)
    h = torch.ones(K, *shape, dtype=torch.double)
    spatial.solve_field(h, g, dim=3, membrane=1, optim='relax',
                        max_iter=2, tolerance=0)
    assert seen == [shape]

    # colouring check: red/black voxels, each group covers all channels
    groups = orig(shape, 'checkerboard')
    colour = torch.full((*shape, K), -1, dtype=torch.long)
    for c, group in enumerate(groups):
        for slicer in group:
            colour[slicer] = c
    assert (colour >= 0).all()
    assert (colour[..., :1] == colour).all()     # channels share colour
    colour = colour[..., 0]
    for d in range(3):
        a = colour.narrow(d, 0, shape[d] - 1)
        b = colour.narrow(d, 1, shape[d] - 1)
        assert (a != b).all(), f'neighbours along axis {d} share a colour'


def test_solve_field_relax_converges_3d():
    torch.manual_seed(0)
    K, shape = 2, (8, 8, 8)
    g = torch.randn(K, *shape, dtype=torch.double)
    h = torch.rand(K, *shape, dtype=torch.double) + 0.1
    w = torch.rand(1, *shape, dtype=torch.double) + 0.5
    prm = dict(dim=3, membrane=1, factor=[2., 3.], weights=w)
    ref = spatial.solve_field(h, g, **prm, optim='cg', max_iter=1000,
                              tolerance=0)
    out = spatial.solve_field(h, g, **prm, optim='relax', max_iter=300,
                              tolerance=0)
    rel = (out - ref).norm() / ref.norm()
    # with the checkerboard over (K, X, Y) instead of (X, Y, Z),
    # the error after 300 sweeps was ~5e-7 (now ~5e-8)
    assert rel < 1e-7, rel.item()


def test_solve_field_relax_batched():
    """Leading batch dimensions must not be coloured either."""
    torch.manual_seed(0)
    B, K, shape = 2, 2, (6, 6, 6)
    g = torch.randn(B, K, *shape, dtype=torch.double)
    h = torch.rand(B, K, *shape, dtype=torch.double) + 0.1
    prm = dict(dim=3, membrane=1)
    ref = spatial.solve_field(h, g, **prm, optim='cg', max_iter=1000,
                              tolerance=0)
    out = spatial.solve_field(h, g, **prm, optim='relax', max_iter=300,
                              tolerance=0)
    rel = (out - ref).norm() / ref.norm()
    assert rel < 1e-6, rel.item()
