"""Affine-only fits on a random subset of the fixed voxels (`samples`).

The data terms are evaluated on a fixed, seeded subset of `samples` voxels
of each fixed image, each weighted by the inverse of its probability of
being drawn, so that sums over the subset estimate sums over the image.
"""
import warnings
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import objects, pairwise_run
from nitorch.tools.registration.losses import MSE, MI, LCC
from nitorch.tools.registration.pairwise_impl import (
    PairwiseRegisterStep, _VoxelSample)

dt = torch.double


def smooth(x, n):
    """Box-smooth a (C, *spatial) tensor with circular boundaries"""
    for _ in range(n):
        for d in range(1, x.dim()):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    return x


def make_images(shape=(24, 22, 20), seed=0, mask=False):
    g = torch.Generator().manual_seed(seed)
    fix = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 3)
    mov = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 3)
    msk = None
    if mask:
        msk = torch.zeros([1, *shape], dtype=dt)
        msk[:, 3:-3, 2:-2, 4:-4] = 1
    aff = spatial.affine_default(shape, 1., dtype=dt)
    fix = objects.Image(fix, affine=aff, mask=msk)
    mov = objects.Image(mov, affine=aff.clone())
    return fix, mov


def make_affine(samples=None, seed=5):
    affine = objects.AffineModel('rigid', position='mov', samples=samples)
    affine.set_dat(dim=3, dtype=dt)
    g = torch.Generator().manual_seed(seed)
    prm = torch.randn(affine.dat.dat.shape, dtype=dt, generator=g) * 0.02
    prm[:3] *= 20   # translations (mm)
    affine.dat.dat.copy_(prm)
    return affine


def make_step(loss, fix, mov, samples):
    sim = objects.SumSimilarity([objects.Similarity(loss, mov, fix)])
    affine = make_affine(samples)
    return PairwiseRegisterStep(sim, affine, None, verbose=0), affine


@pytest.mark.parametrize('mask', [False, True])
def test_sample_layout(mask):
    """Coordinates, values and weights of the sampled voxels"""
    fix, _ = make_images(mask=mask)
    n = 500
    sample = _VoxelSample(fix, n, torch.Generator().manual_seed(1))
    ncand = int(fix.mask.sum()) if mask else fix.dat[0].numel()
    assert sample.n == n and sample.ncand == ncand
    assert sample.grid.shape == (n, 1, 1, 3)
    assert sample.dat.shape == (1, n, 1, 1)
    # values are those of the image at the coordinates
    ijk = sample.grid.reshape(n, 3).long()
    assert torch.equal(sample.dat.reshape(n), fix.dat[0][tuple(ijk.T)])
    # distinct voxels, all inside the mask
    assert len(set(map(tuple, ijk.tolist()))) == n
    if mask:
        assert bool((fix.mask[0][tuple(ijk.T)] > 0).all())
    # Horvitz-Thompson weights: they sum to the number of candidates
    assert sample.mask.sum().item() == pytest.approx(ncand)


def test_sum_is_unbiased():
    """A weighted sum over samples estimates the sum over the image"""
    fix, _ = make_images()
    full = fix.dat.sum().item()
    est = []
    for seed in range(200):
        s = _VoxelSample(fix, 200, torch.Generator().manual_seed(seed))
        est.append((s.dat * s.mask).sum().item())
    est = torch.as_tensor(est)
    sem = est.std() / len(est) ** 0.5
    assert abs(est.mean().item() - full) < 4 * sem


@pytest.mark.parametrize('loss', ['mse', 'mi'])
def test_gradient(loss):
    """Gradient of the sampled objective (finite differences)"""
    fix, mov = make_images()
    loss = MSE(lam=1, dim=3) if loss == 'mse' else MI(dim=3, bins=8)
    step, affine = make_step(loss, fix, mov, samples=2000)
    prm = affine.parameters().dat.clone()
    ll, g = step.do_affine_only(prm, grad=True)
    assert step._samples, 'no sample drawn'
    eps = 1e-5
    fd = torch.zeros_like(g)
    for k in range(len(prm)):
        dp = torch.zeros_like(prm)
        dp[k] = eps
        lp = step.do_affine_only(prm + dp, in_line_search=True)
        lm = step.do_affine_only(prm - dp, in_line_search=True)
        fd[k] = (lp - lm) / (2 * eps)
    # The MI gradient ignores the dependence of the histogram range on
    # the warped intensities: it is ~1.5% off on all voxels too.
    rtol = 1e-3 if isinstance(loss, MSE) else 5e-2
    assert torch.allclose(g, fd, rtol=rtol, atol=1e-6 * g.abs().max())


def test_same_sample_every_evaluation():
    """The objective is a deterministic function of the parameters"""
    fix, mov = make_images()
    step, affine = make_step(MSE(lam=1, dim=3), fix, mov, samples=1000)
    prm = affine.parameters().dat.clone()
    l1 = step.do_affine_only(prm, in_line_search=True)
    l2 = step.do_affine_only(prm, in_line_search=True)
    assert l1.item() == l2.item()
    # a new step (e.g., the next run at this level) draws the same sample
    step2, _ = make_step(MSE(lam=1, dim=3), fix, mov, samples=1000)
    assert step2.do_affine_only(prm, in_line_search=True).item() == l1.item()


def test_scale_matches_full():
    """The sampled loss estimates the full loss (same scale as the prior)"""
    fix, mov = make_images(shape=(40, 40, 40))
    prm = make_affine().parameters().dat.clone()
    full, _ = make_step(MSE(lam=1, dim=3), fix, mov, samples=None)
    lf = full.do_affine_only(prm, in_line_search=True).item()
    ls = []
    for seed in range(10):  # one draw scatters by ~2.5%: average a few
        smpl, _ = make_step(MSE(lam=1, dim=3), fix, mov, samples=20000)
        smpl._sample_rng = torch.Generator().manual_seed(seed)
        ls.append(smpl.do_affine_only(prm, in_line_search=True).item())
    assert sum(ls) / len(ls) == pytest.approx(lf, rel=0.02)


def test_all_voxels_when_samples_exceed_image():
    fix, mov = make_images()
    prm = make_affine().parameters().dat.clone()
    full, _ = make_step(MSE(lam=1, dim=3), fix, mov, samples=None)
    big, _ = make_step(MSE(lam=1, dim=3), fix, mov, samples=10**9)
    assert (big.do_affine_only(prm, in_line_search=True).item() ==
            full.do_affine_only(prm, in_line_search=True).item())


def test_spatial_loss_uses_all_voxels():
    fix, mov = make_images()
    prm = make_affine().parameters().dat.clone()
    full, _ = make_step(LCC(dim=3, patch=5), fix, mov, samples=None)
    smpl, _ = make_step(LCC(dim=3, patch=5), fix, mov, samples=100)
    with pytest.warns(UserWarning, match='all voxels'):
        ls = smpl.do_affine_only(prm, in_line_search=True)
    lf = full.do_affine_only(prm, in_line_search=True)
    assert ls.item() == lf.item()


def test_recovers_translation():
    """An affine-only fit on samples recovers a known rigid transform"""
    shape = (32, 32, 32)
    g = torch.Generator().manual_seed(0)
    img = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 4)
    aff = spatial.affine_default(shape, 1., dtype=dt)
    true = torch.zeros(6, dtype=dt)
    true[:3] = torch.as_tensor([1.3, -0.7, 0.5])
    true[3:] = torch.as_tensor([0.02, -0.03, 0.01])
    tmat = spatial.affine_matrix_classic(true, dim=3)
    grid = spatial.affine_grid(tmat.to(dt), shape)
    mov = spatial.grid_pull(img, grid, bound='dct2', extrapolate=True)
    fix = objects.Image(mov, affine=aff)
    mov = objects.Image(img, affine=aff.clone())

    def fit(samples):
        sim = objects.SumSimilarity([objects.Similarity(
            MSE(lam=1, dim=3), mov, fix)])
        affine = objects.AffineModel('rigid', position='mov', samples=samples)
        affine, _ = pairwise_run.run(
            sim, affine, affine_optim='gn', pyramid=False,
            progressive=False, verbose=0)
        return affine.exp().detach()

    a_full, a_smpl = fit(None), fit(4096)
    # the sampled fit lands close to the full fit (<0.05 voxel at the
    # corners of the field of view)
    corners = torch.cartesian_prod(*[torch.as_tensor([0., s - 1]) for s in shape])
    corners = corners.to(dt)
    d = spatial.affine_matvec(a_full, corners) - spatial.affine_matvec(a_smpl, corners)
    assert d.norm(dim=-1).max() < 0.05
