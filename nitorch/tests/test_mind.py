import itertools
import math

import pytest
import torch

from nitorch.spatial import mind, rmind, smooth
from nitorch.core import utils


@pytest.mark.parametrize('fn', [mind, rmind])
def test_mind_2d_channels_are_independent(fn):
    # the variance normalization of each channel should only depend on
    # that channel (i.e., average over spatial dimensions only)
    torch.manual_seed(0)
    x = torch.rand(2, 16, 16, dtype=torch.double)
    x[1] *= 100
    both = fn(x, dim=2)
    first = fn(x[:1], dim=2)
    assert torch.allclose(both[:1], first)


@pytest.mark.parametrize('fn', [mind, rmind])
def test_mind_2d_without_channel(fn):
    x = torch.rand(16, 16, dtype=torch.double)
    assert fn(x, dim=2).shape[:2] == (16, 16)


def test_rmind_radius_too_small():
    x = torch.rand(1, 8, 8, 8, dtype=torch.double)
    with pytest.raises(ValueError, match='two rings'):
        rmind(x, radius=1, dim=3)
    assert rmind(x, radius=1.5, dim=3).shape[-1] == 1
    # without dropping the first ring, one ring is enough
    assert rmind(x, radius=1, dim=3, reduction='var').shape[-1] == 1
    assert rmind(x, radius=1, dim=3, normalize='all').shape[-1] == 1


def _rmind_reference(x, radius, fwhm, reduction, normalize, robust=3,
                     norm='max'):
    """Straightforward (slow) implementation of `rmind`."""
    dim = 3
    r = int(math.floor(radius))
    idx = [i for i in itertools.product(*[range(-r, r + 1)] * dim)
           if 0 < sum(j * j for j in i) <= radius * radius]
    sq = [sum(j * j for j in i) for i in idx]
    rings = [[k for k, s in enumerate(sq) if s == u] for u in sorted(set(sq))]
    d = torch.stack([(utils.roll(x, i, range(-dim, 0), bound='dct2') - x)
                     .square() for i in idx])
    d = smooth(d, fwhm=[fwhm] * dim, dim=dim, bound='dct2')
    v = d[rings[0]].mean(0) if normalize == 'ring1' else d.mean(0)
    mv = v.mean(list(range(-dim, 0)), keepdim=True)
    v = v.clamp(10 ** (-robust) * mv, 10 ** robust * mv)
    if reduction == 'mean':
        f = torch.stack([d[g].mean(0) for g in rings]) / v
        if normalize == 'ring1':
            f = f[1:]
    else:
        f = torch.stack([d[g].var(0, unbiased=False) for g in rings]) / v**2
    f = torch.exp(-f)
    if norm == 'max':
        f = f / f.max(0).values
    return utils.movedim(f, 0, -1)


@pytest.mark.parametrize('reduction', ['mean', 'var'])
@pytest.mark.parametrize('normalize', ['ring1', 'all'])
def test_rmind_matches_reference(reduction, normalize):
    torch.manual_seed(0)
    x = smooth(torch.rand(1, 12, 12, 12, dtype=torch.double),
               fwhm=[1.5] * 3, dim=3)
    out = rmind(x.clone(), dim=3, reduction=reduction, normalize=normalize)
    ref = _rmind_reference(x, 2, 1, reduction, normalize)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('reduction', ['mean', 'var'])
@pytest.mark.parametrize('normalize', ['ring1', 'all'])
def test_rmind_does_not_modify_input(reduction, normalize):
    x = torch.rand(1, 8, 8, 8, dtype=torch.double)
    x0 = x.clone()
    rmind(x, dim=3, reduction=reduction, normalize=normalize)
    assert torch.equal(x, x0)


@pytest.mark.parametrize('reduction', ['mean', 'var'])
@pytest.mark.parametrize('normalize', ['ring1', 'all'])
def test_rmind_invariant_to_axis_permutations_and_flips(reduction, normalize):
    # rings are invariant to the symmetries of the voxel lattice
    torch.manual_seed(0)
    x = torch.rand(1, 10, 11, 12, dtype=torch.double)
    kw = dict(dim=3, reduction=reduction, normalize=normalize)
    m = rmind(x, **kw)
    xt = x.permute(0, 3, 1, 2).flip(-1)
    mt = rmind(xt, **kw)
    assert torch.allclose(mt, m.permute(0, 3, 1, 2, 4).flip(-2))


@pytest.mark.parametrize('reduction', ['mean', 'var'])
@pytest.mark.parametrize('normalize', ['ring1', 'all'])
def test_rmind_is_constant_on_linear_ramps(reduction, normalize):
    # documents the leading-order behaviour: on a locally linear image,
    # rotation-invariant MIND features do not depend on the direction
    # or magnitude of the gradient (see Notes in `rmind`)
    torch.manual_seed(0)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(16, dtype=torch.double)] * 3, indexing='ij'), -1)
    feats = []
    for _ in range(3):
        g = torch.randn(3, dtype=torch.double) * torch.rand([]) * 10
        f = rmind(grid @ g, dim=3, bound='replicate', robust=0,
                  reduction=reduction, normalize=normalize)
        feats.append(f[5:-5, 5:-5, 5:-5].reshape(-1, f.shape[-1]))
    feats = torch.cat(feats)
    if reduction == 'mean':
        # lattice rings have isotropic second moments: exactly constant
        assert torch.allclose(feats, feats[:1].expand_as(feats))
    else:
        # angular variance depends (weakly) on the lattice orientation
        # but not on the gradient magnitude
        assert (feats.std(0) < 0.2).all()
