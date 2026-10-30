"""Patch-size units of local losses through the registration pyramid
(`pairwise_pyramid.sequential_pyramid` / `concurrent_pyramid`)."""
import pytest
import torch
from nitorch import spatial
from nitorch.tools.registration import objects, losses
from nitorch.tools.registration.losses import SliceWiseLoss
from nitorch.tools.registration.pairwise_pyramid import (
    sequential_pyramid, concurrent_pyramid)

VX = [1., 2., 4.]
SHAPE = [32, 16, 8]
NLEVELS = 3


def make_sim(loss):
    aff = spatial.affine_default(SHAPE, VX)
    g = torch.Generator().manual_seed(0)
    fix = objects.ImagePyramid(torch.rand([1, *SHAPE], generator=g),
                               levels=NLEVELS, affine=aff, method='gauss')
    mov = objects.ImagePyramid(torch.rand([1, *SHAPE], generator=g),
                               levels=NLEVELS, affine=aff, method='gauss')
    return objects.Similarity(loss, mov, fix)


def expected(patch, unit, lvl):
    """Patch in voxels at pyramid level `lvl` (gauss pyramid halves shape,
    doubles voxel size)."""
    patch = patch * 3 if len(patch) == 1 else patch
    if unit == 'vox':
        return [p / 2**lvl for p in patch]
    if unit == 'mm':
        return [p / (v * 2**lvl) for p, v in zip(patch, VX)]
    return [p / 100 * s / 2**lvl for p, s in zip(patch, SHAPE)]


def seq_levels(sim):
    # sequential_pyramid returns coarse -> fine
    return sequential_pyramid(sim)[::-1]


def conc_levels(sim):
    return list(concurrent_pyramid(sim))


CASES = [([5.], 'vox'), ([5.], 'mm'), ([8., 4., 2.], 'mm'), ([10.], 'pct')]


def _patch_arg(patch, unit):
    return patch + [unit] if unit != 'pct' else patch  # pct is the default


@pytest.mark.parametrize('patch, unit', CASES)
@pytest.mark.parametrize('levels', [seq_levels, conc_levels],
                         ids=['sequential', 'concurrent'])
def test_patch_per_level(patch, unit, levels):
    sim = make_sim(losses.LCC(patch=_patch_arg(patch, unit)))
    lvls = levels(sim)
    assert len(lvls) == NLEVELS
    for lvl, s in enumerate(lvls):
        assert s.loss.patch == pytest.approx(expected(patch, unit, lvl)), lvl
    # the user-facing loss object is left untouched
    assert sim.loss.patch == _patch_arg(patch, unit)


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('patch, unit', CASES)
@pytest.mark.parametrize('levels', [seq_levels, conc_levels],
                         ids=['sequential', 'concurrent'])
def test_slicewise_patch_per_level(axis, patch, unit, levels):
    base = losses.LCC(patch=_patch_arg(patch, unit))
    sim = make_sim(SliceWiseLoss(base, axis))
    lvls = levels(sim)
    assert len(lvls) == NLEVELS
    for lvl, s in enumerate(lvls):
        assert isinstance(s.loss, SliceWiseLoss)
        full = expected(patch, unit, lvl)
        del full[axis % 3]
        assert s.loss.base_loss.patch == pytest.approx(full), lvl


@pytest.mark.slow
def test_slicewise_loss_matches_inplane_vox():
    """Slicewise lcc (mm patch, anisotropic voxels) == slicewise lcc with
    the in-plane patch converted by hand to voxels."""
    axis = 0
    sim = make_sim(SliceWiseLoss(losses.LCC(patch=[4., 'mm']), axis))
    s = seq_levels(sim)[0]
    ref = SliceWiseLoss(losses.LCC(patch=[4 / VX[1], 4 / VX[2]]), axis)
    a = s.loss.loss_grad(s.moving.dat, s.fixed.dat, dim=3)
    b = ref.loss_grad(s.moving.dat, s.fixed.dat, dim=3)
    for x, y in zip(a, b):
        assert torch.allclose(x, y)
