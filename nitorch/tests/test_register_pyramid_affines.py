"""Pyramid levels in `nitorch register`: each level's voxel-to-world affine
must be the one induced by the operator that produced its data (#114).

The image pyramid and the velocity pyramid may use different lattices:
the forward model connects them through their voxel-to-world affines
(the `left` / `right` affines), so what matters is that each affine
describes its data exactly, and that the world-space transform is kept
when the velocity is moved across levels."""
import math
import pytest
import torch
from nitorch import spatial
from nitorch.tools.registration import objects, pairwise_run
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep
from nitorch.tools.registration.pairwise_pyramid import sequential_pyramid

dt = torch.double
VX = [1.2, 1.0, 3.0]


def smooth(x, n):
    """Box-smooth a (C, *spatial) tensor with circular boundaries"""
    for _ in range(n):
        for d in range(1, x.dim()):
            if x.shape[d] > 2:
                x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    return x


def smooth_field(shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn([3, *shape], dtype=dt, generator=g)
    return smooth(v, 3).movedim(0, -1).contiguous()


def spline(dat, vox):
    """Cubic (circular) spline of a (*shape, 3) displacement at `vox`,
    the interpolation used by the transfers"""
    return spatial.grid_pull(dat.movedim(-1, 0)[None], vox[None],
                             interpolation=3, bound='dft', extrapolate=True,
                             prefilter=True)[0].movedim(0, -1)


# ----------------------------------------------------------------------
#   Velocity: affine consistent with the data, world transform preserved
# ----------------------------------------------------------------------

SHAPES = [(16, 16, 16), (17, 15, 9), (16, 16, 2), (17, 15, 1), (20, 3, 5)]


def _check_world(src, dst):
    """`dst` represents the transform of `src` at its own voxels"""
    vox = spatial.identity_grid(list(dst.shape), dtype=dt)
    vox_src = spatial.affine_matvec(
        torch.linalg.solve(src.affine, dst.affine), vox)
    world_src = spatial.affine_matvec(src.affine,
                                      vox_src + spline(src.dat, vox_src))
    world_dst = spatial.affine_matvec(dst.affine, vox + dst.dat)
    assert torch.allclose(world_src, world_dst, atol=1e-10)


@pytest.mark.parametrize('shape', SHAPES)
def test_displacement_transfer_world(shape):
    """`Displacement.{down,up}sample` resample the data and update the
    affine with the same voxel-to-voxel map, and rescale the displacement
    accordingly: the world-space transform at the new voxels is the one
    of the source field (including odd sizes, singleton axes and axes
    that become singleton)."""
    aff = spatial.affine_default(shape, VX, dtype=dt)
    disp = objects.Displacement(smooth_field(shape), affine=aff)
    coarse = disp.downsample()
    assert list(coarse.shape) == [int(math.ceil(s / 2)) for s in shape]
    _check_world(disp, coarse)
    # back to the original shape, as in `run_pyramid`
    fine = coarse.upsample(shape=list(shape))
    _check_world(coarse, fine)
    assert torch.allclose(fine.affine, aff, atol=1e-10)
    _check_world(disp, disp.upsample())


@pytest.mark.parametrize('shape', SHAPES)
def test_displacement_transfer_fov(shape):
    """The edges convention keeps the world-space field of view"""
    aff = spatial.affine_default(shape, VX, dtype=dt)
    disp = objects.Displacement(smooth_field(shape), affine=aff)
    for out in (disp.downsample(), disp.downsample().downsample()):
        trf = torch.linalg.solve(aff, out.affine)
        lo = spatial.affine_matvec(trf, torch.full([3], -0.5, dtype=dt))
        hi = spatial.affine_matvec(
            trf, torch.as_tensor(out.shape, dtype=dt) - 0.5)
        assert torch.allclose(lo, torch.full([3], -0.5, dtype=dt))
        assert torch.allclose(hi, torch.as_tensor(shape, dtype=dt) - 0.5)


# ----------------------------------------------------------------------
#   Image pyramid: affine consistent with the data
# ----------------------------------------------------------------------

@pytest.mark.parametrize('method', ['gauss', 'average', 'stride'])
@pytest.mark.parametrize('shape', [(16, 16, 16), (17, 15, 9), (17, 15, 1)])
def test_image_pyramid_affine(shape, method):
    """A ramp along each axis, pyramided, takes at each coarse voxel the
    fine coordinate that the level's affine assigns to it (interior
    voxels: the boundary condition bends the ramp at the border).
    The median pyramid is left out: `math.median` returns the lower of
    the two middle values, which biases intensities, not positions."""
    aff = spatial.affine_default(shape, VX, dtype=dt)
    for d in range(3):
        if shape[d] == 1:
            continue
        ramp = spatial.identity_grid(shape, dtype=dt)[..., d][None]
        pyr = objects.ImagePyramid(ramp, levels=[0, 1, 2], affine=aff,
                                   dim=3, method=method)
        for level in (1, 2):
            img = pyr[level]
            assert list(img.dat.shape[-3:]) == list(img.shape)
            vox = spatial.identity_grid(list(img.shape), dtype=dt)
            pred = spatial.affine_matvec(
                torch.linalg.solve(aff, img.affine), vox)[..., d]
            n = img.shape[d]
            if n <= 4:
                continue
            sl = [slice(None)] * 3
            sl[d] = slice(1, n - 1)
            sl = tuple(sl)
            assert torch.allclose(img.dat[0][sl], pred[sl], atol=1e-3)


# ----------------------------------------------------------------------
#   Level offsets between the two pyramids (measured, not required)
# ----------------------------------------------------------------------

class _Recorder:
    """Stands for PairwiseRegister in `run_pyramid`: records the
    lattices of each level and fits nothing"""
    records = []

    def __init__(self, loss, affine, nonlin, optim, **kwargs):
        fix = list(loss)[0].fixed
        self.records.append((fix.affine.clone(), nonlin.affine.clone(),
                             list(nonlin.shape)))
        self.figure = None

    def fit(self):
        pass


class _NoOptim:
    def reset_state(self):
        pass


def make_losses(shape, method, levels=(0, 1, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    aff = spatial.affine_default(shape, VX, dtype=dt)
    fix = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 2)
    mov = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 2)
    fix = objects.ImagePyramid(fix, levels=list(levels), affine=aff,
                               dim=3, method=method)
    mov = objects.ImagePyramid(mov, levels=list(levels), affine=aff,
                               dim=3, method=method)
    loss = objects.SumSimilarity([objects.Similarity(MSE(lam=1, dim=3), mov, fix)])
    return sequential_pyramid(loss), aff


def nonlin_model(shape, aff, name='smalldef'):
    nl = objects.NonLinModel.subclass(name)(penalty=dict(
        absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
    nl.set_dat(list(shape), affine=aff, dtype=dt)
    return nl


@pytest.mark.parametrize('method,offsets', [
    ('gauss', (-0.25, -0.375)), ('stride', (-0.25, -0.375)),
    ('average', (0., 0.))])
def test_level_offsets(monkeypatch, method, offsets):
    """Through `run_pyramid` (even sizes): the velocity keeps its world
    field of view at every level, and its voxels are offset from the
    fixed image's by -(1 - 2**-n)/2 coarse voxels at level n with the
    gaussian / stride pyramid (0 with average pooling). The `right`
    affine accounts for this offset."""
    monkeypatch.setattr(pairwise_run, 'PairwiseRegister', _Recorder)
    _Recorder.records = []
    shape = (16, 16, 16)
    losses, aff = make_losses(shape, method)
    nonlin = nonlin_model(shape, aff)
    pairwise_run.run_pyramid(losses, None, nonlin, _NoOptim(), verbose=0)
    # records: levels 2, 1, 0 (coarse to fine)
    assert len(_Recorder.records) == 3
    nl0 = _Recorder.records[-1][1]
    for (aff_fix, aff_nl, _), off in zip(_Recorder.records[:2],
                                         offsets[::-1]):
        trf = torch.linalg.solve(aff_nl, aff_fix)
        assert torch.allclose(trf[:3, :3], torch.eye(3, dtype=dt))
        assert torch.allclose(trf[:3, -1], torch.full([3], off, dtype=dt))
        # world field of view of the velocity is kept
        fov = torch.linalg.solve(aff, aff_nl)
        lo = spatial.affine_matvec(fov, torch.full([3], -0.5, dtype=dt))
        assert torch.allclose(lo, torch.full([3], -0.5, dtype=dt))
    assert torch.allclose(nl0, aff)
    assert list(nonlin.shape) == list(shape)


# ----------------------------------------------------------------------
#   Gradients stay exact at coarse levels (misaligned lattices,
#   coarse singleton axis)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(12, 12, 12), (12, 12, 2)])
def test_coarse_level_grad_fd(shape):
    losses, aff = make_losses(shape, 'gauss', levels=(0, 1))
    nonlin = nonlin_model(shape, aff)
    nonlin.downsample_(shape=[int(math.ceil(s / 2)) for s in shape])
    nonlin.dat.dat.copy_(0.3 * smooth_field(nonlin.shape, seed=3))
    step = PairwiseRegisterStep(losses[0], None, nonlin, verbose=0)
    x = nonlin.dat.dat.clone()
    _, g = step.do_vel(x, grad=True, in_line_search=True)
    gen = torch.Generator().manual_seed(2)
    dirs = [g] + [smooth(torch.randn([3, *x.shape[:-1]], dtype=dt,
                                     generator=gen), 2).movedim(0, -1)
                  for _ in range(2)]
    eps = 1e-6
    for d in dirs:
        d = d / d.norm()
        fd = (step.do_vel(x + eps * d, in_line_search=True) -
              step.do_vel(x - eps * d, in_line_search=True)) / (2 * eps)
        an = (g * d).sum()
        assert abs(fd - an) < 1e-5 * g.norm(), (fd.item(), an.item())
