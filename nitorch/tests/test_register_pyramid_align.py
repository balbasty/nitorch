"""Alignment of the nonlinear lattice with the image pyramid (#114).

The image pyramid (gaussian / stride) puts coarse voxel `j` on fine voxel
`2*j` (`ceil(N/2)` voxels); the pooling pyramids (average / median) on
`2*j + 0.5`. The velocity must follow the same convention, so that at
each level its voxel centres are those of the image lattice. It used to
be resized with `anchor='edges'`, i.e., offset by -0.25 / -0.375 voxel at
levels 1 / 2 with respect to the gaussian pyramid, and the zero boundary
of the fixed-to-nonlin sampling then dropped part of the border gradient
(all of the voxels of a singleton axis are border voxels)."""
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


def make_losses(shape, method, levels=(0, 1, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    aff = spatial.affine_default(shape, VX, dtype=dt)
    fix = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 2)
    mov = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 2)
    fix = objects.ImagePyramid(fix, levels=list(levels), affine=aff,
                               dim=3, method=method)
    mov = objects.ImagePyramid(mov, levels=list(levels), affine=aff,
                               dim=3, method=method)
    loss = objects.SumSimilarity([objects.Similarity(MSE(dim=3), mov, fix)])
    return sequential_pyramid(loss), aff


def nonlin_model(shape, aff, name='smalldef'):
    nl = objects.NonLinModel.subclass(name)(penalty=dict(
        absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
    nl.set_dat(list(shape), affine=aff, dtype=dt)
    return nl


# ----------------------------------------------------------------------
#   (a) per level, the velocity lattice is the image lattice
# ----------------------------------------------------------------------

class _Recorder:
    """Stands for PairwiseRegister in `run_pyramid`: records the
    lattices of each level and fits nothing"""
    records = []

    def __init__(self, loss, affine, nonlin, optim, **kwargs):
        fix = list(loss)[0].fixed
        self.records.append((fix.affine.clone(), list(fix.shape),
                             nonlin.affine.clone(), list(nonlin.shape)))
        self.figure = None

    def fit(self):
        pass


class _NoOptim:
    def reset_state(self):
        pass


SHAPES = [(16, 16, 16), (17, 15, 1), (16, 16, 2), (20, 2, 9)]


@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.parametrize('method', ['gauss', 'stride', 'average'])
@pytest.mark.parametrize('shape', SHAPES)
def test_level_lattices_aligned(monkeypatch, shape, method, frozen):
    monkeypatch.setattr(pairwise_run, 'PairwiseRegister', _Recorder)
    _Recorder.records = []
    losses, aff = make_losses(shape, method)
    nonlin = nonlin_model(shape, aff)
    nonlin.frozen = frozen
    pairwise_run.run_pyramid(losses, None, nonlin, _NoOptim(), verbose=0)
    assert len(_Recorder.records) == 3
    for aff_fix, shp_fix, aff_nl, shp_nl in _Recorder.records:
        # image voxel -> nonlin voxel
        trf = torch.linalg.solve(aff_nl, aff_fix)
        # same voxel centres: no offset, ...
        assert trf[:3, -1].abs().max() < 1e-8, trf
        lin = trf[:3, :3]
        assert (lin - torch.diag(lin.diagonal())).abs().max() < 1e-8
        for d in range(3):
            if shp_nl[d] > 1:
                # ... and same voxel size along non-singleton axes
                assert abs(lin[d, d] - 1) < 1e-8, (d, trf)
            # the nonlin field of view covers the image
            assert shp_nl[d] >= shp_fix[d]
    # the final field is back on the original lattice
    assert list(nonlin.shape) == list(shape)
    assert torch.allclose(nonlin.affine, aff)


def test_pyramid_shift():
    for method, shift in (('gauss', 0), ('stride', 0),
                          ('average', 0.5), ('median', 0.5)):
        losses, _ = make_losses((8, 8, 8), method, levels=(0, 1))
        assert abs(pairwise_run._pyramid_shift(losses) - shift) < 1e-8


# ----------------------------------------------------------------------
#   (b) coarse <-> fine transfers of the velocity
# ----------------------------------------------------------------------

def smooth_field(shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn([3, *shape], dtype=dt, generator=g)
    return smooth(v, 3).movedim(0, -1).contiguous()


@pytest.mark.parametrize('shape', [(16, 16, 16), (17, 15, 1), (16, 16, 2)])
def test_downsample_is_subsampling(shape):
    """With the gaussian-pyramid convention, coarse voxel `j` is fine
    voxel `2*j`: downsampling picks these voxels (cubic splines
    interpolate their nodes) and expresses them in coarse voxels."""
    aff = spatial.affine_default(shape, VX, dtype=dt)
    disp = objects.Displacement(smooth_field(shape), affine=aff)
    coarse = disp.downsample()
    cshape = [int(math.ceil(s / 2)) for s in shape]
    assert list(coarse.shape) == cshape
    scale = torch.as_tensor([1. if s == 1 else 2. for s in shape], dtype=dt)
    sub = disp.dat[tuple(slice(None, None, 2) if s > 1 else slice(None)
                         for s in shape)]
    assert torch.allclose(coarse.dat, sub / scale, atol=1e-10)
    # affine: coarse voxel j -> fine voxel scale * j
    trf = torch.linalg.solve(aff, coarse.affine)
    assert torch.allclose(trf[:3, :3], torch.diag(scale), atol=1e-10)
    assert trf[:3, -1].abs().max() < 1e-10


def _world_warp(disp, vox):
    """World coordinates of x + d(x) for voxel coordinates `vox`, where
    the displacement is interpolated (cubic splines, circular) in the
    voxels of `disp`."""
    d = spatial.grid_pull(disp.dat.movedim(-1, 0)[None], vox[None],
                          interpolation=3, bound='dft', extrapolate=True,
                          prefilter=True)[0].movedim(0, -1)
    return spatial.affine_matvec(disp.affine, vox + d)


@pytest.mark.parametrize('shape', [(9, 8, 7), (9, 8, 1)])
def test_upsample_reproduces_transform(shape):
    """A coarse field upsampled onto the fine lattice represents the same
    transform: exactly on the fine voxels that are coarse voxels, and up
    to interpolation accuracy elsewhere."""
    aff = spatial.affine_default([2 * s - 1 if s > 1 else 1 for s in shape],
                                 VX, dtype=dt)
    fshape = [2 * s - 1 if s > 1 else 1 for s in shape]
    disp = objects.Displacement(smooth_field(fshape), affine=aff)
    coarse = disp.downsample()
    assert list(coarse.shape) == list(shape)
    fine = coarse.upsample(shape=fshape)
    assert torch.allclose(fine.affine, aff, atol=1e-10)
    vox = spatial.identity_grid(fshape, dtype=dt)
    vox_c = spatial.affine_matvec(torch.linalg.solve(coarse.affine, aff), vox)
    world_c = _world_warp(coarse, vox_c)
    world_f = spatial.affine_matvec(aff, vox + fine.dat)
    nodes = tuple(slice(None, None, 2) if s > 1 else slice(None)
                  for s in fshape)
    assert torch.allclose(world_f[nodes], world_c[nodes], atol=1e-10)
    # in between nodes: same cubic spline, sampled at the same points
    assert torch.allclose(world_f, world_c, atol=1e-8)
    # and back down: the round trip is exact
    assert torch.allclose(fine.downsample().dat, coarse.dat, atol=1e-10)


def test_transfer_pooling_convention():
    """With `shift=0.5` (average / median pyramids), coarse voxel j sits on
    fine voxel 2j + 0.5"""
    shape = (8, 8, 8)
    aff = spatial.affine_default(shape, VX, dtype=dt)
    disp = objects.Displacement(smooth_field(shape), affine=aff)
    coarse = disp.downsample(shift=0.5)
    trf = torch.linalg.solve(aff, coarse.affine)
    assert torch.allclose(trf[:3, -1], torch.full([3], 0.5, dtype=dt))
    fine = coarse.upsample(shape=shape, shift=0.5)
    assert torch.allclose(fine.affine, aff, atol=1e-10)


def test_legacy_anchor():
    """`anchor=...` selects the former `resize_grid` transfer"""
    shape = (8, 8, 8)
    aff = spatial.affine_default(shape, VX, dtype=dt)
    disp = objects.Displacement(smooth_field(shape), affine=aff)
    coarse = disp.downsample(anchor='e')
    ref, ref_aff = spatial.resize_grid(
        disp.dat, shape=[4, 4, 4], affine=aff, type='displacement',
        anchor='e', interpolation=3, bound='dft', extrapolate=True)
    assert torch.allclose(coarse.dat, ref)
    assert torch.allclose(coarse.affine, ref_aff)


# ----------------------------------------------------------------------
#   (c) border gradient
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(12, 12, 12), (12, 12, 2), (12, 12, 1)])
def test_border_gradient_not_lost(shape):
    """At a coarse level, the velocity gradient is the image-lattice
    gradient: nothing is dropped by the zero boundary, at the border nor
    along a singleton axis."""
    losses, aff = make_losses(shape, 'gauss', levels=(0, 1))
    level1 = losses[0]
    fix1 = list(level1)[0].fixed
    shift = pairwise_run._pyramid_shift(losses)
    shapes = [list(shape), [int(math.ceil(s / 2)) for s in shape]]
    nonlin = nonlin_model(shape, aff)
    nonlin.downsample_(**pairwise_run._level_map(shapes, 1, shift))

    # reference: a field that lives on the image lattice of level 1
    ref = nonlin_model(fix1.shape, fix1.affine)

    def grad(nl):
        step = PairwiseRegisterStep(level1, None, nl, verbose=0)
        x = torch.zeros_like(nl.dat.dat)
        _, g = step.do_vel(x, grad=True, in_line_search=True)
        # gradient wrt displacements in mm
        return g / nl.voxel_size.to(g)

    g, g_ref = grad(nonlin), grad(ref)
    assert g.shape == g_ref.shape
    assert g_ref.abs().sum() > 0
    # mass along each in-plane axis, on the first and last slabs, and
    # everywhere
    for d in range(3):
        if shape[d] == 1:
            continue
        for sl in (0, -1):
            gb = g.select(d, sl)
            gb_ref = g_ref.select(d, sl)
            assert torch.allclose(gb.sum(), gb_ref.sum(), rtol=1e-8)
    assert torch.allclose(g, g_ref, rtol=1e-8, atol=1e-12)
