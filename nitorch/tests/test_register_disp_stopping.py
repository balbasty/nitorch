"""Early stopping on the size of an update (#266).

`OptimIterator(disp_tol=...)` stops when the accepted update is smaller
than `disp_tol`, as measured by a `step_size` hook that `PairwiseRegister`
sets: the RMS, over the fixed image, of the displacement the update
causes, in voxels of the current level (exact, closed form, for affine;
RMS of the displacement/velocity update for nonlinear models).
"""
import glob
import os
import numpy as np
import torch
import pytest

from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.optim.base import OptimIterator
from nitorch.tools.registration.optim.gd import GradientDescent
from nitorch.tools.registration.pairwise_impl import (
    PairwiseRegister, _affine_step_size, _nonlin_step_size)
from nitorch.tools.registration.pairwise_makeobj import (
    make_affine_optim, make_nonlin_optim,
    AFFINE_DISP_TOLERANCE, NONLIN_DISP_TOLERANCE)

dt = torch.double


# ----------------------------------------------------------------------
#   OptimIterator
# ----------------------------------------------------------------------

def _gd(disp_tol, tol=0, max_iter=50):
    def closure(x, grad=False):
        f = 0.5 * x.square().sum()
        return (f, x.clone()) if grad else f
    gd = GradientDescent(lr=0.5)
    gd.iter = OptimIterator(max_iter=max_iter, tol=tol, disp_tol=disp_tol,
                            step_size=lambda a, b: float((b - a).abs().max()))
    sizes = []
    orig = gd.step

    def step(*a, **k):
        out = orig(*a, **k)
        sizes.append(out[0].clone())
        return out
    gd.step = step
    gd.iter(torch.ones(1, dtype=dt), closure)
    return sizes


def test_stops_when_update_is_small():
    # x_k = 0.5^k: the update is 0.5^k; stops at the first one < 1e-3
    sizes = _gd(disp_tol=1e-3)
    assert len(sizes) == 10, len(sizes)          # 0.5^10 < 1e-3 <= 0.5^9


def test_zero_disables_displacement_criterion():
    assert len(_gd(disp_tol=0, max_iter=30)) == 30


def test_either_criterion_stops():
    # the loss criterion fires first here (diff of 0.5 x^2 < 1e-2)
    n_loss = len(_gd(disp_tol=0, tol=1e-2))
    assert n_loss < len(_gd(disp_tol=1e-6, tol=0))
    assert len(_gd(disp_tol=1e-6, tol=1e-2)) == n_loss


def test_no_hook_falls_back_to_rms_of_parameters():
    from nitorch.tools.registration.optim.base import rms_step
    gd = GradientDescent(lr=0.5)
    gd.iter = OptimIterator(max_iter=50, tol=0, disp_tol=1e-3)

    def closure(x, grad=False):
        f = 0.5 * x.square().sum()
        return (f, x.clone()) if grad else f
    x0 = torch.full([4], 2., dtype=dt)
    gd.iter(x0.clone(), closure)
    assert gd.iter.last_step_size < 1e-3
    # rms over all parameters, of lists and of parameter wrappers
    a, b = torch.zeros(3, dtype=dt), torch.full([3], 2., dtype=dt)
    assert rms_step(a, b) == 2.
    assert rms_step([a, a[:1]], [b, b[:1]]) == 2.
    assert abs(rms_step([a, a[:1]], [b, b[:1] + 2]) - 7 ** 0.5) < 1e-12

    class Wrapped:
        def __init__(self, dat):
            self.dat = dat
    assert rms_step(Wrapped(a), Wrapped(b)) == 2.


# ----------------------------------------------------------------------
#   step-size measures
# ----------------------------------------------------------------------

def _image(shape=(9, 10, 11), masked=True, vx=(0.8, 1.0, 1.2), shift=0.):
    aff = spatial.affine_default(shape, list(vx), dtype=dt)
    aff[:3, -1] += shift
    dat = torch.rand([1, *shape], dtype=dt)
    mask = None
    if masked:
        mask = torch.zeros([1, *shape], dtype=dt)
        mask[:, 2:7, 1:8, 3:10] = 1
    return objects.Image(dat, affine=aff, mask=mask)


def _affine_ref(affine, image, q0, q1, backward=False):
    """Explicit sum of squared displacements (in the image's voxels) and
    number of (masked) voxels."""
    x = spatial.affine_grid(image.affine, image.shape)          # world
    x = torch.cat([x, torch.ones_like(x[..., :1])], -1)[..., None]
    exp = affine.dat.iexp if backward else affine.dat.exp
    d = ((exp(q1) - exp(q0)).matmul(x))[..., :3, 0]
    d = d.matmul(torch.linalg.inv(image.affine[:3, :3]).T)      # world -> vox
    d = d.square().sum(-1)
    w = image.mask[0] if image.masked else torch.ones_like(d)
    return (d * w).sum(), w.sum()


def _rand_q(affine, seed=0):
    g = torch.Generator().manual_seed(seed)
    q0 = 0.05 * torch.randn(affine.parameters().shape, dtype=dt, generator=g)
    return q0, q0 + 0.01 * torch.randn(q0.shape, dtype=dt, generator=g)


@pytest.mark.parametrize('masked', [True, False])
@pytest.mark.parametrize('basis', ['rigid', 'affine'])
@pytest.mark.parametrize('backward', [False, True])
def test_affine_step_size_is_exact(masked, basis, backward):
    # anisotropic voxels: RAS displacements are converted to voxels per axis
    image = _image(masked=masked)
    affine = objects.AffineModel(basis).set_dat(dim=3, dtype=dt)
    q0, q1 = _rand_q(affine)
    size = _affine_step_size(affine, [(image, backward)])(q0, q1)
    ss, n = _affine_ref(affine, image, q0, q1, backward)
    ref = (ss / n).sqrt().item()
    assert abs(size - ref) <= 1e-10 * max(1, ref), (size, ref)
    # the parameter wrapper is accepted too
    p = affine.parameters()
    assert _affine_step_size(affine, [(image, backward)])(p.clone(), p) == 0


def test_affine_step_size_symmetric():
    # both images (forward on fixed, inverse on moving): RMS over all
    # their voxels, each in its own voxel coordinates
    fix = _image(vx=(0.8, 1.0, 1.2))
    mov = _image(shape=(8, 9, 10), vx=(1.5, 1.5, 1.5), masked=False, shift=3.)
    affine = objects.AffineModel('affine').set_dat(dim=3, dtype=dt)
    q0, q1 = _rand_q(affine, seed=1)
    size = _affine_step_size(affine, [(fix, False), (mov, True)])(q0, q1)
    ss1, n1 = _affine_ref(affine, fix, q0, q1, False)
    ss2, n2 = _affine_ref(affine, mov, q0, q1, True)
    ref = ((ss1 + ss2) / (n1 + n2)).sqrt().item()
    assert abs(size - ref) <= 1e-10 * max(1, ref), (size, ref)


@pytest.mark.parametrize('lattice', ['same', 'padded', 'coarser'])
def test_nonlin_step_size_is_masked_rms(lattice):
    image = _image()
    nonlin = objects.SmallDefModel()
    if lattice == 'same':
        nonlin.set_dat(image.shape, affine=image.affine, dtype=dt)
    elif lattice == 'padded':
        shape = [s + 4 for s in image.shape]
        aff = image.affine.matmul(spatial.affine_matrix_classic(
            torch.tensor([-2., -2., -2.], dtype=dt)))   # padded by 2 voxels
        nonlin.set_dat(shape, affine=aff, dtype=dt)
    else:
        # lattice voxels twice as large: a 1-voxel update moves the
        # image's voxels by 2 voxels, and covers 8 of them
        nonlin.set_dat(image.shape, affine=image.affine.matmul(
            torch.diag(torch.tensor([2., 2., 2., 1.], dtype=dt))), dtype=dt)
    g = torch.Generator().manual_seed(0)
    v0 = torch.randn(nonlin.parameters().shape, dtype=dt, generator=g)
    v1 = v0 + torch.randn(v0.shape, dtype=dt, generator=g)
    size = _nonlin_step_size(nonlin, [(image, False)])(v0, v1)
    d = (v1 - v0).square().sum(-1)
    m = image.mask[0]
    if lattice == 'padded':
        d = d[2:-2, 2:-2, 2:-2]                    # back on the image grid
        ref = ((d * m).sum() / m.sum()).sqrt()
    elif lattice == 'same':
        ref = ((d * m).sum() / m.sum()).sqrt()
    else:
        # lattice voxel i covers image voxels 2i-0.5 .. 2i+1.5: compare
        # with the lattice voxels whose centre falls inside the mask
        ref = None
    if ref is not None:
        assert abs(size - ref.item()) <= 1e-8 * ref.item(), (size, ref)
    else:
        same = _nonlin_step_size(nonlin, [(image, False)])
        # a uniform update of 1 lattice voxel along x moves voxels by 2
        u = torch.zeros_like(v0); u[..., 0] = 1
        assert abs(same(torch.zeros_like(v0), u) - 2) < 1e-8


def test_nonlin_step_size_symmetric_weights_by_voxel_count():
    # two images with different resolutions: each lattice voxel counts as
    # many target voxels as it covers (here, the coarse image's field of
    # view contains the whole lattice, so that no voxel is on its border)
    nonlin = objects.SmallDefModel()
    fine = _image(shape=(8, 8, 8), vx=(1., 1., 1.), masked=False)
    coarse = _image(shape=(6, 6, 6), vx=(2., 2., 2.), masked=False)
    centre = lambda img: img.affine.matmul(torch.tensor(
        [*[(n - 1) / 2 for n in img.shape], 1.], dtype=dt))
    assert torch.allclose(centre(fine), centre(coarse))     # same centre
    nonlin.set_dat(fine.shape, affine=fine.affine, dtype=dt)
    u = torch.zeros(nonlin.parameters().shape, dtype=dt)
    u[..., 0] = 1                              # 1 mm = 1 fine = 0.5 coarse
    size = _nonlin_step_size(nonlin, [(fine, False), (coarse, True)])(
        torch.zeros_like(u), u)
    n_fine, n_coarse = 8 ** 3, 8 ** 3 / 8     # lattice voxels in coarse units
    ref = ((n_fine * 1. + n_coarse * 0.25) / (n_fine + n_coarse)) ** 0.5
    assert abs(size - ref) < 1e-8, (size, ref)


# ----------------------------------------------------------------------
#   PairwiseRegister / CLI
# ----------------------------------------------------------------------

def _pair(shape=(12, 12, 12)):
    g = torch.Generator().manual_seed(0)
    x = torch.rand([1, *shape], dtype=dt, generator=g)
    for _ in range(2):
        for d in range(1, 4):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    aff = spatial.affine_default(shape, 1., dtype=dt)
    return objects.Image(x, affine=aff), objects.Image(x.roll(1, 3), affine=aff)


@pytest.mark.parametrize('kind', ['affine', 'nonlin'])
def test_register_stops_on_displacement(kind):
    def run(disp_tol):
        fix, mov = _pair()
        sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
        affine = nonlin = None
        if kind == 'affine':
            affine = objects.AffineModel('rigid', position='sym')
            affine.frozen = False
            optim = make_affine_optim('gn', 2, max_iter=30, tolerance=0,
                                      disp_tolerance=disp_tol)
        else:
            nonlin = objects.SVFModel(steps=4)
            nonlin.set_dat(fix.shape, affine=fix.affine, dtype=dt)
            nonlin.frozen = False
            optim = make_nonlin_optim('gn', 2, max_iter=30, tolerance=0,
                                      disp_tolerance=disp_tol, nonlin=nonlin)
        PairwiseRegister([sim], affine, nonlin, optim, verbose=0).fit()
        return optim.iter.last_step_size
    assert run(0) is None                          # not measured when off
    last = run(1e-2)
    assert last is not None and last < 1e-2


def test_defaults():
    assert make_affine_optim('gn', 2).iter.disp_tol == AFFINE_DISP_TOLERANCE
    assert make_nonlin_optim('gn', 2).iter.disp_tol == NONLIN_DISP_TOLERANCE
    assert make_nonlin_optim('gn', 2, disp_tolerance=0).iter.disp_tol == 0


def test_cli_option():
    from nitorch.cli.registration.register.parser import parser
    base = ['@loss', 'mse', '@@fix', 'f.nii', '@@mov', 'm.nii']
    opt = parser.parse(base + ['@nonlin', '@@optim', 'gn', '-d', '0.05'])
    assert opt.nonlin.optim.disp_tolerance == 0.05
    opt = parser.parse(base + ['@affine', '@@optim', 'gn'])
    assert opt.affine.optim.disp_tolerance is None     # model default
