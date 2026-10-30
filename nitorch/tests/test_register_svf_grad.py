"""SVF velocity gradients in `nitorch register` when the fixed lattice
differs from the lattice of the nonlinear field.

The forward pass samples the displacement field at the fixed voxels with
zero boundary conditions (the displacement is zero outside the field of
view of the nonlinear lattice). The gradient must be pushed back to the
nonlinear lattice with the same boundary conditions."""
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep

dt = torch.double


def smooth(x, n):
    """Box-smooth a (C, *spatial) tensor with circular boundaries"""
    for _ in range(n):
        for d in range(1, x.dim()):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    return x


def translation(shift):
    aff = torch.eye(4, dtype=dt)
    aff[:3, -1] = torch.as_tensor(shift, dtype=dt)
    return aff


def make_step(nl_shape, nl_affine, shape=(12, 12, 12), seed=0):
    g = torch.Generator().manual_seed(seed)
    affine = spatial.affine_default(shape, 1., dtype=dt)
    fix = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 2)
    mov = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 2)
    fix = objects.Image(fix, affine=affine)
    mov = objects.Image(mov, affine=affine)
    nonlin = objects.SVFModel(steps=8, penalty=dict(
        absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
    nonlin.set_dat(list(nl_shape), affine=affine @ nl_affine, dtype=dt)
    vel = smooth(torch.randn([3, *nl_shape], dtype=dt, generator=g), 3)
    nonlin.dat.dat.copy_(vel.movedim(0, -1))
    sim = objects.SumSimilarity([objects.Similarity(MSE(lam=1, dim=3), mov, fix)])
    return PairwiseRegisterStep(sim, None, nonlin, verbose=0), nonlin


def check_directional(step, x, tol):
    _, g = step.do_vel(x, grad=True, in_line_search=True)
    gen = torch.Generator().manual_seed(2)
    dirs = [g] + [smooth(torch.randn(x.movedim(-1, 0).shape, dtype=dt,
                                     generator=gen), 2).movedim(0, -1)
                  for _ in range(3)]
    eps = 1e-6
    for d in dirs:
        d = d / d.norm()
        fd = (step.do_vel(x + eps * d, in_line_search=True) -
              step.do_vel(x - eps * d, in_line_search=True)) / (2 * eps)
        an = (g * d).sum()
        assert abs(fd - an) < tol * g.norm(), (fd.item(), an.item(), g.norm().item())


@pytest.mark.parametrize('shape, shift', [
    ((12, 12, 8), (0, 0, 2)),       # field shorter than the image along z
    ((8, 10, 12), (3, -1, 0)),      # shifted field of view
])
def test_vel_grad_smaller_fov(shape, shift):
    """Nonlinear field of view smaller than the fixed image, with the
    same voxels: fixed voxels outside the field must not contribute to
    the gradient (master pushed them with circulant boundaries)."""
    step, nonlin = make_step(shape, translation(shift))
    check_directional(step, nonlin.dat.dat.clone(), 0.05)


@pytest.mark.xfail(strict=True, reason=(
    'Not fixed yet: the image gradient is sampled at the nodes of the '
    'nonlinear lattice instead of at the fixed voxels (see #114).'))
def test_vel_grad_coarser_field():
    """Nonlinear field with 2 mm voxels on 1 mm images (same field of view)"""
    shape = (12, 12, 12)
    aff = spatial.affine_default(shape, 1., dtype=dt)
    aff_nl = spatial.affine_default([7, 7, 7], 2., dtype=dt)
    step, nonlin = make_step([7, 7, 7], torch.linalg.solve(aff, aff_nl), shape)
    check_directional(step, nonlin.dat.dat.clone(), 0.05)
