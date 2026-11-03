"""`spatial.shoot(..., forward_method='fixedpoint')` computes the forward
transform as the fixed-point inverse of the (more accurate) inverse
transform, which makes both transforms inverse-consistent (#171).
The default (`forward_method='compose'`) must be unchanged."""
import pytest
import torch

from nitorch import spatial
from nitorch.core import py, linalg
from nitorch.spatial import _shoot

PEN = dict(absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2))


def make_vel(shape, amp, batch=None, seed=0, dtype=torch.float32):
    """Smooth random velocity (Greens function applied to random
    momentum, with `register`'s default penalties), of max norm `amp`."""
    dim = len(shape)
    factor = 1. / py.prod(shape)
    g = torch.Generator().manual_seed(seed)
    nbatch = batch or 1
    mom = torch.randn([nbatch, *shape, dim], generator=g, dtype=torch.float64)
    greens = spatial.greens(shape, **PEN, factor=factor, dtype=torch.float64)
    vel = spatial.greens_apply(mom, greens, factor=factor)
    vel *= amp / vel.square().sum(-1).sqrt().flatten(1).max(-1).values \
        .reshape([nbatch] + [1] * (dim + 1))
    vel = vel.to(dtype)
    return vel if batch else vel[0]


def shoot_prm(shape, **kwargs):
    return dict(PEN, factor=1. / py.prod(shape), **kwargs)


def compose(a, b):
    """Displacement of (id + a) o (id + b)"""
    dim = a.shape[-1]
    id = spatial.identity_grid(a.shape[-dim-1:-1], dtype=a.dtype)
    return b + _shoot._pull_grid(a, id + b, bound='dft', extrapolate=True,
                                 interpolation=1)


def maxnorm(d):
    return d.square().sum(-1).sqrt().max().item()


def reference_shoot(vel, absolute, membrane, bending, lame, factor,
                    return_inverse=False, displacement=False, steps=8,
                    fast=True):
    """Verbatim copy of the algorithm of `spatial.shoot` before the
    `forward_method` option was added (greens=None, voxel_size=1)."""
    from nitorch.core import utils
    backend = utils.backend(vel)
    dim = vel.shape[-1]
    shape = vel.shape[-dim-1:-1]
    prm = dict(absolute=absolute, membrane=membrane, bending=bending,
               lame=lame, voxel_size=1, factor=factor)
    pull_prm = dict(bound='dft', interpolation=1, extrapolate=True)
    greens = _shoot._greens(shape, **prm, **backend)
    id = spatial.identity_grid(shape, **backend)
    mom = mom0 = _shoot.regulariser_grid(vel, **prm, bound='dft')
    vel = vel / steps
    disp = -vel
    if return_inverse or not fast:
        idisp = vel.clone()
    for i in range(1, abs(steps)):
        if fast:
            jac = _shoot._jacobian(-vel)
            mom = linalg.matvec(jac.transpose(-1, -2), mom)
            mom = _shoot._push_grid(mom, id + vel, **pull_prm)
        else:
            jac = _shoot._jacobian(idisp).inverse()
            mom = linalg.matvec(jac.transpose(-1, -2), mom0)
            mom = _shoot._push_grid(mom, id + idisp, **pull_prm)
        vel = _shoot.greens_apply(mom, greens, factor=factor, voxel_size=1)
        vel = vel.div_(steps)
        disp = _shoot._pull_grid(disp, id - vel, **pull_prm).sub_(vel)
        if return_inverse or not fast:
            idisp += _shoot._pull_grid(vel, id + idisp, **pull_prm)
    if not displacement:
        disp += id
        if return_inverse:
            idisp += id
    return (disp, idisp) if return_inverse else disp


# ----------------------------------------------------------------------
# default unchanged (bit for bit)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(12, 10), (8, 7, 6)])
@pytest.mark.parametrize('batch', [None, 2])
@pytest.mark.parametrize('return_inverse', [False, True])
@pytest.mark.parametrize('displacement', [False, True])
@pytest.mark.parametrize('fast', [True, False])
def test_default_is_unchanged(shape, batch, return_inverse, displacement,
                              fast):
    vel = make_vel(shape, 3., batch)
    prm = shoot_prm(shape, return_inverse=return_inverse,
                    displacement=displacement, steps=4, fast=fast)
    ref = reference_shoot(vel, **prm)
    for kwargs in ({}, {'forward_method': 'compose'}):
        out = spatial.shoot(vel, **prm, **kwargs)
        if return_inverse:
            assert torch.equal(out[0], ref[0])
            assert torch.equal(out[1], ref[1])
        else:
            assert torch.equal(out, ref)


def test_invalid_forward_method():
    with pytest.raises(ValueError):
        spatial.shoot(make_vel((8, 8), 1.), forward_method='inverse')


# ----------------------------------------------------------------------
# fixed-point forward transform
# ----------------------------------------------------------------------
@pytest.mark.parametrize('shape,amp', [((32, 32), 8.), ((16, 16, 16), 10.)])
def test_fixedpoint_improves_consistency(shape, amp):
    vel = make_vel(shape, amp)
    prm = shoot_prm(shape, return_inverse=True, displacement=True)
    psi0, phi0 = spatial.shoot(vel, **prm)
    psi, phi = spatial.shoot(vel, **prm, forward_method='fixedpoint')
    # same inverse
    assert torch.equal(phi, phi0)
    # psi is (nearly) the inverse of phi, on both sides
    err0 = maxnorm(compose(psi0, phi0)), maxnorm(compose(phi0, psi0))
    err = maxnorm(compose(psi, phi)), maxnorm(compose(phi, psi))
    assert err0[0] > 0.05 and err0[1] > 0.05   # the test is meaningful
    assert err[0] < 0.4 * err0[0]
    assert err[1] < 0.1 * err0[1]


@pytest.mark.parametrize('shape', [(16, 14), (10, 9, 8)])
@pytest.mark.parametrize('fast', [True, False])
def test_fixedpoint_outputs(shape, fast):
    vel = make_vel(shape, 4.)
    dim = len(shape)
    id = spatial.identity_grid(shape)
    prm = shoot_prm(shape, steps=6, fast=fast, forward_method='fixedpoint')
    psi_d, phi_d = spatial.shoot(vel, **prm, displacement=True,
                                 return_inverse=True)
    psi_g, phi_g = spatial.shoot(vel, **prm, displacement=False,
                                 return_inverse=True)
    psi_only = spatial.shoot(vel, **prm, displacement=True)
    assert psi_d.shape == phi_d.shape == psi_g.shape == vel.shape
    # transformation = identity + displacement
    assert torch.allclose(psi_g, psi_d + id, atol=1e-5)
    assert torch.allclose(phi_g, phi_d + id, atol=1e-5)
    # psi does not depend on whether the inverse is returned
    assert torch.equal(psi_only, psi_d)
    # same inverse as the default mode
    _, phi_ref = spatial.shoot(vel, **shoot_prm(shape, steps=6, fast=fast),
                               displacement=True, return_inverse=True)
    assert torch.equal(phi_d, phi_ref)
    # fixed point of psi = id - (phi - id) o psi
    assert maxnorm(compose(phi_d, psi_d)) < 1e-2
    assert psi_d.shape[-1] == dim


@pytest.mark.parametrize('shape', [(16, 14), (8, 9, 10)])
def test_fixedpoint_batched(shape):
    vel = make_vel(shape, 4., batch=3)
    prm = shoot_prm(shape, steps=5, displacement=True, return_inverse=True,
                    forward_method='fixedpoint')
    psi, phi = spatial.shoot(vel, **prm)
    assert psi.shape == phi.shape == vel.shape
    for b in range(3):
        psi1, phi1 = spatial.shoot(vel[b], **prm)
        assert torch.allclose(psi[b], psi1, atol=1e-5)
        assert torch.allclose(phi[b], phi1, atol=1e-5)


def test_fixedpoint_iterations_and_tolerance():
    shape = (24, 24)
    vel = make_vel(shape, 6.)
    prm = shoot_prm(shape, displacement=True, return_inverse=True)
    _, phi = spatial.shoot(vel, **prm)
    # zero iterations: initial guess psi = 2*id - phi
    psi, _ = spatial.shoot(vel, **prm, forward_method='fixedpoint',
                           fixedpoint_iter=0)
    assert torch.equal(psi, -phi)
    # explicit iterations of the fixed-point equation
    id = spatial.identity_grid(shape)
    psi_ref = -phi
    for _ in range(3):
        psi_ref = -_shoot._pull_grid(phi, id + psi_ref, bound='dft',
                                     interpolation=1, extrapolate=True)
    psi, _ = spatial.shoot(vel, **prm, forward_method='fixedpoint',
                           fixedpoint_iter=3)
    assert torch.allclose(psi, psi_ref, atol=1e-6)
    # a huge tolerance stops after the first iteration
    psi1, _ = spatial.shoot(vel, **prm, forward_method='fixedpoint',
                            fixedpoint_iter=1)
    psi, _ = spatial.shoot(vel, **prm, forward_method='fixedpoint',
                           fixedpoint_iter=10, fixedpoint_tol=1e3)
    assert torch.equal(psi, psi1)
    # a tiny tolerance runs all iterations
    psi10, _ = spatial.shoot(vel, **prm, forward_method='fixedpoint',
                             fixedpoint_iter=10)
    psi, _ = spatial.shoot(vel, **prm, forward_method='fixedpoint',
                           fixedpoint_iter=10, fixedpoint_tol=1e-12)
    assert torch.equal(psi, psi10)
    # more iterations -> closer to the fixed point
    assert maxnorm(compose(phi, psi10)) < maxnorm(compose(phi, psi1))


def test_fixedpoint_double_and_autograd():
    shape = (12, 12)
    vel = make_vel(shape, 3., dtype=torch.float64).requires_grad_()
    psi = spatial.shoot(vel, **shoot_prm(shape, steps=4),
                        displacement=True, forward_method='fixedpoint',
                        fixedpoint_tol=1e-8)
    assert psi.dtype == torch.float64
    psi.square().sum().backward()
    assert vel.grad is not None and torch.isfinite(vel.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_fixedpoint_cuda():
    shape = (10, 9, 8)
    vel = make_vel(shape, 4.)
    prm = shoot_prm(shape, steps=4, displacement=True, return_inverse=True,
                    forward_method='fixedpoint')
    psi, phi = spatial.shoot(vel.cuda(), **prm)
    psi_cpu, phi_cpu = spatial.shoot(vel, **prm)
    assert psi.is_cuda
    assert torch.allclose(psi.cpu(), psi_cpu, atol=1e-4)
    assert torch.allclose(phi.cpu(), phi_cpu, atol=1e-4)
