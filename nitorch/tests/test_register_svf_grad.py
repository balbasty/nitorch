"""Velocity / displacement gradients in `nitorch register` when the fixed
lattice differs from the lattice of the nonlinear field.

The forward pass samples the displacement field at the fixed voxels, by
linear interpolation with zero boundary conditions (the displacement is
zero outside the field of view of the nonlinear lattice). The gradient
must be the adjoint of that: image gradients sampled at the fixed voxels,
then pushed back to the nonlinear lattice with the same boundary."""
import math
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration import utils as regutils
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep

dt = torch.double
SHAPE = (12, 12, 12)  # fixed and moving images, 1 mm


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


def rotation_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    aff = torch.eye(4, dtype=dt)
    aff[:2, :2] = torch.as_tensor([[c, -s], [s, c]], dtype=dt)
    return aff


def image_affine():
    return spatial.affine_default(SHAPE, 1., dtype=dt)


def nonlin_lattice(name):
    """(shape, vox2vox from the nonlinear lattice to the image lattice)"""
    aff = image_affine()
    if name == 'fov':       # field shorter than the image along z
        return (12, 12, 8), translation([0, 0, 2])
    if name == 'shift':     # shifted field of view
        return (8, 10, 12), translation([3, -1, 0])
    if name == 'subvoxel':  # smaller field of view, sub-voxel shift
        return (12, 12, 8), translation([0.3, -0.4, 2.5])
    if name == 'coarse':    # 2 mm field on 1 mm images, same field of view
        aff_nl = spatial.affine_default([7, 7, 7], 2., dtype=dt)
        return (7, 7, 7), torch.linalg.solve(aff, aff_nl)
    if name == 'rotated':   # 1.3 mm field, rotated and shifted
        aff_nl = spatial.affine_default([10, 10, 10], 1.3, dtype=dt)
        aff_nl = translation([0.5, -0.7, 0.3]) @ rotation_z(10) @ aff_nl
        return (10, 10, 10), torch.linalg.solve(aff, aff_nl)
    raise ValueError(name)


def make_step(model, lattice, kind='fwd', seed=0):
    g = torch.Generator().manual_seed(seed)
    affine = image_affine()
    fix = smooth(torch.rand([1, *SHAPE], dtype=dt, generator=g), 2)
    mov = smooth(torch.rand([1, *SHAPE], dtype=dt, generator=g), 2)
    fix = objects.Image(fix, affine=affine)
    mov = objects.Image(mov, affine=affine)
    nl_shape, nl_affine = nonlin_lattice(lattice)
    Model = objects.NonLinModel.subclass(model)
    nonlin = Model(steps=8, penalty=dict(
        absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
    nonlin.set_dat(list(nl_shape), affine=affine @ nl_affine, dtype=dt)
    vel = smooth(torch.randn([3, *nl_shape], dtype=dt, generator=g), 3)
    nonlin.dat.dat.copy_(vel.movedim(0, -1))
    sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
    if kind == 'fwd':
        sim = objects.SumSimilarity([sim])
    elif kind == 'bwd':
        sim = objects.SumSimilarity([sim.reverse()])
    else:
        sim = sim.symmetrize()
    return PairwiseRegisterStep(sim, None, nonlin, verbose=0), nonlin


def check_directional(step, x, tol):
    """Directional finite differences along the gradient and along
    three smooth random directions"""
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


# SVF gradients go through the (approximate) scaling-and-squaring
# backward pass, so they are only checked to 5%. Small-deformation
# gradients are exact.
TOL = dict(svf=0.05, smalldef=1e-5)
LATTICES = ['fov', 'shift', 'subvoxel', 'coarse', 'rotated']


@pytest.mark.parametrize('lattice', LATTICES)
@pytest.mark.parametrize('model', ['svf', 'smalldef'])
def test_vel_grad_lattice(model, lattice):
    step, nonlin = make_step(model, lattice)
    check_directional(step, nonlin.dat.dat.clone(), TOL[model])


@pytest.mark.parametrize('kind', ['bwd', 'sym'])
@pytest.mark.parametrize('model', ['svf', 'smalldef'])
def test_vel_grad_lattice_symmetric(model, kind):
    """Backward and symmetric losses: the backward term is pushed from
    the lattice of the moving image."""
    step, nonlin = make_step(model, 'coarse', kind)
    # smalldef: the backward gradient assumes inv(Id + D) = Id - D, which
    # is off by about 12% here, even when all lattices are the same.
    tol = 0.05 if model == 'svf' else 0.2
    check_directional(step, nonlin.dat.dat.clone(), tol)


def forward_loss_smalldef(step, disp):
    """Autograd-friendly copy of the forward pass of `do_vel` (one
    forward term, no affine, small deformation)."""
    loss, = step.losses
    fixed, moving, nonlin = loss.fixed, loss.moving, step.nonlin
    right = torch.linalg.solve(nonlin.affine, fixed.affine)
    left = torch.linalg.solve(moving.affine, nonlin.affine)
    right = spatial.affine_grid(right, fixed.shape)
    phi = regutils.smart_pull_grid(disp, right, bound='zero') + right
    phi = spatial.affine_matvec(left, phi)
    return loss.loss.loss(moving.pull(phi), fixed.dat, dim=3)


@pytest.mark.parametrize('lattice', ['coarse', 'rotated'])
def test_smalldef_grad_autograd(lattice):
    """The small-deformation gradient is the exact gradient of the
    discretized objective"""
    step, nonlin = make_step('smalldef', lattice)
    x = nonlin.dat.dat.clone()
    ll, g = step.do_vel(x.clone(), grad=True, in_line_search=True)
    x.requires_grad_()
    ll_ref = forward_loss_smalldef(step, x)
    ll_ref.backward()
    assert torch.allclose(ll, ll_ref, rtol=1e-10, atol=0)
    assert (g - x.grad).norm() < 1e-8 * x.grad.norm()


@pytest.mark.parametrize('model', ['svf', 'smalldef'])
def test_hess_lattice(model):
    """The pushed Gauss-Newton Hessian keeps its shape and is PSD on
    its diagonal"""
    step, nonlin = make_step(model, 'rotated')
    _, g, h = step.do_vel(nonlin.dat.dat.clone(), grad=True, hess=True,
                          in_line_search=True)
    assert g.shape == nonlin.dat.dat.shape
    assert h.shape == (*nonlin.shape, 6)
    assert torch.isfinite(h).all()
    assert (h[..., :3] >= 0).all()


def _push_grad_fixed_master(g, h, moving, disp, left, right, shape):
    """Reference: the formula of `_push_grad_fixed` before the left
    rotation was moved after the chain rule (rotate all K channels)."""
    dim = disp.shape[-1]
    right = objects._right_grid(right, g.shape[-dim:])
    psi = regutils.smart_pull_grid(disp, right, bound='zero') + right
    if left is not None:
        psi = spatial.affine_matvec(left, psi)
    mugrad = objects._rotate_grad(moving.pull_grad(psi, rotate=False), left)
    kwargs = dict(shape=shape, bound='zero', extrapolate=True)
    g = regutils.jg(mugrad, g)
    g = regutils.smart_push(g.movedim(-1, 0), right, **kwargs)
    h = regutils.jhj(mugrad, h)
    h = regutils.smart_push(h.movedim(-1, 0), right, **kwargs)
    return g, h


def _push_inputs(lattice, nchannels=1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mov = smooth(torch.rand([nchannels, *SHAPE], dtype=dt, generator=gen), 2)
    mov = objects.Image(mov, affine=image_affine())
    shape, nl2fix = nonlin_lattice(lattice)
    disp = smooth(torch.randn([3, *shape], dtype=dt, generator=gen), 3)
    disp = disp.movedim(0, -1)
    right = torch.linalg.inv(nl2fix)  # fixed voxels -> nonlinear voxels
    return mov, disp, right, list(shape), gen


@pytest.mark.parametrize('lattice', ['coarse', 'rotated', 'subvoxel'])
def test_push_grad_fixed_left_affine(lattice):
    """Rotating after the chain rule matches rotating the image gradients"""
    mov, disp, right, shape, gen = _push_inputs(lattice, nchannels=2)
    left = translation([0.2, -0.3, 0.1]) @ rotation_z(12)
    left[:3, :3] *= 1.1
    g = torch.randn([2, *SHAPE], dtype=dt, generator=gen)
    h = torch.rand([3, *SHAPE], dtype=dt, generator=gen)
    gn, hn = objects._push_grad_fixed(g, h, mov, disp, left, right, shape)
    gr, hr = _push_grad_fixed_master(g, h, mov, disp, left, right, shape)
    assert (gn - gr).abs().max() <= 1e-12 * gr.abs().max()
    assert (hn - hr).abs().max() <= 1e-12 * hr.abs().max()


def _exact_gn(mov, disp, right):
    """Exact Gauss-Newton Hessian J'J of the warped image wrt the
    displacement nodes (unit weights), and pushed blocks as (N, 3, 3)"""
    rgrid = spatial.affine_grid(right, SHAPE)

    def warp(d):
        psi = regutils.smart_pull_grid(d, rgrid, bound='zero') + rgrid
        return mov.pull(psi).reshape(-1)

    J = torch.autograd.functional.jacobian(warp, disp).reshape(
        -1, disp.numel())
    H = J.T @ J
    g = torch.zeros([1, *SHAPE], dtype=dt)
    h = torch.ones([1, *SHAPE], dtype=dt)
    _, hp = objects._push_grad_fixed(g, h, mov, disp, None, right,
                                     list(disp.shape[:-1]))
    hp = hp.reshape(6, -1).T
    blocks = torch.zeros([hp.shape[0], 3, 3], dtype=dt)
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        blocks[:, i, j] = blocks[:, j, i] = hp[:, k]
    return H, blocks


@pytest.mark.slow
def test_hess_exact_shift():
    """Integer shift: the pushed blocks are the exact GN diagonal blocks"""
    mov, disp, right, _, _ = _push_inputs('shift')
    H, blocks = _exact_gn(mov, disp, right)
    n = blocks.shape[0]
    exact = H.reshape(n, 3, n, 3).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
    assert (blocks - exact).abs().max() <= 1e-10 * exact.abs().max()


@pytest.mark.slow
def test_hess_exact_coarse_majorises():
    """2x coarser field: blockdiag(pushed) - J'J is PSD"""
    mov, disp, right, _, _ = _push_inputs('coarse')
    H, blocks = _exact_gn(mov, disp, right)
    diff = torch.block_diag(*blocks) - H
    lmax = torch.linalg.eigvalsh(H).max()
    assert torch.linalg.eigvalsh(diff).min() >= -1e-12 * lmax
