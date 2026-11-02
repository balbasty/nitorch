"""Symmetric losses (`--symmetric`) in `nitorch register` / `nitorch affreg`.

A symmetric objective is the sum of a forward term (fixed vs moving
warped by T) and a backward term (moving vs fixed warped by T^-1)."""
import os
import torch
import pytest
from nitorch import io, spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep
from nitorch.tools.registration.pairwise_postproc import warp_one_image

dt = torch.double


def smooth(x, n):
    """Box-smooth a (C, *spatial) tensor with circular boundaries"""
    for _ in range(n):
        for d in range(1, x.dim()):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    return x


def make_images(shape, vx_mov=1.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 3)
    mov = smooth(torch.rand([1, *shape], dtype=dt, generator=g), 3)
    fix = objects.Image(fix, affine=spatial.affine_default(shape, 1., dtype=dt))
    mov = objects.Image(mov, affine=spatial.affine_default(shape, vx_mov, dtype=dt))
    return fix, mov


def make_affine(pos):
    affine = objects.AffineModel('rigid', position=pos)
    affine.set_dat(dim=3, dtype=dt)
    affine.dat.dat.copy_(torch.as_tensor([.5, -.3, .4, .05, -.04, .03]))
    return affine


def make_svf(fix, scale=1., seed=1, band=2):
    """SVF on the fixed lattice, with a zero band along the borders"""
    shape = fix.shape
    g = torch.Generator().manual_seed(seed)
    vel = torch.randn([3, *shape], dtype=dt, generator=g)
    vel = smooth(vel, 4) * scale
    if band:
        window = torch.zeros(shape, dtype=dt)
        window[tuple(slice(band, -band) for _ in shape)] = 1
        vel = vel * window
    vel = vel.movedim(0, -1)
    nonlin = objects.SVFModel(steps=8, penalty=dict(
        absolute=0, membrane=0, bending=0, lame=(0, 0), factor=0))
    nonlin.set_dat(shape, affine=fix.affine, dtype=dt)
    nonlin.dat.dat.copy_(vel)
    return nonlin


def make_sim(mov, fix, kind):
    sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
    if kind == 'fwd':
        return objects.SumSimilarity([sim])
    if kind == 'bwd':
        return objects.SumSimilarity([sim.reverse()])
    return sim.symmetrize()


def fd_grad(fn, x, eps=1e-6):
    fd = torch.zeros_like(x).view(-1)
    for i in range(x.numel()):
        xp, xm = x.clone().view(-1), x.clone().view(-1)
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (fn(xp.view_as(x), in_line_search=True) -
                 fn(xm.view_as(x), in_line_search=True)) / (2 * eps)
    return fd.view_as(x)


def test_reverse_swaps_images():
    fix, mov = make_images([4, 4, 4])
    sim = objects.Similarity('mse', mov, fix)
    rev = sim.reverse()
    assert rev.moving is fix and rev.fixed is mov and rev.backward
    fwd, bwd = sim.symmetrize()
    assert fwd.moving is mov and not fwd.backward
    assert bwd.moving is fix and bwd.backward
    assert fwd.factor == bwd.factor == 0.5


@pytest.mark.parametrize('add_identity', [False, True])
def test_svf_exp2(add_identity):
    """exp2 returns the same (displacement) fields as exp and iexp"""
    fix, _ = make_images([8, 8, 8])
    nonlin = make_svf(fix)
    phi, iphi = nonlin.exp2(add_identity=add_identity)
    assert torch.allclose(phi, nonlin.exp(add_identity=add_identity))
    assert torch.allclose(iphi, nonlin.iexp(add_identity=add_identity))


@pytest.mark.parametrize('kind', ['fwd', 'bwd', 'sym'])
def test_affine_only_grad(kind):
    fix, mov = make_images([10, 10, 10])
    affine = make_affine('m')
    step = PairwiseRegisterStep(make_sim(mov, fix, kind), affine, verbose=0)
    x = affine.dat.dat.clone()
    _, g = step.do_affine_only(x, grad=True, in_line_search=True)
    fd = fd_grad(step.do_affine_only, x)
    assert (fd - g).norm() / fd.norm() < 1e-5


@pytest.mark.parametrize('kind, pos', [('fwd', 'f'), ('bwd', 'm')])
def test_affine_grad_with_nonlin(kind, pos):
    """Affine gradients when a (fixed) nonlinear component is present.

    In both cases, the affine is applied on the fixed side of the
    transform used by the loss, where its gradient is not approximated."""
    fix, mov = make_images([12, 12, 12])
    affine = make_affine(pos)
    nonlin = make_svf(fix)
    step = PairwiseRegisterStep(make_sim(mov, fix, kind), affine, nonlin,
                                verbose=0)
    x = affine.dat.dat.clone()
    _, g = step.do_affine(x, grad=True, in_line_search=True)
    fd = fd_grad(step.do_affine, x, eps=1e-5)
    err = (fd - g).norm() / fd.norm()
    assert err < 1e-3, err


@pytest.mark.parametrize('kind', ['fwd', 'bwd', 'sym'])
def test_vel_grad(kind):
    """Velocity gradients (images and velocity on the same lattice).

    The SVF gradient is approximate, so we only check the directional
    derivative along the gradient and along a few smooth directions."""
    fix, mov = make_images([8, 8, 8], vx_mov=1.)
    nonlin = make_svf(fix, band=0)
    step = PairwiseRegisterStep(make_sim(mov, fix, kind), None, nonlin,
                                verbose=0)
    x = nonlin.dat.dat.clone()
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
        assert abs(fd - an) < 0.05 * g.norm(), (kind, fd, an, g.norm())


@pytest.mark.parametrize('pos', ['s', 'm', 'f'])
@pytest.mark.parametrize('step_name', ['do_affine', 'do_vel'])
def test_symmetric_is_sum(pos, step_name):
    """symmetrize() == (forward + backward) / 2, values and gradients"""
    fix, mov = make_images([12, 12, 12])
    out = {}
    for kind in ('fwd', 'bwd', 'sym'):
        affine, nonlin = make_affine(pos), make_svf(fix)
        step = PairwiseRegisterStep(make_sim(mov, fix, kind), affine, nonlin,
                                    verbose=0)
        x = (affine if step_name == 'do_affine' else nonlin).dat.dat
        out[kind] = getattr(step, step_name)(x.clone(), grad=True,
                                             in_line_search=True)
    for i in range(2):
        ref = (out['fwd'][i] + out['bwd'][i]) / 2
        assert torch.allclose(out['sym'][i], ref, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize('pos', ['s', 'm', 'f'])
@pytest.mark.parametrize('backward', [False, True])
def test_composition(pos, backward):
    """The backward term uses the inverse of the forward transform.

    The loss computed during registration must match the one obtained
    by warping the image with the transform used to write the outputs."""
    fix, mov = make_images([12, 12, 12])
    affine, nonlin = make_affine(pos), make_svf(fix, scale=3.)
    sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
    if backward:
        sim = sim.reverse()
    step = PairwiseRegisterStep(objects.SumSimilarity([sim]), affine, nonlin,
                                verbose=0)
    ll = step.do_vel(nonlin.parameters().clone(), in_line_search=True)

    warped = warp_one_image(sim.moving, sim.fixed.affine, sim.fixed.shape,
                            affine=affine, nonlin=nonlin, backward=backward,
                            reslice=True)
    ref = MSE(lam=1, dim=3).loss(warped[None], sim.fixed.dat, dim=3)
    assert torch.allclose(ll, ref, rtol=1e-9, atol=0), (ll, ref)


# ----------------------------------------------------------------------
#   End-to-end: recover a known translation with the command line tools
# ----------------------------------------------------------------------
SHIFT = [3., -2., 1.5]


@pytest.fixture(scope='module')
def shifted_pair(tmp_path_factory):
    odir = tmp_path_factory.mktemp('symmetric')
    shape = [24, 24, 24]
    grid = spatial.identity_grid(shape, dtype=dt)
    center = torch.as_tensor(shape, dtype=dt) / 2

    def blobs(shift):
        x = grid - (center + torch.as_tensor(shift, dtype=dt))
        a = (x / torch.as_tensor([5., 3.5, 4.])).square().sum(-1).neg().exp()
        b = ((x - torch.as_tensor([4., -3., 2.])) / 2.5).square().sum(-1)
        return (a + 0.7 * b.neg().exp()).float()

    affine = spatial.affine_default(shape, voxel_size=1.)
    fix, mov = os.path.join(odir, 'fix.nii.gz'), os.path.join(odir, 'mov.nii.gz')
    io.savef(blobs([0, 0, 0]), fix, affine=affine)
    io.savef(blobs(SHIFT), mov, affine=affine)
    return str(odir), fix, mov


def recovered_translation(fname):
    return io.transforms.loadf(fname).reshape([4, 4])[:3, -1].tolist()


def test_register_symmetric_translation(shifted_pair):
    from nitorch.cli.registration.register.cli import _cli as register
    odir, fix, mov = shifted_pair
    lta = os.path.join(odir, 'register.lta')
    register(['-v', '0', '--cpu', '@loss', 'mse', '--symmetric',
              '@@fix', fix, '-o', 'false', '-r', 'false',
              '@@mov', mov, '-o', 'false', '-r', 'false',
              '@affine', 'translation', '-o', lta,
              '@@optim', 'gn', '-n', '50', '@pyramid', '--levels', '1'])
    t = recovered_translation(lta)
    assert all(abs(a - b) < 0.1 for a, b in zip(t, SHIFT)), t


def test_affreg_symmetric_translation(shifted_pair):
    from nitorch.cli.registration.affreg.cli import _cli as affreg
    odir, fix, mov = shifted_pair
    lta = os.path.join(odir, 'affreg.lta')
    affreg(['-v', '0', '--cpu', '-l', 'mse', '--symmetric', '-f', fix,
            '-m', mov, '-o', lta, '-x', 'false', '-r', 'false', '-p', '1'])
    t = recovered_translation(lta)
    assert all(abs(a - b) < 0.1 for a, b in zip(t, SHIFT)), t


def _penalised_affine(basis, optim, pos='m'):
    affine = objects.AffineModel(basis, penalty=2., optim=optim,
                                 position=pos)
    affine.set_dat(dim=3, dtype=dt)
    g = torch.Generator().manual_seed(3)
    affine.dat.dat.copy_(0.05 * torch.randn([12], dtype=dt, generator=g))
    return affine


@pytest.mark.parametrize('optim', ['rigid', 'similitude'])
@pytest.mark.parametrize('step_name', ['do_affine_only', 'do_affine'])
def test_affine_penalty_progressive(optim, step_name):
    """Affine penalty with `-g`: grad/hess in the optimised sub-basis"""
    fix, mov = make_images([8, 8, 8])
    affine = _penalised_affine('affine', optim)
    nonlin = make_svf(fix) if step_name == 'do_affine' else None
    step = PairwiseRegisterStep(make_sim(mov, fix, 'fwd'), affine, nonlin,
                                verbose=0)
    fn = getattr(step, step_name)
    par = affine.parameters()
    x0 = affine.dat.dat.clone()
    _, g, h = fn(x0, grad=True, hess=True, in_line_search=True)
    n = len(par)
    assert g.shape == (n,) and h.shape == (n, n)
    # finite differences along the optimised parameters
    M = par._gram(x0)                     # model = M @ optim
    fd = torch.zeros(n, dtype=dt)
    eps = 1e-5
    for i in range(n):
        fd[i] = (fn(x0 + eps * M[:, i], in_line_search=True) -
                 fn(x0 - eps * M[:, i], in_line_search=True)) / (2 * eps)
    assert (fd - g).norm() / fd.norm() < 1e-3
    # the penalty part alone is exact
    P = affine.penalty
    gp, hp = torch.zeros(n, dtype=dt), torch.zeros(n, n, dtype=dt)
    val = step._affine_penalty(x0, gp, hp)
    assert torch.allclose(val, 0.5 * P * x0.dot(x0))
    assert torch.allclose(gp, M.T @ (P * x0))
    assert torch.allclose(hp, P * M.T @ M)


def test_affine_penalty_full_basis_unchanged():
    """Without `-g`, the penalty is added as before (bitwise)"""
    fix, mov = make_images([8, 8, 8])
    affine = _penalised_affine('affine', None)
    step = PairwiseRegisterStep(make_sim(mov, fix, 'fwd'), affine, verbose=0)
    x0 = affine.dat.dat.clone()
    ll, g, h = step.do_affine_only(x0, grad=True, hess=True,
                                   in_line_search=True)
    P = affine.penalty
    affine._penalty = None
    ll0, g0, h0 = step.do_affine_only(x0, grad=True, hess=True,
                                      in_line_search=True)
    lla = x0 * P
    g0 += lla
    h0.diagonal(0, -1, -2).add_(P)
    ll0 += 0.5 * lla.dot(x0)   # added once (not twice)
    assert torch.equal(g, g0) and torch.equal(h, h0) and torch.equal(ll, ll0)


def _penalised_step(step_name, optim, penalty, verbose=0):
    """Joint/affine-only/nonlin-only step, with or without penalties"""
    fix, mov = make_images([8, 8, 8], vx_mov=1.)
    affine = nonlin = None
    if step_name in ('do_affine', 'do_affine_only'):
        affine = _penalised_affine('affine', optim)
        if not penalty:
            affine._penalty = None
    else:
        # (the verbose display of the nonlinear step needs an affine)
        affine = make_affine('f')
    if step_name in ('do_affine', 'do_vel'):
        nonlin = make_svf(fix, band=0)
        if penalty:
            nonlin.penalty.update(membrane=1., bending=1., factor=1.)
    step = PairwiseRegisterStep(make_sim(mov, fix, 'fwd'), affine, nonlin,
                                verbose=verbose)
    x = (nonlin if step_name == 'do_vel' else affine).dat.dat.clone()
    return step, getattr(step, step_name), x


@pytest.mark.parametrize('optim', [None, 'rigid'])
@pytest.mark.parametrize('step_name', ['do_affine_only', 'do_affine', 'do_vel'])
def test_penalty_value_matches_grad(step_name, optim):
    """Penalties are added to the objective once, consistently with grad.

    The penalised and unpenalised objectives differ by a quadratic,
    whose directional derivative is exactly given by the gradient
    difference (central differences are exact for quadratics)."""
    if step_name == 'do_vel' and optim:
        pytest.skip('-g does not apply to the nonlinear step')
    _, f1, x = _penalised_step(step_name, optim, True)
    _, f0, _ = _penalised_step(step_name, optim, False)
    _, g1 = f1(x, grad=True, in_line_search=True)
    _, g0 = f0(x, grad=True, in_line_search=True)
    gen = torch.Generator().manual_seed(4)
    d = torch.randn(x.shape, dtype=dt, generator=gen)
    if step_name != 'do_vel' and optim:
        dg = torch.randn(g1.shape, dtype=dt, generator=gen)
        d = objects._affine_gram(optim, 'affine', 3, dtype=dt) @ dg
        an = ((g1 - g0) * dg).sum()
    else:
        an = ((g1 - g0) * d).sum()
    eps = 1e-3
    pen = lambda y: f1(y, in_line_search=True) - f0(y, in_line_search=True)
    fd = (pen(x + eps * d) - pen(x - eps * d)) / (2 * eps)
    assert torch.allclose(fd, an, rtol=1e-6, atol=1e-9), (fd, an)
    assert pen(torch.zeros_like(x)).abs() < 1e-12   # zero at the origin


@pytest.mark.parametrize('optim', [None, 'rigid'])
@pytest.mark.parametrize('step_name', ['do_affine_only', 'do_affine', 'do_vel'])
def test_verbose_does_not_change_objective(step_name, optim, capsys):
    """Printing is side-effect free: same objective for -v 0/1/3"""
    outs = []
    if step_name == 'do_vel' and optim:
        pytest.skip('-g does not apply to the nonlinear step')
    for verbose in (0, 1, 3):
        for line_search in (True, False):
            step, f, x = _penalised_step(step_name, optim, True, verbose)
            outs.append(f(x, grad=True, hess=True, in_line_search=line_search))
    # Not bitwise: on torch 1.13 the first evaluation runs the TorchScript
    # kernels unfused and later ones fused, which changes the last bits
    # (~1e-22 here) even without any printing.
    for out in outs[1:]:
        for a, b in zip(outs[0], out):
            assert torch.allclose(a, b, rtol=1e-12, atol=1e-12 * b.abs().max())
    capsys.readouterr()
