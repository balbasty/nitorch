"""`inversion` option of nonlinear warps (#171).

`spatial.grid_inv(method='newton')` refines the push inverse with
safeguarded Newton steps. The `inversion` option ('default', 'push',
'newton') of `spatial.shoot`, of the registration models (smalldef,
svf, shoot) and of `register @nonlin --inversion` computes one side of
the warp by inverting the other. The default must be unchanged, bit for
bit."""
import json
import math
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
    `inversion` option was added (greens=None, voxel_size=1)."""
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


def reference_grid_inv(grid, type='grid', bound='dft', extrapolate=True,
                       **prm):
    """Verbatim copy of `spatial.grid_inv` before the `method` option."""
    from nitorch.core import utils
    from nitorch.spatial._solvers import solve_grid_fmg
    prm = prm or dict(membrane=0.1)
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    shape = grid.shape[-(dim + 1):-1]
    batch = grid.shape[:-(dim + 1)]
    grid = grid.reshape([-1, *shape, dim])
    backend = dict(dtype=grid.dtype, device=grid.device)
    identity = spatial.identity_grid(shape, **backend)
    if type == 'grid':
        disp = grid - identity
    else:
        disp = grid
        grid = disp + identity
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = utils.movedim(disp, -1, 1)
    disp = spatial.grid_push(disp, grid, **push_opt)
    count = spatial.grid_count(grid, **push_opt)
    disp = utils.movedim(disp, 1, -1)
    count = utils.movedim(count, 1, -1)
    hess = count.new_zeros([*count.shape[:-1], dim * (dim + 1) // 2])
    hess[..., :dim] = count
    disp = solve_grid_fmg(hess, disp, bound=bound, **prm)
    disp = disp.reshape([*batch, *shape, dim])
    if type == 'grid':
        return identity - disp
    else:
        return -disp


# ----------------------------------------------------------------------
# steep analytic warps (displacements in voxels, with exact inverses)
# ----------------------------------------------------------------------
def _bump(s):
    out = torch.zeros_like(s)
    m = s < 1
    out[m] = torch.exp(1 - 1 / (1 - s[m] ** 2))
    return out


def _centred(shape, dtype):
    n = shape[0]
    y = spatial.identity_grid(shape, dtype=dtype) - (n - 1) / 2
    return y, _bump(y.square().sum(-1).sqrt() / (0.45 * n))


def swirl(shape, turns=0.25, dtype=torch.float64):
    """Swirl (rotation of up to 90 deg at the centre) and its inverse"""
    y, b = _centred(shape, dtype)

    def rot(t):
        ang = t * turns * 2 * math.pi * b
        z = y.clone()
        z[..., 0] = ang.cos() * y[..., 0] - ang.sin() * y[..., 1]
        z[..., 1] = ang.sin() * y[..., 0] + ang.cos() * y[..., 1]
        return z - y
    return rot(1), rot(-1)


def swirl_vel(shape, turns=0.25, dtype=torch.float32):
    """Stationary velocity whose exponential is `swirl`"""
    y, b = _centred(shape, dtype)
    w = turns * 2 * math.pi * b
    v = torch.zeros_like(y)
    v[..., 0] = -w * y[..., 1]
    v[..., 1] = w * y[..., 0]
    return v


def radial_vel(shape, scale=5., dtype=torch.float32):
    """Stationary velocity of a radial expansion (x `scale` at the centre)"""
    y, b = _centred(shape, dtype)
    return math.log(scale) * b[..., None] * y


def radial(shape, scale=5., dtype=torch.float64, nstep=100):
    """Radial expansion (x `scale` at the centre) and its inverse (RK4)"""
    beta = math.log(scale)
    n = shape[0]
    y, _ = _centred(shape, dtype)
    r0 = y.square().sum(-1).sqrt()

    def flow(t):
        r = r0.clone()
        h = t / nstep
        f = lambda r: beta * r * _bump(r / (0.45 * n))  # noqa: E731
        for _ in range(nstep):
            k1 = f(r)
            k2 = f(r + h / 2 * k1)
            k3 = f(r + h / 2 * k2)
            k4 = f(r + h * k3)
            r = r + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        scl = torch.where(r0 > 0, r / r0.clamp_min(1e-12),
                          torch.full_like(r0, math.exp(beta * t)))
        return y * scl[..., None] - y
    return flow(1), flow(-1)


def compose3(a, b):
    """Displacement of (id + a) o (id + b), cubic interpolation of `a`
    (so that the metric is not dominated by interpolation errors)"""
    dim = a.shape[-1]
    id = spatial.identity_grid(a.shape[-dim-1:-1], dtype=a.dtype)
    out = spatial.grid_pull(a.movedim(-1, 0)[None], (id + b)[None],
                            bound='dft', extrapolate=True, interpolation=3,
                            prefilter=True)
    return b + out[0].movedim(0, -1)


def norms(d, margin=4):
    """Voxel-wise norm, away from the border"""
    dim = d.shape[-1]
    d = d[(slice(margin, -margin),) * dim]
    return d.double().square().sum(-1).sqrt()


# ----------------------------------------------------------------------
# grid_inv(method='push'|'newton')
# ----------------------------------------------------------------------
@pytest.mark.parametrize('type', ['disp', 'grid'])
@pytest.mark.parametrize('batch', [None, 2])
def test_grid_inv_push_is_unchanged(type, batch):
    shape = (12, 10)
    disp = make_vel(shape, 3., batch)
    inp = disp if type == 'disp' else disp + spatial.identity_grid(shape)
    ref = reference_grid_inv(inp, type=type)
    assert torch.equal(spatial.grid_inv(inp, type=type), ref)
    assert torch.equal(spatial.grid_inv(inp, type=type, method='push'), ref)
    # zero Newton steps = push
    assert torch.equal(spatial.grid_inv(inp, type=type, method='newton',
                                        newton_iter=0), ref)
    # with explicit regularisation
    prm = dict(membrane=0.01, bending=0.1)
    assert torch.equal(spatial.grid_inv(inp, type=type, **prm),
                       reference_grid_inv(inp, type=type, **prm))


def test_grid_inv_invalid_method():
    with pytest.raises(ValueError):
        spatial.grid_inv(make_vel((8, 8), 1.), type='disp', method='fixed')


@pytest.mark.parametrize('warp', ['swirl', 'radial'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(48, 48), (24, 24, 24)])
def test_grid_inv_newton_improves(warp, dtype, shape):
    u, ref = (swirl if warp == 'swirl' else radial)(shape)
    u = u.to(dtype)
    w0 = spatial.grid_inv(u, type='disp')
    w = spatial.grid_inv(u, type='disp', method='newton')
    assert w.dtype == dtype and w.shape == u.shape
    # residual |phi o psi - id| (the equation solved, linear interpolation)
    res0 = compose(u, w0).square().sum(-1).sqrt()
    res = compose(u, w).square().sum(-1).sqrt()
    assert res0.max() > 0.5                      # the test is meaningful
    assert res.max() < 0.05 * res0.max()
    # never worse than push, voxel-wise
    assert (res <= res0 + 1e-5).all()
    # closer to the exact inverse
    err0, err = norms(w0 - ref), norms(w - ref)
    assert err.max() < 0.5 * err0.max()
    assert err.mean() < 0.7 * err0.mean()


@pytest.mark.parametrize('bound', ['dft', 'dct2', 'zero'])
def test_grid_inv_newton_batched_and_bounds(bound):
    shape = (24, 20)
    disp = make_vel(shape, 5., batch=3)
    w = spatial.grid_inv(disp, type='disp', bound=bound, method='newton')
    w0 = spatial.grid_inv(disp, type='disp', bound=bound)
    assert w.shape == disp.shape
    assert torch.isfinite(w).all()
    # (with 'zero', the push inverse itself is not exactly batch-invariant:
    # its multigrid solve differs by ~1e-2 between batched and single)
    atol = 1e-3 if bound != 'zero' else 0.2
    for b in range(3):
        wb = spatial.grid_inv(disp[b], type='disp', bound=bound,
                              method='newton')
        assert torch.allclose(w[b], wb, atol=atol)
    # transformation grids are supported too
    id = spatial.identity_grid(shape)
    g = spatial.grid_inv(disp + id, bound=bound, method='newton')
    assert torch.allclose(g, w + id, atol=atol)
    # (with dft) the residual decreases
    if bound == 'dft':
        res0 = compose(disp[0], w0[0]).square().sum(-1).sqrt()
        res = compose(disp[0], w[0]).square().sum(-1).sqrt()
        assert (res <= res0 + 1e-5).all()


def test_grid_inv_newton_folded():
    # a folding displacement: the safeguards keep the result finite
    shape = (32, 32)
    u, _ = swirl(shape, turns=1.)
    u = u.float()
    u[10:14, 10:14] += 5
    w0 = spatial.grid_inv(u, type='disp')
    w = spatial.grid_inv(u, type='disp', method='newton')
    assert torch.isfinite(w).all()
    res0 = compose(u, w0).square().sum(-1).sqrt()
    res = compose(u, w).square().sum(-1).sqrt()
    assert (res <= res0 + 1e-5).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('shape', [(24, 24), (12, 12, 12)])
def test_grid_inv_newton_cuda(shape):
    u, _ = swirl(shape)
    u = u.float()
    w = spatial.grid_inv(u.cuda(), type='disp', method='newton')
    assert w.is_cuda
    assert torch.allclose(w.cpu(), spatial.grid_inv(u, type='disp',
                                                    method='newton'),
                          atol=1e-3)


# ----------------------------------------------------------------------
# spatial.shoot(inversion=...)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(12, 10), (8, 7, 6)])
@pytest.mark.parametrize('batch', [None, 2])
@pytest.mark.parametrize('return_inverse', [False, True])
@pytest.mark.parametrize('displacement', [False, True])
@pytest.mark.parametrize('fast', [True, False])
def test_shoot_default_is_unchanged(shape, batch, return_inverse,
                                    displacement, fast):
    vel = make_vel(shape, 3., batch)
    prm = shoot_prm(shape, return_inverse=return_inverse,
                    displacement=displacement, steps=4, fast=fast)
    ref = reference_shoot(vel, **prm)
    for kwargs in ({}, {'inversion': 'default'}, {'inversion': None}):
        out = spatial.shoot(vel, **prm, **kwargs)
        if return_inverse:
            assert torch.equal(out[0], ref[0])
            assert torch.equal(out[1], ref[1])
        else:
            assert torch.equal(out, ref)


def test_shoot_invalid_inversion():
    for inversion in ('fixedpoint', 'invert', 'Newton'):
        with pytest.raises(ValueError):
            spatial.shoot(make_vel((8, 8), 1.), inversion=inversion)
    with pytest.raises(TypeError):
        spatial.shoot(make_vel((8, 8), 1.), forward_method='fixedpoint')


@pytest.mark.parametrize('shape', [(16, 14), (10, 9, 8)])
@pytest.mark.parametrize('fast', [True, False])
@pytest.mark.parametrize('inversion', ['push', 'newton'])
@pytest.mark.parametrize('batch', [None, 2])
def test_shoot_inversion_outputs(shape, fast, inversion, batch):
    vel = make_vel(shape, 4., batch)
    id = spatial.identity_grid(shape)
    prm = shoot_prm(shape, steps=6, fast=fast)
    _, phi_ref = spatial.shoot(vel, **prm, displacement=True,
                               return_inverse=True)
    psi_d, phi_d = spatial.shoot(vel, **prm, displacement=True,
                                 return_inverse=True, inversion=inversion)
    psi_g, phi_g = spatial.shoot(vel, **prm, displacement=False,
                                 return_inverse=True, inversion=inversion)
    psi_only = spatial.shoot(vel, **prm, displacement=True,
                             inversion=inversion)
    assert psi_d.shape == phi_d.shape == vel.shape
    # same (directly integrated) inverse as the default
    assert torch.equal(phi_d, phi_ref)
    # the forward is its numerical inverse
    assert torch.equal(psi_d, spatial.grid_inv(phi_ref, type='disp',
                                               method=inversion))
    assert torch.equal(psi_only, psi_d)
    assert torch.allclose(psi_g, psi_d + id, atol=1e-5)
    assert torch.allclose(phi_g, phi_d + id, atol=1e-5)


def test_shoot_inversion_double():
    shape = (12, 12)
    vel = make_vel(shape, 3., dtype=torch.float64)
    psi = spatial.shoot(vel, **shoot_prm(shape, steps=4),
                        displacement=True, inversion='newton')
    assert psi.dtype == torch.float64 and torch.isfinite(psi).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_shoot_inversion_cuda():
    shape = (10, 9, 8)
    vel = make_vel(shape, 4.)
    prm = shoot_prm(shape, steps=4, displacement=True, return_inverse=True,
                    inversion='newton')
    psi, phi = spatial.shoot(vel.cuda(), **prm)
    psi_cpu, phi_cpu = spatial.shoot(vel, **prm)
    assert psi.is_cuda
    assert torch.allclose(psi.cpu(), psi_cpu, atol=1e-3)
    assert torch.allclose(phi.cpu(), phi_cpu, atol=1e-4)


# ----------------------------------------------------------------------
# registration models
# ----------------------------------------------------------------------
def _models():
    from nitorch.tools.registration import objects
    return objects.SmallDefModel, objects.SVFModel, objects.ShootModel


def _default_exp2(model, v):
    """What each model computed before the `inversion` option"""
    from nitorch.tools.registration import objects
    if isinstance(model, objects.SmallDefModel):
        return v, spatial.grid_inv(v, type='disp', **model.penalty)
    if isinstance(model, objects.SVFModel):
        return (spatial.exp_forward(v, steps=model.steps, displacement=True),
                spatial.exp_forward(v, steps=model.steps, inverse=True,
                                    displacement=True))
    penalty = dict(model.penalty)
    factor = penalty.pop('factor')
    return spatial.shoot(v, model.kernel, steps=model.steps,
                         factor=factor / py.prod(model.shape),
                         voxel_size=model.voxel_size, **penalty,
                         return_inverse=True, displacement=True)


@pytest.mark.parametrize('model', ['smalldef', 'svf', 'shoot'])
def test_model_default_is_unchanged(model):
    from nitorch.tools.registration import objects
    shape = [10, 10, 10]
    vel = make_vel(shape, 3.)
    for kwargs in ({}, {'inversion': 'default'}):
        nl = objects.NonLinModel.subclass(model)(vel.clone(), factor=1, **kwargs)
        assert nl.inversion == 'default'
        grid, igrid = _default_exp2(nl, vel)
        assert torch.equal(nl.exp(), grid)
        assert torch.equal(nl.iexp(), igrid)
        g, ig = nl.exp2()
        assert torch.equal(g, grid) and torch.equal(ig, igrid)
        g, ig, jac, ijac = nl.exp2(jacobian=True)
        assert torch.equal(g, grid) and torch.equal(ig, igrid)


@pytest.mark.parametrize('model', ['smalldef', 'svf', 'shoot'])
@pytest.mark.parametrize('inversion', ['push', 'newton'])
def test_model_inversion(model, inversion):
    from nitorch.tools.registration import objects
    shape = [10, 10, 10]
    vel = make_vel(shape, 3.)
    nl = objects.NonLinModel.subclass(model)(vel.clone(), factor=1,
                             inversion=inversion)
    ref = objects.NonLinModel.subclass(model)(vel.clone(), factor=1)
    grid, igrid = _default_exp2(ref, vel)
    g, ig = nl.exp2()
    if model == 'shoot':
        # the inverse is direct, the forward is inverted
        assert torch.equal(ig, igrid)
        assert torch.equal(g, spatial.grid_inv(igrid, type='disp',
                                               method=inversion))
        assert torch.equal(nl.iexp(), igrid)
        assert torch.equal(nl.exp(), g)
    else:
        # the forward is direct, the inverse is inverted
        assert torch.equal(g, grid)
        kw = dict(**nl.penalty) if model == 'smalldef' else {}
        if model == 'svf' or inversion == 'newton':
            kw['method'] = inversion
        assert torch.equal(ig, spatial.grid_inv(grid, type='disp', **kw))
        assert torch.equal(nl.exp(), grid)
        assert torch.equal(nl.iexp(), ig)
        if model == 'smalldef' and inversion == 'push':
            # smalldef: 'push' is the default
            assert torch.equal(ig, igrid)
    # jacobians
    g, ig, jac, ijac = nl.exp2(jacobian=True)
    assert torch.allclose(ijac, spatial.grid_jacobian(ig, type='disp'),
                          atol=1e-5)
    ig2, ijac2 = nl.iexp(jacobian=True)
    assert torch.equal(ig2, ig)
    # pyramid levels and copies keep the option
    assert nl.downsample(shape=[5, 5, 5]).inversion == inversion
    assert nl.upsample(shape=[12, 12, 12]).inversion == inversion
    assert nl.to('cpu').inversion == inversion


def test_model_invalid_inversion():
    for Model in _models():
        with pytest.raises(ValueError):
            Model([4, 4, 4], inversion='fixedpoint')


def _consistency(model):
    """Inverse consistency of the two sides of the warp.

    Returns {name: (max, mean)} for
    - 'solved': |direct o inverted - id| with linear interpolation (as
      `reslice`/`compose` compose warps): the equation that the
      inversion solves (fwd o inv for smalldef/svf, inv o fwd for shoot);
    - 'solved3', 'other3': both orders with cubic interpolation (a
      metric less dominated by interpolation errors).
    """
    g, ig = model.exp2()
    direct, inverted = (ig, g) if model.model == 'shoot' else (g, ig)
    out = {}
    for name, d in (('solved', compose(direct, inverted)),
                    ('solved3', compose3(direct, inverted)),
                    ('other3', compose3(inverted, direct))):
        d = norms(d)
        out[name] = (d.max().item(), d.mean().item())
    return out


@pytest.mark.parametrize('warp', ['swirl', 'radial'])
@pytest.mark.parametrize('model', ['smalldef', 'svf', 'shoot'])
def test_model_inversion_improves_consistency(warp, model):
    """90 deg swirl and x5 radial expansion (2D, 64^2)."""
    from nitorch.tools.registration import objects
    shape = [64, 64]
    if model == 'smalldef':
        dat = (swirl(shape) if warp == 'swirl' else radial(shape))[0].float()
    else:
        dat = swirl_vel(shape) if warp == 'swirl' else radial_vel(shape)
    Model = objects.NonLinModel.subclass(model)
    err = {inv: _consistency(Model(dat.clone(), factor=1, inversion=inv))
           for inv in ('default', 'push', 'newton')}
    new, push, ref = err['newton'], err['push'], err['default']

    # newton: (nearly) exact on the equation it solves
    assert ref['solved'][0] > 0.05                  # the test is meaningful
    assert new['solved'][0] < 0.1 * ref['solved'][0]
    assert new['solved'][1] < 0.1 * ref['solved'][1]
    # newton is no worse than push
    for key in ('solved', 'solved3'):
        assert new[key][0] <= push[key][0] and new[key][1] <= push[key][1]
    assert new['other3'][1] <= push['other3'][1]

    if model in ('smalldef', 'shoot'):
        # ... and more consistent in both orders than the default
        for key in ('solved3', 'other3'):
            assert new[key][0] < 0.6 * ref[key][0]
            assert new[key][1] < 0.6 * ref[key][1]
    else:
        # svf: the default integrates exp(-v) as accurately as exp(v), so
        # the other order is about as consistent as by default
        assert new["other3"][0] < 2 * ref["other3"][0]
        assert new['other3'][1] < 1.5 * ref['other3'][1]
    if model == 'shoot':
        # shoot: push is already much better than the composed forward
        for key in ('solved', 'solved3', 'other3'):
            assert push[key][1] < 0.5 * ref[key][1]
    if model == 'smalldef':
        assert push == ref                          # same push inverse


# ----------------------------------------------------------------------
# command line: register @nonlin --inversion, JSON, reslice/compose/vexp
# ----------------------------------------------------------------------
REGISTER_BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii',
                 '@nonlin']


@pytest.mark.parametrize('model', ['svf', 'shoot', 'smalldef'])
@pytest.mark.parametrize('args,expected', [
    ([], 'default'),
    (['--inversion', 'default'], 'default'),
    (['--inversion', 'push'], 'push'),
    (['--inversion', 'newton'], 'newton'),
])
def test_register_parse_inversion(model, args, expected):
    from nitorch.cli.registration.register.parser import parser
    options = parser.parse(REGISTER_BASE + [model, *args])
    assert options.nonlin.inversion == expected


@pytest.mark.parametrize('args', [
    ['--inversion', 'invert'],
    ['--inversion', 'fixedpoint'],
    ['--inversion'],
    ['--fixedpoint'],
    ['--fixedpoint', '3'],
])
def test_register_parse_inversion_invalid(args):
    from nitorch.cli.registration.register.parser import parser
    with pytest.raises(Exception):
        parser.parse(REGISTER_BASE + ['shoot', *args])


def _write(path, dat, affine=None):
    import numpy as np
    nib = pytest.importorskip('nibabel')
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype='float32'),
                             np.eye(4) if affine is None else affine),
             str(path))
    return str(path)


def _spy(monkeypatch, name):
    calls = []
    func = getattr(spatial, name)

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return func(*args, **kwargs)
    monkeypatch.setattr(spatial, name, spy)
    return calls


@pytest.mark.parametrize('model', ['shoot', 'svf', 'smalldef'])
@pytest.mark.parametrize('inversion', [None, 'push', 'newton'])
def test_register_inversion_json_and_reslice(tmp_path, monkeypatch, model,
                                             inversion):
    import numpy as np
    pytest.importorskip('nibabel')
    from nitorch.cli.registration.register.cli import _cli
    from nitorch.cli.registration.reslice import main as reslice_main
    from nitorch.cli.registration.reslice import parser as reslice_parser
    from nitorch.cli.registration import helpers

    rng = np.random.default_rng(0)
    fix = _write(tmp_path / 'fix.nii.gz', rng.random([12, 12, 12]))
    mov = _write(tmp_path / 'mov.nii.gz', rng.random([12, 12, 12]))
    out = str(tmp_path / 'nl.nii.gz')
    flag = ['--inversion', inversion] if inversion else []
    expected = inversion or 'default'

    shoot_calls = _spy(monkeypatch, 'shoot')
    inv_calls = _spy(monkeypatch, 'grid_inv')
    # symmetric loss: the registration uses both sides of the warp
    _cli(['@loss', 'mse', '-s', '@@fix', fix, '-o', 'false', '@@mov', mov,
          '-o', 'false', '@nonlin', model, '-s', '3', *flag, '-o', out,
          '@@optim', 'gd', '-n', '1', '@optim', '-n', '1',
          '@pyramid', '-l', '0'])
    # the registration used the option
    if model == 'shoot':
        fwd = [kw for kw in shoot_calls if 'inversion' in kw]
        assert fwd and all(kw['inversion'] == expected for kw in fwd)
    methods = {kw.get('method', 'push') for kw in inv_calls}
    if model == 'shoot':
        pass  # (`spatial.shoot` calls `grid_inv` internally)
    elif expected == 'default' and model != 'smalldef':
        assert not inv_calls
    elif expected == 'newton':
        assert methods == {'newton'}
    else:
        assert methods == {'push'}
    # ... wrote it next to the transform
    with open(tmp_path / 'nl.json') as f:
        prm = json.load(f)
    assert prm['inversion'] == expected
    assert prm['model'] == model
    assert ('bending' in prm) == (model == 'shoot')

    # ... and reslice / compose rebuild the same warp
    shoot_calls.clear()
    inv_calls.clear()
    if model == 'smalldef':
        # the sidecar is found from the displacement file
        disp = reslice_main.io.volumes.map(out).fdata().reshape([12] * 3 + [3])
        helpers.invert_displacement(disp, out)
        assert len(inv_calls) == 1
        assert inv_calls[0].get('method', 'push') == (
            'newton' if expected == 'newton' else 'push')
        return
    for inv in (False, True):
        shoot_calls.clear()
        inv_calls.clear()
        options = reslice_parser.parse(
            [fix, '-iv' if inv else '-v', out, '-t', fix])
        trf = options.transformations[0]
        assert trf.json == str(tmp_path / 'nl.json')
        reslice_main.exponentiate_transforms([trf])
        if model == 'shoot':
            assert len(shoot_calls) == 1
            if inv or expected == 'default':
                # the inverse is direct (and old JSONs: unchanged)
                assert 'inversion' not in shoot_calls[0]
            else:
                assert shoot_calls[0]['inversion'] == expected
            assert not inv_calls
        else:
            assert not shoot_calls
            if inv and expected != 'default':
                assert [kw['method'] for kw in inv_calls] == [expected]
            else:
                assert not inv_calls


@pytest.mark.parametrize('model', ['shoot', 'svf'])
def test_json_without_inversion_is_unchanged(tmp_path, model):
    """Old sidecars (no "inversion" key) behave exactly as before"""
    from nitorch.cli.registration import helpers
    shape = (12, 12, 12)
    vel = make_vel(shape, 3.)
    aff = torch.eye(4, dtype=torch.float64)
    if model == 'shoot':
        old = shoot_prm(shape, steps=4, fast=True)
        old['lame'] = list(old['lame'])
        new = dict(old, inversion='default', model='shoot')
    else:
        old, new = None, dict(model='svf', inversion='default')
    fold = None
    if old is not None:
        fold = str(tmp_path / 'old.json')
        with open(fold, 'w') as f:
            json.dump(old, f)
    fnew = str(tmp_path / 'new.json')
    with open(fnew, 'w') as f:
        json.dump(new, f)
    for t0, t1 in ((0, 1), (1, 0), (0.25, 1), (1, 0.5), (0, 0.5)):
        a = helpers.exponentiate_velocity(vel, aff, json=fold, t0=t0, t1=t1)
        b = helpers.exponentiate_velocity(vel, aff, json=fnew, t0=t0, t1=t1)
        assert torch.equal(a, b)


@pytest.mark.parametrize('inversion', ['push', 'newton'])
def test_exponentiate_velocity_time_bounds(tmp_path, inversion):
    """Inversion is applied to each map that is built, as at t = 1"""
    from nitorch.cli.registration import helpers
    shape = (12, 12, 12)
    vel = make_vel(shape, 3.)
    aff = torch.eye(4, dtype=torch.float64)
    sprm = shoot_prm(shape, steps=4, fast=True)
    sprm['lame'] = list(sprm['lame'])
    fshoot = str(tmp_path / 'shoot.json')
    with open(fshoot, 'w') as f:
        json.dump(dict(sprm, inversion=inversion, model='shoot'), f)
    fsvf = str(tmp_path / 'svf.json')
    with open(fsvf, 'w') as f:
        json.dump(dict(model='svf', inversion=inversion), f)
    sprm['voxel_size'] = [1., 1., 1.]
    id = spatial.identity_grid(shape)

    def shoot(t, inverse=False):
        psi, phi = spatial.shoot(vel * t, **sprm, displacement=True,
                                 return_inverse=True)
        if inverse:
            return phi
        return spatial.grid_inv(phi, type='disp', method=inversion)

    def exp(t):
        return helpers.exponentiate_velocity(vel, aff, json=fshoot,
                                             t0=t[0], t1=t[1])

    assert torch.allclose(exp((0, 1)), shoot(1), atol=1e-5)
    assert torch.allclose(exp((0, 0.5)), shoot(0.5), atol=1e-5)
    assert torch.allclose(exp((1, 0)), shoot(1, inverse=True), atol=1e-5)
    # inv(phi_t0) o phi_t1: phi_t1 inverted, inv(phi_t0) direct
    d1, i0 = shoot(1), shoot(0.5, inverse=True)
    ref = d1 + _shoot._pull_grid(i0, d1 + id, bound='dft',
                                 extrapolate=True, interpolation=1)
    assert torch.allclose(exp((0.5, 1)), ref, atol=1e-5)
    # svf: the negative-time map is the inverse of the forward
    fwd = spatial.exp(vel[None], displacement=True)[0]
    ref = spatial.grid_inv(fwd, type='disp', method=inversion)
    for t0, t1 in ((1, 0), (1.5, 0.5)):
        out = helpers.exponentiate_velocity(vel, aff, json=fsvf,
                                            t0=t0, t1=t1)
        assert torch.allclose(out, ref, atol=1e-5)
    out = helpers.exponentiate_velocity(vel, aff, json=fsvf, t0=0, t1=1)
    assert torch.equal(out, fwd)


def test_reslice_compose_find_sidecars(tmp_path):
    from nitorch.cli.registration import helpers
    vel = _write(tmp_path / 'vel.nii.gz', torch.zeros([4, 4, 4, 3]))
    sidecar = str(tmp_path / 'vel.json')
    for prm, as_vel, as_disp in (
            (dict(bending=0.2), True, False),                 # old shoot
            (dict(bending=0.2, model='shoot', inversion='push'), True, False),
            (dict(model='svf', inversion='newton'), True, False),
            (dict(model='smalldef', inversion='newton'), False, True),
            (dict(other=1), False, False)):
        with open(sidecar, 'w') as f:
            json.dump(prm, f)
        assert (helpers.find_shoot_json(vel) == sidecar) == as_vel
        assert (helpers.find_displacement_json(vel) == sidecar) == as_disp
    # a smalldef JSON is not a velocity
    with open(sidecar, 'w') as f:
        json.dump(dict(model='smalldef', inversion='newton'), f)
    with pytest.raises(ValueError):
        helpers.exponentiate_velocity(torch.zeros([4, 4, 4, 3]),
                                      torch.eye(4), json=sidecar)
    with open(sidecar, 'w') as f:
        json.dump(dict(model='svf', inversion='bad'), f)
    with pytest.raises(ValueError):
        helpers.exponentiate_velocity(torch.zeros([4, 4, 4, 3]),
                                      torch.eye(4), json=sidecar)


def test_reslice_inverse_displacement_newton(tmp_path):
    """`reslice -id DISP` uses the inversion of the smalldef sidecar"""
    import numpy as np
    from nitorch.cli.registration.reslice.main import _reslice as reslice
    nib = pytest.importorskip('nibabel')
    shape = [16, 16, 16]
    disp = make_vel(shape, 3.)
    img = _write(tmp_path / 'img.nii.gz', np.random.default_rng(0).random(shape))
    fdisp = _write(tmp_path / 'disp.nii.gz', disp)
    outs = {}
    for inversion in (None, 'default', 'push', 'newton'):
        if inversion:
            with open(tmp_path / 'disp.json', 'w') as f:
                json.dump(dict(model='smalldef', inversion=inversion), f)
        out = str(tmp_path / f'out_{inversion}.nii.gz')
        reslice([img, '-id', fdisp, '-t', img, '-o', out])
        outs[inversion] = np.asarray(nib.load(out).dataobj)
    assert np.array_equal(outs[None], outs['default'])
    assert np.array_equal(outs[None], outs['push'])
    assert not np.array_equal(outs[None], outs['newton'])


def test_vexp_inversion(tmp_path):
    import numpy as np
    from nitorch.cli.registration.vexp.cli import _cli as vexp_cli
    from nitorch.cli.registration.vexp.parser import parse, ParseError
    nib = pytest.importorskip('nibabel')
    assert parse(['vel.nii']).inversion is None
    assert parse(['vel.nii', '--inversion', 'newton']).inversion == 'newton'
    with pytest.raises(ParseError):
        parse(['vel.nii', '--inversion', 'invert'])

    shape = [12, 12, 12]
    vel = make_vel(shape, 3.)
    fvel = _write(tmp_path / 'vel.nii.gz', vel)

    def run(*args):
        out = str(tmp_path / 'out.nii.gz')
        vexp_cli([fvel, '-inv', *args, '-o', out])
        return torch.as_tensor(np.asarray(nib.load(out).dataobj)).squeeze()

    default = run()
    assert torch.allclose(default, spatial.exp(vel[None], inverse=True,
                                               displacement=True)[0],
                          atol=1e-6)
    fwd = spatial.exp(vel[None], displacement=True)[0]
    newton = spatial.grid_inv(fwd, type='disp', method='newton')
    assert torch.allclose(run('--inversion', 'newton'), newton, atol=1e-6)
    # read from the sidecar written by `register @nonlin svf`
    with open(tmp_path / 'vel.json', 'w') as f:
        json.dump(dict(model='svf', inversion='newton'), f)
    assert torch.allclose(run(), newton, atol=1e-6)
    assert torch.equal(run('--inversion', 'default'), default)
    # a shooting sidecar is ignored (vexp exponentiates stationary fields)
    with open(tmp_path / 'vel.json', 'w') as f:
        json.dump(dict(bending=0.2, inversion='newton'), f)
    assert torch.equal(run(), default)
