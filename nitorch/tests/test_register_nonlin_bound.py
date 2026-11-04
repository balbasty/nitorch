"""Boundary conditions of the nonlinear field in `nitorch register`
(`@nonlin --bound`).

The boundary condition of the field must be the same in every place
that touches it: the regulariser (value and gradient), the Gauss-Newton
solver (whose linear system includes the regulariser), the scaling and
squaring exponentiation and its backward pass, the numerical inversion,
the Jacobian and the resampling across pyramid levels. If one of them
uses another boundary condition, the gradient no longer matches the
objective, or the Gauss-Newton step does not solve the system of the
objective.
"""
import json
import pytest
import torch
from nitorch import spatial
from nitorch.core.linalg import sym_matvec
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep
from nitorch.tools.registration.pairwise_makeobj import make_nonlin_optim
from nitorch.tools.registration.pairwise_run import _sync_nonlin_optim

dt = torch.double
SHAPE = (10, 10, 10)
BOUNDS = ['dft', 'dct2', 'dct1', 'dst2', 'dst1', 'zero']
PENALTY = dict(absolute=1e-2, membrane=1e-1, bending=1.,
               lame=(0.05, 0.2), factor=1.)


def ramps():
    """(3, *SHAPE) coordinates in [-1, 1]: smooth, not periodic"""
    x = torch.linspace(-1, 1, SHAPE[0], dtype=dt)
    return torch.stack(torch.meshgrid(x, x, x, indexing='ij'))


def make_step(model, bound, penalty=PENALTY, amplitude=1., seed=0):
    """Pairwise step whose images and velocity do not wrap around: they
    differ on opposite faces of the field of view, so that the boundary
    condition of the field matters."""
    g = torch.Generator().manual_seed(seed)
    affine = spatial.affine_default(SHAPE, 1., dtype=dt)
    X = ramps()
    noise = lambda: 0.1 * torch.rand(SHAPE, dtype=dt, generator=g)
    fix = X[0] + 0.5 * X[1] ** 2 - 0.3 * X[2] + noise()
    mov = 0.8 * X[0] - 0.2 * X[1] + 0.6 * X[2] ** 2 + noise()
    fix = objects.Image(fix[None], affine=affine, bound='dct2')
    mov = objects.Image(mov[None], affine=affine, bound='dct2')
    Model = objects.NonLinModel.subclass(model)
    nonlin = Model(steps=8, penalty=dict(penalty), bound=bound)
    nonlin.set_dat(list(SHAPE), affine=affine, dtype=dt)
    vel = torch.stack([0.5 * X[1] + 0.3, -0.4 * X[0], 0.3 * X[2] ** 2])
    nonlin.dat.dat.copy_(amplitude * vel.movedim(0, -1))
    sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
    sim = objects.SumSimilarity([sim])
    return PairwiseRegisterStep(sim, None, nonlin, verbose=0), nonlin


def edge_mask():
    """Two voxels on each face of the field of view"""
    mask = torch.zeros(SHAPE, dtype=dt)
    for d in range(3):
        mask.narrow(d, 0, 2).fill_(1)
        mask.narrow(d, SHAPE[d] - 2, 2).fill_(1)
    return mask[..., None]


def directional_error(step, x):
    """Largest relative error between the analytical gradient and
    central finite differences, along the gradient and random
    directions, restricted (or not) to the edges of the field"""
    _, g = step.do_vel(x, grad=True, in_line_search=True)
    gen = torch.Generator().manual_seed(2)
    dirs = [g] + [torch.randn(x.shape, dtype=dt, generator=gen)
                  for _ in range(3)]
    dirs += [d * edge_mask() for d in dirs]
    eps = 1e-6
    err = 0
    for d in dirs:
        d = d / d.norm()
        fd = (step.do_vel(x + eps * d, in_line_search=True) -
              step.do_vel(x - eps * d, in_line_search=True)) / (2 * eps)
        an = (g * d).sum()
        err = max(err, abs(fd - an).item() / g.norm().item())
    return err


# SVF gradients go through the (approximate) scaling-and-squaring
# backward pass, whose error is concentrated in the voxels at the edges
# of the field (for all boundary conditions, two to four times as large
# with non-circulant ones). Small-deformation gradients are exact.
TOL = dict(svf=0.02, smalldef=1e-6)
SUB_ITER, RATIO = 64, 0.1  # see test_gauss_newton_bound_solves_model_system


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('model', ['svf', 'smalldef'])
def test_grad_bound(model, bound):
    """The gradient of the objective (data term + regulariser) matches
    finite differences, whatever the boundary condition"""
    step, nonlin = make_step(model, bound)
    assert nonlin.bound == bound
    assert directional_error(step, nonlin.dat.dat.clone()) < TOL[model]


@pytest.mark.parametrize('bound', BOUNDS)
def test_exp_backward_bound(monkeypatch, bound):
    """The backward pass of scaling and squaring uses the boundary
    condition of the model. (Checked directly: with dct/zero boundaries,
    its approximation error at the edges hides the effect of the
    boundary condition from finite differences.)"""
    captured = []
    exp_backward = spatial.exp_backward

    def spy(*args, **kwargs):
        captured.append(kwargs.get('bound'))
        return exp_backward(*args, **kwargs)

    monkeypatch.setattr(spatial, 'exp_backward', spy)
    step, nonlin = make_step('svf', bound)
    step.do_vel(nonlin.dat.dat.clone(), grad=True, in_line_search=True)
    assert captured == [bound]


@pytest.mark.parametrize('bound', ['dct2', 'zero'])
def test_regulariser_bound_mismatch_is_detected(bound):
    """The finite-difference check is sensitive to the boundary
    condition: a gradient whose regulariser uses another boundary
    condition (dft) does not match finite differences of the objective
    along a direction supported on the edges of the field."""
    step, nonlin = make_step('smalldef', bound)
    x = nonlin.dat.dat.clone()
    _, g = step.do_vel(x, grad=True, in_line_search=True)
    penalty = dict(nonlin.penalty)
    factor = penalty.pop('factor') / x[..., 0].numel()
    # the regulariser of the model is the one of its boundary condition
    assert torch.allclose(nonlin.regulariser(x), spatial.regulariser_grid(
        x, **penalty, factor=factor, bound=bound))
    # gradient with a dft regulariser instead
    g_wrong = g - nonlin.regulariser(x) + spatial.regulariser_grid(
        x, **penalty, factor=factor, bound='dft')
    gen = torch.Generator().manual_seed(3)
    d = torch.randn(x.shape, dtype=dt, generator=gen) * edge_mask()
    d = d / d.norm()
    eps = 1e-6
    fd = (step.do_vel(x + eps * d, in_line_search=True) -
          step.do_vel(x - eps * d, in_line_search=True)) / (2 * eps)
    assert abs(fd - (g * d).sum()) < TOL['smalldef'] * g.norm()
    assert abs(fd - (g_wrong * d).sum()) > 1e-3 * g.norm()


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('fmg', [0, 2])
@pytest.mark.parametrize('solver', ['cg', 'relax'])
def test_gauss_newton_bound(monkeypatch, bound, solver, fmg):
    """The Gauss-Newton optimizer of the field solves its linear system
    with the boundary condition of the model"""
    captured = []
    for name in ('solve_grid_fmg', 'solve_grid_sym'):
        def spy(*args, _solve=getattr(spatial, name), **kwargs):
            captured.append(kwargs.get('bound'))
            return _solve(*args, **kwargs)
        monkeypatch.setattr(spatial, name, spy)
    step, nonlin = make_step('svf', bound)
    optim = make_nonlin_optim('gn', nonlin=nonlin, solver=solver, fmg=fmg)
    assert optim.bound == bound
    _sync_nonlin_optim(optim, nonlin)  # as at each pyramid level
    assert optim.bound == bound
    _, g, h = step.do_vel(nonlin.dat.dat.clone(), grad=True, hess=True,
                          in_line_search=True)
    optim.search_direction(g, h)
    assert captured == [bound]


@pytest.mark.parametrize('bound', ['dct2', 'zero'])
def test_gauss_newton_bound_solves_model_system(bound):
    """The Gauss-Newton step solves (H + L) step = -g, with L the
    regulariser of the model (the one used in the gradient), better
    than with the dft regulariser"""
    step, nonlin = make_step('svf', bound)
    optim = make_nonlin_optim('gn', nonlin=nonlin, solver='cg', fmg=0,
                              sub_iter=SUB_ITER, marquardt=False)
    _sync_nonlin_optim(optim, nonlin)
    _, g, h = step.do_vel(nonlin.dat.dat.clone(), grad=True, hess=True,
                          in_line_search=True)
    h[..., :3] += 1e-3  # well-posed
    delta = -optim.search_direction(g.clone(), h.clone())
    penalty = dict(nonlin.penalty)
    factor = penalty.pop('factor') / delta[..., 0].numel()
    residual = lambda reg: (sym_matvec(h, delta) + reg - g).norm()
    own = residual(nonlin.regulariser(delta))
    dft = residual(spatial.regulariser_grid(delta, **penalty, factor=factor,
                                            bound='dft'))
    assert own < RATIO * dft


def test_gauss_newton_bound_synced():
    """The optimizer takes the boundary condition of the model at each
    pyramid level"""
    _, nonlin = make_step('svf', 'dct2')
    optim = make_nonlin_optim('gn', nonlin=nonlin)
    optim.bound = 'dft'
    _sync_nonlin_optim(optim, nonlin)
    assert optim.bound == 'dct2'


@pytest.mark.parametrize('bound', BOUNDS)
def test_exp_bound(bound):
    """The exponentiation (forward, inverse, Jacobian) uses the boundary
    condition of the model"""
    _, nonlin = make_step('svf', bound)
    v = nonlin.dat.dat
    kw = dict(steps=nonlin.steps, displacement=True, bound=bound)
    phi, iphi, jac, ijac = nonlin.exp2(jacobian=True)
    phi_ref, jac_ref = spatial.exp_forward(v, jacobian=True, **kw)
    iphi_ref, ijac_ref = spatial.exp_forward(v, jacobian=True, inverse=True,
                                             **kw)
    assert torch.allclose(phi, phi_ref)
    assert torch.allclose(iphi, iphi_ref)
    assert torch.allclose(jac, jac_ref)
    assert torch.allclose(ijac, ijac_ref)
    if bound != 'dft':
        phi_dft = spatial.exp_forward(v, steps=nonlin.steps,
                                      displacement=True, bound='dft')
        assert not torch.allclose(phi, phi_dft)


@pytest.mark.parametrize('model', ['svf', 'smalldef'])
@pytest.mark.parametrize('bound', ['dft', 'dct2'])
def test_inversion_bound(monkeypatch, model, bound):
    """Numerical inversion uses the boundary condition of the model"""
    captured = []
    grid_inv = spatial.grid_inv

    def spy(*args, **kwargs):
        captured.append(kwargs.get('bound'))
        return grid_inv(*args, **kwargs)

    monkeypatch.setattr(spatial, 'grid_inv', spy)
    _, nonlin = make_step(model, bound, amplitude=0.5)
    nonlin.inversion = 'push'
    nonlin.iexp()
    nonlin.exp2()
    assert captured == [bound, bound]


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('inplace', [True, False])
def test_resize_bound(bound, inplace):
    """Resampling across pyramid levels uses the boundary condition of
    the model, and keeps it"""
    _, nonlin = make_step('svf', bound)
    v, affine = nonlin.dat.dat.clone(), nonlin.affine.clone()
    shape = [6, 6, 6]
    if inplace:
        out = nonlin.downsample_(shape=shape)
        assert out is nonlin
    else:
        out = nonlin.downsample(shape=shape)
    assert out.bound == bound
    kwargs = dict(objects.Displacement._upkwargs, bound=bound)
    ref, _ = spatial.resize_grid(v, affine=affine, shape=shape, **kwargs)
    assert torch.allclose(out.dat.dat, ref)
    if bound != 'dft':
        kwargs['bound'] = 'dft'
        ref, _ = spatial.resize_grid(v, affine=affine, shape=shape, **kwargs)
        assert not torch.allclose(out.dat.dat, ref)


def test_shoot_only_dft():
    Shoot = objects.NonLinModel.subclass('shoot')
    with pytest.raises(ValueError, match='shooting'):
        Shoot(bound='dct2')
    assert Shoot(bound='dft').bound == 'dft'


def test_default_is_dft():
    for model in ('svf', 'smalldef', 'shoot'):
        assert objects.NonLinModel.subclass(model)().bound == 'dft'


# ----------------------------------------------------------------------
#   command line
# ----------------------------------------------------------------------

nib = pytest.importorskip('nibabel')


def _write(path, dat, affine=None):
    import numpy as np
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype='float32'), affine),
             str(path))
    return str(path)


@pytest.mark.parametrize('args, expected', [
    ([], 'dft'), (['--bound', 'dct2'], 'dct2'), (['--bound', 'zero'], 'zero'),
])
def test_parse_bound(args, expected):
    from nitorch.cli.registration.register.parser import parser
    options = parser.parse(['@loss', 'mse', '@@fix', 'fix.nii',
                            '@@mov', 'mov.nii', '@nonlin', 'svf', *args])
    assert options.nonlin.bound == expected


def test_parse_bound_invalid():
    from nitorch.core.cli import ParseError
    from nitorch.cli.registration.register.parser import parser
    with pytest.raises(ParseError):
        parser.parse(['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii',
                      '@nonlin', 'svf', '--bound', 'nope'])


def test_parse_bound_shoot_rejected():
    from nitorch.core.cli import ParseError
    from nitorch.cli.registration.register.parser import parser
    from nitorch.cli.registration.register.cli import check_options
    options = parser.parse(['@loss', 'mse', '@@fix', 'fix.nii',
                            '@@mov', 'mov.nii', '@nonlin', 'shoot',
                            '--bound', 'dct2'])
    with pytest.raises(ParseError, match='shoot'):
        check_options(options)


@pytest.mark.parametrize('model', ['svf', 'smalldef'])
def test_register_bound_json_and_reuse(tmp_path, model):
    """`register @nonlin --bound` builds the model with it, saves it in
    the JSON sidecar, and `reslice`/`compose`/`vexp` use it"""
    import numpy as np
    from nitorch.cli.registration.register.cli import _cli
    from nitorch.cli.registration import helpers
    from nitorch.cli.registration.vexp.main import vexp
    rng = np.random.default_rng(0)
    fix = _write(tmp_path / 'fix.nii.gz', rng.random([12, 12, 12]))
    mov = _write(tmp_path / 'mov.nii.gz', rng.random([12, 12, 12]))
    out = str(tmp_path / 'nl.nii.gz')
    _cli(['@loss', 'mse', '@@fix', fix, '-o', 'false', '@@mov', mov,
          '-o', 'false', '@nonlin', model, '--bound', 'dct2', '-o', out,
          '@@optim', 'gn', '-n', '2', '@optim', '-n', '1',
          '@pyramid', '-l', '0'])
    sidecar = str(tmp_path / 'nl.json')
    with open(sidecar) as f:
        prm = json.load(f)
    assert prm['bound'] == 'dct2'
    assert helpers.load_sidecar_bound(sidecar) == 'dct2'

    dat = torch.as_tensor(np.asarray(nib.load(out).dataobj), dtype=torch.float)
    dat = dat.reshape([12, 12, 12, 3])
    if model == 'svf':
        # reslice/compose
        for inversion in ('default', 'push'):
            prm['inversion'] = inversion
            with open(sidecar, 'w') as f:
                json.dump(prm, f)
            for t0, t1 in ((0, 1), (1, 0)):
                disp = helpers.exponentiate_velocity(
                    dat, torch.eye(4), json=sidecar, t0=t0, t1=t1)
                if t1 < t0 and inversion != 'default':
                    ref = spatial.exp(dat[None], displacement=True,
                                      bound='dct2')[0]
                    ref = spatial.grid_inv(ref, type='disp', bound='dct2',
                                           method=inversion)
                else:
                    ref = spatial.exp(dat[None], displacement=True,
                                      bound='dct2', inverse=t1 < t0)[0]
                assert torch.allclose(disp, ref)
        # vexp: from the sidecar, unless given
        prm['inversion'] = 'default'
        with open(sidecar, 'w') as f:
            json.dump(prm, f)
        for bound, ref_bound in ((None, 'dct2'), ('dft', 'dft')):
            fphi = vexp(out, output=str(tmp_path / 'phi.nii.gz'),
                        bound=bound)
            phi = torch.as_tensor(np.asarray(nib.load(fphi).dataobj))
            phi = phi.reshape([12, 12, 12, 3]).float()
            ref = spatial.exp(dat[None], displacement=True,
                              bound=ref_bound)[0]
            assert torch.allclose(phi, ref, atol=1e-5)
    else:
        idisp = helpers.invert_displacement(dat, out)
        ref = spatial.grid_inv(dat, type='disp', bound='dct2')
        assert torch.allclose(idisp, ref)


def test_old_sidecar_is_dft(tmp_path):
    """Sidecars without "bound" (older ones, shooting) are dft"""
    from nitorch.cli.registration import helpers
    sidecar = str(tmp_path / 'vel.json')
    with open(sidecar, 'w') as f:
        json.dump(dict(model='svf', inversion='default'), f)
    assert helpers.load_sidecar_bound(sidecar) == 'dft'
