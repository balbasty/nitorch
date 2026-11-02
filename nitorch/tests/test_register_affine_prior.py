"""Scaled-identity Gaussian prior on the affine Lie parameters (`@affine FACTOR`).

penalty = 1/2 lambda ||q||^2, gradient lambda q, Hessian lambda I
(mapped with the Gram matrix M into the `-g` sub-basis)."""
import torch
import pytest
from nitorch import io, spatial
from nitorch.cli.registration.register import cli as register_cli
from nitorch.cli.registration.register.parser import parser
from nitorch.tools.registration import objects
from nitorch.tools.registration.pairwise_impl import PairwiseRegisterStep
from nitorch.tests.test_register_symmetric import (
    make_images, make_sim, make_svf, SHIFT)

dt = torch.double
IMAGES = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii']


def prior_affine(lam, optim=None, basis='affine'):
    affine = objects.AffineModel(basis, penalty=lam, optim=optim, position='m')
    affine.set_dat(dim=3, dtype=dt)
    g = torch.Generator().manual_seed(5)
    affine.dat.dat.copy_(0.05 * torch.randn([len(affine.dat.dat)], dtype=dt,
                                            generator=g))
    return affine


@pytest.mark.parametrize('args, expected', [
    ([], 0.), (['rigid', '3.5'], 3.5), (['affine', '2'], 2.)])
def test_factor_parsed(args, expected):
    options = parser.parse(IMAGES + ['@affine'] + args)
    assert options.affine.factor == [expected]


@pytest.mark.parametrize('factor, expected', [(None, None), ('0', None),
                                              ('7.5', 7.5)])
def test_factor_reaches_model(tmp_path, monkeypatch, factor, expected):
    shape = [8, 8, 8]
    aff = spatial.affine_default(shape)
    fix, mov = str(tmp_path / 'f.nii.gz'), str(tmp_path / 'm.nii.gz')
    io.savef(torch.rand(shape), fix, affine=aff)
    io.savef(torch.rand(shape), mov, affine=aff)
    seen = {}

    class Stop(Exception):
        pass

    def fake_run(losses, affine, *a, **k):
        seen['affine'] = affine
        raise Stop
    monkeypatch.setattr(register_cli, 'run', fake_run)
    args = ['-v', '0', '--cpu', '@loss', 'mse', '@@fix', fix, '@@mov', mov,
            '@affine', 'rigid'] + ([factor] if factor else [])
    with pytest.raises(Stop):
        register_cli._cli(args)
    assert isinstance(seen['affine'], objects.AffineModel)
    assert seen['affine'].penalty == expected


def test_model_penalty_scalar():
    assert objects.AffineModel('rigid').penalty is None
    assert objects.AffineModel('rigid', penalty=0).penalty is None
    a = objects.AffineModel('rigid', penalty=torch.tensor(2.))
    assert a.penalty == 2. and isinstance(a.penalty, float)
    assert a.switch_basis('affine').penalty == 2.
    with pytest.raises(ValueError):
        objects.AffineModel('rigid', penalty=-1)


@pytest.mark.parametrize('optim', [None, 'rigid', 'similitude'])
def test_penalty_value_grad_hess(optim):
    fix, mov = make_images([6, 6, 6])
    lam = 3.
    affine = prior_affine(lam, optim)
    step = PairwiseRegisterStep(make_sim(mov, fix, 'fwd'), affine, verbose=0)
    par = affine.parameters()
    q = affine.dat.dat.clone()
    n = len(par)
    g, h = torch.zeros(n, dtype=dt), torch.zeros(n, n, dtype=dt)
    val = step._affine_penalty(q, g, h)
    assert torch.allclose(val, 0.5 * lam * q.dot(q))
    if optim is None:
        assert torch.allclose(g, lam * q)
        assert torch.allclose(h, lam * torch.eye(n, dtype=dt))
    else:
        M = par._gram(q)
        assert torch.allclose(g, M.T @ (lam * q))
        assert torch.allclose(h, lam * M.T @ M)
        assert torch.allclose(h, lam * torch.eye(n, dtype=dt))  # orthonormal


@pytest.mark.parametrize('optim', [None, 'rigid'])
@pytest.mark.parametrize('step_name', ['do_affine_only', 'do_affine'])
def test_objective_vs_fd(optim, step_name):
    lam = 50.
    fix, mov = make_images([8, 8, 8])
    affine = prior_affine(lam, optim)
    nonlin = make_svf(fix) if step_name == 'do_affine' else None
    step = PairwiseRegisterStep(make_sim(mov, fix, 'fwd'), affine, nonlin,
                                verbose=0)
    fn = getattr(step, step_name)
    par = affine.parameters()
    x0 = affine.dat.dat.clone()
    _, g, _ = fn(x0, grad=True, hess=True, in_line_search=True)
    M = torch.eye(len(x0), dtype=dt) if optim is None else par._gram(x0)
    fd = torch.zeros(len(par), dtype=dt)
    eps = 1e-5
    for i in range(len(par)):
        fd[i] = (fn(x0 + eps * M[:, i], in_line_search=True) -
                 fn(x0 - eps * M[:, i], in_line_search=True)) / (2 * eps)
    assert (fd - g).norm() / fd.norm() < 1e-4
    # the prior is counted exactly once in the objective
    ll = fn(x0, in_line_search=True)
    affine._penalty = None
    ll0 = fn(x0, in_line_search=True)
    assert torch.allclose(ll - ll0, 0.5 * lam * x0.dot(x0))


@pytest.mark.parametrize('lam', [None, 0.])
def test_off_is_bitwise_unchanged(lam):
    fix, mov = make_images([8, 8, 8])
    out = []
    for pen in (lam, 'none'):
        affine = prior_affine(None if pen == 'none' else pen)
        assert affine.penalty is None
        step = PairwiseRegisterStep(make_sim(mov, fix, 'fwd'), affine,
                                    verbose=0)
        out.append(step.do_affine_only(affine.dat.dat.clone(), grad=True,
                                       hess=True, in_line_search=True))
    for a, b in zip(*out):
        assert torch.equal(a, b)


def test_large_prior_shrinks_estimate(tmp_path):
    shape = [24, 24, 24]
    grid = spatial.identity_grid(shape, dtype=dt)
    center = torch.as_tensor(shape, dtype=dt) / 2

    def blob(shift):
        x = grid - (center + torch.as_tensor(shift, dtype=dt))
        x = x / torch.as_tensor([5., 3.5, 4.], dtype=dt)
        return x.square().sum(-1).neg().exp().float()
    aff = spatial.affine_default(shape, voxel_size=1.)
    fix, mov = str(tmp_path / 'fix.nii.gz'), str(tmp_path / 'mov.nii.gz')
    io.savef(blob([0, 0, 0]), fix, affine=aff)
    io.savef(blob(SHIFT), mov, affine=aff)

    def shift(factor):
        lta = str(tmp_path / f'aff{factor}.lta')
        register_cli._cli(
            ['-v', '0', '--cpu', '@loss', 'mse',
             '@@fix', fix, '-o', 'false', '-r', 'false',
             '@@mov', mov, '-o', 'false', '-r', 'false',
             '@affine', 'translation', factor, '-o', lta,
             '@@optim', 'gn', '-n', '50', '@pyramid', '--levels', '1'])
        t = io.transforms.loadf(lta).reshape([4, 4])[:3, -1]
        return t.norm().item()
    t0, t1 = shift('0'), shift('1e6')
    assert t0 > 0.5 * torch.as_tensor(SHIFT).norm()
    assert t1 < 0.1 * t0
