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


def test_group_options_parsed():
    options = parser.parse(IMAGES + ['@affine', 'affine', '2',
                                     '-t', '5', '-s', '7'])
    assert options.affine.factor == [2.]
    assert options.affine.translation == 5.
    assert options.affine.shear == 7.
    assert options.affine.rotation is None
    assert options.affine.zoom is None
    options = parser.parse(IMAGES + ['@affine', 'affine', '2'])
    for g in ('translation', 'rotation', 'zoom', 'shear'):
        assert getattr(options.affine, g) is None
    options = parser.parse(IMAGES + ['@affine', '--rotation-penalty', '3',
                                     '--zoom-penalty', '4',
                                     '--translation-penalty', '1',
                                     '--shear-penalty', '6'])
    assert (options.affine.translation, options.affine.rotation,
            options.affine.zoom, options.affine.shear) == (1., 3., 4., 6.)


def _cli_affine(tmp_path, monkeypatch, affine_args):
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
            '@affine'] + affine_args
    with pytest.raises(Stop):
        register_cli._cli(args)
    return seen['affine']


@pytest.mark.parametrize('basis, args, values', [
    # group option on its own: FACTOR (default 0) fills the rest
    ('affine', ['-t', '5'], dict(t=5., r=0., z=0., s=0.)),
    # FACTOR fills the groups not given
    ('affine', ['2', '-r', '3', '-s', '7'], dict(t=2., r=3., z=2., s=7.)),
    ('affine', ['1', '--translation-penalty', '4', '--zoom-penalty', '6'],
     dict(t=4., r=1., z=6., s=1.)),
    ('similitude', ['2', '-z', '9'], dict(t=2., r=2., z=9.)),
    ('rigid', ['-r', '8'], dict(t=0., r=8.)),
])
def test_group_options_reach_model(tmp_path, monkeypatch, basis, args,
                                   values):
    model = _cli_affine(tmp_path, monkeypatch, [basis] + args)
    assert isinstance(model, objects.AffineModel)
    # model.dat is only allocated by the fit: map the stored penalty to
    # per-parameter indices on an allocated model of the same basis
    expected = torch.as_tensor([values[c] for c in GROUPS[3, basis]],
                               dtype=dt)
    assert torch.equal(_model(basis, 3, model._penalty).penalty, expected)


@pytest.mark.parametrize('args, expected', [([], None), (['2.5'], 2.5)])
def test_no_group_options_unchanged(tmp_path, monkeypatch, args, expected):
    # without group options the penalty is the scalar FACTOR, as before
    model = _cli_affine(tmp_path, monkeypatch, ['affine'] + args)
    assert model.penalty == expected
    assert isinstance(model.penalty, (float, type(None)))


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


# ---------------------------------------------------------------------
#   Per-component (diagonal) precision: vector and dict forms
# ---------------------------------------------------------------------

GROUPS = {
    (3, 'translation'): 'ttt', (3, 'rotation'): 'rrr', (3, 'rigid'): 'tttrrr',
    (3, 'similitude'): 'tttrrrz', (3, 'affine'): 'tttrrrzzzsss',
    (2, 'translation'): 'tt', (2, 'rotation'): 'r', (2, 'rigid'): 'ttr',
    (2, 'similitude'): 'ttrz', (2, 'affine'): 'ttrzzs',
}
PDICT = dict(translation=1., rotation=2., zoom=3., shear=4.)
PVAL = dict(t=1., r=2., z=3., s=4.)


def _model(basis, dim, penalty):
    if dim == 3:
        a = objects.AffineModel(basis, penalty=penalty)
    else:
        a = objects.Affine2dModel(basis, plane=2,
                                  ref_affine=torch.eye(4, dtype=dt),
                                  penalty=penalty)
    a.set_dat(dim=3, dtype=dt)
    return a


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('basis', ['translation', 'rotation', 'rigid',
                                   'similitude', 'affine'])
def test_dict_penalty_indices(dim, basis):
    a = _model(basis, dim, PDICT)
    expected = torch.as_tensor([PVAL[c] for c in GROUPS[dim, basis]], dtype=dt)
    assert len(a.dat.dat) == len(expected)
    assert torch.equal(a.penalty, expected)
    # short keys and partial dicts (missing groups = 0)
    a = _model(basis, dim, dict(t=5., sh=7.))
    expected = torch.as_tensor([{'t': 5., 's': 7.}.get(c, 0.)
                                for c in GROUPS[dim, basis]], dtype=dt)
    assert torch.equal(a.penalty, expected)
    # 's' is a short key for shear
    a = _model(basis, dim, dict(t=5., s=7.))
    assert torch.equal(a.penalty, expected)


def test_dict_penalty_bad_key():
    with pytest.raises(ValueError):
        objects.AffineModel('affine', penalty=dict(foo=1.))


def test_vector_penalty():
    vec = torch.arange(1., 13., dtype=dt)
    a = _model('affine', 3, vec.tolist())
    assert torch.equal(a.penalty, vec)
    # an affine-basis vector, mapped group-wise to similitude and rigid
    s = a.switch_basis('similitude')
    assert torch.equal(s.penalty,
                       torch.as_tensor([1, 2, 3, 4, 5, 6, 8], dtype=dt))
    r = a.switch_basis('rigid')
    assert torch.equal(r.penalty, vec[:6])
    # similitude-basis vector -> affine: iso zoom broadcast, no shear
    b = _model('affine', 3, [1., 2, 3, 4, 5, 6, 9])
    assert torch.equal(b.penalty, torch.as_tensor(
        [1, 2, 3, 4, 5, 6, 9, 9, 9, 0, 0, 0], dtype=dt))
    with pytest.raises(ValueError):
        _model('affine', 3, [1., 2.]).penalty
    with pytest.raises(ValueError):
        objects.AffineModel('affine', penalty=[1., -1.])


def test_penalty_cache_invalidation():
    a = _model('affine', 3, PDICT)
    p1 = a.penalty
    assert a.penalty is p1                       # cached
    a.penalty = dict(rotation=9.)
    assert a.penalty is not p1 and a.penalty[3] == 9. and a.penalty[0] == 0.
    s = a.switch_basis('rigid')
    assert len(s.penalty) == 6 and len(a.penalty) == 12
    a.set_dat(dim=3, dtype=torch.float)
    assert a.penalty.dtype == torch.float
    a.penalty = 2.
    assert a.penalty == 2.


@pytest.mark.parametrize('optim', [None, 'rigid', 'similitude'])
@pytest.mark.parametrize('step_name', ['do_affine_only', 'do_affine'])
def test_objective_vs_fd_nonuniform(optim, step_name):
    fix, mov = make_images([8, 8, 8])
    P = torch.linspace(5., 60., 12, dtype=dt)
    affine = prior_affine(P.tolist(), optim)
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
    # exact penalty terms
    n = len(par)
    gp, hp = torch.zeros(n, dtype=dt), torch.zeros(n, n, dtype=dt)
    val = step._affine_penalty(x0, gp, hp)
    assert torch.allclose(val, 0.5 * (P * x0 * x0).sum())
    assert torch.allclose(gp, M.T @ (P * x0))
    assert torch.allclose(hp, M.T @ torch.diag(P) @ M)
