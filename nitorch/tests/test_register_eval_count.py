"""Number of loss evaluations spent by `nitorch register` (#155).

- The strong Wolfe line search stops zooming once the loss is flat to
  numerical precision (near convergence it used to spend up to 25
  gradient evaluations for gains ~1e-8).
- `PairwiseRegister.fit` no longer evaluates the data term at the initial
  parameters (the optimiser does it anyway), unless a loss keeps a state
  that this evaluation initialises.
"""
import torch
import pytest
from nitorch import spatial
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE, MAD
from nitorch.tools.registration.optim.wolfe import StrongWolfe
from nitorch.tools.registration.pairwise_impl import (
    PairwiseRegister, PairwiseRegisterStep)
from nitorch.tools.registration.pairwise_makeobj import (
    make_affine_optim, make_nonlin_optim)
from nitorch.tools.registration.pairwise_run import build_joint_optim

dt = torch.double


# ----------------------------------------------------------------------
#                        Wolfe: stagnation stop
# ----------------------------------------------------------------------

def _wolfe(closure, **kwargs):
    f0, g0 = closure(0, grad=True)
    return StrongWolfe(**kwargs).step(0, closure, torch.ones(1, dtype=dt),
                                      loss=f0, grad=g0)


def _counted(fn):
    calls = []

    def closure(a, grad=False):
        calls.append(float(a))
        f, g = fn(float(a))
        f, g = torch.tensor(f, dtype=dt), torch.tensor([g], dtype=dt)
        return (f, g) if grad else f
    return closure, calls


def _flat(a):
    # Loss flat to ~1e-10 (relative), with a gradient that never meets
    # the curvature condition -- what float32 losses look like near
    # convergence.
    return 1 + 1e-10 * (a - 0.3) ** 2, -1.


def test_wolfe_stops_when_loss_is_flat():
    closure, calls = _counted(_flat)
    a, f, g = _wolfe(closure)
    n_new = len(calls)
    assert f <= 1 + 1e-10 * 0.09       # never worse than the start
    closure, calls = _counted(_flat)
    _wolfe(closure, ftol=0)            # previous behaviour
    n_old = len(calls)
    assert n_new <= 3 < n_old, (n_new, n_old)


def test_wolfe_still_backtracks_after_overshoot():
    # f(a) = (a - 0.1)^2: the first trial (a = 1) is much worse than f0;
    # the zoom must still find the minimum (worse trials are not
    # "stagnation").
    closure, calls = _counted(lambda a: ((a - 0.1) ** 2, 2 * (a - 0.1)))
    a, f, g = _wolfe(closure)
    assert abs(a - 0.1) < 1e-2 and f < 0.1 ** 2


def test_wolfe_still_expands():
    # f(a) = (a - 50)^2 from a = 0 with lr = 1: the step must grow
    closure, calls = _counted(lambda a: ((a - 50) ** 2, 2 * (a - 50)))
    a, f, g = _wolfe(closure)
    assert a > 10 and f < 40 ** 2


def test_wolfe_line_search_exposes_ftol():
    from nitorch.tools.registration.optim.wolfe import StrongWolfeLineSearch
    ls = StrongWolfeLineSearch(ftol=1e-3)
    assert ls.ftol == 1e-3 and ls.wolfe.ftol == 1e-3
    ls.ftol = 0
    assert ls.wolfe.ftol == 0


# ----------------------------------------------------------------------
#                     fit(): initial evaluation
# ----------------------------------------------------------------------

def _images(shape=(10, 10, 10)):
    g = torch.Generator().manual_seed(0)
    x = torch.rand([1, *shape], dtype=dt, generator=g)
    for _ in range(2):
        for d in range(1, 4):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    aff = spatial.affine_default(shape, 1., dtype=dt)
    fix = objects.Image(x, affine=aff)
    mov = objects.Image(x.roll(1, 3), affine=aff)
    return fix, mov


def _setup(kind, loss='mse', max_iter=3):
    fix, mov = _images()
    lossobj = MSE(lam=1, dim=3) if loss == 'mse' else MAD(dim=3)
    sim = objects.Similarity(lossobj, mov, fix)
    affine = nonlin = None
    if kind in ('affine', 'joint'):
        affine = objects.AffineModel('rigid', position='sym',
                                     penalty=[1e-5] * 12)
        affine.frozen = False
    if kind in ('nonlin', 'joint'):
        nonlin = objects.SVFModel(steps=4, penalty=dict(
            absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2),
            factor=1.))
        nonlin.set_dat(fix.shape, affine=fix.affine, dtype=dt)
        nonlin.frozen = False
    aopt = make_affine_optim(None, 2, max_iter=max_iter) if affine else None
    nopt = (make_nonlin_optim(None, 2, max_iter=max_iter, nonlin=nonlin)
            if nonlin else None)
    optim = build_joint_optim(aopt, nopt, max_iter=2, sequential=False)
    return sim, affine, nonlin, optim


def _record(monkeypatch):
    calls = []
    for name in ('do_affine_only', 'do_affine', 'do_vel'):
        orig = getattr(PairwiseRegisterStep, name)

        def wrapper(self, x, grad=False, hess=False, in_line_search=False,
                    _orig=orig, _name=name):
            calls.append((_name, grad, hess, in_line_search))
            return _orig(self, x, grad=grad, hess=hess,
                         in_line_search=in_line_search)
        monkeypatch.setattr(PairwiseRegisterStep, name, wrapper)
    return calls


def _fit(kind, loss='mse', force_init=False, monkeypatch=None):
    sim, affine, nonlin, optim = _setup(kind, loss)
    if force_init:
        # pretend the loss is stateful: the initial evaluation runs, as
        # before #155
        monkeypatch.setattr(sim.loss, 'get_state', lambda: {})
    PairwiseRegister([sim], affine, nonlin, optim, verbose=0).fit()
    return [m.dat.dat.clone() for m in (affine, nonlin) if m is not None]


@pytest.mark.parametrize('kind', ['affine', 'nonlin', 'joint'])
def test_no_duplicate_initial_evaluation(kind, monkeypatch):
    calls = _record(monkeypatch)
    _fit(kind)
    # the first evaluation is the optimiser's (with derivatives), not a
    # loss-only evaluation at the same parameters
    assert calls[0][1], calls[:3]


@pytest.mark.parametrize('kind', ['affine', 'nonlin', 'joint'])
def test_initial_evaluation_kept_for_stateful_losses(kind, monkeypatch):
    calls = _record(monkeypatch)
    _fit(kind, loss='mad')
    assert not calls[0][1] and not calls[0][2], calls[:3]


@pytest.mark.parametrize('kind', ['affine', 'nonlin', 'joint'])
def test_removing_initial_evaluation_does_not_change_result(kind, monkeypatch):
    new = _fit(kind)
    with monkeypatch.context() as m:
        old = _fit(kind, force_init=True, monkeypatch=m)
    assert max(float(x.abs().max()) for x in new) > 1e-4  # it did move
    for x, y in zip(new, old):
        assert torch.allclose(x, y, rtol=0, atol=1e-12), (x - y).abs().max()


# ----------------------------------------------------------------------
#        L-BFGS: side effects committed at the accepted point
# ----------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['affine', 'nonlin'])
@pytest.mark.parametrize('loss', ['mse', 'mad'])
def test_lbfgs_commits_at_accepted_point(kind, loss):
    # `pairwise_postproc` writes the outputs from the cached transforms
    # (`exp(recompute=False)`), set by the normal evaluation that follows
    # each line search: it must be at the final parameters, not the
    # previous ones
    fix, mov = _images()
    lossobj = MSE(lam=1, dim=3) if loss == 'mse' else MAD(dim=3)
    sim = objects.Similarity(lossobj, mov, fix)
    affine = nonlin = None
    if kind == 'affine':
        affine = objects.AffineModel('rigid', position='sym',
                                     penalty=[1e-5] * 12)
        affine.frozen = False
        optim = make_affine_optim('lbfgs', max_iter=3)
    else:
        nonlin = objects.SVFModel(steps=4, penalty=dict(
            absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2),
            factor=1.))
        nonlin.set_dat(fix.shape, affine=fix.affine, dtype=dt)
        nonlin.frozen = False
        optim = make_nonlin_optim('lbfgs', max_iter=3, nonlin=nonlin)
    optim.iter.tol = 0
    PairwiseRegister([sim], affine, nonlin, optim, verbose=0).fit()
    model = affine if kind == 'affine' else nonlin
    assert float(model.dat.dat.abs().max()) > 0      # it did move
    assert torch.equal(model.exp(recompute=False), model.exp())
