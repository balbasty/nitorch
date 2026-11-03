"""Number of loss evaluations spent by `nitorch register` (#155).

- The strong Wolfe line search stops zooming once the loss is flat to
  numerical precision (near convergence it used to spend up to 25
  gradient evaluations for gains ~1e-8).
- `PairwiseRegister.fit` no longer evaluates the data term at the initial
  parameters (the optimiser does it anyway), unless a loss keeps a state
  that this evaluation initialises.
- Optimisation steps only return derivatives evaluated at the parameters
  they return. After a strong Wolfe line search (which does not compute
  the Hessian), the next Gauss-Newton step evaluates a fresh Hessian
  instead of reusing the first one, and that normal evaluation commits
  the side effects of the accepted point. The loss-only commit runs once,
  after the last step. Interleaved (joint) blocks start each turn with a
  fresh evaluation.
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
#        Fresh derivatives after a Wolfe line search
# ----------------------------------------------------------------------

def _trace(kind, loss='mse', max_iter=3, outer=2):
    calls = []
    orig = {}
    for name in ('do_affine_only', 'do_affine', 'do_vel'):
        orig[name] = getattr(PairwiseRegisterStep, name)
    sim, affine, nonlin, optim = _setup(kind, loss, max_iter=max_iter)
    for opt in (optim if kind == 'joint' else [optim]):
        opt.iter.tol = 0                # exactly `max_iter` steps
    if kind == 'joint':
        optim.max_iter = outer
    try:
        for name, fn in orig.items():
            def wrapper(self, x, grad=False, hess=False, in_line_search=False,
                        _orig=fn, _name=name):
                calls.append((_name, grad, hess, in_line_search))
                return _orig(self, x, grad=grad, hess=hess,
                             in_line_search=in_line_search)
            setattr(PairwiseRegisterStep, name, wrapper)
        PairwiseRegister([sim], affine, nonlin, optim, verbose=0).fit()
    finally:
        for name, fn in orig.items():
            setattr(PairwiseRegisterStep, name, fn)
    return calls, affine, nonlin


def _runs(calls):
    # split the trace into runs of the same block
    runs = []
    for c in calls:
        if not runs or runs[-1][0][0] != c[0]:
            runs.append([])
        runs[-1].append(c)
    return runs


@pytest.mark.parametrize('kind', ['affine', 'nonlin'])
def test_hessian_is_recomputed_at_each_step(kind):
    calls, *_ = _trace(kind, max_iter=3)
    normal = [c for c in calls if not c[3]]
    # one normal evaluation with the Hessian per step (no line-search
    # trial asks for it), then one loss-only commit after the last step
    assert all(c[2] for c in normal[:-1]), normal
    assert not (normal[-1][1] or normal[-1][2]), normal
    assert len(normal) - 1 == 3, normal
    assert not any(c[2] for c in calls if c[3])


def test_joint_blocks_start_each_turn_with_fresh_derivatives():
    calls, *_ = _trace('joint', max_iter=2, outer=2)
    runs = _runs(calls)
    assert [r[0][0] for r in runs] == ['do_affine', 'do_vel'] * 2, runs
    for run in runs:
        # each turn starts with a normal evaluation of loss, gradient and
        # Hessian at the current parameters (not a line-search trial from
        # stale derivatives), and ends with the loss-only commit
        assert run[0][1:] == (True, True, False), run
        assert run[-1][1:] == (False, False, False), run
        assert sum(1 for c in run if not (c[1] or c[2] or c[3])) == 1, run


@pytest.mark.parametrize('kind', ['affine', 'nonlin', 'joint'])
@pytest.mark.parametrize('loss', ['mse', 'mad'])
def test_cached_transforms_match_final_parameters(kind, loss):
    # `pairwise_postproc` writes the outputs from the cached transforms
    # (`exp(recompute=False)`): they must be those of the final parameters
    calls, affine, nonlin = _trace(kind, loss)
    if affine is not None:
        assert torch.equal(affine.exp(recompute=False), affine.exp())
    if nonlin is not None:
        assert torch.equal(nonlin.exp(recompute=False), nonlin.exp())


def test_first_order_wolfe_still_reuses_gradient():
    # gradient descent + Wolfe: no Hessian, the trial gradient is valid at
    # the accepted point and reused; side effects are committed after each
    # line search
    from nitorch.tools.registration.optim.gd import GradientDescent
    from nitorch.tools.registration.optim.wolfe import StrongWolfeLineSearch
    from nitorch.tools.registration.optim.base import OptimIterator
    calls = []

    def closure(x, grad=False, in_line_search=False):
        calls.append((grad, in_line_search))
        f = 0.25 * x.square().sum()
        return (f, 0.5 * x) if grad else f

    gd = GradientDescent(lr=1)
    gd.search = StrongWolfeLineSearch()
    gd.iter = OptimIterator(max_iter=3, tol=0)
    gd.iter(torch.ones(1, dtype=dt), closure)
    assert calls[0] == (True, False)
    assert sum(1 for c in calls if c == (True, False)) == 1, calls
    assert calls.count((False, False)) == 3, calls       # one per step
