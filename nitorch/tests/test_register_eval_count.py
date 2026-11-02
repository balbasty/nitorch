"""Number of loss evaluations spent by `nitorch register` (#155).

- The strong Wolfe line search stops zooming once the loss is flat to
  numerical precision (near convergence it used to spend up to 25
  gradient evaluations for gains ~1e-8).
"""
import torch
from nitorch.tools.registration.optim.wolfe import StrongWolfe

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
