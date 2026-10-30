import torch
import pytest

from nitorch.cli.registration.helpers import BacktrackingLineSearch


def make_problem(**kwargs):
    param = torch.tensor([3., -2.], requires_grad=True)
    optim = BacktrackingLineSearch(torch.optim.SGD([param], **kwargs))

    def closure():
        optim.zero_grad()
        loss = param.square().sum()
        loss.backward()
        return loss

    return param, optim, closure


def test_line_search_decreases_loss():
    param, optim, closure = make_problem(lr=0.1)
    loss0 = param.detach().square().sum()
    loss = optim.step(closure)
    assert optim.last_ok
    assert loss < loss0
    assert torch.allclose(loss, param.detach().square().sum())


def test_line_search_restores_params_on_failure():
    # steps go uphill, so no backtracking step can decrease the loss
    param, optim, closure = make_problem(lr=0.1, maximize=True)
    param0 = param.detach().clone()
    loss0 = param0.square().sum()
    loss = optim.step(closure)
    assert not optim.last_ok
    assert torch.equal(param.detach(), param0)
    assert torch.allclose(loss, loss0)


# --- nitorch.tools.registration.optim line searches -------------------------

from nitorch.tools.registration.optim.gd import GradientDescent
from nitorch.tools.registration.optim.linesearch import (
    StepSizeLineSearch, BacktrackingLineSearch as RegBacktracking)


def _piecewise_closure(points):
    # f(x) = x^2/4 for x > 0.25, else 10 (bad): from x=1 with GD lr=1, the
    # step to 0.5 is good but twice that step (to 0) is bad.
    def closure(param, grad=False, hess=False):
        x = param.item()
        points.append(x)
        f = torch.tensor(x * x / 4 if x > 0.25 else 10.)
        g = torch.tensor([x / 2])
        return (f, g) if grad else f
    return closure


def test_stepsize_line_search_evaluates_accepted_point():
    points = []
    search = StepSizeLineSearch(GradientDescent(lr=1), max_iter=4)
    param = torch.tensor([1.])
    param, loss = search.step(param, _piecewise_closure(points))[:2]
    assert param.item() == 0.5          # full step of lr accepted
    assert 0.5 in points and 0. not in points
    assert loss.item() == 0.5 ** 2 / 4


def test_stepsize_line_search_store_value():
    for store, expected in [(False, 1.), (0.5, 0.5)]:
        gd = GradientDescent(lr=1)
        search = StepSizeLineSearch(gd, max_iter=4, store_value=store)
        search.step(torch.tensor([1.]), _piecewise_closure([]))
        assert gd.lr == expected


def test_backtracking_line_search_prints_nothing(capsys):
    search = RegBacktracking(GradientDescent(lr=1), max_iter=4)
    search.step(torch.tensor([1.]), _piecewise_closure([]))
    search.step(torch.tensor([0.3]), _piecewise_closure([]))  # failure
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('verbose', [1, 2, 3])
def test_register_progress_output_clean(tmp_path, capsys, monkeypatch, verbose):
    # -v 1/2: progress rows are rewritten in place ('\r'); -v 3: one row per
    # line. Line searches must not add their own lines in either case.
    import sys
    import types
    import numpy as np
    import nibabel as nib
    from nitorch.cli.registration.register.cli import _cli
    if verbose > 1:  # -v >= 2 selects an interactive matplotlib backend
        mpl = pytest.importorskip('matplotlib')
        mpl.use('Agg')
        monkeypatch.setattr(mpl, 'use', lambda *a, **k: None)
    g = np.random.default_rng(0)
    for name, sh in [('fix', 0), ('mov', 1)]:
        dat = np.zeros([12] * 3, np.float32)
        dat[4:8, 4:8, 4 + sh:8 + sh] = 1
        dat += 0.01 * g.random([12] * 3, dtype=np.float32)
        nib.save(nib.Nifti1Image(dat, np.eye(4)), str(tmp_path / f'{name}.nii'))
    _cli(['-v', str(verbose), '-o', str(tmp_path), '@loss', 'mse',
          '@@fix', str(tmp_path / 'fix.nii'), '@@mov', str(tmp_path / 'mov.nii'),
          '@affine', 'translation', '@@optim', 'gn', '-n', '3',
          '@pyramid', '--levels', '1'])
    out = capsys.readouterr().out
    assert 'success' not in out and 'failure' not in out
    rows = [l for l in out.split('\n') if '(affine)' in l]
    if verbose < 3:
        assert len(rows) == 1           # all progress on one in-place line
    else:
        assert len(rows) > 1
