"""Scaling of the nonlinear regularisation across pyramid levels."""
import pytest
import torch
from nitorch import spatial
from nitorch.tools.registration import pairwise_run
from nitorch.tools.registration.objects import SVFModel, ShootModel
from nitorch.tools.registration.pairwise_makeobj import make_nonlin_optim


class _RecordFit:
    """Stand-in for PairwiseRegister: records the regularisation seen
    by the model, the optimiser and (shoot) the Greens kernel."""
    records = []

    def __init__(self, losses, affine, nonlin, optim, **kwargs):
        self.nonlin, self.optim, self.figure = nonlin, optim, None

    def fit(self):
        kernel = self.nonlin.kernel
        self.records.append(dict(
            shape=list(self.nonlin.shape),
            factor=self.nonlin.factor,
            penalty=self.nonlin.penalty['factor'],
            optim=self.optim.factor,
            kernel=None if kernel is None else kernel.clone(),
        ))


def _run(model, eps, nb_levels=3, monkeypatch=None):
    _RecordFit.records = []
    monkeypatch.setattr(pairwise_run, 'PairwiseRegister', _RecordFit)
    shape = [16, 16, 16]
    aff = spatial.affine_default(shape, voxel_size=1.5)
    nonlin = model(shape, factor=2.)
    nonlin.set_dat(torch.zeros([*shape, 3]), affine=aff)
    optim = make_nonlin_optim('gn', 2, nonlin=nonlin)
    losses = [None] * nb_levels  # never evaluated
    pairwise_run.run_pyramid(losses, None, nonlin, optim,
                             verbose=False, reg_eps=eps)
    return nonlin, _RecordFit.records


@pytest.mark.parametrize('model', [SVFModel, ShootModel])
@pytest.mark.parametrize('eps', [0, 1, 1.5])
def test_reg_scaling(model, eps, monkeypatch):
    nonlin, records = _run(model, eps, monkeypatch=monkeypatch)
    # coarse to fine: voxel size ratio 4, 2, 1
    expected = [2. * 4 ** eps, 2. * 2 ** eps, 2.]
    assert [r['shape'] for r in records] == [[4] * 3, [8] * 3, [16] * 3]
    for r, f in zip(records, expected):
        assert r['factor'] == pytest.approx(f)
        assert r['penalty'] == pytest.approx(f)
        assert r['optim'] == pytest.approx(f)
    # the finest level and the returned model use the user's factor
    assert records[-1]['factor'] == 2.
    assert nonlin.factor == 2. and nonlin.penalty['factor'] == 2.


def test_reg_scaling_shoot_kernel(monkeypatch):
    """The Greens kernel must be rebuilt with the scaled factor"""
    _, records = _run(ShootModel, 1, nb_levels=2, monkeypatch=monkeypatch)
    coarse = records[0]
    ref = ShootModel([8] * 3, factor=coarse['factor'])
    aff = spatial.affine_default([8] * 3, voxel_size=3.)
    ref.set_dat(torch.zeros([8, 8, 8, 3]), affine=aff)
    assert torch.allclose(coarse['kernel'], ref.kernel)


def test_single_level_untouched(monkeypatch):
    nonlin, records = _run(SVFModel, 1, nb_levels=1, monkeypatch=monkeypatch)
    assert records[0]['factor'] == 2.


def test_voxel_size_ratio_ignores_singleton_axes():
    ratio = pairwise_run._voxel_size_ratio([2., 2., 1.], [1., 1., 1.],
                                           [64, 64, 1])
    assert ratio == pytest.approx(2.)
