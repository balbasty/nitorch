"""`@nonlin -h/--hessian`: Hessian of the objective at the solution.

The CLI writes `nonlin.hess`. It used to be the Hessian returned by the
optimiser, which, after a strong Wolfe line search, was the one of its
first iteration (the line search reused it). Steps now only return the
derivatives evaluated at the parameters they return, so when the Hessian
is requested it is evaluated once at the final parameters.
"""
import glob
import os
import numpy as np
import torch
import pytest

nib = pytest.importorskip('nibabel')

from nitorch import io, spatial
from nitorch.cli.registration.register.cli import _cli
from nitorch.tools.registration import objects
from nitorch.tools.registration.losses import MSE
from nitorch.tools.registration.pairwise_impl import (
    PairwiseRegister, PairwiseRegisterStep)
from nitorch.tools.registration.pairwise_makeobj import make_nonlin_optim


def _save(path, seed=0, shift=0.):
    g = np.random.default_rng(seed)
    x = np.stack(np.meshgrid(*[np.arange(10.)] * 3, indexing='ij'), -1)
    dat = np.exp(-((x - 4.5 - shift) ** 2).sum(-1) / 8)
    dat += 0.01 * g.random(dat.shape)
    nib.save(nib.Nifti1Image(dat.astype(np.float32), np.eye(4)), path)
    return path


@pytest.fixture
def pair(tmp_path):
    fix = _save(str(tmp_path / 'fix.nii.gz'), 0)
    mov = _save(str(tmp_path / 'mov.nii.gz'), 1, shift=0.7)
    return fix, mov, str(tmp_path / 'out')


def _run(pair, *args, levels='0'):
    fix, mov, odir = pair
    _cli(['-o', odir, '@loss', 'mse', '@@fix', fix, '-o', 'false',
          '@@mov', mov, '-o', 'false', *args, '@pyramid', '-l', levels])
    return odir


@pytest.mark.parametrize('levels', ['0', '0:2'])
def test_cli_writes_hessian(pair, levels):
    odir = _run(pair, '@nonlin', 'svf', '-h', '@@optim', 'gn', '-n', '3',
                levels=levels)
    files = glob.glob(os.path.join(odir, '*hessian*'))
    assert files == [os.path.join(odir, 'svf_hessian.nii.gz')], files
    vel = io.map(os.path.join(odir, 'svf.nii.gz'))
    hess = io.map(files[0])
    # finest lattice, one channel per unique Hessian element
    assert tuple(hess.shape[:3]) == tuple(vel.shape[:3])
    assert hess.shape[-1] == 6
    assert torch.allclose(hess.affine, vel.affine)
    assert hess.fdata().abs().max() > 0


def test_cli_no_hessian_by_default(pair):
    odir = _run(pair, '@nonlin', 'svf', '@@optim', 'gn', '-n', '2')
    assert not glob.glob(os.path.join(odir, '*hessian*'))


def test_hessian_is_at_final_parameters():
    dt = torch.double
    g = torch.Generator().manual_seed(0)
    x = torch.rand([1, 10, 10, 10], dtype=dt, generator=g)
    for _ in range(2):
        for d in range(1, 4):
            x = (x + x.roll(1, d) + x.roll(-1, d)) / 3
    aff = spatial.affine_default(x.shape[1:], 1., dtype=dt)
    fix = objects.Image(x, affine=aff)
    mov = objects.Image(x.roll(1, 3), affine=aff)
    sim = objects.Similarity(MSE(lam=1, dim=3), mov, fix)
    nonlin = objects.SVFModel(steps=4, penalty=dict(
        absolute=1e-4, membrane=1e-3, bending=0.2, lame=(0.05, 0.2),
        factor=1.))
    nonlin.set_dat(fix.shape, affine=fix.affine, dtype=dt)
    nonlin.frozen = False
    optim = make_nonlin_optim('gn', 2, max_iter=3, nonlin=nonlin)

    PairwiseRegister([sim], None, nonlin, optim, verbose=0).fit()
    assert nonlin.hess is None          # not requested, not returned

    PairwiseRegister([sim], None, nonlin, optim, verbose=0,
                     hessian=True).fit()
    assert nonlin.hess is not None
    step = PairwiseRegisterStep([sim], None, nonlin, verbose=0)
    _, _, ref = step.do_vel(nonlin.parameters(), grad=True, hess=True)
    assert torch.allclose(nonlin.hess, ref)
