"""`@@optim nes|ogm -o/--output`: final momentum map of the nonlinear optimizer.

The option used to be parsed and passed to the optimizer constructor,
where it was silently swallowed: nothing was ever written. It now writes
the optimizer's final momentum state (`delta`) once, after the fit, next
to the velocity. For `@affine`, the momentum is a vector of Lie-algebra
coefficients and is written to a JSON file instead.
"""
import glob
import json
import os
import numpy as np
import torch
import pytest

nib = pytest.importorskip('nibabel')

from nitorch import io
from nitorch.cli.registration.register import cli as register_cli
from nitorch.cli.registration.register.parser import parser
from nitorch.cli.registration.register.cli import _cli


BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii']


@pytest.mark.parametrize('name', ['nes', 'ogm'])
def test_parser_default(name):
    options = parser.parse(BASE + ['@nonlin', '@@optim', name])
    assert not options.nonlin.optim.output
    options = parser.parse(BASE + ['@nonlin', '@@optim', name, '-o'])
    assert options.nonlin.optim.output == '{dir}{sep}momentum.nii.gz'


# ----------------------------------------------------------------------
#   end-to-end
# ----------------------------------------------------------------------

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
    odir = str(tmp_path / 'out')
    return fix, mov, odir


@pytest.fixture
def optims(monkeypatch):
    """Record the nonlinear optimizers built by the CLI"""
    built = []
    make = register_cli.make_nonlin_optim

    def record(*args, **kwargs):
        built.append(make(*args, **kwargs))
        return built[-1]
    monkeypatch.setattr(register_cli, 'make_nonlin_optim', record)
    return built


def _run(pair, *args, levels='0'):
    fix, mov, odir = pair
    _cli(['-o', odir, '@loss', 'mse', '@@fix', fix, '-o', 'false',
          '@@mov', mov, '-o', 'false', *args, '@pyramid', '-l', levels])
    return odir


def _check_final_momentum(fname, odir, optim):
    vel = io.map(os.path.join(odir, 'svf.nii.gz'))
    mom = io.map(fname)
    assert tuple(mom.shape) == tuple(vel.shape)
    assert torch.allclose(mom.affine, vel.affine)
    dat = mom.fdata(dtype=torch.double)
    ref = optim.delta.detach().double().cpu()
    assert dat.abs().max() > 0
    # written as float32
    assert torch.allclose(dat, ref, rtol=1e-6, atol=1e-6 * ref.abs().max())


@pytest.mark.parametrize('name', ['nes', 'ogm'])
def test_cli_writes_final_momentum(pair, optims, name):
    odir = _run(pair, '@nonlin', 'svf', '@@optim', name, '-n', '3', '-o')
    files = glob.glob(os.path.join(odir, 'momentum*'))
    assert files == [os.path.join(odir, 'momentum.nii.gz')]
    assert len(optims) == 1
    _check_final_momentum(files[0], odir, optims[0])


def test_cli_momentum_custom_path_and_pyramid(pair, optims):
    """Written once, at the end: on the finest lattice, like the velocity"""
    _, _, odir = pair
    out = os.path.join(odir, 'mom', '{name}_momentum.nii.gz')
    os.makedirs(os.path.dirname(out))
    _run(pair, '@nonlin', 'svf', '@@optim', 'nes', '-n', '2', '-o', out,
         levels='0:2')
    files = glob.glob(os.path.join(odir, 'mom', '*'))
    assert files == [os.path.join(odir, 'mom', 'svf_momentum.nii.gz')]
    _check_final_momentum(files[0], odir, optims[0])


def test_cli_no_momentum_by_default(pair):
    odir = _run(pair, '@nonlin', 'svf', '@@optim', 'nes', '-n', '2')
    assert not glob.glob(os.path.join(odir, 'momentum*'))
    assert os.path.exists(os.path.join(odir, 'svf.nii.gz'))


@pytest.mark.parametrize('name', ['gd', 'mom'])
def test_cli_other_optimizers_unaffected(pair, name):
    odir = _run(pair, '@nonlin', 'svf', '@@optim', name, '-n', '2')
    assert not glob.glob(os.path.join(odir, 'momentum*'))
    assert os.path.exists(os.path.join(odir, 'svf.nii.gz'))


@pytest.fixture
def affine_optims(monkeypatch):
    """Record the affine optimizers built by the CLI"""
    built = []
    make = register_cli.make_affine_optim

    def record(*args, **kwargs):
        built.append(make(*args, **kwargs))
        return built[-1]
    monkeypatch.setattr(register_cli, 'make_affine_optim', record)
    return built


def _check_affine_momentum(fname, optim, basis):
    with open(fname) as f:
        prm = json.load(f)
    assert set(prm) == {'basis', 'momentum'}
    assert prm['basis'] == basis
    ref = optim.delta.detach().double().cpu()
    mom = torch.as_tensor(prm['momentum'], dtype=torch.double)
    assert mom.shape == ref.shape
    assert mom.abs().max() > 0
    assert torch.allclose(mom, ref)


@pytest.mark.parametrize('name', ['nes', 'ogm'])
def test_cli_affine_writes_final_momentum(pair, affine_optims, name):
    odir = _run(pair, '@affine', 'rigid',
                '@@optim', name, '-n', '2', '-o')
    assert os.path.exists(os.path.join(odir, 'rigid.lta'))
    assert not glob.glob(os.path.join(odir, 'momentum*'))
    files = glob.glob(os.path.join(odir, '*momentum*'))
    assert files == [os.path.join(odir, 'affine_momentum.json')]
    assert len(affine_optims) == 1
    _check_affine_momentum(files[0], affine_optims[0], 'rigid')
    assert len(affine_optims[0].delta) == 6


def test_cli_affine_momentum_custom_path(pair, affine_optims):
    _, _, odir = pair
    out = os.path.join(odir, '{name}_mom.json')
    _run(pair, '@affine', 'translation', '@@optim', 'nes', '-n', '2',
         '-o', out)
    fname = os.path.join(odir, 'translation_mom.json')
    _check_affine_momentum(fname, affine_optims[0], 'translation')


def test_cli_affine_and_nonlin_momentum_do_not_collide(
        pair, optims, affine_optims):
    odir = _run(pair, '@affine', 'translation',
                '@@optim', 'nes', '-n', '2', '-o',
                '@nonlin', 'svf', '@@optim', 'nes', '-n', '2', '-o')
    files = sorted(glob.glob(os.path.join(odir, '*momentum*')))
    assert files == [os.path.join(odir, 'affine_momentum.json'),
                     os.path.join(odir, 'momentum.nii.gz')]
    _check_affine_momentum(files[0], affine_optims[0], 'translation')
    _check_final_momentum(files[1], odir, optims[0])


def test_cli_affine_no_momentum_by_default(pair):
    odir = _run(pair, '@affine', 'translation', '@@optim', 'nes', '-n', '2')
    assert not glob.glob(os.path.join(odir, '*momentum*'))
    assert os.path.exists(os.path.join(odir, 'translation.lta'))
