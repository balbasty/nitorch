"""`nitorch affreg`: affine-only wrapper around `nitorch register`."""
import os

import numpy as np
import pytest

import nitorch.cli.registration.affreg.cli as affreg_cli
from nitorch.cli.registration.register.parser import parser as register_parser

nib = pytest.importorskip('nibabel')


def _register_args(monkeypatch, *args):
    """affreg args -> list of arguments passed to `nitorch register`"""
    captured = []
    monkeypatch.setattr(affreg_cli, 'cli_register',
                        lambda a: captured.append(a))
    affreg_cli._cli(['-m', 'mov.nii', '-f', 'fix.nii', *args])
    assert len(captured) == 1
    return captured[0]


def test_affreg_register_args_are_strings(monkeypatch):
    args = _register_args(monkeypatch, '-n', '7', '-t', '1e-3')
    assert all(isinstance(a, str) for a in args), args
    opt = register_parser.parse(args)
    assert opt.affine.optim.max_iter == 7
    assert opt.affine.optim.tolerance == 1e-3
    assert list(opt.pyramid.levels) == [0, 1, 2]


def test_affreg_no_pyramid_args(monkeypatch):
    args = _register_args(monkeypatch, '--no-pyramid')
    opt = register_parser.parse(args)
    assert list(opt.pyramid.levels) == [0]


def test_affreg_output_false_args(monkeypatch):
    args = _register_args(monkeypatch, '-o', 'false')
    assert not register_parser.parse(args).affine.output
    args = _register_args(monkeypatch, '-o', 'aff.lta')
    assert register_parser.parse(args).affine.output == 'aff.lta'
    args = _register_args(monkeypatch)
    assert register_parser.parse(args).affine.output == '{dir}/{name}.lta'


@pytest.fixture
def images(tmp_path):
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(0)
    fnames = []
    for name, shift in (('fix', 0), ('mov', 1)):
        dat = gaussian_filter(rng.rand(12, 12, 12), 2).astype('float32')
        dat = np.roll(dat, shift, 0) * 100 + 1
        fname = str(tmp_path / f'{name}.nii.gz')
        nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
        fnames.append(fname)
    return fnames


def _ltas(path):
    return sorted(f for f in os.listdir(path) if f.endswith('.lta'))


def test_affreg_run_no_pyramid_no_output(tmp_path, images):
    fix, mov = images
    affreg_cli._cli(['-m', mov, '-f', fix, '-l', 'mse', '--no-pyramid',
                     '-n', '2', '-o', 'false', '-x', 'false', '-v', '0'])
    assert _ltas(tmp_path) == []


def test_affreg_run_writes_lta(tmp_path, images):
    fix, mov = images
    affreg_cli._cli(['-m', mov, '-f', fix, '-l', 'mse', '-p', '0',
                     '-n', '2', '-x', 'false', '-v', '0'])
    assert _ltas(tmp_path) == ['rigid.lta']


# ----------------------------------------------------------------------
#   #171: only the moving image's outputs are written
# ----------------------------------------------------------------------

def test_affreg_fixed_image_outputs_disabled(monkeypatch):
    opt = register_parser.parse(_register_args(monkeypatch))
    assert opt.loss[0].fix.output is False
    assert opt.loss[0].fix.resliced is False
    assert opt.loss[0].mov.output == '{dir}/{base}.moved{ext}'


def test_affreg_run_writes_no_fixed_moved(tmp_path, images):
    fix, mov = images
    affreg_cli._cli(['-m', mov, '-f', fix, '-l', 'mse', '-p', '0',
                     '-n', '2', '-o', 'false', '-v', '0'])
    assert not (tmp_path / 'fix.moved.nii.gz').exists()
    assert (tmp_path / 'mov.moved.nii.gz').exists()


def test_affreg_run_moving_outputs_off_writes_nothing(tmp_path, images):
    fix, mov = images
    affreg_cli._cli(['-m', mov, '-f', fix, '-l', 'mse', '-p', '0',
                     '-n', '2', '-o', 'false', '-x', 'false', '-v', '0'])
    assert sorted(os.listdir(tmp_path)) == ['fix.nii.gz', 'mov.nii.gz']
