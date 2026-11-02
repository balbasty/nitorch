"""`nitorch autoreg`: autograd-based registration (issue #171)."""
import os
import signal
import numpy as np
import pytest
import torch
import nibabel as nib
import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands
from nitorch import io
from nitorch.cli.registration.autoreg import struct
from nitorch.cli.registration.autoreg.parser import parse, ParseError, help
from nitorch.cli.registration.autoreg.main import _autoreg


SMOKE_TIMEOUT = 120  # seconds


@pytest.fixture
def timeout():
    """Fail (instead of hanging) if a smoke run does not terminate."""
    if not hasattr(signal, 'SIGALRM'):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f'autoreg did not terminate '
                           f'within {SMOKE_TIMEOUT} s')

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(SMOKE_TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def images(tmp_path, monkeypatch):
    """Small 12^3 intensity and label images, in a temporary cwd."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    x = np.zeros([12] * 3, np.float32)
    x[3:9, 3:9, 3:9] = 1
    x += 0.1 * rng.random(x.shape).astype(np.float32)
    nib.save(nib.Nifti1Image(x, np.eye(4)), 'fix.nii')
    nib.save(nib.Nifti1Image(np.roll(x, 1, 0), np.eye(4)), 'mov.nii')
    lab = np.zeros([12] * 3, np.int16)
    lab[3:9, 3:9, 3:9] = 1
    lab[3:6, 3:9, 3:9] = 2
    nib.save(nib.Nifti1Image(lab, np.eye(4)), 'lfix.nii')
    nib.save(nib.Nifti1Image(np.roll(lab, 1, 0), np.eye(4)), 'lmov.nii')
    return tmp_path


def _run(*args):
    _autoreg([*args, '-all', '-nit', '2', '-v', '0'])


def test_autoreg_is_registered():
    assert 'autoreg' in commands


# --- parser ----------------------------------------------------------

def test_autoreg_requires_transformation():
    with pytest.raises(ParseError, match='[Nn]o transformation'):
        parse(['-mi', '-fix', 'fix.nii', '-mov', 'mov.nii'])


def test_autoreg_help_still_works_without_transformation(capsys):
    assert not parse(['-h'])
    assert 'AUTOgrad REGistration' in capsys.readouterr().out


def test_autoreg_help_examples_have_a_transformation():
    lines = help.split('examples:')[1].splitlines()
    cmds, cur = [], ''
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        cur += ' ' + line.rstrip('\\')
        if not line.endswith('\\'):
            cmds.append(cur.split()[1:])  # drop 'autoreg'
            cur = ''
    assert cmds
    for cmd in cmds:
        opt = parse(cmd)
        assert opt.transformations, cmd


def test_autoreg_all_stop_keeps_next_arguments():
    opt = parse(['-all', '-stop', '1e-3', '-nit', '7',
                 '-mse', '-fix', 'fix.nii', '-mov', 'mov.nii', '-rig'])
    assert opt.defaults.stop == 1e-3
    assert opt.defaults.max_iter == 7
    assert len(opt.losses) == 1
    assert opt.losses[0].fixed.files == ['fix.nii']
    assert isinstance(opt.transformations[0], struct.Rigid)


def test_autoreg_dice_keeps_all_weights():
    opt = parse(['-dice', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-w', '1', '2', '3', '-rig'])
    assert opt.losses[0].weights == [1., 2., 3.]


def test_autoreg_dice_bare_weights():
    opt = parse(['-dice', '-fix', 'f.nii', '-mov', 'm.nii', '-w', '-rig'])
    assert opt.losses[0].weights is True


# --- smoke runs ------------------------------------------------------

def test_autoreg_init_linear(images, timeout):
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii', '-rig', '-o', 'r0.lta')
    r0 = io.transforms.loadf('r0.lta').reshape([4, 4])
    assert not torch.allclose(r0, torch.eye(4))
    # with no iteration, the output is the initial transform
    _autoreg(['-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
              '-aff', '-init', 'r0.lta', '-o', 'r1.lta',
              '-all', '-nit', '0', '-v', '0'])
    r1 = io.transforms.loadf('r1.lta').reshape([4, 4])
    assert torch.allclose(r0, r1, atol=1e-5)
    # and the moving image header is updated with it
    aff = torch.as_tensor(nib.load('mov.registered.nii').affine,
                          dtype=torch.float32)
    assert torch.allclose(aff, r0, atol=1e-5)
    # with iterations, it runs from there
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
         '-rig', '-init', 'r0.lta', '-o', 'r2.lta')
    assert os.path.exists('r2.lta')


def test_autoreg_init_nonlinear(images, timeout):
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
         '-dif', '-o', 'd0.nii.gz')
    d0 = io.volumes.loadf('d0.nii.gz')
    assert d0.abs().max() > 0
    _autoreg(['-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
              '-dif', '-init', 'd0.nii.gz', '-o', 'd1.nii.gz',
              '-all', '-nit', '0', '-v', '0'])
    d1 = io.volumes.loadf('d1.nii.gz')
    assert torch.allclose(d0, d1, atol=1e-5)
    # several pyramid levels: downsampled to the coarsest, then back up
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
         '-dif', '-init', 'd0.nii.gz', '-pyr', '1', '2', '-o', 'd2.nii.gz')
    assert list(io.volumes.map('d2.nii.gz').shape[:3]) == [12, 12, 12]


def test_autoreg_integer_labels(images, timeout):
    _run('-dice', '-fix', 'lfix.nii', '-mov', 'lmov.nii', '-w', '1', '2',
         '-rig', '-o', 'rig.lta')
    assert os.path.exists('rig.lta')
    assert os.path.exists('lmov.registered.nii')


def test_autoreg_non_progressive_pyramid_terminates(images, timeout):
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
         '-rig', '-dif', '-pyr', '1', '2', '-prg', 'false')
    assert os.path.exists('rigid.lta')
    # the field ends at the finest level
    assert list(io.volumes.map('diffeo.nii.gz').shape[:3]) == [12, 12, 12]


def test_autoreg_help_example_runs(images, timeout):
    # first example of the help (with -mi)
    _run('-mi', '-fix', 'fix.nii', '-mov', 'mov.nii', '-rig')
    assert os.path.exists('rigid.lta')


# --- follow-up fixes -------------------------------------------------

def test_autoreg_negative_numbers_are_values():
    opt = parse(['-mse', '-.5', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-rig', '-1'])
    assert opt.losses[0].factor == -0.5
    assert opt.losses[0].fixed.files == ['f.nii']
    assert opt.transformations[0].factor == -1.
    opt = parse(['-dice', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-w', '-1', '2', '-rig'])
    assert opt.losses[0].weights == [-1., 2.]
    opt = parse(['-mi', '-t', '-1e-3', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-rig'])
    assert opt.losses[0].threshold == -1e-3


def test_autoreg_empty_argv_does_not_read_sys_argv(monkeypatch, capsys):
    import sys
    from nitorch.cli.registration.autoreg.main import autoreg
    monkeypatch.setattr(sys, 'argv', ['nitorch', '-mse', '-fix', 'nofile.nii',
                                      '-mov', 'nofile.nii', '-rig'])
    # an empty list is an empty command line -> help, not sys.argv
    assert not autoreg([])
    assert 'AUTOgrad REGistration' in capsys.readouterr().out
    _autoreg([])
    assert 'AUTOgrad REGistration' in capsys.readouterr().out


def test_autoreg_missing_file_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = commands['autoreg'](['-mse', '-fix', 'nofile.nii',
                              '-mov', 'nofile.nii', '-rig', '-v', '0'])
    assert rc == 1
    err = capsys.readouterr().err
    assert '[ERROR]' in err
    assert 'Traceback' not in err


def test_autoreg_parse_error_returns_1(capsys):
    rc = commands['autoreg'](['-mse', '-fix', 'f.nii', '-mov', 'm.nii'])
    assert rc == 1
    assert '[ERROR]' in capsys.readouterr().err


def test_autoreg_oth_default_verbosity(images, timeout, capsys):
    _autoreg(['-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
              '-oth', '-mov', 'lmov.nii',
              '-rig', '-all', '-nit', '2'])
    assert os.path.exists('lmov.registered.nii')
    assert 'Losses:' in capsys.readouterr().out


def test_autoreg_ffd_grid(images, timeout, monkeypatch):
    from nitorch.cli.registration.autoreg import main
    shapes = []
    ffd_exp = main.ffd_exp

    def spy(prm, shape, *a, **k):
        shapes.append((list(prm.shape[:-1]), list(shape)))
        return ffd_exp(prm, shape, *a, **k)

    monkeypatch.setattr(main, 'ffd_exp', spy)
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
         '-ffd', '-g', '5', '4', '3', '-o', 'ffd.nii.gz')
    assert shapes
    assert all(p == [5, 4, 3] and s == [12, 12, 12] for p, s in shapes)
    assert list(io.volumes.map('ffd.nii.gz').shape[:3]) == [5, 4, 3]
    # a single value is used for all dimensions
    shapes.clear()
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii', '-ffd', '-g', '6')
    assert all(p == [6, 6, 6] for p, _ in shapes)


def test_autoreg_ffd_grid_pyramid(images, timeout):
    # the grid of nodes keeps its size; the dense field is upsampled
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii',
         '-ffd', '-g', '5', '-pyr', '1', '2', '-o', 'ffd.nii.gz')
    assert list(io.volumes.map('ffd.nii.gz').shape[:3]) == [5, 5, 5]
    assert os.path.exists('mov.registered.nii')
