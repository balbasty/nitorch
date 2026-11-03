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


# --- true/false for output options -----------------------------------

def test_autoreg_transform_output_bool():
    opt = parse(['-mse', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-rig', '-o', 'false', '-dif', '-o', 'yes'])
    assert opt.transformations[0].output is False
    assert opt.transformations[1].output is True
    opt.propagate_defaults()
    assert opt.transformations[0].output is False
    assert opt.transformations[1].output == opt.defaults.output
    opt = parse(['-mse', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-rig', '-o', '{name}_out{ext}'])
    assert opt.transformations[0].output == '{name}_out{ext}'
    # -all -o false: no transform written by default
    opt = parse(['-all', '-o', 'no', '-mse', '-fix', 'f.nii',
                 '-mov', 'm.nii', '-rig'])
    opt.propagate_defaults()
    assert opt.transformations[0].output is False
    # -all -o true: keep the default template
    opt = parse(['-all', '-o', 'true', '-mse', '-fix', 'f.nii',
                 '-mov', 'm.nii', '-rig'])
    assert opt.defaults.output == struct.Defaults.output


def test_autoreg_image_output_bool():
    opt = parse(['-mse', '-fix', 'f.nii', '-r', 'true',
                 '-mov', 'm.nii', '-o', 'false', '-r', '0', '-rig'])
    assert opt.losses[0].fixed.resliced is True
    assert opt.losses[0].moving.updated is False
    assert opt.losses[0].moving.resliced is False
    opt = parse(['-mse', '-fix', 'f.nii', '-mov', 'm1.nii', 'm2.nii',
                 '-o', 'out.nii', 'no', '-rig'])
    assert opt.losses[0].moving.updated == ['out.nii', False]


def test_autoreg_output_template_validated():
    with pytest.raises(ParseError, match='[Tt]emplate'):
        parse(['-mse', '-fix', 'f.nii', '-mov', 'm.nii',
               '-rig', '-o', '{nmae}.lta'])
    with pytest.raises(ParseError, match='[Tt]emplate'):
        parse(['-mse', '-fix', 'f.nii', '-mov', 'm.nii',
               '-o', '{bse}.nii', '-rig'])
    with pytest.raises(ParseError, match='[Tt]emplate'):
        parse(['-all', '-o', '{oops', '-mse', '-fix', 'f.nii',
               '-mov', 'm.nii', '-rig'])


def test_autoreg_output_false_writes_nothing(images, timeout):
    before = set(os.listdir('.'))
    _run('-mse', '-fix', 'fix.nii', '-mov', 'mov.nii', '-o', 'false',
         '-r', 'true', '-rig', '-o', 'false')
    new = set(os.listdir('.')) - before
    assert new == {'mov.resliced.nii'}


# --- garbled values, bool words, -pyr, -o/-r count (#171) ------------

_FM = ['-fix', 'f.nii', '-mov', 'm.nii']


@pytest.mark.parametrize('argv,msg', [
    (['-mi', *_FM, '-p', '3', 'abc', '-rig'],
     "-p/--patch expects <int>, got 'abc'"),
    (['-dice', *_FM, '-w', '1', 'abc', '-rig'],
     "-w/--weights expects <float>, got 'abc'"),
    (['-dice', *_FM, '-l', '1', 'abc', '-rig'],
     "-l/--labels expects <int>, got 'abc'"),
    (['-mse', *_FM, '-ffd', '-g', '5', 'abc'],
     "-g/--grid expects <int>, got 'abc'"),
    (['-mse', *_FM, '-dif', '-ben', 'abc'],
     "-ben expects <float>, got 'abc'"),
    (['-mse', *_FM, '-aff', '-le', '1', 'abc'],
     "-le expects <float>, got 'abc'"),
    (['-mse', 'abc', *_FM, '-rig'],
     "-mse expects <factor: float>, got 'abc'"),
    (['-mse', *_FM, '-rig', 'abc'],
     "-rig expects <factor: float>, got 'abc'"),
    (['-mse', *_FM, '-rig', '-pyr', 'abc'],
     "-pyr/--pyramid expects <int> or [start]:stop[:step], got 'abc'"),
    (['-mse', *_FM, '-pyr', '1:x', '-rig'],
     "-pyr/--pyramid expects <int> or [start]:stop[:step], got '1:x'"),
    (['-mse', '-fix', 'f.nii', '-pyr', '2', 'x', '-mov', 'm.nii', '-rig'],
     "-pyr/--pyramid expects <int> or [start]:stop[:step], got 'x'"),
    (['-all', '-pyr', 'x', '-mse', *_FM, '-rig'],
     "-pyr/--pyramid expects <int> or [start]:stop[:step], got 'x'"),
])
def test_autoreg_garbled_values_are_named_parse_errors(argv, msg):
    with pytest.raises(ParseError) as e:
        parse(argv)
    assert str(e.value) == msg


def test_autoreg_variable_length_grammar_unchanged():
    opt = parse(['-mi', '2', *_FM, '-p', '3', '4', '-pyr', '1:3',
                 '-dice', *_FM, '-l', '1', '2', '-w', '0.5', '-1',
                 '-ffd', '0.5', '-g', '4', '5', '6', '-ben', '2',
                 '-aff', '-le', '1', '2', '-pyr', '2', '3'])
    mi, dc = opt.losses
    assert mi.factor == 2. and mi.patch == [3, 4] and mi.pyramid == [1, 2, 3]
    assert dc.labels == [1, 2] and dc.weights == [0.5, -1.]
    ffd, aff = opt.transformations
    assert ffd.factor == 0.5 and ffd.grid == [4, 5, 6]
    assert ffd.losses[0].factor == 2.
    assert aff.losses[0].factor == [1., 2.]
    assert aff.pyramid == [2, 3]


@pytest.mark.parametrize('word,value', [
    ('1', True), ('0', False), ('true', True), ('False', False),
    ('yes', True), ('no', False), ('t', True), ('n', False)])
def test_autoreg_bool_words(word, value):
    opt = parse(['-mse', *_FM, '-exc', word, '-aff', '-z', word,
                 '-dif', '-sd', word, '-prg', word])
    assert opt.losses[0].exclude is value
    aff, dif = opt.transformations
    assert aff.shift is value
    assert dif.smalldef is value
    assert opt.progressive is value


@pytest.mark.parametrize('argv,name', [
    (['-mse', *_FM, '-exc', 'maybe', '-rig'], '-exc/--exclude'),
    (['-mse', *_FM, '-aff', '-z', 'maybe'], '-z/--zero'),
    (['-mse', *_FM, '-dif', '-sd', 'maybe'], '-sd/--smalldef'),
    (['-mse', *_FM, '-rig', '-prg', 'maybe'], '-prg/--progressive'),
])
def test_autoreg_bad_bool_is_parse_error(argv, name):
    with pytest.raises(ParseError) as e:
        parse(argv)
    assert str(e.value) == f"{name} expects <bool>, got 'maybe'"


def test_autoreg_bare_bool_flags():
    opt = parse(['-mse', *_FM, '-exc', '-aff', '-z', '-dif', '-sd', '-prg'])
    assert opt.losses[0].exclude is True
    assert opt.transformations[0].shift is True
    assert opt.transformations[1].smalldef is True
    assert opt.progressive is True


def test_autoreg_transform_pyramid_levels():
    opt = parse(['-mse', *_FM, '-rig', '-pyr', '1:3', '-lr', '0.5'])
    rig, = opt.transformations
    assert rig.pyramid == [1, 2, 3]
    assert rig.lr == 0.5
    opt = parse(['-mse', *_FM, '-rig', '-pyr'])
    assert opt.transformations[0].pyramid == [1]


@pytest.mark.parametrize('tag', ['-o', '-r'])
def test_autoreg_image_output_count(tag):
    with pytest.raises(ParseError) as e:
        parse(['-mse', '-fix', 'f.nii', '-mov', 'a.nii', 'b.nii', 'c.nii',
               tag, 'x.nii', 'y.nii', '-rig'])
    assert str(e.value) == (f'{tag} expects 1 value per input file (3), '
                            f'or a single template; got 2')
    with pytest.raises(ParseError):
        parse(['-mse', '-fix', 'f.nii', '-mov', 'a.nii',
               tag, 'x.nii', 'y.nii', '-rig'])


def test_autoreg_image_output_single_template():
    opt = parse(['-mse', '-fix', 'f.nii', '-mov', 'a.nii', 'b.nii',
                 '-o', '{base}.out{ext}', '-r', 'no', '-rig'])
    assert opt.losses[0].moving.updated == ['{base}.out{ext}'] * 2
    assert opt.losses[0].moving.resliced is False
    opt = parse(['-mse', '-fix', 'f.nii', '-mov', 'a.nii', 'b.nii',
                 '-o', 'true', 'x.nii', '-rig'])
    assert opt.losses[0].moving.updated == [True, 'x.nii']


def test_autoreg_check_next_isvalue_removed():
    from nitorch.cli.registration.autoreg import parser
    assert not hasattr(parser, 'check_next_isvalue')
