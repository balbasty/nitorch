"""Error messages and exit status of `vopt`, `optinorm`, `resize` and
`autoreg` (#171)."""
import pytest

import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands

MISSING = '/nonexistent/nitorch-missing-input.nii.gz'


def test_vopt_stray_value_names_problem(capsys):
    rc = commands['vopt']([MISSING])
    _, err = capsys.readouterr()
    assert rc == 1
    assert f"Unexpected value '{MISSING}'" in err
    assert '-i/--input' in err
    assert 'Unknown tag' not in err


def test_vopt_unknown_tag_still_reported(capsys):
    rc = commands['vopt'](['--nope'])
    _, err = capsys.readouterr()
    assert rc == 1
    assert 'Unknown tag --nope' in err


def test_optinorm_runtime_error_is_reported(tmp_path, capsys):
    # two missing inputs: parsing succeeds, the pipeline fails
    rc = commands['optinorm']([MISSING, MISSING, '-o', str(tmp_path),
                               '--steps', 'affopt'])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert 'Traceback' not in out + err


def test_optinorm_parse_error_returns_1(capsys):
    # option without its value
    rc = commands['optinorm']([MISSING, MISSING, '-a'])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert out.strip()  # help


@pytest.mark.parametrize('args,message', [
    (['-i', 'a', 'b'], '-i/--input expects 3 values: <fix> <mov> <path>'),
    (['-i', 'a', 'b', '-o', 'x'],
     '-i/--input expects 3 values: <fix> <mov> <path>'),
    (['-i', 'a', 'b', 'c', '-w'],
     '-w/--weight expects 3 values: <fix> <mov> <path>'),
    (['-i', 'a', 'b', 'c', '-o'], '-o/--output expects 1 value: <path>'),
    (['-i', 'a', 'b', 'c', '-x'], '-x/--bound expects 1 value'),
    (['-i', 'a', 'b', 'c', '-a'], '-a/--absolute expects 1 value'),
    (['-i', 'a', 'b', 'c', '-m'], '-m/--membrane expects 1 value'),
    (['-i', 'a', 'b', 'c', '-b'], '-b/--bending expects 1 value'),
    (['-i', 'a', 'b', 'c', '-l'], '-l/--lame expects 1 value'),
    (['-i', 'a', 'b', 'c', '-a', 'abc'], "-a/--absolute expects <float>"),
])
def test_vopt_missing_values(args, message, capsys):
    rc = commands['vopt'](args)
    _, err = capsys.readouterr()
    assert rc == 1
    assert message in err
    assert 'pop from empty list' not in err


def test_vopt_lame_values():
    from nitorch.cli.registration.vopt.cli import parse
    *_, prm = parse(['-i', 'a', 'b', 'c', '-l', '1', '2'])
    assert prm['lame'] == [1., 2.]
    *_, prm = parse(['-i', 'a', 'b', 'c', '-l', '1', '-o', 'x'])
    assert prm['lame'] == [1.]


@pytest.mark.parametrize('args', [
    ['--nope', 'a', 'b'],
    ['a', '--nope', 'b'],
    ['a', 'b', '--nope'],
])
def test_optinorm_unknown_option(args, capsys):
    rc = commands['optinorm'](args)
    _, err = capsys.readouterr()
    assert rc == 1
    assert 'Unknown option --nope' in err


@pytest.mark.parametrize('args,got', [
    (['a', 'b', '--affine'], 'nothing'),
    (['a', 'b', '--affine', '-a', '1'], "'-a'"),
    (['a', 'b', '--affine', 'foo'], "'foo'"),
])
def test_optinorm_affine_needs_valid_value(args, got, capsys):
    rc = commands['optinorm'](args)
    _, err = capsys.readouterr()
    assert rc == 1
    assert ('--affine expects one of {rigid, similitude, affine}, got '
            + got) in err


def test_optinorm_valid_options_still_parse():
    from nitorch.cli.registration.optinorm.parser import parser
    opt = parser.parse(['a', 'b', '--affine', 'similitude', '-a', '-1',
                        '-h'])
    assert opt.help
    opt = parser.parse(['a', 'b', '--affine', 'affine', '-a', '-1'])
    assert opt.affine == 'affine' and opt.absolute == -1.
    assert opt.input == ['a', 'b']
