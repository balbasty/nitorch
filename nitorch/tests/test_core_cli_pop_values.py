"""`nitorch.core.cli.pop_values` and the hand-written parsers that use it
(#171)."""
import pytest

import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError, pop_values


# ----------------------------------------------------------------------
#   pop_values
# ----------------------------------------------------------------------
def test_pop_values_pops_from_front():
    args = ['a', 'b', 'c', '-o', 'x']
    assert pop_values(args, '-i/--input', 3) == ['a', 'b', 'c']
    assert args == ['-o', 'x']


def test_pop_values_default_one():
    args = ['x']
    assert pop_values(args, '-o') == ['x']
    assert args == []


@pytest.mark.parametrize('args', [[], ['-x']])
def test_pop_values_missing_single(args):
    with pytest.raises(ParseError) as e:
        pop_values(list(args), '-o/--output', 1, names='<path>')
    assert str(e.value) == '-o/--output expects 1 value: <path>'


@pytest.mark.parametrize('args', [['a', 'b'], ['a', 'b', '-o', 'x']])
def test_pop_values_missing_several(args):
    with pytest.raises(ParseError) as e:
        pop_values(list(args), ('-i', '--input'), 3,
                   names='<fix> <mov> <path>')
    assert str(e.value) == '-i/--input expects 3 values: <fix> <mov> <path>'


@pytest.mark.parametrize('value', ['-1', '-.5', '-1e-3', '-inf', '-', '1'])
def test_pop_values_numbers_and_dash_are_values(value):
    args = [value]
    assert pop_values(args, '-x') == [value]


@pytest.mark.parametrize('value,expected', [
    ('-1', -1.), ('-.5', -.5), ('-inf', -float('inf')), ('2', 2.)])
def test_pop_values_convert(value, expected):
    assert pop_values([value], '-a', 1, float) == [expected]


def test_pop_values_convert_error():
    with pytest.raises(ParseError) as e:
        pop_values(['abc'], '-a/--absolute', 1, float)
    assert str(e.value) == "-a/--absolute expects <float>, got 'abc'"
    with pytest.raises(ParseError) as e:
        pop_values(['abc'], '-n', 1, int)
    assert str(e.value) == "-n expects <int>, got 'abc'"


def test_pop_values_convert_error_names_list():
    with pytest.raises(ParseError) as e:
        pop_values(['1', 'x'], '-p', 2, float, ['<lo>', '<hi>'])
    assert str(e.value) == "-p expects <hi>, got 'x'"
    with pytest.raises(ParseError) as e:
        pop_values(['1'], '-p', 2, float, ['<lo>', '<hi>'])
    assert str(e.value) == '-p expects 2 values: <lo> <hi>'


def test_pop_values_nmin():
    args = ['1', '-o']
    assert pop_values(args, '-l', 2, float, nmin=1) == [1.]
    assert args == ['-o']
    args = ['1', '2', '3']
    assert pop_values(args, '-l', 2, float, nmin=1) == [1., 2.]
    assert args == ['3']
    with pytest.raises(ParseError) as e:
        pop_values(['-o'], '-l', 2, float, '<float> [<float>]', nmin=1)
    assert str(e.value) == '-l expects 1 to 2 values: <float> [<float>]'
    with pytest.raises(ParseError) as e:
        pop_values([], '-l', float('inf'), nmin=1)
    assert str(e.value) == '-l expects at least 1 value: <value>'
    assert pop_values(['a', 'b', '-o'], '-l', float('inf'), nmin=1) \
        == ['a', 'b']


def test_pop_values_custom_istag():
    with pytest.raises(ParseError):
        pop_values(['-'], '-x', istag=lambda x: x.startswith('-'))


# ----------------------------------------------------------------------
#   Commands: a missing or garbled value is a ParseError naming the
#   option (used to be IndexError / ValueError, or a vaguer message)
# ----------------------------------------------------------------------
def _run(command, args, capsys):
    rc = commands[command](args)
    _, err = capsys.readouterr()
    assert rc == 1
    assert 'pop from empty list' not in err
    assert 'could not convert' not in err
    assert 'invalid literal' not in err
    return err


@pytest.mark.parametrize('args,message', [
    (['f.nii', '-k'], '-k/--like expects 1 value: <file>'),
    (['f.nii', '-k', '-o', 'x'], '-k/--like expects 1 value: <file>'),
    (['f.nii', '-b', 'abc'], "-b/--bounding-box expects <float>, got 'abc'"),
])
def test_crop_missing_values(args, message, capsys):
    assert message in _run('crop', args, capsys)


def test_crop_grammar_unchanged():
    from nitorch.cli.misc.crop.parser import parse
    assert parse(['f.nii', '-b']).bbox is True
    assert parse(['f.nii', '-b', '2']).bbox == 2.
    assert parse(['f.nii', '-k', 'ref.nii']).like == 'ref.nii'


@pytest.mark.parametrize('args,message', [
    (['f.nii', '-b'], '-b/--bound expects 1 value: <bound>'),
    (['f.nii', '-b', '-o', 'x'], '-b/--bound expects 1 value: <bound>'),
    (['f.nii', '-k'], '-k/--like expects 1 value: <file>'),
])
def test_pad_missing_values(args, message, capsys):
    assert message in _run('pad', args, capsys)


@pytest.mark.parametrize('value,expected', [
    ('-.5', -.5), ('-1', -1.), ('-inf', -float('inf')), ('dft', 'dft'),
])
def test_pad_bound_grammar_unchanged(value, expected):
    from nitorch.cli.misc.pad.parser import parse
    assert parse(['f.nii', '-b', value, '-s', '2']).bound == expected


@pytest.mark.parametrize('args,message', [
    ([], 'extract_patches expects 1 value: <file>'),
    (['-n', '3'], 'extract_patches expects 1 value: <file>'),
    (['f.nii', '-n', 'abc'], "-n/--size expects <int>, got 'abc'"),
    (['f.nii', '-s', '2', 'x'], "-s/--stride expects <int>, got 'x'"),
])
def test_extract_patches_missing_values(args, message, capsys):
    assert message in _run('extract_patches', args, capsys)


def test_extract_patches_grammar_unchanged():
    from nitorch.cli.misc.extract_patches.parser import parse
    opt = parse(['f.nii', '-n', '3', '4', '-s', '2', '-o', 'x'])
    assert opt.file == 'f.nii'
    assert opt.size == [3, 4] and opt.stride == [2] and opt.output == ['x']


@pytest.mark.parametrize('args,message', [
    (['-i', 'a', 'b'], '-i/--input expects 3 values: <fix> <mov> <path>'),
    (['-i', 'a', 'b', '-o', 'x'],
     '-i/--input expects 3 values: <fix> <mov> <path>'),
    (['-i', 'a', 'b', 'c', '-o'], '-o/--output expects 1 value: <path>'),
])
def test_affopt_missing_values(args, message, capsys):
    assert message in _run('affopt', args, capsys)


def test_affopt_grammar_unchanged():
    from nitorch.cli.registration.affopt.cli import parse
    inputs, output, log, basis = parse(['-i', 'a', 'b', 'c.lta',
                                        '-o', 'out.lta', '-r'])
    assert inputs == {('a', 'b'): 'c.lta'}
    assert output == 'out.lta' and basis == 'SE'


@pytest.mark.parametrize('args,message', [
    (['-rig', '-lr'], '-lr/--learning-rate expects 1 value: <float>'),
    (['-rig', '-lr', 'abc'], "-lr/--learning-rate expects <float>, got 'abc'"),
    (['-rig', '-adam', '-nit'], '-nit/--max-iter expects 1 value: <int>'),
    (['-rig', '-adam', '-nit', 'x'], "-nit/--max-iter expects <int>, got 'x'"),
    (['-rig', '-lbfgs', '-h'], '-h/--history expects 1 value: <int>'),
    (['-rig', '-pad'], '-pad/--padding expects 1 value: <float>'),
    (['-rig', '-v', 'x'], "-v/--verbose expects <int>, got 'x'"),
    (['-rig', '-gpu', 'x'], "-gpu/--gpu expects <int>, got 'x'"),
    (['-mi', '-n', 'x', '-rig'], "-n/--order expects <int> or 'inf', got 'x'"),
    (['-rig', '-all', '-bnd'], '-bnd/--bound expects 1 value'),
])
def test_autoreg_missing_values(args, message, capsys):
    assert message in _run('autoreg', args, capsys)


def test_autoreg_grammar_unchanged():
    from nitorch.cli.registration.autoreg.parser import parse, ParseError
    opt = parse(['-mi', '-n', 'inf', '-rig', '-lr', '-1e-3',
                 '-lbfgs', '-h', '5', '-ls', '-v', '-gpu', '1'])
    assert opt.losses[0].order == 'inf'
    assert opt.transformations[0].lr == -1e-3
    assert opt.optimizers[0].history == 5 and opt.optimizers[0].ls is True
    assert opt.verbose == 2 and opt.device == 'cuda:1'
    # autoreg's own rule: a lone dash is a tag, not a value
    with pytest.raises(ParseError):
        parse(['-rig', '-all', '-bnd', '-'])
