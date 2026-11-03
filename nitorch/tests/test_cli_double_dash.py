"""A bare `--` ends option parsing (Unix convention, #171).

Every argument after `--` is a value, even if it starts with a dash
(e.g. `-odd.nii`) or spells an option (`-h`, `-o`):

- `CommandParser`: values after `--` are positionals of the top-level
  command (never of a group).
- Hand-written parsers with positional inputs (crop, pad, reslice, ...):
  values after `--` are input files.
- Hand-written parsers without positional inputs (vopt, affopt, compose,
  autoreg): values after `--` complete the option just before `--`.
"""
import os

import numpy as np
import pytest

import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands
from nitorch.core import cli
from nitorch.core.cli import ParseError


# ----------------------------------------------------------------------
#   core helpers
# ----------------------------------------------------------------------

def test_split_double_dash():
    assert cli.split_double_dash(['a', '-o', 'b']) == (['a', '-o', 'b'], [])
    assert cli.split_double_dash(['a', '--', '-b', '-h']) == (['a'], ['-b', '-h'])
    # only the first `--` is a marker
    assert cli.split_double_dash(['--', '--', 'x']) == ([], ['--', 'x'])
    # trailing `--`
    assert cli.split_double_dash(['a', '--']) == (['a'], [])
    # input not modified
    args = ['a', '--', 'b']
    cli.split_double_dash(args)
    assert args == ['a', '--', 'b']


def test_pop_values_rest():
    rest = ['-odd.nii', 'x']
    args = ['fix', 'mov']
    assert cli.pop_values(args, '-i', 3, rest=rest) == ['fix', 'mov', '-odd.nii']
    assert args == [] and rest == ['x']
    # rest is only used once args is exhausted
    args, rest = ['-o', 'out'], ['-odd.nii']
    with pytest.raises(ParseError, match='-i expects 1 value'):
        cli.pop_values(args, '-i', 1, rest=rest)
    assert rest == ['-odd.nii']
    # negative numbers are still values
    assert cli.pop_values(['-1', '-.5'], '-a', 2, float, rest=[]) == [-1., -.5]
    # without rest: unchanged (a dash-prefixed value is a missing value)
    with pytest.raises(ParseError, match='-o expects 1 value'):
        cli.pop_values(['--', '-odd.nii'], '-o', 1)


def test_check_rest_used():
    cli.check_rest_used([])
    with pytest.raises(ParseError, match="Unexpected value 'x' after '--'"):
        cli.check_rest_used(['x'])


def _parser(**kw):
    parser = cli.CommandParser('demo', **kw)
    parser.add_positional('files', nargs='+')
    parser.add_option('output', ('-o', '--output'), nargs=1)
    parser.add_option('value', ('-v', '--value'), nargs=1, convert=float)
    group = cli.Group('grp', '@grp', n='?')
    group.add_positional('names', nargs='*')
    group.add_option('flag', ('-f',), nargs=0, default=False)
    parser.add_group(group)
    return parser


def test_command_parser_positionals_after_double_dash():
    parser = _parser()
    out = parser.parse(['a.nii', '-o', 'out.nii', '--', '-odd.nii', '-o'])
    assert out.files == ['a.nii', '-odd.nii', '-o']
    assert out.output == 'out.nii'
    out = parser.parse(['--', '-odd.nii'])
    assert out.files == ['-odd.nii']
    # options and groups before `--` are still parsed
    out = parser.parse(['-o', 'x', '@grp', 'n', '-f', '--', '-a', '-b'])
    assert out.files == ['-a', '-b']
    assert out.output == 'x' and out.grp.names == ['n'] and out.grp.flag


def test_command_parser_double_dash_not_unknown():
    parser = _parser()
    out = parser.parse(['a.nii', '--'])
    assert out.files == ['a.nii']
    # dash-prefixed values after `--` are not checked
    out = parser.parse(['a.nii', '--', '--nope', '-x'])
    assert out.files == ['a.nii', '--nope', '-x']
    # but they still are before `--`
    with pytest.raises(ParseError, match='Unknown option --nope'):
        parser.parse(['a.nii', '--nope', '--', 'b.nii'])


def test_command_parser_negative_numbers_unchanged():
    parser = _parser()
    out = parser.parse(['a.nii', '-v', '-1.5', '--', '-2'])
    assert out.value == -1.5
    assert out.files == ['a.nii', '-2']
    out = parser.parse(['a.nii', '-v', '-1e-3'])
    assert out.value == -1e-3


def test_command_parser_help_before_and_after():
    parser = _parser()
    assert parser.parse(['a.nii', '-h', '--', 'b.nii']).help is True
    out = parser.parse(['a.nii', '--', '-h', '--help'])
    assert out.files == ['a.nii', '-h', '--help']
    assert out.help is False


def test_command_parser_double_dash_errors():
    parser = _parser()
    # positionals come first: a value between options and `--` is stray
    with pytest.raises(ParseError):
        parser.parse(['-o', 'x', 'stray', '--', 'a'])
    # no positional at all
    noarg = cli.CommandParser('noarg')
    noarg.add_option('output', ('-o',), nargs=1)
    with pytest.raises(ParseError, match="after '--'.*no positional"):
        noarg.parse(['-o', 'x', '--', '-odd'])
    # too many positionals
    one = cli.CommandParser('one')
    one.add_positional('file', nargs=1)
    with pytest.raises(ParseError, match="Unexpected value 'b' after '--'"):
        one.parse(['--', 'a', 'b'])
    assert one.parse(['--', '-a']).file == '-a'


# ----------------------------------------------------------------------
#   CommandParser-based commands
# ----------------------------------------------------------------------

def test_smooth_double_dash():
    from nitorch.cli.misc.smooth.parser import parser
    out = parser.parse(['a.nii', '-f', '2', '--', '-odd.nii'])
    assert out.files == ['a.nii', '-odd.nii']


def test_topup_double_dash():
    from nitorch.cli.qmri.topup.parser import parser_fit, parser_apply
    # options (even before the positionals) + `--`: the nargs='+' -p
    # cannot swallow the positionals placed after `--`
    out = parser_fit.parse(['-p', '100', '-o', 'o.nii', '--',
                            '-pos.nii', '-neg.nii'])
    assert out.pos_file == '-pos.nii' and out.neg_file == '-neg.nii'
    assert out.penalty == [100.] and out.output == 'o.nii'
    out = parser_fit.parse(['pos.nii', '-p', '100', '--', 'neg.nii'])
    assert out.pos_file == 'pos.nii' and out.neg_file == 'neg.nii'
    out = parser_apply.parse(['-p', 'a.nii', 'b.nii', '--', 'dist.nii'])
    assert out.dist_file == 'dist.nii'
    assert out.file_pos == ['a.nii', 'b.nii']
    # unchanged without `--`
    out = parser_apply.parse(['dist.nii', '-p', 'a.nii', 'b.nii'])
    assert out.dist_file == 'dist.nii'


def test_register_double_dash_is_reported():
    # register has no top-level positionals: files live in @@fix/@@mov
    from nitorch.cli.registration.register.parser import parser
    with pytest.raises(ParseError, match="after '--'.*\\./-name"):
        parser.parse(['@loss', 'mse', '@@fix', 'f.nii',
                      '@@mov', 'm.nii', '--', '-x.nii'])


# ----------------------------------------------------------------------
#   hand-written parsers with positional inputs
# ----------------------------------------------------------------------

def _files_parsers():
    from nitorch.cli.misc.chunk.parser import parse as chunk
    from nitorch.cli.misc.convert.parser import parse as convert
    from nitorch.cli.misc.crop.parser import parse as crop
    from nitorch.cli.misc.info.parser import parse as info
    from nitorch.cli.misc.inpaint.parser import parse as inpaint
    from nitorch.cli.misc.pad.parser import parse as pad
    from nitorch.cli.misc.pool.parser import parse as pool
    from nitorch.cli.misc.unstack.parser import parse as unstack
    from nitorch.cli.registration.orient.parser import parse as orient
    from nitorch.cli.registration.reorient.parser import parse as reorient
    from nitorch.cli.registration.vexp.parser import parse as vexp
    from nitorch.cli.denoising.denoise_mri.parser import parse as dmri
    from nitorch.cli.viewer.parser import parse as view
    from nitorch.cli.registration.reslice.parser import parse as reslice
    return {
        'chunk': (chunk, ['-n', '2']),
        'convert': (convert, ['-o', 'out.nii']),
        'crop': (crop, ['-s', '4', '4', '4']),
        'info': (info, []),
        'inpaint': (inpaint, ['-o', 'out.nii']),
        'pad': (pad, ['-s', '2']),
        'pool': (pool, ['-w', '2']),
        'unstack': (unstack, ['-d', '-1']),
        'orient': (orient, ['-l', 'RAS']),
        'reorient': (reorient, ['-l', 'RAS']),
        'vexp': (vexp, ['-o', 'out.nii']),
        'denoise_mri': (dmri, ['-o', 'out.nii']),
        'view': (view, []),
        'reslice': (reslice, ['-l', 'aff.lta']),
    }


FILES_COMMANDS = sorted(_files_parsers())


@pytest.mark.parametrize('cmd', FILES_COMMANDS)
def test_files_after_double_dash(cmd):
    parse, opts = _files_parsers()[cmd]
    out = parse(['a.nii', *opts, '--', '-odd.nii', '-h'])
    assert list(out.files) == ['a.nii', '-odd.nii', '-h']
    out = parse([*opts, '--', '-odd.nii'])
    assert list(out.files) == ['-odd.nii']


@pytest.mark.parametrize('cmd', FILES_COMMANDS)
def test_files_help_before_double_dash(cmd, capsys):
    parse, _ = _files_parsers()[cmd]
    assert not parse(['a.nii', '-h', '--', '-odd.nii'])
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize('cmd', FILES_COMMANDS)
def test_files_no_double_dash_unchanged(cmd):
    parse, opts = _files_parsers()[cmd]
    out = parse(['a.nii', *opts])
    assert list(out.files) == ['a.nii']
    with pytest.raises(ParseError):
        parse(['a.nii', '-odd.nii'])


def test_extract_patches_double_dash():
    from nitorch.cli.misc.extract_patches.parser import parse
    out = parse(['-n', '8', '--', '-odd.nii'])
    assert out.file == '-odd.nii' and out.size == [8]
    assert parse(['--', '-h']).file == '-h'
    assert parse(['a.nii', '-n', '8']).file == 'a.nii'
    with pytest.raises(ParseError, match="after '--'"):
        parse(['--', 'a.nii', 'b.nii'])


def test_reslice_negative_numbers_unchanged():
    from nitorch.cli.registration.reslice.parser import parse
    out = parse(['a.nii', '-vx', '-1', '2', '--', '-b.nii'])
    assert out.voxel_size == [-1., 2.]
    assert out.files == ['a.nii', '-b.nii']


# ----------------------------------------------------------------------
#   hand-written parsers without positional inputs
# ----------------------------------------------------------------------

def test_vopt_double_dash():
    from nitorch.cli.registration.vopt.cli import parse
    inputs, _, output, _ = parse(['-i', 'a', 'b', 'a.nii',
                                  '-i', 'b', 'a', '--', '-odd.nii'])
    assert inputs == {('a', 'b'): 'a.nii', ('b', 'a'): '-odd.nii'}
    _, _, output, _ = parse(['-i', 'a', 'b', 'x.nii', '-o', '--', '-out'])
    assert output == '-out'
    # nothing left to complete
    with pytest.raises(ParseError, match="after '--'"):
        parse(['-i', 'a', 'b', 'x.nii', '--', '-odd.nii'])
    # `-h` after `--` is a value
    inputs, *_ = parse(['-i', 'a', 'b', '--', '-h'])
    assert inputs == {('a', 'b'): '-h'}


def test_affopt_double_dash():
    from nitorch.cli.registration.affopt.cli import parse
    inputs, output, *_ = parse(['-i', 'a', 'b', '--', '-odd.lta'])
    assert inputs == {('a', 'b'): '-odd.lta'}
    inputs, output, *_ = parse(['-i', 'a', 'b', 'x.lta', '-o', '--', '-o'])
    assert output == '-o'
    with pytest.raises(ParseError, match="after '--'"):
        parse(['-i', 'a', 'b', 'x.lta', '-r', '--', 'y'])


def test_compose_double_dash():
    from nitorch.cli.registration.compose.parser import parse
    out = parse(['-l', 'a.lta', '-o', '--', '-out.nii.gz'])
    assert out.output == '-out.nii.gz'
    out = parse(['-l', 'a.lta', '-il', '--', '-b.lta'])
    assert [t.file for t in out.transformations] == ['a.lta', '-b.lta']
    assert out.transformations[1].inv
    with pytest.raises(ParseError, match="after '--'"):
        parse(['-l', 'a.lta', '--', 'x'])


def test_autoreg_double_dash():
    from nitorch.cli.registration.autoreg.parser import parse
    out = parse(['-aff', '-mse', '-fix', 'f.nii', '-mov', '--',
                 '-m1.nii', '-m2.nii'])
    assert out.losses[0].fixed.files == ['f.nii']
    assert out.losses[0].moving.files == ['-m1.nii', '-m2.nii']
    out = parse(['-aff', '-mse', '-mov', 'm.nii', '-fix', 'f.nii', '--',
                 '-g.nii'])
    assert out.losses[0].fixed.files == ['f.nii', '-g.nii']
    out = parse(['-aff', '-mse', '-fix', 'f.nii', '-mov', 'm.nii',
                 '-o', '--', '-out.nii'])
    assert out.losses[0].moving.updated == ['-out.nii']
    with pytest.raises(ParseError, match="after '--'"):
        parse(['-mse', '-fix', 'f.nii', '-mov', 'm.nii', '-aff',
               '--', '-x.nii'])


# ----------------------------------------------------------------------
#   end-to-end, with a file whose name starts with a dash
# ----------------------------------------------------------------------

nib = pytest.importorskip('nibabel')


@pytest.fixture
def odd_volume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dat = np.arange(10 * 12 * 8, dtype='float32').reshape([10, 12, 8])
    nib.save(nib.Nifti1Image(dat, np.eye(4)), '-odd.nii.gz')
    return '-odd.nii.gz'


def test_crop_end_to_end(odd_volume, capsys):
    rc = commands['crop'](['-s', '4', '6', '2', '-o', 'out.nii.gz',
                           '--', odd_volume])
    assert not rc, capsys.readouterr().err
    assert nib.load('out.nii.gz').shape == (4, 6, 2)


def test_crop_end_to_end_without_double_dash_fails(odd_volume, capsys):
    rc = commands['crop']([odd_volume, '-s', '4', '6', '2'])
    assert rc == 1
    assert 'Unknown tag -odd.nii.gz' in capsys.readouterr().err


def test_smooth_end_to_end(odd_volume, capsys):
    rc = commands['smooth'](['-f', '2', '-o', 'out.nii.gz',
                             '--', odd_volume])
    assert not rc, capsys.readouterr().err
    assert os.path.exists('out.nii.gz')


def test_info_end_to_end(odd_volume, capsys):
    rc = commands['info'](['--', odd_volume])
    assert not rc
    out = capsys.readouterr().out
    assert '10' in out and '12' in out
