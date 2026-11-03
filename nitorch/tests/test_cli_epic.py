"""Tests for the `nitorch epic` command (issue #171)."""
import importlib
import os

import pytest
import torch

from nitorch import io
from nitorch.cli.qmri.epic.parser import parser, ParseError
from nitorch.tools.denoising.epic import epic

epic_main = importlib.import_module('nitorch.cli.qmri.epic.main')


# ----------------------------------------------------------------------
#   parser
# ----------------------------------------------------------------------

def test_short_p_is_polarity():
    opt = parser.parse(['a.nii', '-p', '-'])
    assert opt.polarity == '-'
    assert opt.penalty == 100


def test_penalty_long_tag():
    opt = parser.parse(['a.nii', '--penalty', '5'])
    assert opt.penalty == 5
    assert opt.polarity == '+'


def test_penalty_not_registered_under_p():
    tags = [t for o in parser.options for t in o.tags if o.name == 'penalty']
    assert '-p' not in tags


def test_default_device_is_cpu():
    opt = parser.parse(['a.nii'])
    assert epic_main.get_device(opt.gpu) == torch.device('cpu')


@pytest.mark.parametrize('args,expected', [
    (['--cpu'], ('cpu', None)),
    (['--gpu', '1'], ('gpu', 1)),
])
def test_explicit_device(args, expected):
    opt = parser.parse(['a.nii', *args])
    assert tuple(opt.gpu) == expected


def test_fieldmap_optional_in_parser():
    # The algorithm supports fieldmap=None (denoising only), so the
    # parser must accept a missing -f and the help must say so.
    opt = parser.parse(['a.nii'])
    assert opt.fieldmap is None
    from nitorch.cli.qmri.epic.parser import help
    assert '[-f FIELDMAP]' in help


# ----------------------------------------------------------------------
#   default output path
# ----------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, args):
    monkeypatch.chdir(tmp_path)
    io.savef(torch.rand(4, 5, 6), 'x.nii', affine=torch.eye(4))
    captured = {}

    def fake_epic(echoes, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 4, 5, 6)

    saved = []
    monkeypatch.setattr(epic_main, 'epic', fake_epic)
    monkeypatch.setattr(epic_main.io, 'savef',
                        lambda dat, fname, **kw: saved.append(fname))
    # single echo: reverse-echo synthesis is impossible, disable it
    epic_main._cli(['x.nii', '-q', '-x', *args])
    return captured, saved


def test_default_output_dir_no_directory(monkeypatch, tmp_path):
    _, saved = _run_main(monkeypatch, tmp_path, ['-f', 'x.nii'])
    assert len(saved) == 1
    assert os.path.dirname(saved[0]) in ('.', '')
    assert not saved[0].startswith(os.sep)
    assert os.path.basename(saved[0]) == 'x_epic.nii'


def test_main_runs_without_fieldmap(monkeypatch, tmp_path):
    captured, saved = _run_main(monkeypatch, tmp_path, [])
    assert captured['fieldmap'] is None
    assert captured['device'] == torch.device('cpu')
    assert len(saved) == 1


def test_main_readout_direction(monkeypatch, tmp_path):
    # RAS volume, `-d l` must map onto the first axis (index 0)
    captured, _ = _run_main(monkeypatch, tmp_path, ['-d', 'l'])
    assert captured['readout'] == 0


# ----------------------------------------------------------------------
#   readout axis
# ----------------------------------------------------------------------

@pytest.mark.parametrize('direction,expected', [
    ('r', 0), ('l', 0), ('rl', 0), ('lr', 0),
    ('a', 1), ('p', 1), ('pa', 1),
    ('s', 2), ('i', 2), ('is', 2),
])
def test_get_readout_ras(direction, expected):
    assert epic_main.get_readout(direction, torch.eye(4), (8, 9, 10), 0) \
           == expected


def test_get_readout_4d_and_largest():
    # 4D input (echoes stacked along the 4th dimension): the echo axis
    # must never be picked, and the index must be a spatial index.
    assert epic_main.get_readout(None, torch.eye(4), (8, 12, 10, 20), 0) == 1
    assert epic_main.get_readout('s', torch.eye(4), (8, 12, 10, 20), 0) == 2


def _synthetic():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 6, 7, 8, generator=g) + 1
    fm = torch.randn(6, 7, 8, generator=g) * 0.5
    return x, fm


def test_readout_axis0_equivariant():
    x, fm = _synthetic()
    opt = dict(max_iter=(2, 4), verbose=0, device='cpu')
    fit0 = epic(x, fieldmap=fm, readout=0, **opt)
    assert fit0.shape == x.shape
    fit_neg = epic(x, fieldmap=fm, readout=-3, **opt)
    assert torch.allclose(fit0, fit_neg)
    # same problem with axes 0 and 2 swapped, readout along axis 2
    fit2 = epic(x.transpose(1, 3), fieldmap=fm.transpose(0, 2),
                readout=2, **opt).transpose(1, 3)
    assert torch.allclose(fit0, fit2, rtol=1e-3, atol=1e-3)


# ----------------------------------------------------------------------
#   number of outputs
# ----------------------------------------------------------------------

def _run_outputs(monkeypatch, tmp_path, inputs, outputs, called=None):
    """`inputs` is a list of echo counts (one 3D/4D file per entry)."""
    monkeypatch.chdir(tmp_path)
    names = []
    for i, ne in enumerate(inputs):
        shape = (4, 5, 6) if ne == 1 else (4, 5, 6, ne)
        names.append(f'in{i}.nii')
        io.savef(torch.rand(shape), names[-1], affine=torch.eye(4))
    called = [] if called is None else called

    def fake_epic(echoes, **kwargs):
        called.append(True)
        return torch.zeros(sum(inputs), 4, 5, 6)

    saved = []
    monkeypatch.setattr(epic_main, 'epic', fake_epic)
    monkeypatch.setattr(epic_main.io, 'savef',
                        lambda dat, fname, **kw: saved.append(
                            (fname, tuple(dat.shape))))
    # few echoes: disable reverse-echo synthesis (tested separately)
    epic_main._cli([*names, '-q', '-x', '-o', *outputs])
    return called, saved


@pytest.mark.parametrize('inputs,outputs', [
    ([1, 1], ['a.nii', 'b.nii', 'c.nii']),       # 2 inputs, 2 echoes
    ([1], ['a.nii', 'b.nii']),                   # 1 input, 1 echo
    ([3, 2], ['a.nii', 'b.nii', 'c.nii']),       # 2 inputs, 5 echoes
])
def test_bad_output_count_is_parse_error(monkeypatch, tmp_path,
                                         inputs, outputs):
    called = []
    with pytest.raises(ParseError) as e:
        _run_outputs(monkeypatch, tmp_path, inputs, outputs, called)
    assert not called, 'output count must be checked before the fit'
    msg = str(e.value)
    assert f'one per input ({len(inputs)})' in msg
    assert f'one per echo ({sum(inputs)})' in msg
    assert f'got {len(outputs)}' in msg


def test_bad_output_count_cli_message(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    io.savef(torch.rand(4, 5, 6), 'x.nii', affine=torch.eye(4))
    monkeypatch.setattr(epic_main, 'epic',
                        lambda *a, **k: torch.zeros(1, 4, 5, 6))
    assert epic_main.cli(['x.nii', '-q', '-o', 'a.nii', 'b.nii']) == 1
    err = capsys.readouterr().err
    assert 'Expected 1 output' in err
    assert 'got 2' in err


@pytest.mark.parametrize('inputs,outputs,expected', [
    # one output: single 4D file
    ([3, 2], ['out.nii'], [('out.nii', (4, 5, 6, 5))]),
    # one output with {n}: one file per echo
    ([3], ['e{n}.nii'], [(f'e{n}.nii', (4, 5, 6)) for n in range(3)]),
    # one output with {base}: one file per input
    ([3, 2], ['{base}_x.nii'], [('in0_x.nii', (4, 5, 6, 3)),
                                ('in1_x.nii', (4, 5, 6, 2))]),
    # one output per input
    ([3, 2], ['a.nii', 'b.nii'], [('a.nii', (4, 5, 6, 3)),
                                  ('b.nii', (4, 5, 6, 2))]),
    # one output per echo
    ([2, 1], ['a.nii', 'b.nii', 'c.nii'], [('a.nii', (4, 5, 6)),
                                           ('b.nii', (4, 5, 6)),
                                           ('c.nii', (4, 5, 6))]),
])
def test_accepted_output_counts(monkeypatch, tmp_path,
                                inputs, outputs, expected):
    called, saved = _run_outputs(monkeypatch, tmp_path, inputs, outputs)
    assert called
    assert saved == expected


# ----------------------------------------------------------------------
#   reverse-echo synthesis needs >= 2 echoes per polarity
# ----------------------------------------------------------------------

def _run_synth(monkeypatch, tmp_path, inputs, args=()):
    monkeypatch.chdir(tmp_path)
    names = []
    for i, ne in enumerate(inputs):
        shape = (4, 5, 6) if ne == 1 else (4, 5, 6, ne)
        names.append(f'in{i}.nii')
        io.savef(torch.rand(shape) + 1, names[-1], affine=torch.eye(4))
    called = []

    def fake_epic(echoes, **kwargs):
        called.append(kwargs)
        return torch.zeros(sum(inputs), 4, 5, 6)

    monkeypatch.setattr(epic_main, 'epic', fake_epic)
    monkeypatch.setattr(epic_main.io, 'savef', lambda *a, **k: None)
    epic_main._cli([*names, '-q', *args])
    return called


@pytest.mark.parametrize('inputs,args', [
    ([1, 1], []),           # two 3D inputs (one echo per polarity)
    ([1, 1], ['-e']),       # extrapolation also needs two echoes
    ([3], []),              # only one echo of the second polarity
])
def test_synth_too_few_echoes_is_parse_error(monkeypatch, tmp_path,
                                             inputs, args):
    with pytest.raises(ParseError) as e:
        _run_synth(monkeypatch, tmp_path, inputs, args)
    msg = str(e.value)
    assert 'at least 2 echoes per polarity' in msg
    assert f'got {sum(inputs)}' in msg
    assert '-x/--no-synth' in msg


@pytest.mark.parametrize('inputs,args', [
    ([1, 1], ['-x']),
    ([1, 1], ['-r', 'in0.nii', 'in1.nii']),
    ([2, 2], []),
])
def test_synth_enough_echoes_or_no_synth(monkeypatch, tmp_path, inputs, args):
    assert _run_synth(monkeypatch, tmp_path, inputs, args)


def test_synth_too_few_echoes_cli_message(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    io.savef(torch.rand(4, 5, 6), 'a.nii', affine=torch.eye(4))
    io.savef(torch.rand(4, 5, 6), 'b.nii', affine=torch.eye(4))
    assert epic_main.cli(['a.nii', 'b.nii', '-q']) == 1
    assert 'at least 2 echoes per polarity' in capsys.readouterr().err


@pytest.mark.parametrize('ne', [1, 2, 3])
def test_tools_synth_too_few_echoes_is_value_error(ne):
    x = torch.rand(ne, 4, 5, 6) + 1
    for slicewise in (False, 2):
        with pytest.raises(ValueError, match='at least 2 echoes per'):
            epic(x, max_iter=(1, 1), verbose=0, device='cpu',
                 slicewise=slicewise)


def test_tools_synthesize_single_echo_is_value_error():
    from nitorch.tools.denoising.epic import synthesize_neg, synthesize_pos
    x = [torch.rand(4, 5)]
    with pytest.raises(ValueError):
        synthesize_neg(x)
    with pytest.raises(ValueError):
        synthesize_pos(x)


def test_tools_reversed_fit_has_forward_echoes_only():
    # with acquired reversed echoes, the fit only holds the N forward
    # echoes, so `-r` files must not be counted among the outputs
    x = torch.rand(2, 4, 5, 6) + 1
    fit = epic(x, reverse_echoes=x.flip(0), max_iter=(1, 1),
               verbose=0, device='cpu')
    assert fit.shape == x.shape


# ----------------------------------------------------------------------
#   map_files
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(4, 5, 6, 1, 3), (4, 5, 6, 3, 1),
                                   (4, 5, 6, 3), (4, 5, 6)])
def test_map_files_shapes(tmp_path, shape):
    from nitorch.tools.denoising.epic import map_files
    fname = str(tmp_path / 'x.nii')
    io.savef(torch.rand(shape), fname, affine=torch.eye(4))
    ne = 1 if len(shape) == 3 else max(shape[3:])
    assert tuple(map_files(fname).shape) == (ne, 4, 5, 6)
