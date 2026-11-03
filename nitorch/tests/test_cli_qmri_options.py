"""Option-preparation tests for the `estatics` and `greeq` CLIs (#171).

The fitting functions are monkeypatched so that no model is fitted:
we only check that the command line is turned into the right arguments.
"""
import numpy as np
import nibabel as nib
import pytest

from nitorch.cli.qmri.estatics import main as estatics_main
from nitorch.cli.qmri.greeq import main as greeq_main
from nitorch.cli.qmri.greeq.parser import parser as greeq_parser


class _Called(Exception):
    pass


def _capture(module, name, monkeypatch):
    captured = {}

    def fake(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        raise _Called

    monkeypatch.setattr(module, name, fake)
    return captured


@pytest.fixture
def vols(tmp_path):
    paths = {}
    rng = np.random.default_rng(0)
    for name in ('e1', 'e2', 'b0', 'rf'):
        dat = rng.random((4, 4, 4)).astype('float32') + 1
        path = str(tmp_path / f'{name}.nii')
        nib.save(nib.Nifti1Image(dat, np.eye(4)), path)
        paths[name] = path
    return paths


def _run_greeq(args, monkeypatch):
    captured = _capture(greeq_main, 'greeq', monkeypatch)
    with pytest.raises(_Called):
        greeq_main._cli(args)
    return captured


def _greeq_contrast(vols):
    return ['-c', '-e', vols['e1'], '-fa', '20', '-tr', '25', 'ms',
            '-te', '2', 'ms']


def test_estatics_b0_is_mapped(vols, monkeypatch):
    captured = _capture(estatics_main, 'estatics', monkeypatch)
    with pytest.raises(_Called):
        estatics_main._cli(['-c', '-e', vols['e1'], vols['e2'],
                            '-te', '2', '4', 'ms', '-b0', vols['b0'], 'vox'])
    contrasts, distortion = captured['args'][:2]
    assert len(distortion) == 1 and distortion[0] is not None
    assert tuple(distortion[0].shape) == (4, 4, 4)


def test_greeq_shared_receive(vols, monkeypatch):
    captured = _run_greeq(['-rf', vols['rf']] + _greeq_contrast(vols),
                          monkeypatch)
    contrasts, transmits, receives = captured['args'][:3]
    assert len(receives) == 1 and receives[0] is not None


@pytest.mark.parametrize('opt', [
    ['-b0', 'b0.nii', 'hz'],
    ['--b0-field', 'b0.nii'],
    ['--meetup'],
    ['--lam-meetup', '1e5'],
])
def test_greeq_unimplemented_options_are_rejected(opt):
    # B0 maps and MEETUP distortion correction are not implemented in greeq
    with pytest.raises(greeq_main.ParseError, match='Unknown option'):
        greeq_parser.parse(opt + ['-c', '-e', 'x.nii'])
    with pytest.raises(greeq_main.ParseError, match='Unknown option'):
        greeq_parser.parse(['-c', '-e', 'x.nii'] + opt)


def test_greeq_verbose_default():
    opt = greeq_parser.parse(['-c', '-e', 'x.nii'])
    assert opt.verbose == 1
    opt = greeq_parser.parse(['-v', '0', '-c', '-e', 'x.nii'])
    assert opt.verbose == 0


def test_greeq_zero_lam_is_honoured(vols, monkeypatch):
    captured = _run_greeq(['--lam', '5', '--lam-pd', '0', '--lam-t2', '3']
                          + _greeq_contrast(vols), monkeypatch)
    opt = captured['args'][3]
    assert opt.penalty.factor == {'pd': 0., 'r1': 5., 'r2s': 3., 'mt': 5.}
