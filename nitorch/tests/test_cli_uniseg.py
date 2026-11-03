"""CPU-only tests for the `uniseg` command-line interface (issue #171).

The heavy `uniseg` fit is monkeypatched: it records its arguments and
returns small fake outputs, so that we can check what the CLI passes to
the fit and which output files each flag writes.
"""
import os
import pytest
import torch
import numpy as np
import nibabel as nib

from nitorch.core.cli import ParseError
from nitorch.cli.segmentation.uniseg.parser import parser
from nitorch.cli.segmentation.uniseg import cli as uniseg_cli

SHAPE = (6, 6, 6)
K = 3
BIAS = 2.


def parse(*args):
    return parser.parse(list(args))


# ----------------------------------------------------------------------
# parser-level tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize('args,expected', [
    ((), 'always'),
    (('-a',), 'always'),
    (('-a', 'yes'), 'always'),
    (('-a', 'always'), 'always'),
    (('-a', 'once'), 'once'),
    (('-a', 'no'), False),
    (('--align', 'false'), False),
    (('--no-align',), False),
])
def test_align_parsing(args, expected):
    assert parse('in.nii', *args).align == expected


@pytest.mark.parametrize('args,flag', [
    (('-a', 'bogus'), '-a/--align'),
    (('--mrf', 'maybe'), '--mrf'),
])
def test_choice_invalid(args, flag):
    with pytest.raises(ParseError, match=flag):
        uniseg_cli.cli_parse(['in.nii', *args])


@pytest.mark.parametrize('args,expected', [
    ((), 'once'),
    (('--mrf',), 'learn'),
    (('--mrf', 'yes'), 'learn'),
    (('--mrf', 'learn'), 'learn'),
    (('--mrf', 'always'), 'always'),
    (('--mrf', 'once'), 'once'),
    (('--mrf', 'no'), False),
    (('--mrf', 'false'), False),
    (('--no-mrf',), False),
])
def test_mrf_parsing(args, expected):
    assert parse('in.nii', *args).mrf == expected


def test_mask_parsing():
    assert parse('in.nii').mask is None
    assert parse('in.nii', '-m', 'mask.nii').mask == 'mask.nii'
    with pytest.raises(ParseError):
        parse('in.nii', '-m')


def test_plot_parsing():
    assert parse('in.nii').plot == 0
    assert parse('in.nii', '-p').plot == 1
    assert parse('in.nii', '-p', '3').plot == 3


# ----------------------------------------------------------------------
# end-to-end with a fake fit
# ----------------------------------------------------------------------

@pytest.fixture
def setup(tmp_path, monkeypatch):
    aff = np.eye(4)
    img = np.random.rand(*SHAPE).astype('float32') + 1
    nib.save(nib.Nifti1Image(img, aff), str(tmp_path / 'in.nii'))
    for c in (1, 2):
        img = np.random.rand(*SHAPE).astype('float32') + c
        nib.save(nib.Nifti1Image(img, aff), str(tmp_path / f'ch{c}.nii'))
    tpm = np.random.rand(*SHAPE, K).astype('float32')
    nib.save(nib.Nifti1Image(tpm, aff), str(tmp_path / 'tpm.nii'))

    captured = {}

    def fake_uniseg(x, w=None, **kwargs):
        captured['x'] = x
        captured['w'] = w
        captured.update(kwargs)
        z = torch.rand(K, *SHAPE).softmax(0)
        prm = dict(
            affine=torch.eye(4),
            warp=torch.zeros(*SHAPE, 3),
            bias=torch.full([len(x), *SHAPE], BIAS),
            warped=torch.rand(K, *SHAPE).softmax(0),
        )
        if kwargs.get('do_affine') is False:
            del prm['affine']
        return z, None, prm

    monkeypatch.setattr(uniseg_cli, 'uniseg', fake_uniseg)

    def run(*args, inputs=('in.nii',)):
        argv = [*(str(tmp_path / i) for i in inputs),
                '-t', str(tmp_path / 'tpm.nii'),
                '-o', str(tmp_path / 'out'), '-v', '0', *args]
        os.makedirs(tmp_path / 'out', exist_ok=True)
        uniseg_cli.main(parser.parse(argv))
        return sorted(os.listdir(tmp_path / 'out'))

    return run, captured


def test_align_and_mrf_passed_to_fit(setup):
    run, captured = setup
    run('-a', 'once', '--mrf', 'no')
    assert captured['do_affine'] == 'once'
    assert captured['do_mrf'] is False
    run('--no-align')
    assert captured['do_affine'] is False


def test_mask_passed_to_fit(setup, tmp_path):
    run, captured = setup
    run()
    assert captured['w'] is None


def test_default_outputs(setup):
    run, _ = setup
    assert run() == ['in.labels.nat.nii', 'in.prob.nat.nii']


@pytest.mark.parametrize('flag', [
    'prob', 'labels', 'prior', 'bias', 'nobias', 'warp'])
@pytest.mark.parametrize('space', ['mni', 'wrp'])
def test_single_output_flag(setup, flag, space):
    run, _ = setup
    if flag == 'warp' and space == 'wrp':
        pytest.skip('warp is only written in nat/mni space')
    out = run('--no-prob-nat', '--no-labels-nat', f'--{flag}-{space}')
    assert out == [f'in.{flag}.{space}.nii']


def test_custom_filename(setup, tmp_path):
    run, _ = setup
    out = run('--no-prob-nat', '--no-labels-nat',
              '--prior-mni', '{dir}{sep}myprior{ext}',
              '--nobias-mni', '{dir}{sep}mynobias{ext}')
    assert out == ['mynobias.nii', 'myprior.nii']


@pytest.mark.parametrize('space', ['mni', 'wrp'])
def test_all_space(setup, space):
    run, _ = setup
    out = run('--no-prob-nat', '--no-labels-nat', f'--all-{space}')
    expected = {f'in.{f}.{space}.nii'
                for f in ('prob', 'labels', 'prior', 'bias', 'nobias')}
    if space == 'mni':
        expected.add('in.warp.mni.nii')
    assert set(out) == expected


def test_no_align_mni_output(setup):
    run, _ = setup
    out = run('--no-align', '--no-prob-nat', '--no-labels-nat', '--prob-mni')
    assert out == ['in.prob.mni.nii']


@pytest.mark.parametrize('flag', ['bias', 'nobias'])
@pytest.mark.parametrize('space', ['nat', 'mni', 'wrp'])
def test_multichannel(setup, tmp_path, flag, space):
    run, _ = setup
    out = run('--no-prob-nat', '--no-labels-nat', f'--{flag}-{space}',
              inputs=('ch1.nii', 'ch2.nii'))
    assert out == [f'ch1.{flag}.{space}.nii', f'ch2.{flag}.{space}.nii']
    for c in (1, 2):
        written = nib.load(str(tmp_path / 'out' / out[c-1])).get_fdata()
        written = written.reshape(SHAPE)
        if flag == 'bias':
            expected = np.full(SHAPE, BIAS)
        else:
            dat = nib.load(str(tmp_path / f'ch{c}.nii')).get_fdata()
            expected = BIAS * dat
        np.testing.assert_allclose(written, expected, rtol=1e-3, atol=1e-3)


def test_multichannel_custom_name(setup):
    run, _ = setup
    out = run('--no-prob-nat', '--no-labels-nat',
              '--bias-nat', '{dir}{sep}mybias{ext}',
              '--nobias-nat', '{dir}{sep}nobias_{c}{ext}',
              inputs=('ch1.nii', 'ch2.nii'))
    assert out == ['mybias.1.nii', 'mybias.2.nii',
                   'nobias_1.nii', 'nobias_2.nii']


def test_parse_error_prints_full_help(capsys):
    """On a ParseError, uniseg prints the whole help text, not one character."""
    from nitorch.cli.segmentation.uniseg.cli import cli
    from nitorch.cli.segmentation.uniseg.parser import help as uniseg_help
    rc = cli(['x.nii', '-a', 'sometimes'])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert out.strip() == uniseg_help.strip()
