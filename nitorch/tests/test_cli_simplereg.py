"""`nitorch simplereg`: the command is registered and its help is accurate."""
import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands, cli


def test_simplereg_is_registered():
    assert 'simplereg' in commands


def test_simplereg_help(capsys):
    cli(['simplereg', '-h'])
    captured = capsys.readouterr()
    assert '[nitorch] simplereg' in captured.out
    assert 'Unknown command' not in captured.err


def test_simplereg_help_advertises_no_unparsed_options():
    from nitorch.cli.registration.simplereg.parser import help
    for opt in ('--fwhm', '--bound', '--order', '--pad', '--rescale',
                '--discretize', 'emmi', '[m]', 'nonlinear registration.'):
        assert opt not in help, opt


def test_simplereg_smoke(tmp_path):
    import numpy as np
    import pytest
    nib = pytest.importorskip('nibabel')
    rng = np.random.RandomState(0)
    for name in ('mov', 'fix'):
        img = rng.rand(8, 8, 8).astype('float32')
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(tmp_path / f'{name}.nii.gz'))
    cli(['simplereg', '-m', str(tmp_path / 'mov.nii.gz'),
         '-f', str(tmp_path / 'fix.nii.gz'), '-n', '2', '-p', '0',
         '-o', str(tmp_path / 'xfm.lta'),
         '-r', str(tmp_path / 'res.nii.gz')])
    assert (tmp_path / 'xfm.lta').exists()
    assert (tmp_path / 'res.nii.gz').exists()
