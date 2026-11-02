"""`denoise_mri` command: device default/fallback, verbosity, output names."""
import os
import numpy as np
import pytest
import torch

from nitorch.cli.denoising.denoise_mri import cli as dmri_cli
from nitorch.cli.denoising.denoise_mri import main as dmri_main
from nitorch.cli.denoising.denoise_mri.parser import parse

nib = pytest.importorskip('nibabel')


def _volume(tmp_path, name, seed=0):
    fname = str(tmp_path / name)
    dat = np.random.RandomState(seed).rand(4, 5, 6).astype('float32')
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
    return fname


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_denoise_mri(*files, **kwargs):
        calls.append((files, kwargs))

    monkeypatch.setattr(dmri_cli, 'denoise_mri', fake_denoise_mri)
    return calls


@pytest.fixture
def fast_noise(monkeypatch):
    """Replace the (slow) mixture-model noise estimate."""
    def fake_estimate_noise(dat, **kwargs):
        one = torch.ones([], dtype=dat.dtype)
        return {'sd': one}, {'mean': one}
    monkeypatch.setattr(dmri_main, 'estimate_noise', fake_estimate_noise)


def test_default_device_is_cpu(tmp_path, captured):
    f = _volume(tmp_path, 'a.nii.gz')
    dmri_cli._cli([f])
    assert len(captured) == 1
    assert torch.device(captured[0][1]['device']).type == 'cpu'


def test_gpu_without_cuda_falls_back(tmp_path, captured, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    f = _volume(tmp_path, 'a.nii.gz')
    with pytest.warns(UserWarning, match='CUDA not available'):
        dmri_cli._cli([f, '--gpu'])
    assert torch.device(captured[0][1]['device']).type == 'cpu'


def test_verbose_zero_silences(tmp_path, fast_noise, capsys):
    assert not parse(['a.nii', '-v', '0']).verbose
    assert parse(['a.nii', '-v', '1']).verbose
    f = _volume(tmp_path, 'a.nii.gz')
    dmri_cli._cli([f, '-v', '0', '-n', '30', '-t', '0'])
    assert 'n_iter' not in capsys.readouterr().out


def test_outputs_named_after_own_inputs(tmp_path, fast_noise):
    f1 = _volume(tmp_path, 'T1w.nii.gz', 0)
    f2 = _volume(tmp_path, 'T2w.nii.gz', 1)
    out = str(tmp_path / 'out')
    dmri_cli._cli([f1, f2, '-n', '2', '-v', '0', '-o', out])
    assert sorted(os.listdir(out)) == ['den_T1w.nii.gz', 'den_T2w.nii.gz']
