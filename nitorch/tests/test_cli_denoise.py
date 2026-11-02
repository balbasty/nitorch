"""`denoise` command: option pass-through, device fallback, input shape."""
import numpy as np
import pytest
import torch

from nitorch.cli.denoising.denoise import cli as denoise_cli
from nitorch.cli.denoising.denoise import main as denoise_main

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.random.RandomState(0).rand(4, 5, 6).astype('float32')
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
    return fname


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_tv_denoise(dat, **kwargs):
        calls.append(kwargs)
        return dat

    monkeypatch.setattr(denoise_main.tv, 'denoise', fake_tv_denoise)
    return calls


@pytest.mark.parametrize('optim', ['relax', 'fmg+cg'])
def test_optim_reaches_solver(volume, captured, optim):
    denoise_cli._cli([volume, '-s', '1', '-x', optim])
    assert len(captured) == 1
    assert captured[0]['optim'] == optim


def test_gpu_without_cuda_falls_back(volume, captured, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.warns(UserWarning, match='CUDA not available'):
        denoise_cli._cli([volume, '-s', '1', '--gpu'])
    assert len(captured) == 1


def test_ensure_4d_squeezes_5d_singleton(tmp_path, captured):
    fname = str(tmp_path / 'img5d.nii.gz')
    dat = np.random.RandomState(0).rand(4, 5, 6, 1, 2).astype('float32')
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
    out = denoise_main.denoise(fname, lam=1, sigma=1)
    assert nib.load(out).shape[:3] == (4, 5, 6)
    assert np.prod(nib.load(out).shape) == 4 * 5 * 6 * 2
