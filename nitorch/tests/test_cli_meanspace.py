"""`meanspace` command: layout, device, pad unit and `-o no`."""
import os

import numpy as np
import pytest
import torch

from nitorch.cli.registration.meanspace import cli as meanspace_cli
from nitorch.cli.registration.meanspace.cli import _cli as meanspace

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.zeros([4, 6, 8], dtype='float32')
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


def load(fname):
    f = nib.load(fname)
    return f.shape, f.affine


def test_layout(tmp_path, volume):
    out = str(tmp_path / 'ms.nii.gz')
    meanspace([volume, '-o', out, '-l', 'LPS'])
    _, aff = load(out)
    assert ''.join(nib.aff2axcodes(aff)) == 'LPS'


def test_default_layout(tmp_path, volume):
    out = str(tmp_path / 'ms.nii.gz')
    meanspace([volume, '-o', out])
    shape, aff = load(out)
    assert ''.join(nib.aff2axcodes(aff)) == 'RAS'
    assert shape == (4, 6, 8)


@pytest.mark.parametrize('unit,expected', [
    ('mm', (6, 8, 10)),     # 2 mm = 1 voxel (of 2 mm) on each side
    ('vox', (8, 10, 12)),   # 2 voxels on each side
])
def test_pad_unit(tmp_path, volume, unit, expected):
    out = str(tmp_path / 'ms.nii.gz')
    meanspace([volume, '-o', out, '-p', '2', unit])
    shape, _ = load(out)
    assert shape == expected


def test_pad_pct_unchanged(tmp_path, volume):
    out = str(tmp_path / 'ms.nii.gz')
    meanspace([volume, '-o', out, '-p', '50', '%'])
    shape, _ = load(out)
    assert shape == (6, 10, 12)


def test_output_no_writes_nothing(tmp_path, volume, monkeypatch):
    tmpdir = tmp_path / 'tmp'
    tmpdir.mkdir()
    monkeypatch.setenv('TMPDIR', str(tmpdir))
    monkeypatch.setattr('tempfile.tempdir', None)
    before = sorted(os.listdir(tmp_path))
    meanspace([volume, '-o', 'no'])
    assert sorted(os.listdir(tmp_path)) == before
    assert os.listdir(tmpdir) == []


def test_output_no_with_reslice(tmp_path, volume, monkeypatch):
    tmpdir = tmp_path / 'tmp'
    tmpdir.mkdir()
    monkeypatch.setenv('TMPDIR', str(tmpdir))
    monkeypatch.setattr('tempfile.tempdir', None)
    meanspace([volume, '-o', 'no', '-r'])
    assert os.path.exists(tmp_path / 'img.meanspace.nii.gz')
    assert not os.path.exists(tmp_path / 'meanspace.nii.gz')
    assert os.listdir(tmpdir) == []


@pytest.mark.parametrize('flag,expected', [
    ([], ('cpu', None)),
    (['--cpu'], ('cpu', None)),
    (['--gpu'], ('gpu', None)),
    (['--gpu', '1'], ('gpu', 1)),
])
def test_device(tmp_path, volume, monkeypatch, flag, expected):
    seen = {}

    def fake_setup_device(device='cpu', ndevice=None):
        seen['option'] = (device, ndevice)
        # 'meta' tensors need no CUDA and are never allocated
        return torch.device('meta')

    def fake_mean_space(affines, shapes, **kwargs):
        seen['devices'] = {a.device.type for a in affines}
        raise StopIteration

    monkeypatch.setattr(meanspace_cli, 'setup_device', fake_setup_device)
    monkeypatch.setattr(meanspace_cli, 'mean_space', fake_mean_space)
    with pytest.raises(StopIteration):
        meanspace([volume, '-o', str(tmp_path / 'ms.nii.gz'), *flag])
    assert seen['option'] == expected
    assert seen['devices'] == {'meta'}


def test_setup_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert meanspace_cli.setup_device('cpu', None) == torch.device('cpu')
    assert meanspace_cli.setup_device('gpu', None) == torch.device('cuda')
    assert meanspace_cli.setup_device('gpu', 1) == torch.device('cuda:1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.warns(UserWarning):
        assert meanspace_cli.setup_device('gpu', 1) == torch.device('cpu')


def test_resliced_names(tmp_path, volume, monkeypatch):
    other = str(tmp_path / 'sub' / 'other.nii')
    os.makedirs(os.path.dirname(other))
    nib.save(nib.load(volume), other)
    calls = []
    monkeypatch.setattr(meanspace_cli, 'reslice', lambda a: calls.append(a))
    meanspace([volume, other, '-o', '{dir}/{base}.ms{ext}',
               '-r', '{dir}/{base}.r{ext}', '--cpu'])
    out = str(tmp_path / 'img.ms.nii.gz')
    assert os.path.exists(out)
    assert calls == [
        [volume, '-o', str(tmp_path / 'img.r.nii.gz'), '-t', out, '--cpu'],
        [other, '-o', str(tmp_path / 'sub' / 'other.r.nii'), '-t', out,
         '--cpu'],
    ]
