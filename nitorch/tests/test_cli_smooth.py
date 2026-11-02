"""Options of the `smooth` and `metrics` commands (issue #171)."""
import numpy as np
import pytest

from nitorch.cli.misc.smooth import cli as smooth_cli
from nitorch.cli.misc.smooth.cli import _cli as smooth
from nitorch.cli.misc.smooth.parser import parser as smooth_parser
from nitorch.cli.misc.metrics.parser import parser as metrics_parser

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.zeros([9, 9, 9], dtype='float32')
    dat[4, 4, 4] = 1
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
    return fname


def run(volume, tmp_path, name, *opts):
    out = str(tmp_path / f'{name}.nii.gz')
    smooth([volume, '-f', '3', 'vox', '-o', out, *opts])
    return np.asarray(nib.load(out).dataobj)


def test_smooth_parser_device():
    assert smooth_parser(['x.nii', '--gpu', '1']).gpu == ('gpu', 1)
    assert smooth_parser(['x.nii', '--gpu']).gpu == ('gpu', 0)
    assert smooth_parser(['x.nii', '--cpu', '4']).gpu == ('cpu', 4)
    assert smooth_parser(['x.nii', '--cpu']).gpu == ('cpu', 0)
    assert smooth_parser(['x.nii']).gpu == ('cpu', 0)


def test_metrics_parser_device():
    base = ['-p', 'a.nii', '-r', 'b.nii', '-m', 'mse']
    assert metrics_parser(base + ['--gpu', '1']).device == ('gpu', 1)
    assert metrics_parser(base + ['--gpu']).device == ('gpu', 0)
    assert metrics_parser(base + ['--cpu', '4']).device == ('cpu', 4)
    assert metrics_parser(base + ['--cpu']).device == ('cpu', 0)
    assert metrics_parser(base).device == ('cpu', 0)


def test_smooth_device_string(monkeypatch):
    monkeypatch.setattr(smooth_cli.torch.cuda, 'is_available', lambda: True)
    assert smooth_cli.get_device(('gpu', 1)) == 'cuda:1'
    assert smooth_cli.get_device(('cpu', 0)) == 'cpu'


def test_smooth_method_tri(tmp_path, volume):
    out = run(volume, tmp_path, 'tri', '-m', 'tri')
    assert np.isclose(out.sum(), 1, atol=1e-4)
    assert out[4, 4, 4] < 1


def test_smooth_basis_rect(tmp_path, volume):
    default = run(volume, tmp_path, 'default')
    tri = run(volume, tmp_path, 'tri', '-b', 'tri')
    rect = run(volume, tmp_path, 'rect', '-b', 'rect')
    assert np.allclose(default, tri)
    assert not np.allclose(rect, tri)
    assert np.isclose(rect.sum(), 1, atol=1e-2)
