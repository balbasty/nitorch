"""The `distmap` command."""
import numpy as np
import pytest

from nitorch.cli.cli import cli as nitorch_cli
from nitorch.cli.misc.distmap.cli import _cli as distmap
from nitorch.cli.misc.distmap.parser import parser

nib = pytest.importorskip('nibabel')
ndimage = pytest.importorskip('scipy.ndimage')


@pytest.fixture
def labels(tmp_path):
    """Label map with a 5x5x5 cube of label 1 and a 3x3x3 cube of label 2"""
    fname = str(tmp_path / 'labels.nii.gz')
    dat = np.zeros([16, 16, 12], dtype='uint8')
    dat[2:7, 2:7, 2:7] = 1
    dat[9:12, 9:12, 6:9] = 2
    affine = np.diag([1., 2., 3., 1.])
    nib.save(nib.Nifti1Image(dat, affine), fname)
    return fname, dat


def run(tmp_path, fname, *opts):
    out = str(tmp_path / 'out.nii.gz')
    distmap([fname, '-o', out, *opts])
    img = nib.load(out)
    assert img.get_data_dtype() == np.float32
    return np.asarray(img.dataobj)


def test_parser_defaults():
    opt = parser(['x.nii'])
    assert opt.labels == []
    assert opt.threshold == 0
    assert opt.invert is False
    assert opt.signed is False
    assert opt.metric == 'l2'
    assert opt.unit == 'mm'
    assert opt.gpu == ('cpu', 0)
    opt = parser(['x.nii', '-l', '1', '3', '-i', '-s', '-m', 'l1', '-u', 'vox'])
    assert opt.labels == [1, 3]
    assert opt.invert is True
    assert opt.signed is True
    assert opt.metric == 'l1'
    assert opt.unit == 'vox'


def test_registered():
    assert nitorch_cli(['help', 'distmap']) is None


def test_euclidean_mm(tmp_path, labels):
    fname, dat = labels
    out = run(tmp_path, fname)
    ref = ndimage.distance_transform_edt(dat > 0, sampling=[1, 2, 3])
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_euclidean_vox_label_invert(tmp_path, labels):
    fname, dat = labels
    out = run(tmp_path, fname, '-l', '2', '-i', '-u', 'vox')
    ref = ndimage.distance_transform_edt(dat != 2)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_l1(tmp_path, labels):
    fname, dat = labels
    out = run(tmp_path, fname, '-m', 'l1', '-u', 'vox')
    ref = ndimage.distance_transform_cdt(dat > 0, metric='taxicab')
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_signed(tmp_path, labels):
    fname, dat = labels
    mask = dat == 1
    inside = ndimage.distance_transform_edt(mask, sampling=[1, 2, 3])
    outside = ndimage.distance_transform_edt(~mask, sampling=[1, 2, 3])
    out = run(tmp_path, fname, '-l', '1', '-s')
    np.testing.assert_allclose(out, inside - outside, atol=1e-4)
    out = run(tmp_path, fname, '-l', '1', '-s', '-i')
    np.testing.assert_allclose(out, outside - inside, atol=1e-4)


def test_threshold(tmp_path, labels):
    fname, dat = labels
    out = run(tmp_path, fname, '-t', '1', '-u', 'vox')
    ref = ndimage.distance_transform_edt(dat > 1)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_empty_object_warns(tmp_path, labels):
    fname, _ = labels
    with pytest.warns(UserWarning, match='infinite'):
        out = run(tmp_path, fname, '-l', '7', '-i')
    assert np.isinf(out).all()
