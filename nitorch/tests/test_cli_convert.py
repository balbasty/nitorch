"""`convert` command: header overrides and MGH data types (#171)."""
import numpy as np
import pytest

from nitorch.cli.misc.convert.cli import _cli as convert
from nitorch.cli.misc.convert.parser import parse

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii')
    dat = np.arange(2 * 3 * 4, dtype='int8').reshape([2, 3, 4])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


@pytest.fixture
def series(tmp_path):
    fname = str(tmp_path / 'series.nii')
    dat = np.ones([2, 3, 4, 2], dtype='float32')
    nib.save(nib.Nifti1Image(dat, np.eye(4)), fname)
    return fname


@pytest.mark.parametrize('tag', ['-m', '--meta', '--metadata'])
def test_metadata_tag(tag):
    opt = parse(['img.nii', tag, 'inter=1', 'slope=2'])
    assert opt.meta == {'inter': '1', 'slope': '2'}


def test_mgh_dtype_from_output_extension(tmp_path, volume):
    # int8 is not an MGH type: it must be promoted (to int16) because
    # the *output* is .mgz, even though the input is .nii
    out = str(tmp_path / 'out.mgz')
    convert([volume, '-o', out])
    img = nib.load(out)
    assert img.get_data_dtype().newbyteorder('=') == np.dtype('int16')
    assert np.array_equal(np.asarray(img.dataobj),
                          nib.load(volume).get_fdata())


@pytest.mark.parametrize('ext', ['.nii', '.mgz'])
def test_voxel_size_override(tmp_path, volume, ext):
    out = str(tmp_path / ('out' + ext))
    convert([volume, '-m', 'voxel_size=[3,3,3]', '-o', out])
    img = nib.load(out)
    assert np.allclose(img.header.get_zooms()[:3], 3)
    assert np.allclose(np.linalg.norm(img.affine[:3, :3], axis=0), 3)


def test_tr_override_nifti(tmp_path, series):
    out = str(tmp_path / 'out.nii')
    convert([series, '-m', 'tr=2.5', '-o', out])
    hdr = nib.load(out).header
    assert np.isclose(hdr.get_zooms()[3], 2.5)
    assert hdr.get_xyzt_units()[1] == 'sec'


def test_tr_override_mgh(tmp_path, series):
    out = str(tmp_path / 'out.mgz')
    convert([series, '-m', 'tr=2.5', '-o', out])
    assert np.isclose(float(nib.load(out).header['tr']), 2500)


def test_inter_sets_intercept(tmp_path, series):
    out = str(tmp_path / 'out.nii')
    convert([series, '-m', 'inter=5', '-o', out])
    img = nib.load(out)
    assert np.isclose(img.dataobj.slope, 1)
    assert np.isclose(img.dataobj.inter, 5)
