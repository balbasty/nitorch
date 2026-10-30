"""Tests that should pass with all supported versions of nibabel."""
import os

import numpy as np
import pytest
import torch

from nitorch.io import optionals

nib = pytest.importorskip('nibabel')
if optionals.nibabel is None:
    pytest.skip('nibabel cannot be imported', allow_module_level=True)

from nitorch import io                                       # noqa: E402
from nitorch.io.volumes.babel import _compat                 # noqa: E402

SHAPE = (7, 6, 5)
AFFINE = np.array([[0.9, 0.1, 0, -10],
                   [-0.1, 0.9, 0.05, 5],
                   [0, -0.05, 1.1, 3],
                   [0, 0, 0, 1]])


def make_scaled(fname):
    """Write an int16 nifti with non-trivial slope and intercept"""
    fdat = np.arange(np.prod(SHAPE), dtype='float64').reshape(SHAPE)
    fdat = fdat * 0.37 + 5
    header = nib.Nifti1Header()
    header.set_data_dtype('int16')
    nib.save(nib.Nifti1Image(fdat, AFFINE, header), fname)
    slope, inter = nib.load(fname).dataobj.slope, nib.load(fname).dataobj.inter
    assert slope != 1 and inter != 0, "test file should be scaled"
    return fdat, slope


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_disk_header_has_scaling(tmp_path, ext):
    fname = os.path.join(tmp_path, 'scaled' + ext)
    make_scaled(fname)
    image = nib.load(fname)
    header = _compat.disk_header(image)
    assert header.get_slope_inter() == (image.dataobj.slope,
                                        image.dataobj.inter)


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_metadata_scaling(tmp_path, ext):
    fname = os.path.join(tmp_path, 'scaled' + ext)
    make_scaled(fname)
    image = nib.load(fname)
    meta = io.map(fname).metadata(['slope', 'inter'])
    assert np.isclose(meta['slope'], image.dataobj.slope)
    assert np.isclose(meta['inter'], image.dataobj.inter)


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_savef_like_scaled(tmp_path, ext):
    fname = os.path.join(tmp_path, 'scaled' + ext)
    oname = os.path.join(tmp_path, 'out' + ext)
    fdat, slope = make_scaled(fname)
    io.savef(torch.as_tensor(fdat), oname, like=fname)
    out = io.map(oname)
    assert out.dtype == np.int16
    assert np.abs(out.fdata(numpy=True) - fdat).max() <= slope


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_save_like_scaled(tmp_path, ext):
    fname = os.path.join(tmp_path, 'scaled' + ext)
    oname = os.path.join(tmp_path, 'out' + ext)
    fdat, slope = make_scaled(fname)
    inp = io.map(fname)
    io.save(inp.data(), oname, like=inp)
    out = io.map(oname)
    assert np.array_equal(out.data(numpy=True), inp.data(numpy=True))
    assert np.abs(out.fdata(numpy=True) - fdat).max() <= slope


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz', '.mgh', '.mgz'])
@pytest.mark.parametrize('dtype', ['uint8', 'int16', 'float32'])
def test_partial_write(tmp_path, ext, dtype):
    fname = os.path.join(tmp_path, 'vol' + ext)
    dat = (np.random.default_rng(0).random(SHAPE) * 100).astype(dtype)
    io.save(torch.as_tensor(dat), fname, affine=torch.as_tensor(AFFINE))
    vol = io.map(fname, 'r+')
    vol[2:4, 1:3].set_data(np.full([2, 2, SHAPE[-1]], 7, dtype=vol.dtype))
    del vol
    dat[2:4, 1:3] = 7
    out = io.map(fname)
    assert np.array_equal(out.data(numpy=True), dat)
    assert np.allclose(out.affine.numpy(), AFFINE, atol=1e-4)


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_partial_write_keeps_scaling(tmp_path, ext):
    fname = os.path.join(tmp_path, 'scaled' + ext)
    fdat, slope = make_scaled(fname)
    vol = io.map(fname, 'r+')
    raw = vol.data(numpy=True)
    vol[2:4, 1:3].set_data(torch.as_tensor(raw[2:4, 1:3]))
    del vol
    out = io.map(fname)
    assert np.abs(out.fdata(numpy=True) - fdat).max() <= slope


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
def test_set_metadata_keeps_scaling(tmp_path, ext):
    fname = os.path.join(tmp_path, 'scaled' + ext)
    fdat, slope = make_scaled(fname)
    new_affine = AFFINE.copy()
    new_affine[:3, -1] += [1, 2, 3]
    vol = io.map(fname, 'r+')
    vol.set_metadata(affine=new_affine)
    del vol
    out = io.map(fname)
    assert np.allclose(out.affine.numpy(), new_affine, atol=1e-4)
    assert np.abs(out.fdata(numpy=True) - fdat).max() <= slope


def test_mgh_dtypes():
    dtypes = _compat.mgh_dtypes()
    assert dtypes[0] == np.uint8
    assert dtypes[-1] == np.float32
    for dt in ('uint8', 'int16', 'int32', 'float32'):
        # smallest compatible type is the type itself
        first = next(d for d in dtypes if np.can_cast(dt, d, 'safe'))
        assert first == np.dtype(dt)


def test_quaternion_tolerance(tmp_path):
    fname = os.path.join(tmp_path, 'quat.nii')
    header = nib.Nifti1Header()
    header.set_data_shape(SHAPE)
    header.set_data_dtype('float32')
    header.set_qform(np.eye(4), code=1)
    # |bcd|^2 = 1 + 5e-7: not a unit quaternion because of rounding
    bcd = np.array([1, 0, 0], dtype='float64') * np.sqrt(1 + 5e-7)
    header['quatern_b'], header['quatern_c'], header['quatern_d'] = bcd
    nib.save(nib.Nifti1Image(np.zeros(SHAPE, 'float32'), None, header), fname)
    io.map(fname).affine
