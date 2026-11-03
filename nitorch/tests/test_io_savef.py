"""Tests for `io.savef` dispatch across formats (part of #171)."""
import numpy as np
import pytest
import torch

from nitorch import io


def _affine():
    aff = torch.eye(4, dtype=torch.float64)
    aff[:3, :3] = torch.tensor([[0.9, 0.1, 0.0],
                                [-0.1, 0.9, 0.0],
                                [0.0, 0.0, 1.1]], dtype=torch.float64)
    aff[:3, 3] = torch.tensor([1.5, -2.0, 3.25], dtype=torch.float64)
    return aff


@pytest.mark.parametrize('savef', [io.savef, io.transforms.savef])
def test_savef_affine_lta_roundtrip(tmp_path, savef):
    aff = _affine()
    fname = str(tmp_path / 'xfm.lta')
    savef(aff, fname)
    for loadf in (io.transforms.loadf, io.loadf):
        out = torch.as_tensor(loadf(fname), dtype=torch.float64)
        out = out.reshape(4, 4)
        assert torch.allclose(out, aff, atol=1e-5)


def test_savef_volume_to_lta_raises_clear_error(tmp_path):
    vol = torch.rand(5, 6, 7)
    fname = str(tmp_path / 'vol.lta')
    with pytest.raises(ValueError, match='4x4'):
        io.savef(vol, fname)


def test_savef_nifti_still_works(tmp_path):
    vol = torch.rand(5, 6, 7)
    fname = str(tmp_path / 'vol.nii.gz')
    io.savef(vol, fname)
    out = io.loadf(fname)
    assert tuple(out.shape) == (5, 6, 7)
    assert np.allclose(np.asarray(out), vol.numpy(), atol=1e-5)


# ---------------------------------------------------------------------
# `savef_new` takes a world (RAS-to-RAS) matrix and converts it to the
# format's own convention; `save_new` writes the raw matrix as-is.
# ---------------------------------------------------------------------

from nitorch.io.transforms.freesurfer.lta import LinearTransformArray
from nitorch.io.transforms.freesurfer.registerdat import RegisterDat
from nitorch.io.transforms.freesurfer.fsutils import affine_to_fs


def _vox2ras(shift, vx):
    aff = torch.eye(4, dtype=torch.float64)
    aff[:3, :3] = torch.diag(torch.as_tensor(vx, dtype=torch.float64))
    aff[0, 1] = 0.2
    aff[:3, 3] = torch.as_tensor(shift, dtype=torch.float64)
    return aff


def _volinfo(aff, shape):
    vx, x, y, z, c = affine_to_fs(aff, shape)
    return dict(volume=tuple(shape), voxelsize=tuple(map(float, vx)),
                xras=tuple(map(float, x)), yras=tuple(map(float, y)),
                zras=tuple(map(float, z)), cras=tuple(map(float, c)))


def _spaces():
    src_aff, src_shape = _vox2ras([-10., 5., 3.], [1., 1.2, 0.8]), (10, 11, 12)
    dst_aff, dst_shape = _vox2ras([4., -6., 2.], [0.9, 1., 1.5]), (8, 9, 7)
    return src_aff, src_shape, dst_aff, dst_shape


def _load(fname):
    return io.transforms.map(fname)


def test_savef_new_ras_same_as_save_new(tmp_path):
    aff = _affine()
    f1, f2 = str(tmp_path / 'a.lta'), str(tmp_path / 'b.lta')
    LinearTransformArray.save_new(aff, f1)
    LinearTransformArray.savef_new(aff, f2)
    for f in (f1, f2):
        assert _load(f).type() == ('ras', 'ras')
    d1 = _load(f1).data(dtype=torch.float64)
    d2 = _load(f2).data(dtype=torch.float64)
    assert torch.allclose(d1, d2, atol=1e-6)
    assert torch.allclose(d2.reshape(4, 4), aff, atol=1e-5)


@pytest.mark.parametrize('via_like', [False, True])
def test_savef_new_voxel_lta_converts_from_ras(tmp_path, via_like):
    src_aff, src_shape, dst_aff, dst_shape = _spaces()
    aff_ras = _affine()
    meta = dict(type='voxel', src=_volinfo(src_aff, src_shape),
                dst=_volinfo(dst_aff, dst_shape))
    if via_like:
        like = str(tmp_path / 'like.lta')
        LinearTransformArray.save_new(torch.eye(4), like, **meta)
        assert _load(like).type() == ('voxel', 'voxel')
        kwargs = dict(like=like)
    else:
        kwargs = meta

    fname = str(tmp_path / 'xfm.lta')
    LinearTransformArray.savef_new(aff_ras, fname, **kwargs)
    lta = _load(fname)
    assert lta.type() == ('voxel', 'voxel')
    fdata = lta.fdata(dtype=torch.float64).reshape(4, 4)
    assert torch.allclose(fdata, aff_ras, atol=1e-4)
    expected_vox = dst_aff.inverse() @ aff_ras @ src_aff
    data = lta.data(dtype=torch.float64).reshape(4, 4)
    assert torch.allclose(data, expected_vox, atol=1e-4)

    # save_new writes the raw matrix as-is
    fname_raw = str(tmp_path / 'raw.lta')
    LinearTransformArray.save_new(aff_ras, fname_raw, **kwargs)
    raw = _load(fname_raw)
    assert raw.type() == ('voxel', 'voxel')
    data = raw.data(dtype=torch.float64).reshape(4, 4)
    assert torch.allclose(data, aff_ras, atol=1e-5)


def test_savef_new_voxel_lta_from_mapped_uses_fdata(tmp_path):
    src_aff, src_shape, dst_aff, dst_shape = _spaces()
    aff_ras = _affine()
    meta = dict(type='voxel', src=_volinfo(src_aff, src_shape),
                dst=_volinfo(dst_aff, dst_shape))
    f1, f2 = str(tmp_path / 'a.lta'), str(tmp_path / 'b.lta')
    LinearTransformArray.savef_new(aff_ras, f1, **meta)
    # re-save a mapped voxel LTA with savef_new -> same file content
    LinearTransformArray.savef_new(_load(f1), f2)
    d1 = _load(f1).data(dtype=torch.float64)
    d2 = _load(f2).data(dtype=torch.float64)
    assert _load(f2).type() == ('voxel', 'voxel')
    assert torch.allclose(d1, d2, atol=1e-4)


def test_savef_new_non_ras_missing_volume_info_raises(tmp_path):
    fname = str(tmp_path / 'xfm.lta')
    with pytest.raises(ValueError, match='volume'):
        LinearTransformArray.savef_new(_affine(), fname, type='voxel')
    src_aff, src_shape, _, _ = _spaces()
    with pytest.raises(ValueError, match='dst'):
        LinearTransformArray.savef_new(_affine(), fname, type='physical',
                                       src=_volinfo(src_aff, src_shape))


def test_savef_new_registerdat_roundtrip(tmp_path):
    aff = _affine()
    fname = str(tmp_path / 'register.dat')
    RegisterDat.savef_new(aff, fname, subject='bert')
    dat = _load(fname)
    assert isinstance(dat, RegisterDat)
    assert torch.allclose(dat.fdata(dtype=torch.float64), aff, atol=1e-5)
    assert dat.metadata(['subject'])['subject'] == 'bert'
    fname2 = str(tmp_path / 'register2.dat')
    RegisterDat.savef_new(dat, fname2)
    dat2 = _load(fname2)
    assert torch.allclose(dat2.fdata(dtype=torch.float64), aff, atol=1e-5)
    assert dat2.metadata(['subject'])['subject'] == 'bert'
