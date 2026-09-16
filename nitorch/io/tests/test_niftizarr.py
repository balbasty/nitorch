import numpy as np
import pytest
import torch
import zarr
import nibabel as nib
from nibabel.nifti1 import Nifti1Header

import nitorch.io as nio
from nitorch.io.volumes.zarr.array import NiftiZarrArray
from nitorch.io.volumes.zarr.metadata import _NIFTI1_SIZE


def _write_group(path, arrays, header_bytes=None, multiscales=None):
    root = zarr.open_group(store=str(path), mode='w')
    for name, data in arrays.items():
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype,
                                chunks=data.shape)
        arr[:] = data
    if header_bytes is not None:
        hdr_arr = root.create_array('nifti', shape=(len(header_bytes),), dtype='uint8')
        hdr_arr[:] = np.frombuffer(header_bytes, dtype='uint8')
    if multiscales is not None:
        root.attrs['multiscales'] = multiscales
    return root


def _nifti_header_bytes(shape, affine, dtype='float32'):
    hdr = Nifti1Header()
    hdr.set_data_shape(shape)
    hdr.set_data_dtype(dtype)
    hdr.set_qform(affine)
    hdr.set_sform(affine)
    return hdr.binaryblock


@pytest.fixture
def reference_nifti(tmp_path):
    """A real NIfTI file and the equivalent nifti-zarr store built from it.

    `data`/`affine` describe the store in standard NIfTI (x, y, z) axis
    order. The underlying Zarr array is written in the nifti-zarr spec's
    own on-disk convention -- the reverse of NIfTI order (z, y, x), absent
    OME axes metadata saying otherwise -- so this fixture exercises the
    same axis permutation a real nifti-zarr store requires (a non-cubic
    shape makes a wrong or missing permutation produce visibly wrong data,
    not just a shape mismatch).
    """
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype('float32')
    affine = np.diag([1.5, 2.5, 3.5, 1.0])
    affine[:3, -1] = [10, 20, 30]
    nii_path = tmp_path / 'ref.nii'
    nib.save(nib.Nifti1Image(data, affine), str(nii_path))

    img = nib.load(str(nii_path))
    zarr_path = tmp_path / 'ref.nii.zarr'
    _write_group(zarr_path, {'0': data.transpose(2, 1, 0)},
                 header_bytes=img.header.binaryblock)
    return str(nii_path), str(zarr_path), data, affine


@pytest.fixture
def plain_ome_zarr(tmp_path):
    """A plain OME-Zarr store (no embedded header), single scale.

    `data` is in standard (x, y, z) order; the store's declared axes are
    (z, y, x), so the underlying array is written transposed accordingly
    (see `reference_nifti`).
    """
    data = np.arange(8 * 8 * 8).reshape(8, 8, 8).astype('float32')
    path = tmp_path / 'plain.ome.zarr'
    multiscales = [{
        'axes': [{'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                 {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                 {'name': 'x', 'type': 'space', 'unit': 'micrometer'}],
        'datasets': [{'path': '0', 'coordinateTransformations':
                      [{'type': 'scale', 'scale': [1.0, 2.0, 3.0]}]}],
    }]
    _write_group(path, {'0': data.transpose(2, 1, 0)}, multiscales=multiscales)
    return str(path), data


@pytest.fixture
def multiscale_ome_zarr(tmp_path):
    """A multiscale (3-level) plain OME-Zarr store, no embedded header.

    `arrays` (returned, used for assertions) are in standard (x, y, z)
    order; the declared axes are (z, y, x), so the underlying arrays are
    written transposed accordingly (see `reference_nifti`).
    """
    shapes = [(16, 16, 16), (8, 8, 8), (4, 4, 4)]
    rs = np.random.RandomState(0)
    arrays = {str(i): (rs.rand(*shp) * 100).astype('float32')
              for i, shp in enumerate(shapes)}
    path = tmp_path / 'multi.ome.zarr'
    multiscales = [{
        'axes': [{'name': 'z', 'type': 'space', 'unit': 'millimeter'},
                 {'name': 'y', 'type': 'space', 'unit': 'millimeter'},
                 {'name': 'x', 'type': 'space', 'unit': 'millimeter'}],
        'datasets': [
            {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}]},
            {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2.0, 2.0, 2.0]}]},
            {'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4.0, 4.0, 4.0]}]},
        ],
    }]
    _write_group(path, {k: v.transpose(2, 1, 0) for k, v in arrays.items()},
                 multiscales=multiscales)
    return str(path), arrays


# --- User Story 1 --------------------------------------------------------

def test_map_nifti_zarr_matches_plain_nifti(reference_nifti):
    nii_path, zarr_path, data, affine = reference_nifti
    vol_nii = nio.map(nii_path)
    vol_zarr = nio.map(zarr_path)
    assert isinstance(vol_zarr, NiftiZarrArray)
    assert torch.allclose(vol_nii.affine, vol_zarr.affine)
    assert np.allclose(vol_nii.voxel_size, vol_zarr.voxel_size)
    assert vol_nii.dtype == vol_zarr.dtype
    assert np.allclose(vol_nii.fdata(numpy=True), vol_zarr.fdata(numpy=True))


def test_nifti_zarr_axis_order_matches_disk_layout(reference_nifti):
    # regression: the raw Zarr array is stored in the nifti-zarr spec's own
    # on-disk order (here, no OME axes metadata -> reversed NIfTI order,
    # z/y/x for this 3D case), not the same x/y/z order `.shape`/`.fdata()`
    # report. A missing or wrong `self.permutation` reorders the *shape*
    # correctly (both are length-3 tuples) but returns the wrong *data* --
    # invisible on a cubic fixture, caught here by using a non-cubic shape
    # (2, 3, 4) and comparing raw on-disk storage against the logical data.
    import numpy as np
    import zarr as zarr_module
    nii_path, zarr_path, data, affine = reference_nifti
    vol_zarr = nio.map(zarr_path)
    assert vol_zarr.permutation == (2, 1, 0)
    raw_on_disk = np.asarray(zarr_module.open(store=zarr_path, mode='r')['0'])
    assert raw_on_disk.shape == data.shape[::-1]
    assert np.array_equal(raw_on_disk.transpose(2, 1, 0), data)
    assert np.array_equal(vol_zarr.fdata(numpy=True), data)


def test_map_plain_ome_zarr_derives_header(plain_ome_zarr):
    path, data = plain_ome_zarr
    vol = nio.map(path)
    # z=1um -> 0.001mm, y=2um -> 0.002mm, x=3um -> 0.003mm
    expected_diag = [0.003, 0.002, 0.001]
    assert np.allclose(vol.affine.numpy().diagonal()[:3], expected_diag, atol=1e-6)
    assert np.allclose(vol.fdata(numpy=True), data)


def test_map_unrecognizable_store_raises(tmp_path):
    bad = tmp_path / 'bare.zarr'
    za = zarr.open_array(store=str(bad), mode='w', shape=(4, 4, 4), dtype='float32')
    za[:] = 1.0
    with pytest.raises(Exception):
        nio.map(str(bad))


def test_map_missing_path_raises(tmp_path):
    with pytest.raises(Exception):
        nio.map(str(tmp_path / 'does_not_exist.zarr'))


def test_existing_formats_unaffected(reference_nifti):
    nii_path, _, data, affine = reference_nifti
    vol = nio.map(nii_path)
    assert np.allclose(vol.fdata(numpy=True), data)
    assert torch.allclose(vol.affine, torch.as_tensor(affine, dtype=torch.double))


# --- User Story 2 ----------------------------------------------------------

def test_as_dask_returns_uncomputed_array(plain_ome_zarr):
    path, data = plain_ome_zarr
    vol = nio.map(path)
    lazy = vol.as_dask()
    import dask.array as da
    assert isinstance(lazy, da.Array)


def test_as_dask_partial_read(plain_ome_zarr):
    path, data = plain_ome_zarr
    vol = nio.map(path)
    lazy = vol.as_dask()
    sub = lazy[0:2, 0:2, 0:2].compute()
    assert sub.shape == (2, 2, 2)
    assert np.allclose(sub, data[0:2, 0:2, 0:2])


def test_data_and_fdata_still_work(plain_ome_zarr):
    path, data = plain_ome_zarr
    vol = nio.map(path)
    assert np.allclose(vol.fdata(numpy=True), data)
    assert np.allclose(vol.data(numpy=True), data)


# --- User Story 3 (level-fetch, IO side) -----------------------------------

def test_level_fetch_returns_native_data(multiscale_ome_zarr):
    path, arrays = multiscale_ome_zarr
    vol = nio.map(path)
    assert vol.nb_levels == 3
    for i in range(3):
        lvl = vol.level(i)
        assert lvl.shape == arrays[str(i)].shape
        assert np.allclose(lvl.fdata(numpy=True), arrays[str(i)])
        # the header (and therefore affine) is shared, unchanged, across
        # levels by design -- only shape/data are level-specific
        assert torch.allclose(lvl.affine, vol.affine)


def test_level_fetch_shape_reflects_bound_level_not_header(multiscale_ome_zarr):
    # regression: `.level(i).shape` used to return the shared header's own
    # (finest-level) shape regardless of which level was actually bound,
    # since `_shape` read `self._header.get_data_shape()` unconditionally
    path, arrays = multiscale_ome_zarr
    vol = nio.map(path)
    assert vol.shape == (16, 16, 16)
    assert vol.level(1).shape == (8, 8, 8)
    assert vol.level(2).shape == (4, 4, 4)
