import numpy as np
import pytest
import torch
import zarr

from nitorch.tools.registration.objects import ImagePyramid


@pytest.fixture
def multiscale_ome_zarr(tmp_path):
    """A multiscale (3-level) plain OME-Zarr store, no embedded header.

    `arrays` (returned, used for assertions) are in standard (x, y, z)
    order; the declared axes below are (z, y, x), so the underlying arrays
    are written transposed accordingly -- exercising the same axis
    permutation a real nifti-zarr/OME-Zarr store requires.
    """
    shapes = [(16, 16, 16), (8, 8, 8), (4, 4, 4)]
    rs = np.random.RandomState(0)
    arrays = {}
    path = tmp_path / 'multi.ome.zarr'
    root = zarr.open_group(store=str(path), mode='w')
    for i, shp in enumerate(shapes):
        data = (rs.rand(*shp) * 100).astype('float32')
        arr = root.create_array(str(i), shape=shp, dtype='float32', chunks=shp)
        arr[:] = data.transpose(2, 1, 0)
        arrays[str(i)] = data
    root.attrs['multiscales'] = [{
        'axes': [{'name': 'z', 'type': 'space', 'unit': 'millimeter'},
                 {'name': 'y', 'type': 'space', 'unit': 'millimeter'},
                 {'name': 'x', 'type': 'space', 'unit': 'millimeter'}],
        'datasets': [
            {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}]},
            {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2.0, 2.0, 2.0]}]},
            {'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4.0, 4.0, 4.0]}]},
        ],
    }]
    return str(path), arrays


def test_pyramid_uses_native_levels(multiscale_ome_zarr):
    path, arrays = multiscale_ome_zarr
    pyramid = ImagePyramid(path, levels=range(3))
    assert len(pyramid) == 3
    for i in range(3):
        got = pyramid[i].dat.numpy().squeeze()
        assert np.allclose(got, arrays[str(i)], atol=1e-4)


def test_pyramid_falls_back_beyond_native_levels(multiscale_ome_zarr):
    path, arrays = multiscale_ome_zarr
    # store only has 3 native levels (0, 1, 2); level 3 must be derived
    # by downsampling from the coarsest native level, not from the finest
    pyramid = ImagePyramid(path, levels=range(4))
    assert len(pyramid) == 4
    assert pyramid[3].dat.shape[-3:] == (2, 2, 2)
    for i in range(3):
        got = pyramid[i].dat.numpy().squeeze()
        assert np.allclose(got, arrays[str(i)], atol=1e-4)


def test_pyramid_plain_tensor_unaffected():
    # regression: a plain-tensor (non-multiscale) source must still be
    # downsampled exactly as before -- FR-006
    plain = torch.rand(1, 16, 16, 16)
    pyramid = ImagePyramid(plain, levels=range(3))
    assert len(pyramid) == 3
    assert pyramid[0].dat.shape[-3:] == (16, 16, 16)
    assert pyramid[1].dat.shape[-3:] == (8, 8, 8)
    assert pyramid[2].dat.shape[-3:] == (4, 4, 4)
