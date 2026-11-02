"""Orientation matrices of the pieces written by `chunk` and `unstack` (#171)."""
import os

import numpy as np
import pytest

from nitorch.cli.misc.chunk.cli import _cli as chunk
from nitorch.cli.misc.unstack.cli import _cli as unstack

nib = pytest.importorskip('nibabel')

AFFINE = np.array([[0., -2., 0., 10.],
                   [1.5, 0., 0., -20.],
                   [0., 0., 3., 5.],
                   [0., 0., 0., 1.]])


def make_volume(tmp_path, shape):
    fname = str(tmp_path / 'img.nii')
    dat = np.arange(np.prod(shape), dtype='float32').reshape(shape)
    nib.save(nib.Nifti1Image(dat, AFFINE), fname)
    return fname, dat


def outputs(tmp_path, n):
    return [str(tmp_path / f'img.{i}.nii') for i in range(1, n + 1)]


def check(tmp_path, dat, dim, size):
    ndim = dat.ndim
    pdim = dim % ndim
    starts = list(range(0, dat.shape[pdim], size))
    for start, fname in zip(starts, outputs(tmp_path, len(starts))):
        img = nib.load(fname)
        piece = np.asarray(img.dataobj)
        ref = np.take(dat, range(start, min(start + size, dat.shape[pdim])),
                      axis=pdim)
        assert piece.shape == ref.shape
        np.testing.assert_array_equal(piece, ref)
        vox = np.zeros(4)
        vox[3] = 1
        if pdim < 3:
            vox[pdim] = start
        np.testing.assert_allclose(img.affine[:3, 3], (AFFINE @ vox)[:3])
        np.testing.assert_allclose(img.affine[:3, :3], AFFINE[:3, :3])
    assert not os.path.exists(outputs(tmp_path, len(starts) + 1)[-1])


CASES = [((4, 5, 6), d) for d in (0, 1, 2, -1)] + \
        [((4, 5, 6, 3), d) for d in (None, 3, -2, -1, 0)]


@pytest.mark.parametrize('shape,dim', CASES)
def test_unstack_affine(tmp_path, shape, dim):
    fname, dat = make_volume(tmp_path, shape)
    unstack([fname] + ([] if dim is None else ['-d', str(dim)]))
    check(tmp_path, dat, -1 if dim is None else dim, 1)


@pytest.mark.parametrize('shape,dim', CASES)
def test_chunk_affine(tmp_path, shape, dim):
    fname, dat = make_volume(tmp_path, shape)
    chunk([fname, '-n', '2'] + ([] if dim is None else ['-d', str(dim)]))
    check(tmp_path, dat, -1 if dim is None else dim, 2)
