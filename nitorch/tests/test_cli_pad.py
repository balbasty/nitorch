"""`nitorch pad` command-line interface (issue #171)."""
import os

import numpy as np
import pytest

from nitorch.cli.misc.pad.cli import _cli as pad
from nitorch.cli.misc.pad.parser import parse, help, ParseError

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(4 * 6 * 8, dtype='float32').reshape([4, 6, 8])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


def load(tmp_path, name='img.pad.nii.gz'):
    return np.asarray(nib.load(str(tmp_path / name)).dataobj)


def test_pad_size_ras(tmp_path, volume):
    # 4 mm with 2 mm voxels -> 2 voxels on each side
    pad([volume, '-s', '4', 'ras'])
    assert load(tmp_path).shape == (8, 10, 12)


def test_pad_default_names(tmp_path, volume):
    assert '.crop' not in help
    pad([volume, '-s', '1', '-t'])
    assert os.path.exists(tmp_path / 'img.pad.nii.gz')
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith('.lta')) \
        == ['img.pad.lta']


@pytest.mark.parametrize('name', ['dct', 'dct2', 'reflect', 'dft', 'wrap',
                                  'replicate', 'nearest', 'zero'])
def test_pad_bound_names(tmp_path, volume, name):
    pad([volume, '-s', '1', '-b', name])
    assert load(tmp_path).shape == (6, 8, 10)


def test_pad_bound_dct_is_dct2(tmp_path, volume):
    pad([volume, '-s', '1', '-b', 'dct'])
    out = load(tmp_path)
    dat = np.asarray(nib.load(volume).dataobj)
    # dct2: edge voxel is repeated
    assert np.allclose(out[0, 1:-1, 1:-1], dat[0])


def test_pad_bound_invalid():
    with pytest.raises(ParseError):
        parse(['img.nii', '-s', '1', '-b', 'foo'])


@pytest.mark.parametrize('pair', [['(0,2)'], ['[0,2]'], ['(0,', '2)'],
                                  ['(0', ',', '2)']])
def test_pad_pair_parsing(pair):
    opt = parse(['img.nii', '-s', '1', *pair, '1', '-o', 'out.nii'])
    assert opt.size == [1, (0, 2), 1]
    assert opt.output == ['out.nii']


def test_pad_pair_one_word(tmp_path, volume):
    pad([volume, '-s', '0', '0', '(0,2)', '-o', str(tmp_path / 'o.nii.gz')])
    assert load(tmp_path, 'o.nii.gz').shape == (4, 6, 10)


def test_pad_pair_unclosed():
    with pytest.raises(ParseError):
        parse(['img.nii', '-s', '(0,', '2', '-o', 'out.nii'])


def test_pad_size_ras_anisotropic_permuted(tmp_path):
    # voxel axes (i, j, k) -> (y, -x, z) with voxel sizes (1, 2, 4) mm
    fname = str(tmp_path / 'img.nii.gz')
    aff = np.array([[0., -2., 0., 0.], [1., 0., 0., 0.],
                    [0., 0., 4., 0.], [0., 0., 0., 1.]])
    nib.save(nib.Nifti1Image(np.zeros([4, 6, 8], dtype='float32'), aff),
             fname)
    # x: (0, 4) mm, y: 2 mm, z: 8 mm (given in RAS order)
    pad([fname, '-s', '(0,4)', '2', '8', 'ras'])
    out = nib.load(str(tmp_path / 'img.pad.nii.gz'))
    # i <- y: 2 mm / 1 mm = 2 each side; j <- -x: (4, 0) mm / 2 = (2, 0);
    # k <- z: 8 mm / 4 mm = 2 each side
    assert out.shape == (4 + 4, 6 + 2, 8 + 4)
    # world position of voxel 0 of the input is preserved
    assert np.allclose(np.linalg.inv(out.affine) @ aff @ [0, 0, 0, 1],
                       [2, 2, 2, 1])
