"""`pool` command: padding parsing, `-p same`, and the `ssq` method."""
import numpy as np
import pytest

from nitorch.cli.misc.pool.cli import _cli as pool
from nitorch.cli.misc.pool.parser import parse, ParseError

nib = pytest.importorskip('nibabel')


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(4 * 6 * 8, dtype='float32').reshape([4, 6, 8])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname, dat


def run(tmp_path, volume, *args):
    out = str(tmp_path / 'out.nii.gz')
    pool([volume[0], *args, '-o', out])
    img = nib.load(out)
    return np.asarray(img.dataobj), img.affine


# --- padding parser ---------------------------------------------------------

def test_parse_type_after_size():
    s = parse(['img.nii', '-p', '1', 'zero', '2', 'dft'])
    assert s.padding == [1, 2]
    assert s.bound == ['zero', 'dft']


def test_parse_default_type_then_sizes():
    s = parse(['img.nii', '-p', 'replicate', '1', '2', 'zero', '3'])
    assert s.padding == [1, 2, 3]
    assert s.bound == ['replicate', 'zero', 'replicate']


def test_parse_type_without_size_errors():
    with pytest.raises(ParseError):
        parse(['img.nii', '-p', '1', 'zero', 'dft'])
    with pytest.raises(ParseError):
        parse(['img.nii', '-p', 'zero'])
    with pytest.raises(ParseError):
        parse(['img.nii', '-p', 'zero', 'dft', '1'])


def test_padding_size_and_type_applied(tmp_path, volume):
    # window 3 / stride 1 with 1 voxel of zero padding keeps the shape
    # and the corner voxel is the sum over the 2x2x2 real voxels
    out, _ = run(tmp_path, volume, '-w', '3', '-s', '1', '-p', '1', 'zero',
                 '-m', 'sum')
    dat = volume[1]
    assert out.shape == dat.shape
    assert np.isclose(out[0, 0, 0], dat[:2, :2, :2].sum())


def test_padding_type_per_axis_applied(tmp_path, volume):
    # zero padding on axis 0, replicate on the others (last value repeated)
    out, _ = run(tmp_path, volume, '-w', '3', '-s', '1',
                 '-p', '1', 'zero', '1', 'replicate', '-m', 'sum')
    dat = volume[1]
    ref = np.pad(dat, [(1, 1), (0, 0), (0, 0)], mode='constant')
    ref = np.pad(ref, [(0, 0), (1, 1), (1, 1)], mode='edge')
    assert out.shape == dat.shape
    assert np.isclose(out[0, 0, 0], ref[:3, :3, :3].sum())
    assert np.isclose(out[-1, -1, -1], ref[-3:, -3:, -3:].sum())


# --- -p same ----------------------------------------------------------------

@pytest.mark.parametrize('window', [2, 3, 4])
def test_same_padding_shape(tmp_path, volume, window):
    out, aff = run(tmp_path, volume, '-w', str(window), '-s', '1',
                   '-p', 'same')
    assert out.shape == volume[1].shape
    # output voxel i is centred on the window centre (K-1)/2 - P_low
    plow = (window - 1) // 2
    shift = (window - 1) / 2 - plow
    assert np.allclose(aff[:3, 3], 2 * shift)
    assert np.allclose(np.diag(aff)[:3], 2)


# --- ssq --------------------------------------------------------------------

def test_ssq_is_sum_of_squares(tmp_path, volume):
    out, _ = run(tmp_path, volume, '-w', '2', '-m', 'ssq')
    dat = volume[1]
    ref = (dat ** 2).reshape([2, 2, 3, 2, 4, 2]).sum(axis=(1, 3, 5))
    assert out.shape == (2, 3, 4)
    assert np.allclose(out, ref, rtol=1e-5)
