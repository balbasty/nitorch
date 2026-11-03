"""`crop` command: `-k` alone, `-b` bounding box, `-t` input/output."""
import os

import numpy as np
import pytest

from nitorch.cli.misc.crop.cli import _cli as crop

nib = pytest.importorskip('nibabel')

AFFINE = np.array([[2., 0, 0, -10],
                   [0, 2., 0, -20],
                   [0, 0, 2., -30],
                   [0, 0, 0, 1]])


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(10 * 12 * 8, dtype='float32').reshape([10, 12, 8])
    nib.save(nib.Nifti1Image(dat, AFFINE), fname)
    return fname


@pytest.fixture
def blob(tmp_path):
    # nonzero block at [3:7, 5:9, 1:4] (odd first index along x and y)
    fname = str(tmp_path / 'blob.nii.gz')
    dat = np.zeros([10, 12, 8], dtype='float32')
    dat[3:7, 5:9, 1:4] = np.arange(4 * 4 * 3).reshape([4, 4, 3]) + 1
    nib.save(nib.Nifti1Image(dat, AFFINE), fname)
    return fname, dat


def load(fname):
    f = nib.load(fname)
    return np.asarray(f.dataobj), f.affine


def shifted(first):
    aff = AFFINE.copy()
    aff[:3, 3] += AFFINE[:3, :3] @ np.asarray(first, dtype=float)
    return aff


def test_crop_like_alone(tmp_path, volume):
    ref = str(tmp_path / 'ref.nii.gz')
    crop([volume, '-s', '4', '6', '2', '-c', '3', '5', '4', '-o', ref])
    out = str(tmp_path / 'out.nii.gz')
    crop([volume, '-k', ref, '-o', out])
    assert os.path.exists(out)
    dat, aff = load(out)
    rdat, raff = load(ref)
    assert dat.shape == rdat.shape == (4, 6, 2)
    np.testing.assert_array_equal(dat, rdat)
    np.testing.assert_allclose(aff, raff)


@pytest.mark.parametrize('thresh', [None, '0.5'])
def test_crop_bbox(tmp_path, blob, thresh):
    fname, ref = blob
    out = str(tmp_path / 'out.nii.gz')
    args = [fname, '-b'] + ([thresh] if thresh else []) + ['-o', out]
    crop(args)
    assert os.path.exists(out)
    dat, aff = load(out)
    np.testing.assert_array_equal(dat, ref[3:7, 5:9, 1:4])
    np.testing.assert_allclose(aff, shifted([3, 5, 1]))


def test_crop_bbox_threshold(tmp_path, blob):
    fname, ref = blob
    out = str(tmp_path / 'out.nii.gz')
    # values 1..48; > 40 keeps the last 8 values, at x=6, y=7..8, z=1..3
    crop([fname, '-b', '40', '-o', out])
    dat, aff = load(out)
    idx = np.argwhere(ref > 40)
    lo, hi = idx.min(0), idx.max(0) + 1
    np.testing.assert_array_equal(
        dat, ref[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    np.testing.assert_allclose(aff, shifted(lo))


def test_crop_bbox_writes_transform(tmp_path, blob):
    fname, _ = blob
    out = str(tmp_path / 'out.nii.gz')
    lta = str(tmp_path / 'bbox.lta')
    crop([fname, '-b', '-o', out, '-t', lta])
    assert os.path.exists(lta)


def test_crop_size_transform_is_output(tmp_path, volume):
    out = str(tmp_path / 'out.nii.gz')
    lta = str(tmp_path / 'crop.lta')
    crop([volume, '-s', '4', '6', '2', '-o', out, '-t', lta])
    assert os.path.exists(out)
    assert os.path.exists(lta)


def test_crop_size_default_transform_name(tmp_path, volume):
    out = str(tmp_path / 'out.nii.gz')
    crop([volume, '-s', '4', '-o', out, '-t'])
    assert os.path.exists(tmp_path / 'img.crop.lta')


def test_crop_transform_roundtrip(tmp_path, volume):
    out1 = str(tmp_path / 'out1.nii.gz')
    lta = str(tmp_path / 'crop.lta')
    crop([volume, '-s', '4', '6', '2', '-c', '3', '5', '4',
          '-o', out1, '-t', lta])
    out2 = str(tmp_path / 'out2.nii.gz')
    crop([volume, '-t', lta, '-o', out2])
    dat1, aff1 = load(out1)
    dat2, aff2 = load(out2)
    assert dat1.shape == (4, 6, 2)
    np.testing.assert_array_equal(dat1, dat2)
    np.testing.assert_allclose(aff1, aff2)


@pytest.mark.parametrize('opts', [
    ['-s', '4', '-k', 'REF'],
    ['-s', '4', '-b'],
    ['-k', 'REF', '-b', '0'],
    ['-b', '-s', '4', '-k', 'REF'],
    ['-s', '4', '-b', '-t', 'T'],
])
def test_crop_exclusive_options(tmp_path, volume, capsys, opts):
    from nitorch.cli.misc.crop.cli import cli
    from nitorch.cli.misc.crop.parser import parse, ParseError
    ref = str(tmp_path / 'ref.nii.gz')
    crop([volume, '-s', '4', '-o', ref])
    capsys.readouterr()
    opts = [ref if o == 'REF' else str(tmp_path / 't.lta') if o == 'T'
            else o for o in opts]
    out = str(tmp_path / 'out.nii.gz')
    args = [volume] + opts + ['-o', out]
    with pytest.raises(ParseError):
        parse(list(args))
    cli(list(args))  # clean error, no traceback
    captured = capsys.readouterr()
    assert '[ERROR]' in captured.err
    assert 'Only one of' in captured.err
    assert 'usage:' in captured.out
    assert not os.path.exists(out)
    assert not os.path.exists(tmp_path / 't.lta')


def test_crop_center_with_size_still_allowed(tmp_path, volume):
    out = str(tmp_path / 'out.nii.gz')
    crop([volume, '-s', '4', '6', '2', '-c', '3', '5', '4', '-o', out])
    dat, aff = load(out)
    assert dat.shape == (4, 6, 2)


@pytest.mark.parametrize('axcodes', [('L', 'P', 'S'), ('A', 'S', 'R'),
                                     ('L', 'I', 'A')])
def test_crop_like_other_layout(tmp_path, volume, axcodes):
    ref = str(tmp_path / 'ref.nii.gz')
    crop([volume, '-s', '4', '6', '2', '-c', '3', '5', '4', '-o', ref])
    rdat, raff = load(ref)
    # same world FOV, stored in another voxel layout
    img = nib.Nifti1Image(rdat, raff)
    xform = nib.orientations.ornt_transform(
        nib.io_orientation(raff), nib.orientations.axcodes2ornt(axcodes))
    img = img.as_reoriented(xform)
    ref2 = str(tmp_path / 'ref2.nii.gz')
    nib.save(img, ref2)
    out = str(tmp_path / 'out.nii.gz')
    crop([volume, '-k', ref2, '-o', out])
    f = nib.load(out)
    assert f.shape == img.shape
    np.testing.assert_allclose(f.affine, img.affine)
    np.testing.assert_array_equal(np.asarray(f.dataobj),
                                  np.asarray(img.dataobj))


# --- parser: garbled values (#171) -----------------------------------

@pytest.mark.parametrize('tag,name', [('-s', '-s/--size'),
                                      ('--size', '-s/--size'),
                                      ('-c', '-c/--center'),
                                      ('--center', '-c/--center')])
def test_crop_garbled_value_is_parse_error(tag, name):
    from nitorch.cli.misc.crop.parser import parse
    from nitorch.core.cli import ParseError
    with pytest.raises(ParseError) as e:
        parse(['in.nii', tag, '4', 'abc', 'ras'])
    assert str(e.value) == f"{name} expects <float> or vox|ras, got 'abc'"


def test_crop_size_center_grammar_unchanged():
    from nitorch.cli.misc.crop.parser import parse
    opt = parse(['in.nii', '-s', '4', '-6.5', 'RAS', '-c', '1', '2', '3',
                 'vox', '-o', 'out.nii'])
    assert opt.size == [4., -6.5]
    assert opt.size_space == 'RAS'
    assert opt.center == [1., 2., 3.]
    assert opt.center_space == 'vox'
