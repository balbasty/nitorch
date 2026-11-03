"""`nitorch resize`: grid mode (-g) and option checks."""
import numpy as np
import pytest

from nitorch.cli.registration.resize.main import resize

nib = pytest.importorskip('nibabel')


@pytest.fixture
def disp(tmp_path):
    fname = str(tmp_path / 'disp.nii.gz')
    # constant displacement field (in voxels)
    dat = np.zeros([6, 6, 6, 3], dtype='float32')
    dat[..., 0] = 1.
    dat[..., 1] = -2.
    dat[..., 2] = 0.5
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


@pytest.fixture
def volume(tmp_path):
    fname = str(tmp_path / 'img.nii.gz')
    dat = np.arange(6 * 6 * 6, dtype='float32').reshape([6, 6, 6])
    nib.save(nib.Nifti1Image(dat, np.diag([2., 2., 2., 1.])), fname)
    return fname


@pytest.mark.parametrize('anchor', ['centers', 'first'])
def test_resize_grid_disp(tmp_path, disp, anchor):
    out = str(tmp_path / 'out.nii.gz')
    resize([disp, '-g', 'd', '-f', '2', '-a', anchor, '-o', out, '--cpu'])
    img = nib.load(out)
    # 'centers' keeps floor(N*f) = 12; 'first' gives floor((N-1)*f)+1 = 11
    # so that the last output voxel stays inside the input FOV
    n = 12 if anchor == 'centers' else 11
    dat = np.asarray(img.dataobj).reshape([n, n, n, 3])
    # voxel size is (about) halved
    vx = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
    expected_vx = 10. / 11. if anchor == 'centers' else 1.
    assert np.allclose(vx, expected_vx)
    # first voxel centre is preserved with both anchors
    assert np.allclose(img.affine[:3, 3], 0.)
    # displacements (in voxels) are expressed in the new voxel units,
    # i.e. multiplied by input_vx / output_vx (no zeroed border).
    scl = 2. / expected_vx
    assert np.allclose(dat[..., 0], 1. * scl)
    assert np.allclose(dat[..., 1], -2. * scl)
    assert np.allclose(dat[..., 2], 0.5 * scl)


def test_resize_grid_trf(tmp_path, tmp_path_factory):
    # identity transformation field: resized field is the identity
    fname = str(tmp_path / 'trf.nii.gz')
    dat = np.stack(np.meshgrid(*[np.arange(6.)] * 3, indexing='ij'), -1)
    nib.save(nib.Nifti1Image(dat.astype('float32'), np.eye(4)), fname)
    out = str(tmp_path / 'out.nii.gz')
    resize([fname, '-g', 't', '-f', '2', '-a', 'first', '-i', '1',
            '-b', 'replicate', '-o', out, '--cpu'])
    # 'first': floor((6-1)*2)+1 = 11 voxels, all inside the input FOV
    res = np.asarray(nib.load(out).dataobj).reshape([11, 11, 11, 3])
    ident = np.stack(np.meshgrid(*[np.arange(11.)] * 3, indexing='ij'), -1)
    assert np.allclose(res, ident, atol=1e-4)


def test_resize_factor_and_voxel_size(tmp_path, volume, capsys):
    out = str(tmp_path / 'out.nii.gz')
    resize([volume, '-f', '2', '-v', '1', '-o', out, '--cpu'])
    err = capsys.readouterr().err
    assert '[ERROR]' in err
    assert 'Traceback' not in err
    assert not (tmp_path / 'out.nii.gz').exists()


def test_resize_no_size_option(tmp_path, volume, capsys):
    out = str(tmp_path / 'out.nii.gz')
    resize([volume, '-o', out, '--cpu'])
    captured = capsys.readouterr()
    # parse-time error: usage is shown, then a clean [ERROR] line
    assert 'usage' in captured.out.lower() or '-f, --factor' in captured.out
    assert '[ERROR]' in captured.err
    assert 'Traceback' not in captured.err
    assert not (tmp_path / 'out.nii.gz').exists()


def test_resize_failure_returns_nonzero_status(tmp_path, capsys):
    """A failing resize returns 1, so the dispatcher exits non-zero."""
    from nitorch.cli.registration.resize.main import resize
    # missing -f/-v/-s: ParseError
    assert resize([str(tmp_path / 'x.nii')]) == 1
    # unreadable input: generic error
    assert resize([str(tmp_path / 'missing.nii'), '-f', '2']) == 1
    err = capsys.readouterr().err
    assert '[ERROR]' in err


def test_resize_grid_disp_edges_no_zero_border(tmp_path, disp):
    # 'edges' samples up to half a voxel outside the FOV when upsampling:
    # resize_grid extrapolates with dct2 instead of zeroing them.
    out = str(tmp_path / 'out.nii.gz')
    resize([disp, '-g', 'd', '-f', '2', '-a', 'edges', '-o', out, '--cpu'])
    dat = np.asarray(nib.load(out).dataobj).reshape([12, 12, 12, 3])
    assert np.allclose(dat[..., 0], 2.)
    assert np.allclose(dat[..., 1], -4.)
    assert np.allclose(dat[..., 2], 1.)
