"""`nitorch vexp`: command-line behaviour."""
import numpy as np
import pytest

from nitorch.cli.registration.vexp.cli import _cli as vexp
from nitorch.cli.registration.vexp.parser import parse

nib = pytest.importorskip('nibabel')

# anisotropic voxels and a translation, to tell the linear part apart
AFFINE = np.asarray([[2., 0., 0., -10.],
                     [0., 3., 0., 20.],
                     [0., 0., 4., 5.],
                     [0., 0., 0., 1.]])


@pytest.fixture
def velocity(tmp_path):
    rng = np.random.RandomState(0)
    vel = rng.randn(10, 10, 10, 3).astype('float32') * 0.3
    fname = str(tmp_path / 'vel.nii.gz')
    nib.save(nib.Nifti1Image(vel, AFFINE), fname)
    fneg = str(tmp_path / 'negvel.nii.gz')
    nib.save(nib.Nifti1Image(-vel, AFFINE), fneg)
    return fname, fneg


def load(fname):
    return np.asarray(nib.load(str(fname)).dataobj).squeeze()


@pytest.mark.parametrize('args,expected', [
    ([], False),
    (['-inv'], True),
    (['-inv', 'true'], True),
    (['-inv', 'false'], False),
    (['--inverse', '0'], False),
])
def test_parse_inverse(args, expected):
    assert parse(['vel.nii.gz', *args]).inv is expected


def test_inverse_is_exp_of_negated_velocity(tmp_path, velocity):
    fvel, fneg = velocity
    vexp([fvel, '-inv', '-o', str(tmp_path / 'inv.nii.gz')])
    vexp([fneg, '-o', str(tmp_path / 'neg.nii.gz')])
    vexp([fvel, '-o', str(tmp_path / 'fwd.nii.gz')])
    inv = load(tmp_path / 'inv.nii.gz')
    neg = load(tmp_path / 'neg.nii.gz')
    fwd = load(tmp_path / 'fwd.nii.gz')
    np.testing.assert_allclose(inv, neg, atol=1e-5)
    assert not np.allclose(inv, fwd, atol=1e-3)


def test_displacement_in_mm(tmp_path, velocity):
    fvel, _ = velocity
    vexp([fvel, '-o', str(tmp_path / 'vox.nii.gz')])
    vexp([fvel, '-u', 'mm', '-o', str(tmp_path / 'mm.nii.gz')])
    vox = load(tmp_path / 'vox.nii.gz')
    mm = load(tmp_path / 'mm.nii.gz')
    # scaled by the voxel size, without the translation
    np.testing.assert_allclose(mm, vox * np.asarray([2., 3., 4.]), atol=1e-5)
