"""Transform options shared by `nitorch reslice` and `nitorch compose`."""
import numpy as np
import pytest
import torch

from nitorch import io
from nitorch.cli.registration.reslice.main import reslice
from nitorch.cli.registration.compose.cli import _cli as compose
from nitorch.cli.registration.compose.parser import parse as parse_compose
from nitorch.cli.registration.reslice.parser import parse as parse_reslice

nib = pytest.importorskip('nibabel')

AFFINE = np.diag([2., 3., 4., 1.])


@pytest.fixture
def files(tmp_path):
    rng = np.random.RandomState(0)
    img = str(tmp_path / 'img.nii.gz')
    nib.save(nib.Nifti1Image(rng.rand(8, 8, 8).astype('float32'), AFFINE),
             img)
    # constant displacement of (1, 1, 1) voxels, i.e., (2, 3, 4) mm
    vox = np.ones([8, 8, 8, 3], dtype='float32')
    mm = vox * np.asarray([2., 3., 4.], dtype='float32')
    fvox = str(tmp_path / 'disp_vox.nii.gz')
    fmm = str(tmp_path / 'disp_mm.nii.gz')
    nib.save(nib.Nifti1Image(vox, AFFINE), fvox)
    nib.save(nib.Nifti1Image(mm, AFFINE), fmm)
    lta = str(tmp_path / 'aff.lta')
    io.transforms.savef(torch.eye(4), lta)
    return dict(img=img, vox=fvox, mm=fmm, lta=lta, dir=tmp_path)


def load(fname):
    return np.asarray(nib.load(str(fname)).dataobj).squeeze()


@pytest.mark.parametrize('parse', [parse_reslice, parse_compose])
def test_displacement_order_option(parse):
    options = parse(['-d', 'disp.nii.gz', '-n', '3', '-u', 'mm'])
    trf, = options.transformations
    assert trf.order == 3
    assert trf.unit == 'mm'


@pytest.mark.parametrize('tag', ['--linear-inverse', '-il'])
def test_compose_inverse_tags(tag):
    trf, = parse_compose([tag, 'aff.lta']).transformations
    assert trf.inv
    assert type(trf).__name__ == 'Linear'


def test_reslice_displacement_in_mm(files):
    out_vox = files['dir'] / 'out_vox.nii.gz'
    out_mm = files['dir'] / 'out_mm.nii.gz'
    reslice([files['img'], '-d', files['vox'], '-t', files['img'],
             '-o', str(out_vox)])
    reslice([files['img'], '-d', files['mm'], '-u', 'mm', '-t', files['img'],
             '-o', str(out_mm)])
    np.testing.assert_allclose(load(out_mm), load(out_vox), atol=1e-5)


def test_compose_displacement_in_mm(files):
    out_vox = files['dir'] / 'cmp_vox.nii.gz'
    out_mm = files['dir'] / 'cmp_mm.nii.gz'
    compose(['--linear-inverse', files['lta'], '-d', files['vox'],
             '-t', files['img'], '-o', str(out_vox)])
    compose(['--linear-inverse', files['lta'], '-d', files['mm'],
             '-u', 'mm', '-t', files['img'], '-o', str(out_mm)])
    np.testing.assert_allclose(load(out_mm), load(out_vox), atol=1e-5)


def test_compose_displacement_order_is_used(files):
    rng = np.random.RandomState(1)
    disp = str(files['dir'] / 'disp_rand.nii.gz')
    nib.save(nib.Nifti1Image(rng.randn(8, 8, 8, 3).astype('float32'),
                             AFFINE), disp)
    # finer target grid so that the field is sampled between its nodes
    target = str(files['dir'] / 'target.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros([12, 12, 12], dtype='float32'),
                             np.diag([1.3, 1.9, 2.7, 1.])), target)
    out = {}
    for order in (1, 3):
        out[order] = files['dir'] / f'cmp_n{order}.nii.gz'
        compose(['-d', disp, '-n', str(order), '-t', target,
                 '-o', str(out[order])])
    assert not np.allclose(load(out[1]), load(out[3]), atol=1e-3)
