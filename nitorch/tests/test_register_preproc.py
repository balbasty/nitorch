"""Preprocessing options of `nitorch register` (rescale, pad, fwhm, pyramid)."""
import pytest
import torch
from nitorch.tools.registration.pairwise_preproc import (
    preproc_image, rescale_image)
from nitorch.cli.registration.register.parser import parser

BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii']


def make_image(shape=(10, 12, 14), vx=2.):
    g = torch.Generator().manual_seed(0)
    dat = torch.rand([1, *shape], dtype=torch.double, generator=g) + 1
    affine = torch.eye(4, dtype=torch.double)
    affine[:3, :3] *= vx
    return dat, affine


def test_rescale_nonzero_min():
    dat, _ = make_image()
    out = rescale_image(dat.clone(), (5, 95))
    assert torch.isfinite(out).all()
    q = torch.quantile(out.flatten(), torch.tensor([.05, .95], dtype=out.dtype))
    assert torch.allclose(q, torch.tensor([0., 1.], dtype=out.dtype), atol=0.02)


def test_pad_mm():
    dat, affine = make_image(vx=2.)
    out, _, aff = preproc_image(dat, affine=None, world=affine, rescale=0,
                                pad=[4, 'mm'], dim=3)
    assert list(out.shape[1:]) == [14, 16, 18]   # 4mm / 2mm = 2 vox per side
    assert torch.allclose(aff[:3, 3], torch.full([3], -4., dtype=aff.dtype))


def test_fwhm_mm():
    dat, affine = make_image(vx=2.)
    out_mm, *_ = preproc_image(dat.clone(), world=affine, rescale=0,
                               fwhm=[4, 'mm'], dim=3)
    out_vx, *_ = preproc_image(dat.clone(), world=affine, rescale=0,
                               fwhm=[2], dim=3)
    assert torch.allclose(out_mm, out_vx)


@pytest.mark.parametrize('args, expected', [
    ([], False),
    (['@pyramid'], False),
    (['@pyramid', '-c'], True),
    (['@pyramid', '--concurrent'], True),
    (['@pyramid', '-c', 'false'], False),
])
def test_pyramid_concurrent_option(args, expected):
    options = parser.parse(BASE + args)
    assert options.pyramid.concurrent == expected


def test_cli_concurrent_pyramid_runs(tmp_path):
    nib = pytest.importorskip('nibabel')
    import numpy as np
    from nitorch.cli.registration.register.cli import cli
    x = np.zeros([16, 16, 16], dtype='float32')
    x[4:12, 4:12, 4:12] = 1
    fix, mov = str(tmp_path / 'fix.nii'), str(tmp_path / 'mov.nii')
    nib.save(nib.Nifti1Image(x, np.eye(4)), fix)
    nib.save(nib.Nifti1Image(np.roll(x, 1, 0), np.eye(4)), mov)
    out = str(tmp_path / 'aff.lta')
    cli(['@loss', 'mse', '@@fix', fix, '@@mov', mov,
         '@loss', 'mse', '@@fix', fix, '@@mov', mov,
         '@affine', 'rigid', '-o', out, '@optim', '-n', '2',
         '@pyramid', '-c'])
    assert (tmp_path / 'aff.lta').exists()
