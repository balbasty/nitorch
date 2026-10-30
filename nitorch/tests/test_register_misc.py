"""Regression tests for assorted `nitorch register` crashes / defaults."""
import os
import torch
import pytest
import nibabel as nib
import numpy as np
from nitorch import io
from nitorch.cli.registration.register import parser
from nitorch.cli.registration.register.cli import _cli
from nitorch.tools.registration import objects, pairwise_makeobj
from nitorch.tools.registration import optim as opt
from nitorch.tools.registration.pairwise_postproc import warp_one_image


def _save(path, shape, shift=(0, 0, 0), seed=0):
    g = np.random.default_rng(seed)
    dat = np.zeros(shape, dtype=np.float32)
    c = [s // 2 for s in shape]
    dat[c[0]-2:c[0]+2, c[1]-2:c[1]+2, c[2]-2:c[2]+2] = 1
    dat += 0.01 * g.random(shape, dtype=np.float32)
    aff = np.eye(4)
    aff[:3, -1] = shift
    nib.save(nib.Nifti1Image(dat, aff), path)
    return path


def test_optim_sequential(tmp_path):
    fix = _save(str(tmp_path / 'fix.nii'), [12, 12, 12])
    mov = _save(str(tmp_path / 'mov.nii'), [12, 12, 12], seed=1)
    out = str(tmp_path / 'aff.lta')
    _cli(['-o', str(tmp_path), '@loss', 'mse', '@@fix', fix, '@@mov', mov,
          '@affine', 'translation', '-o', out, '@@optim', 'gn', '-n', '2',
          '@optim', 'sequential', '@pyramid', '--levels', '1'])
    assert os.path.exists(out)


def test_lr_default_unset():
    opts = parser.parser.parse(['@loss', 'mse', '@@fix', 'a.nii', '@@mov', 'b.nii',
                         '@affine', '@@optim', 'gd'])
    assert opts.affine.optim.lr is None
    o = pairwise_makeobj.make_affine_optim('gd', 1, lr=opts.affine.optim.lr)
    assert o.lr == 1e-3


def test_lbfgs_positional_lr():
    o = opt.LBFGS(0.5)
    assert o.lr == 0.5 and o.max_iter == 20


def test_get_center_integer_image():
    aff = torch.eye(4, dtype=torch.double)
    aff[:3, :3] *= 0.5
    aff[:3, -1] = 0.25
    img = objects.Image(torch.zeros([1, 4, 4, 4], dtype=torch.int32), aff)
    c = img.get_center()
    assert torch.allclose(c.double(), torch.full([3], 1.0, dtype=torch.double))


def test_fov_init_weighted_by_size(tmp_path):
    # loss 1: small images, shift 2; loss 2: big images, shift 0
    f1 = _save(str(tmp_path / 'f1.nii'), [4, 4, 4])
    m1 = _save(str(tmp_path / 'm1.nii'), [4, 4, 4], shift=(2, 0, 0))
    f2 = _save(str(tmp_path / 'f2.nii'), [8, 8, 8])
    m2 = _save(str(tmp_path / 'm2.nii'), [8, 8, 8])
    out = str(tmp_path / 'aff.lta')
    _cli(['-o', str(tmp_path), '@loss', 'mse', '@@fix', f1, '@@mov', m1,
          '@loss', 'mse', '@@fix', f2, '@@mov', m2,
          '@affine', 'translation', '-i', 'fov', '-o', out,
          '@@optim', 'none', '@pyramid', '--levels', '1'])
    aff = io.transforms.loadf(out)
    expected = 2 * 64 / (64 + 512)
    assert abs(float(aff.reshape(-1, 4, 4)[0, 0, -1]) - expected) < 1e-4, aff


def test_warp_reslice_identity():
    dat = torch.rand([1, 5, 5, 5])
    img = objects.Image(dat, torch.eye(4))
    out = warp_one_image(img, torch.eye(4), [5, 5, 5], reslice=True)
    assert torch.allclose(out, dat[0], atol=1e-5)
