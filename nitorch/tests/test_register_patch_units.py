"""Units for the `patch` option of local losses in `nitorch register`."""
import os
import pytest
import torch
from nitorch import spatial
from nitorch.cli.registration.register.parser import parser
from nitorch.tools.registration import objects
from nitorch.tools.registration.pairwise_makeobj import (
    make_loss, split_patch_unit, patch_to_voxels)
from nitorch.tools.registration.pairwise_impl import _patch_kwargs

BASE = ['@@fix', 'fix.nii', '@@mov', 'mov.nii', '@affine', 'rigid']


@pytest.mark.parametrize('args, expected', [
    ([], ([10.], 'vox')),
    (['-p', '5'], ([5.], 'vox')),
    (['-p', '5', 'vox'], ([5.], 'vox')),
    (['-p', '5', 'mm'], ([5.], 'mm')),
    (['-p', '5', '6', '7', 'mm'], ([5., 6., 7.], 'mm')),
    (['-p', '10', '%'], ([10.], 'pct')),
    (['-p', '10', 'pct'], ([10.], 'pct')),
])
@pytest.mark.parametrize('name', ['lcc', 'lgmm'])
def test_parse_patch_unit(name, args, expected):
    opt = parser.parse(['@loss', name] + args + BASE)
    assert split_patch_unit(opt.loss[0].patch) == expected
    loss = make_loss(name, patch=opt.loss[0].patch)
    assert loss.patch == expected[0]
    assert loss.patch_unit == expected[1]


def test_bad_unit():
    with pytest.raises(ValueError):
        split_patch_unit([5, 'furlong'])


def test_mm_to_vox_anisotropic():
    aff = spatial.affine_default([10, 10, 10], [1., 2., 4.])
    assert patch_to_voxels([8.], 'mm', aff) == [8., 4., 2.]
    assert patch_to_voxels([4., 4., 8.], 'mm', aff) == [4., 2., 2.]
    assert patch_to_voxels([5.], 'vox', aff) == [5.]
    assert patch_to_voxels([10.], 'pct', shape=[20, 40, 10]) == [2., 4., 1.]


def test_per_level_conversion():
    aff = spatial.affine_default([32, 32, 16], [1., 1., 2.])
    pyr = objects.ImagePyramid(torch.rand([1, 32, 32, 16]), levels=3,
                               affine=aff)
    loss = make_loss('lcc', patch=[8, 'mm'])
    got = [_patch_kwargs(loss, lvl)['patch'] for lvl in pyr]
    vxs = [spatial.voxel_size(lvl.affine).tolist() for lvl in pyr]
    assert len(got) == 3
    for p, vx in zip(got, vxs):
        assert p == pytest.approx([8 / v for v in vx])
    # coarser levels -> fewer voxels per patch
    assert got[0][0] > got[1][0] > got[2][0]
    # voxel units are untouched
    assert _patch_kwargs(make_loss('lcc', patch=[5]), pyr[1]) == {}


def test_register_lcc_mm_end_to_end(tmp_path):
    nib = pytest.importorskip('nibabel')
    import numpy as np
    from nitorch.cli.registration.register.cli import cli
    g = np.random.RandomState(0)
    x = g.rand(24, 24, 12).astype('float32')
    x = spatial.smooth(torch.as_tensor(x), fwhm=3, dim=3).numpy()
    aff = np.diag([1., 1., 2., 1.])
    fix, mov = str(tmp_path / 'fix.nii'), str(tmp_path / 'mov.nii')
    nib.save(nib.Nifti1Image(x, aff), fix)
    nib.save(nib.Nifti1Image(np.roll(x, 1, 0), aff), mov)
    cli(['-o', str(tmp_path), '@loss', 'lcc', '-p', '5', 'mm', '@@fix', fix, '@@mov', mov,
         '@affine', 'rigid', '@@optim', 'gn', '-n', '3',
         '@pyramid', '-l', '0:2'])
    assert any(f.endswith('.lta') for f in os.listdir(tmp_path))
