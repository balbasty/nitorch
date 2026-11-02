"""`topup` command: option parsing, bandwidth default, kernel units, Jacobian."""
import os

import numpy as np
import pytest
import torch

from nitorch.cli.qmri.topup import main as topup_main
from nitorch.cli.qmri.topup.parser import parser_fit, parser_apply

nib = pytest.importorskip('nibabel')


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

def test_fit_mask_and_modulation_have_distinct_tags():
    opt = parser_fit.parse(['pos.nii', 'neg.nii', '-m', 'mask.nii',
                            '--modulation', 'no'])
    assert opt.mask == 'mask.nii'
    assert opt.modulation is False
    opt = parser_fit.parse(['pos.nii', 'neg.nii', '-j', 'no'])
    assert opt.mask is None
    assert opt.modulation is False


def test_fit_no_duplicate_short_tags():
    tags = [t for o in parser_fit.options for t in o.tags]
    tags = [t for t in tags if t not in ('--gpu', '--cpu')]
    assert len(tags) == len(set(tags))


@pytest.mark.parametrize('parser,args', [
    (parser_fit, ['pos.nii', 'neg.nii']),
    (parser_apply, ['field.nii', '-p', 'pos.nii']),
])
def test_bandwidth_default(parser, args):
    opt = parser.parse(list(args))
    assert opt.bandwidth == 1


@pytest.mark.parametrize('flag', ['-d', '--diffeo'])
def test_fit_diffeo_flag(flag):
    opt = parser_fit.parse(['pos.nii', 'neg.nii', flag])
    assert opt.diffeo is True
    opt = parser_fit.parse(['pos.nii', 'neg.nii', flag, 'no'])
    assert opt.diffeo is False
    opt = parser_fit.parse(['pos.nii', 'neg.nii', flag, '-l', 'ncc'])
    assert opt.diffeo is True and opt.loss == 'ncc'


@pytest.mark.parametrize('flag', ['-j', '--modulation'])
def test_fit_modulation_flag(flag):
    opt = parser_fit.parse(['pos.nii', 'neg.nii', flag])
    assert opt.modulation is True
    opt = parser_fit.parse(['pos.nii', 'neg.nii', flag, 'no'])
    assert opt.modulation is False


@pytest.mark.parametrize('flag', ['-m', '--modulation', '-d', '--diffeo'])
def test_apply_bool_flags(flag):
    key = 'modulation' if 'm' in flag[:3] else 'diffeo'
    opt = parser_apply.parse(['field.nii', flag, '-p', 'pos.nii'])
    assert getattr(opt, key) is True
    assert opt.file_pos == ['pos.nii']
    opt = parser_apply.parse(['field.nii', flag, 'no', '-p', 'pos.nii'])
    assert getattr(opt, key) is False


# ----------------------------------------------------------------------
# Kernel units
# ----------------------------------------------------------------------

def test_kernel_vox_scaled_by_downsampling_factor():
    aff0 = torch.eye(4, dtype=torch.double)
    aff = aff0.clone()
    aff[:3, :3] *= 4          # level downsampled by 4
    # 16 original voxels == 4 voxels at this level (not 16 / 2**4)
    kernel = topup_main.get_kernel([16, 'vox'], aff, [8, 8, 8], aff0)
    assert kernel == [4, 4, 4]
    # at full resolution, vox kernel is unchanged
    kernel = topup_main.get_kernel([5, 'vox'], aff0, [32, 32, 32], aff0)
    assert kernel == [5, 5, 5]


def test_kernel_vox_matches_mm_for_unit_voxels():
    aff0 = torch.eye(4, dtype=torch.double)
    aff = aff0.clone()
    aff[:3, :3] *= 8
    kvox = topup_main.get_kernel([24, 'vox'], aff, [4, 4, 4], aff0)
    kmm = topup_main.get_kernel([24, 'mm'], aff, [4, 4, 4], aff0)
    assert kvox == kmm == [3, 3, 3]


# ----------------------------------------------------------------------
# fit / apply with default bandwidth
# ----------------------------------------------------------------------

def _save(tmp_path, name, dat):
    fname = str(tmp_path / name)
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype='float32'), np.eye(4)),
             fname)
    return fname


def test_fit_runs_without_bandwidth(tmp_path, monkeypatch):
    dat = np.random.RandomState(0).rand(6, 7, 8)
    pos = _save(tmp_path, 'pos.nii.gz', dat)
    neg = _save(tmp_path, 'neg.nii.gz', dat)

    def fake_fit(d0, d1, **kwargs):
        return torch.full_like(d0, 0.5)

    monkeypatch.setattr(topup_main, 'topup_fit', fake_fit)
    out = str(tmp_path / 'field.nii.gz')
    topup_main._cli(['fit', pos, neg, '-l', 'ncc', '-w', '1', '-o', out])
    field = nib.load(out).get_fdata()
    np.testing.assert_allclose(field, 0.5, atol=1e-5)


def test_apply_runs_without_bandwidth(tmp_path):
    field = _save(tmp_path, 'field.nii.gz', np.zeros([6, 7, 8]))
    img = _save(tmp_path, 'img.nii.gz', np.ones([6, 7, 8]))
    out = str(tmp_path / 'out.nii.gz')
    topup_main._cli(['apply', field, '-p', img, '-o', out])
    np.testing.assert_allclose(nib.load(out).get_fdata(), 1, atol=1e-5)


# ----------------------------------------------------------------------
# Jacobian axis in apply
# ----------------------------------------------------------------------

SHAPE = [16, 12, 10]
READOUT = {'lr': 0, 'ap': 1, 'is': 2}


def _disp_along(axis, amplitude=0.6):
    """Smooth displacement varying only along `axis` (zero slope at
    edges so that the dct2 central difference is exact-ish)."""
    n = SHAPE[axis]
    x = np.arange(n)
    prof = amplitude * np.cos(np.pi * (x + 0.5) / n)
    shape = [1, 1, 1]
    shape[axis] = n
    return np.broadcast_to(prof.reshape(shape), SHAPE).copy(), axis


def _central_diff(disp, axis):
    d = np.pad(disp, [(1, 1) if i == axis else (0, 0) for i in range(3)],
               mode='symmetric')
    sl_p = [slice(None)] * 3
    sl_m = [slice(None)] * 3
    sl_p[axis] = slice(2, None)
    sl_m[axis] = slice(None, -2)
    return (d[tuple(sl_p)] - d[tuple(sl_m)]) / 2


@pytest.mark.parametrize('readout', ['lr', 'ap', 'is'])
@pytest.mark.parametrize('polarity', ['-p', '-n'])
def test_apply_jacobian_along_readout(tmp_path, readout, polarity):
    axis = READOUT[readout]
    disp, _ = _disp_along(axis)
    field = _save(tmp_path, 'field.nii.gz', disp)
    img = _save(tmp_path, 'img.nii.gz', np.ones(SHAPE))
    out = str(tmp_path / 'out.nii.gz')
    topup_main._cli(['apply', field, polarity, img, '-r', readout,
                     '-bw', '1', '-o', out])
    res = nib.load(out).get_fdata()
    sign = 1 if polarity == '-p' else -1
    expected = 1 + sign * _central_diff(disp, axis)
    np.testing.assert_allclose(res, expected, atol=1e-4)
    assert not np.allclose(res, 1)


@pytest.mark.parametrize('readout', ['lr', 'ap', 'is'])
def test_apply_jacobian_ignores_other_axes(tmp_path, readout):
    axis = READOUT[readout]
    other = (axis + 1) % 3
    disp, _ = _disp_along(other)
    field = _save(tmp_path, 'field.nii.gz', disp)
    # constant image: a displacement along the readout that varies only
    # along another axis does not compress/stretch along the readout
    img = _save(tmp_path, 'img.nii.gz', np.ones(SHAPE))
    out = str(tmp_path / 'out.nii.gz')
    topup_main._cli(['apply', field, '-p', img, '-r', readout,
                     '-bw', '1', '-o', out])
    res = nib.load(out).get_fdata()
    np.testing.assert_allclose(res, 1, atol=1e-4)
