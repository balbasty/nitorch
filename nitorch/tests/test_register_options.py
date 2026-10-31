"""Command-line options of `nitorch register` that were parsed but
ignored (or mis-converted) on their way to the registration objects."""
import os
import numpy as np
import torch
import pytest

nib = pytest.importorskip('nibabel')

from nitorch.cli.registration.register.parser import parser
from nitorch.cli.registration.register.cli import build_losses, _cli
from nitorch.core.py import make_list
from nitorch.tools.registration import optim as opt
from nitorch.tools.registration.losses import MI
from nitorch.tools.registration.pairwise_makeobj import (
    make_affine, make_affine_optim, make_nonlin_optim
)
from nitorch.tools.registration.pairwise_postproc import warp_images
from nitorch.tools.registration.pairwise_pyramid import pyramid_levels

BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii']


def parse(*args):
    return parser.parse(BASE + list(args))


def optim_kwargs(group):
    """What `cli._main` passes to `make_*_optim`"""
    return dict(group.optim)


# ----------------------------------------------------------------------
#   Levenberg-Marquardt
# ----------------------------------------------------------------------

def test_gauss_newton_marquardt_only_loads_diagonal():
    hess = torch.tensor([[4., 1., 2.], [1., 5., 3.], [2., 3., 6.]])
    grad = torch.ones(3)
    _, h = opt.GaussNewton(marquardt=0.5)._add_marquardt(grad, hess.clone())
    assert torch.allclose(h, hess + 0.5 * torch.eye(3))


@pytest.mark.parametrize('group', ['@affine', '@nonlin'])
@pytest.mark.parametrize('args, expected', [
    ([], True),                 # auto
    (['-m', 'auto'], True),
    (['-m', '0.5'], 0.5),
    (['-m', '0'], 0),           # disabled on request
])
def test_marquardt_option(group, args, expected):
    options = parse(group, '@@optim', 'gn', *args)
    sub = options.affine if group == '@affine' else options.nonlin
    if expected is True:
        assert sub.optim.marquardt is True
    else:
        assert sub.optim.marquardt == expected
    if group == '@affine':
        optim = make_affine_optim('gn', 2, **optim_kwargs(sub))
    else:
        optim = make_nonlin_optim('gn', 2, **optim_kwargs(sub))
    assert optim.marquardt == expected


def test_gauss_newton_rigid_on_flat_image(tmp_path):
    """Out-of-plane parameters of a 2D image have a zero Hessian:
    this used to raise a 'singular matrix' error."""
    g = np.random.default_rng(0)
    fix = write(str(tmp_path / 'fix.nii.gz'), g.random([12, 12]))
    mov = write(str(tmp_path / 'mov.nii.gz'), g.random([12, 12]))
    odir = str(tmp_path / 'out')
    _cli(['-o', odir, '@loss', 'mse', '@@fix', fix, '-o', 'false',
          '@@mov', mov, '-o', 'false', '@affine', 'rigid',
          '@@optim', 'gn', '-n', '2', '@optim', '-n', '1',
          '@pyramid', '-l', '0'])
    assert os.path.exists(os.path.join(odir, 'rigid.lta'))


# ----------------------------------------------------------------------
#   Other optimizer options
# ----------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    (None, True), ('0', False), ('false', False), ('1', True), ('true', True),
])
def test_nesterov_restart_option(value, expected):
    args = ['@affine', '@@optim', 'nes']
    if value is not None:
        args += ['-r', value]
    options = parse(*args)
    assert options.affine.optim.restart is expected
    optim = make_affine_optim('nes', 1, **optim_kwargs(options.affine))
    assert optim.auto_restart is expected
    optim = make_nonlin_optim('nes', 1, **optim_kwargs(options.affine))
    assert optim.auto_restart is expected


@pytest.mark.parametrize('fmg, sub_iter', [(0, 16), (1, 2), (3, 2)])
def test_nonlin_fmg_option(fmg, sub_iter):
    options = parse('@nonlin', '@@optim', 'gn', '-g', str(fmg))
    optim = make_nonlin_optim('gn', 2, **optim_kwargs(options.nonlin))
    assert optim.fmg == fmg
    assert optim.max_iter == sub_iter


# ----------------------------------------------------------------------
#   Loss options
# ----------------------------------------------------------------------

def write(path, dat, dtype='float32'):
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype=dtype), np.eye(4)), path)
    return path


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(0)
    fix = write(str(tmp_path / 'fix.nii.gz'), rng.random([8, 8, 8]))
    mov = write(str(tmp_path / 'mov.nii.gz'), rng.random([8, 8, 8]))
    return fix, mov


def built_loss(images, name, *args):
    fix, mov = images
    options = parser.parse(['@loss', name, *args, '@@fix', fix,
                            '@@mov', mov, '@affine', '@pyramid', '-l', '0'])
    shapes = [nib.load(f).shape for f in (fix, mov)]
    levels = pyramid_levels([[1.] * 3] * 2, shapes, **options.pyramid)
    levels = list(levels)
    losses, _ = build_losses(options, [{'fix': levels[0], 'mov': levels[1]}],
                             torch.device('cpu'))
    return options, losses[0][0].loss


def test_mi_default_order():
    options = parser.parse(['@loss', 'mi', '@@fix', 'f.nii', '@@mov', 'm.nii'])
    assert options.loss[0].order == [3]


@pytest.mark.parametrize('args, expected', [([], [3]), (['-n', '2'], [2])])
def test_mi_order_passed_to_loss(images, args, expected):
    _, loss = built_loss(images, 'mi', *args)
    assert isinstance(loss, MI)
    assert make_list(loss.spline) == expected


@pytest.mark.parametrize('name', ['mad', 'tuk'])
@pytest.mark.parametrize('flag', [False, True])
def test_robust_joint_passed_to_loss(images, name, flag):
    _, loss = built_loss(images, name, *(['-j'] if flag else []))
    assert loss.joint is flag


@pytest.mark.parametrize('name', ['lcc', 'lgmm'])
def test_local_stride_passed_to_loss(images, name):
    _, loss = built_loss(images, name, '--stride', '2')
    assert list(loss.stride) == [2]


@pytest.mark.parametrize('name', ['gmm', 'lgmm'])
@pytest.mark.parametrize('args, measure', [
    ([], 'mi'), (['-m', 'mi'], 'mi'), (['-m', 'joint'], 'joint')])
def test_gmm_measure_passed_to_loss(images, name, args, measure):
    _, loss = built_loss(images, name, *args)
    assert loss.measure == measure
    assert loss.bins == {'gmm': 6, 'lgmm': 3}[name]  # documented defaults


# ----------------------------------------------------------------------
#   Interpolation order of the warped outputs
# ----------------------------------------------------------------------

def warp_shifted(tmp_path, side, order):
    """Reslice a two-label image (values 1|5) after a sub-voxel
    translation and return the result"""
    dat = np.zeros([8, 4, 4])
    dat[:4] = 1
    dat[4:] = 5
    files = dict(
        fix=write(str(tmp_path / 'fix.nii.gz'), dat if side == 'fix' else 0*dat),
        mov=write(str(tmp_path / 'mov.nii.gz'), dat if side == 'mov' else 0*dat),
    )
    out = str(tmp_path / 'resliced.nii.gz')
    opt = {key: dict(files=[fname]) for key, fname in files.items()}
    opt[side].update(resliced=out, order=order)
    init = torch.eye(4, dtype=torch.double)
    init[0, -1] = 0.3
    affine = make_affine('translation', 'sym', init=init)
    warp_images(opt['fix'], opt['mov'], affine=affine, dim=3,
                device=torch.device('cpu'), verbose=False)
    return np.asarray(nib.load(out).dataobj)


def test_file_order_option():
    options = parser.parse(['@loss', 'mse', '@@fix', 'f.nii', '-n', '0',
                            '@@mov', 'm.nii', '-n', '3'])
    assert options.loss[0].fix.order == 0
    assert options.loss[0].mov.order == 3


@pytest.mark.parametrize('side', ['fix', 'mov'])
def test_warp_order_honoured(tmp_path, side):
    nearest = warp_shifted(tmp_path, side, 0)
    assert set(np.unique(nearest).tolist()) <= {0, 1, 5}
    linear = warp_shifted(tmp_path, side, 1)
    # the 1|5 boundary is interpolated
    assert ((linear > 1) & (linear < 5)).any()


def reslice_identity(tmp_path, dat, label, order):
    dtype = 'int16' if label else 'float32'
    fix = write(str(tmp_path / 'fix.nii.gz'), np.zeros_like(dat))
    mov = write(str(tmp_path / 'mov.nii.gz'), dat, dtype)
    out = str(tmp_path / 'resliced.nii.gz')
    affine = make_affine('translation', 'sym',
                         init=torch.eye(4, dtype=torch.double))
    warp_images(dict(files=[fix]),
                dict(files=[mov], resliced=out, label=label, order=order),
                affine=affine, dim=3, device=torch.device('cpu'),
                verbose=False)
    return np.asarray(nib.load(out).dataobj)


def test_warp_spline_identity_reproduces_intensities(tmp_path):
    """Order > 1 interpolates the spline coefficients of the
    (prefiltered) image, so the identity reproduces the input."""
    dat = np.random.default_rng(0).random([8, 7, 6]).astype('float32')
    out = reslice_identity(tmp_path, dat, False, 3)
    assert np.allclose(out, dat, atol=1e-5), np.abs(out - dat).max()


def test_warp_spline_labels_stay_labels(tmp_path):
    lab = np.random.default_rng(0).choice([0, 1, 4, 7], size=[8, 7, 6])
    out = reslice_identity(tmp_path, lab, True, 3)
    # (no prefilter: order-3 weights vote between labels, so the
    # output need not equal the input, but it only contains labels)
    assert not np.issubdtype(out.dtype, np.floating)
    assert set(np.unique(out).tolist()) <= {0, 1, 4, 7}
