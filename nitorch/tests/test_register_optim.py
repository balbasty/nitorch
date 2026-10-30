"""Affine optimisers / bases of `nitorch register` and `nitorch affreg`
that used to crash: `@@optim powell` (affreg's default for its default
NMI loss), `@@optim ogm` and `@affine rotation`."""
import torch
import pytest
from nitorch import io, spatial
from nitorch.cli.registration.register.cli import cli as register
from nitorch.cli.registration.affreg.cli import cli as affreg
from nitorch.tools.registration import objects, optim
from nitorch.tools.registration.pairwise_makeobj import make_affine_optim


def phantom(x):
    """Smooth "head" (ellipsoid) with some off-centre structure,
    evaluated at world coordinates `x` (in a 24 mm field of view)"""
    dist = lambda c, s: ((x - x.new_tensor(c)) / x.new_tensor(s)).square().sum(-1)
    img = torch.sigmoid(4 * (1 - dist([0, 0, 0], [8, 9.5, 7]).sqrt()))
    img = img * (1 + (-dist([4, 3, -2], [3] * 3)).exp()
                 - 0.6 * (-dist([-3, -4, 2], [3, 4, 3])).exp()
                 + 0.8 * (-dist([-4, 3, 1], [4, 2.5, 2.5])).exp())
    return img


def make_pair(tmp_path, shift=(0., 0., 0.), rot=(0., 0., 0.), shape=[24]*3):
    """Write a fixed image and a moving image `mov(x) = fix(T x)`.
    Return the paths and the world-to-world transform registration
    should recover (`T^{-1}`)."""
    aff = spatial.affine_default(shape, dtype=torch.double)
    T = spatial.affine_matrix_classic(torch.as_tensor([*shift, *rot], dtype=torch.double), dim=3)
    x = spatial.affine_grid(aff, shape)
    fix = phantom(x)
    mov = phantom(spatial.affine_matvec(T, x))
    fnames = dict(fix=str(tmp_path / 'fix.nii.gz'), mov=str(tmp_path / 'mov.nii.gz'),
                  out=str(tmp_path / 'out.lta'))
    io.savef(fix.float(), fnames['fix'], affine=aff)
    io.savef(mov.float(), fnames['mov'], affine=aff)
    return fnames, T.inverse()


SHIFT, ROT = (1.5, -1., 0.5), (0., 0., 0.08)


def check_affine(fname, expected, atol_lin=0.01, atol_shift=0.05):
    """Compare the written transform with the expected one.
    (For reference, identity is off by 0.08 / 1.5 here.)"""
    aff = io.transforms.loadf(fname).squeeze().double()
    assert torch.allclose(aff[:3, :3], expected[:3, :3], atol=atol_lin), (aff, expected)
    assert torch.allclose(aff[:3, -1], expected[:3, -1], atol=atol_shift), (aff, expected)


@pytest.mark.slow
def test_affreg_default(tmp_path):
    """`nitorch affreg` with its default NMI loss and Powell optimiser.
    (A single 2 mm pyramid level keeps Powell fast enough for a test.)"""
    f, expected = make_pair(tmp_path, shift=SHIFT, rot=ROT)
    affreg(['-m', f['mov'], '-f', f['fix'], '-o', f['out'], '-x', 'false',
            '-p', '1'])
    check_affine(f['out'], expected, atol_lin=0.03, atol_shift=0.15)


@pytest.mark.slow
def test_register_affine_powell(tmp_path):
    f, expected = make_pair(tmp_path, shift=SHIFT, rot=ROT)
    register(['@loss', 'mse', '@@fix', f['fix'], '-o', 'false',
              '@@mov', f['mov'], '-o', 'false',
              '@affine', 'rigid', '-o', f['out'], '@@optim', 'powell',
              '@pyramid', '--levels', '1'])
    check_affine(f['out'], expected, atol_lin=0.03, atol_shift=0.15)


@pytest.mark.slow
def test_register_affine_ogm(tmp_path):
    f, expected = make_pair(tmp_path, shift=SHIFT, rot=ROT)
    register(['@loss', 'mse', '@@fix', f['fix'], '-o', 'false',
              '@@mov', f['mov'], '-o', 'false',
              '@affine', 'rigid', '-o', f['out'], '@@optim', 'ogm'])
    check_affine(f['out'], expected)


@pytest.mark.slow
def test_register_affine_rotation(tmp_path):
    f, expected = make_pair(tmp_path, rot=(0.03, -0.02, 0.08))
    register(['@loss', 'mse', '@@fix', f['fix'], '-o', 'false',
              '@@mov', f['mov'], '-o', 'false',
              '@affine', 'rotation', '-o', f['out']])
    check_affine(f['out'], expected, atol_shift=1e-5)


def test_optim_grad_rotation_basis():
    """Gradients w.r.t. a rotation-only basis keep all parameters"""
    dat = objects.LogAffine(basis='rotation', dim=3, dtype=torch.double)
    _, grad = dat.exp(grad=True)
    assert grad.shape[0] == 3


def test_powell_affine_parameters():
    """Powell works on `AffineModel.Parameters` (what `register` passes),
    including when only a subset of the parameters is optimised."""
    target = torch.as_tensor([1., -2., 0.5, 0.1, -0.2, 0.3], dtype=torch.double)
    closure = lambda x, **k: (x.dat if hasattr(x, 'dat') else x).sub(target).square().sum()
    for sub in (None, 'translation'):
        model = objects.AffineModel('rigid', optim=sub).set_dat(dim=3, dtype=torch.double)
        powell = make_affine_optim('powell', max_iter=20, tolerance=0, name='pow', fmg=2)
        powell.iter(model.parameters(), closure)
        expected = target if sub is None else torch.cat([target[:3], torch.zeros(3).double()])
        assert torch.allclose(model.dat.dat, expected, atol=1e-4), (sub, model.dat.dat)


def test_ogm_from_cli_options():
    """OGM accepts the learning rate positionally, plus the CLI options"""
    ogm = make_affine_optim('ogm', lr=0.5, momentum=0, relax=0, restart=True,
                            output='', name='ogm', fmg=2)
    assert isinstance(ogm, optim.OGM) and ogm.lr == 0.5
