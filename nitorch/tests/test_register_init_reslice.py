"""`nitorch register`: center-of-mass initialization (`@affine -i com`)
and fully resliced outputs (`@@mov/@@fix -r`) for all affine positions.

Convention (shared by `-i fov`, the LTA output and `warp_images`): the
affine `A` maps fixed world coordinates to moving world coordinates, i.e.,
`mov_vox = inv(mov_affine) @ A @ fix_affine @ fix_vox`. With
`position='symmetric'`, `AffineModel.exp` returns `sqrtm(A)`.
"""
import os
import pytest
import torch
from nitorch import spatial, io
from nitorch.tools.registration import objects
from nitorch.tools.registration.pairwise_makeobj import make_affine
from nitorch.tools.registration.pairwise_postproc import warp_images
from nitorch.cli.registration.register.cli import _cli as register

nib = pytest.importorskip('nibabel')

SIGMA = 4.
FIX_SHAPE, FIX_VX = [40, 40, 40], [1., 1., 1.]
MOV_SHAPE, MOV_VX = [30, 32, 24], [1.5, 1.3, 2.]
FIX_CENTER = [0., 0., 0.]
MOV_CENTER = [5., -3., 2.]  # => true shift = MOV_CENTER - FIX_CENTER


def make_affines():
    aff_f = spatial.affine_default(FIX_SHAPE, FIX_VX).double()
    aff_m = spatial.affine_default(MOV_SHAPE, MOV_VX).double()
    aff_m[:3, -1] += torch.as_tensor([3., 2., -1.], dtype=torch.double)
    return aff_f, aff_m


def blob_world(xyz, center, scale=100., offset=10.):
    """Gaussian blob evaluated at world coordinates `xyz`"""
    center = torch.as_tensor(center, dtype=xyz.dtype)
    d2 = (xyz - center).square().sum(-1)
    return scale * (-d2 / (2 * SIGMA ** 2)).exp() + offset


def blob(shape, aff, center, **kwargs):
    grid = spatial.identity_grid(shape, dtype=torch.double)
    return blob_world(spatial.affine_matvec(aff, grid), center, **kwargs)


def expected_resliced(src_shape, src_aff, src_center, trg_shape, trg_aff,
                      transform):
    """Analytical value of the source blob resliced onto the target grid,
    where `transform` maps target world -> source world. Also returns a
    mask of target voxels that fall well inside the source field of view."""
    grid = spatial.identity_grid(trg_shape, dtype=torch.double)
    xyz = spatial.affine_matvec(transform.matmul(trg_aff), grid)
    vox = spatial.affine_matvec(spatial.affine_inv(src_aff), xyz)
    shape = torch.as_tensor(src_shape, dtype=torch.double)
    inside = ((vox > 2) & (vox < shape - 3)).all(-1)
    return blob_world(xyz, src_center), inside


def write_images(tmp_path, aff_f, aff_m, fix_center, mov_center):
    fix = blob(FIX_SHAPE, aff_f, fix_center)
    mov = blob(MOV_SHAPE, aff_m, mov_center)
    fname_fix = os.path.join(str(tmp_path), 'fix.nii')
    fname_mov = os.path.join(str(tmp_path), 'mov.nii')
    nib.save(nib.Nifti1Image(fix.float().numpy(), aff_f.numpy()), fname_fix)
    nib.save(nib.Nifti1Image(mov.float().numpy(), aff_m.numpy()), fname_mov)
    return fname_fix, fname_mov


def check_resliced(fname, src_shape, src_aff, src_center, trg_shape,
                   trg_aff, transform):
    dat = io.map(fname)
    assert list(dat.shape[:3]) == list(trg_shape)
    assert torch.allclose(dat.affine.double(), trg_aff, atol=1e-4)
    dat = dat.fdata(dtype=torch.double).reshape(trg_shape)
    ref, inside = expected_resliced(src_shape, src_aff, src_center,
                                    trg_shape, trg_aff, transform)
    assert inside.sum() > 1000
    err = (dat - ref)[inside].abs().max()
    # blob peak = 100: linear interpolation error is ~5, whereas applying
    # the transform twice (5 mm shift) gives errors > 40
    assert err < 10, err


# ----------------------------------------------------------------------
#   Center of mass
# ----------------------------------------------------------------------

@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('offset', [0., 10., -50.])
def test_center_of_mass_world(masked, offset):
    """COM is in world (RAS) coordinates and is independent of the
    intensity offset/scale, for images with different affines."""
    aff_f, aff_m = make_affines()
    for shape, aff, center in ((FIX_SHAPE, aff_f, FIX_CENTER),
                               (MOV_SHAPE, aff_m, MOV_CENTER)):
        dat = blob(shape, aff, center, offset=offset)[None]
        mask = torch.ones_like(dat) if masked else None
        image = objects.Image(dat, affine=aff, mask=mask)
        com = image.get_center_of_mass()
        assert com.shape == (3,)
        assert torch.allclose(com, torch.as_tensor(center).to(com),
                              atol=0.1), (com, center)


def test_center_of_mass_mask():
    """Voxels outside the mask do not contribute"""
    aff_f, _ = make_affines()
    dat = blob(FIX_SHAPE, aff_f, [10., 0., 0.])[None]
    dat += blob(FIX_SHAPE, aff_f, [-10., 0., 0.], offset=0)[None]
    xyz = spatial.affine_matvec(
        aff_f, spatial.identity_grid(FIX_SHAPE, dtype=torch.double))
    mask = (xyz[..., 0] > 0)[None].double()  # excludes the second blob
    com = objects.Image(dat, affine=aff_f).get_center_of_mass()
    assert abs(com[0]) < 0.1, com
    com = objects.Image(dat, affine=aff_f, mask=mask).get_center_of_mass()
    assert com[0] > 8, com


# ----------------------------------------------------------------------
#   Command line: -i com / -i fov, and -r resliced outputs
# ----------------------------------------------------------------------

def run_register(tmp_path, init, position, resliced=False):
    aff_f, aff_m = make_affines()
    fname_fix, fname_mov = write_images(tmp_path, aff_f, aff_m,
                                        FIX_CENTER, MOV_CENTER)
    odir = str(tmp_path)
    fname_lta = os.path.join(odir, 'affine.lta')
    out = ['-o', 'false']
    if resliced:
        out += ['-r', os.path.join(odir, '{base}.resliced{ext}')]
    args = ['@loss', 'mse',
            '@@fix', fname_fix, *out,
            '@@mov', fname_mov, *out,
            '@affine', 'translation', '-i', init, '-p', position,
            '-o', fname_lta,
            '@@optim', 'none',
            '@pyramid', '-l', '0']
    register(args)
    lta = io.transforms.map(fname_lta).fdata(dtype=torch.double).squeeze()
    return lta, aff_f, aff_m


@pytest.mark.parametrize('init', ['com', 'fov'])
def test_register_init_center(tmp_path, init):
    lta, aff_f, aff_m = run_register(tmp_path, init, 'sym')
    if init == 'com':
        shift = torch.as_tensor(MOV_CENTER) - torch.as_tensor(FIX_CENTER)
    else:
        shape_f = torch.as_tensor(FIX_SHAPE, dtype=torch.double)
        shape_m = torch.as_tensor(MOV_SHAPE, dtype=torch.double)
        shift = (spatial.affine_matvec(aff_m, (shape_m - 1) / 2) -
                 spatial.affine_matvec(aff_f, (shape_f - 1) / 2))
    assert torch.allclose(lta[:3, :3], torch.eye(3).double(), atol=1e-5)
    assert torch.allclose(lta[:3, -1], shift.double(), atol=0.1), \
        (lta[:3, -1], shift)


@pytest.mark.slow
@pytest.mark.parametrize('position', ['sym', 'mov', 'fix'])
def test_register_com_resliced(tmp_path, position):
    """`-i com` + `-r`: resliced images match the true transform"""
    lta, aff_f, aff_m = run_register(tmp_path, 'com', position, True)
    shift = torch.as_tensor(MOV_CENTER) - torch.as_tensor(FIX_CENTER)
    assert torch.allclose(lta[:3, -1], shift.double(), atol=0.1)
    true = torch.eye(4, dtype=torch.double)
    true[:3, -1] = shift
    # moving resliced to fixed space
    check_resliced(os.path.join(str(tmp_path), 'mov.resliced.nii'),
                   MOV_SHAPE, aff_m, MOV_CENTER, FIX_SHAPE, aff_f, true)
    # fixed resliced to moving space
    check_resliced(os.path.join(str(tmp_path), 'fix.resliced.nii'),
                   FIX_SHAPE, aff_f, FIX_CENTER, MOV_SHAPE, aff_m,
                   spatial.affine_inv(true))


# ----------------------------------------------------------------------
#   warp_images: -r with a rigid transform, for all positions
# ----------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('position', ['symmetric', 'moving', 'fixed'])
def test_warp_images_resliced(tmp_path, position):
    aff_f, aff_m = make_affines()
    # true transform: fixed world -> moving world
    true = spatial.affine_matrix_classic(
        torch.as_tensor([4., -2., 1., 0.1, -0.05, 0.08], dtype=torch.double),
        dim=3)
    fix_center = torch.as_tensor(FIX_CENTER, dtype=torch.double)
    mov_center = spatial.affine_matvec(true, fix_center).tolist()
    fname_fix, fname_mov = write_images(tmp_path, aff_f, aff_m,
                                        fix_center.tolist(), mov_center)

    affine = make_affine('rigid', position, init=true.float())
    aff = affine.exp()
    if position[0] == 's':
        aff = aff.matmul(aff)
    assert torch.allclose(aff.double(), true, atol=1e-4)

    odir = str(tmp_path)
    out_mov = os.path.join(odir, 'mov.resliced.nii')
    out_fix = os.path.join(odir, 'fix.resliced.nii')
    warp_images(dict(files=[fname_fix], resliced=out_fix),
                dict(files=[fname_mov], resliced=out_mov),
                affine=affine, dim=3, verbose=False)
    check_resliced(out_mov, MOV_SHAPE, aff_m, mov_center,
                   FIX_SHAPE, aff_f, true)
    check_resliced(out_fix, FIX_SHAPE, aff_f, fix_center.tolist(),
                   MOV_SHAPE, aff_m, spatial.affine_inv(true))
