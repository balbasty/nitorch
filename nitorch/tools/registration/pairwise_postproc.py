__all__ = ['warp_images']

from nitorch import io
from nitorch.core.py import fileparts, make_list
from nitorch.core import utils
from nitorch.core.struct import Structure
from nitorch import spatial
from . import objects, pairwise_preproc as preproc, utils as regutils
import torch
import os


def warp_images(fix, mov, affine=None, nonlin=None, dim=None, device=None, odir=None):
    """Warp and save the moving and fixed images from a loss object

    Parameters
    ----------
    fix, mov : dict or Structure with fields
        files : [list of] str
            Path to input images
        output : str
            Path to output moved (but not resliced) file
        resliced : str
            Path to output resliced file
        world : (D+1, D+1) tensor or str, optional
            Voxel-to-world matrix that takes precedence over whatever is
            read from disk.
        affine : [sequence of] (D+1, D+1) tensor or str, optional
            A series of affine transforms that should be applied to the image.
        bound : str, default='dct2'
            Boundary conditions for out-of-bound data
        extrapolate : bool, default=False
            Extrapolate out-of-bound data

    affine : objects.AffineModel, optional
        An affine transform (fitted using `PairwiseRegister`)
    nonlin : objects.NonlinModel, optional
        A non-linear transform (fitted using `PairwiseRegister`)
    dim : int, optional
        Number of spatial dimensions
    device : torch.device
        Device to use to perform the warping
    odir : str, optional
        Output directory

    """

    if isinstance(fix, dict):
        fix = dict(fix)
        fix.setdefault('world', None)
        fix.setdefault('affine', None)
        fix.setdefault('output', None)
        fix.setdefault('resliced', None)
        fix.setdefault('bound', 'dct2')
        fix.setdefault('extrapolate', False)
        fix.setdefault('label', False)
        fix = Structure(fix)
    if isinstance(mov, dict):
        mov = dict(mov)
        mov.setdefault('world', None)
        mov.setdefault('affine', None)
        mov.setdefault('output', None)
        mov.setdefault('resliced', None)
        mov.setdefault('bound', 'dct2')
        mov.setdefault('extrapolate', False)
        mov.setdefault('label', False)
        mov = Structure(mov)

    if not (mov.output or mov.resliced or fix.output or fix.resliced):
        return

    dat_fix, fix_affine = preproc.map_image(fix.files, dim=dim)
    dat_mov, mov_affine = preproc.map_image(mov.files, dim=dim)
    fix_affine = fix_affine.float()
    mov_affine = mov_affine.float()
    dim = dim or (fix.dim - 1)

    if affine:
        affine.to_(device)
        affine.clear_cache()
    if nonlin:
        nonlin.to_(device)
        nonlin.clear_cache()

    if fix.world:  # overwrite orientation matrix
        fix_affine = io.transforms.map(fix.world).fdata().squeeze()
    for transform in (fix.affine or []):
        transform = io.transforms.map(transform).fdata().squeeze()
        fix_affine = spatial.affine_lmdiv(transform, fix_affine)

    if mov.world:  # overwrite orientation matrix
        mov_affine = io.transforms.map(mov.world).fdata().squeeze()
    for transform in (mov.affine or []):
        transform = io.transforms.map(transform).fdata().squeeze()
        mov_affine = spatial.affine_lmdiv(transform, mov_affine)

    # moving
    if mov.output or mov.resliced:
        ifname = make_list(mov.files)[0]
        idir, base, ext = fileparts(ifname)
        odir_mov = odir or idir or '.'

        dat = dat_mov.data(device=device) if mov.label else dat_mov.fdata(rand=True, device=device)
        image = objects.Image(dat, affine=mov_affine, bound=mov.bound,
                              extrapolate=mov.extrapolate)

        if mov.output:
            target_affine = mov_affine
            target_shape = image.shape
            if affine and affine.position[0].lower() in 'ms':
                aff = affine.exp(recompute=False, cache_result=True)
                if not nonlin and affine.position[0].lower() == 's':
                    aff = aff.matmul(aff)
                target_affine = spatial.affine_lmdiv(aff, target_affine)

            fname = mov.output.format(dir=odir_mov, base=base, sep=os.path.sep, ext=ext)
            print(f'Minimal reslice: {ifname} -> {fname} ...', end=' ')
            warped = warp_one_image(image, target_affine, target_shape,
                                    affine=affine, nonlin=nonlin)
            save = io.save if mov.label else io.savef
            save(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

        if mov.resliced:
            target_affine = fix_affine
            target_shape = dat_fix.shape[1:]

            fname = mov.resliced.format(dir=odir_mov, base=base, sep=os.path.sep, ext=ext)
            print(f'Full reslice: {ifname} -> {fname} ...', end=' ')
            warped = warp_one_image(image, target_affine, target_shape,
                                    affine=affine, nonlin=nonlin, reslice=True)
            save = io.save if mov.label else io.savef
            save(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

    # fixed
    if fix.output or fix.resliced:
        ifname = make_list(fix.files)[0]
        idir, base, ext = fileparts(ifname)
        odir_fix = odir or idir or '.'

        dat = dat_fix.data(device=device) if fix.label else dat_fix.fdata(rand=True, device=device)
        image = objects.Image(dat, affine=fix_affine, bound=fix.bound,
                              extrapolate=fix.extrapolate)

        if fix.output:
            target_affine = fix_affine
            target_shape = image.shape
            if affine and affine.position[0].lower() in 'fs':
                aff = affine.exp(recompute=False, cache_result=True)
                if not nonlin and affine.position[0].lower() == 's':
                    aff = aff.matmul(aff)
                target_affine = spatial.affine_matmul(aff, target_affine)

            fname = fix.output.format(dir=odir_fix, base=base, sep=os.path.sep, ext=ext)
            print(f'Minimal reslice: {ifname} -> {fname} ...', end=' ')
            warped = warp_one_image(image, target_affine, target_shape,
                                    affine=affine, nonlin=nonlin, backward=True)
            save = io.save if fix.label else io.savef
            save(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

        if fix.resliced:
            target_affine = mov_affine
            target_shape = dat_mov.shape[1:]

            fname = fix.resliced.format(dir=odir_fix, base=base, sep=os.path.sep, ext=ext)
            print(f'Full reslice: {ifname} -> {fname} ...', end=' ')
            warped = warp_one_image(image, target_affine, target_shape,
                                    affine=affine, nonlin=nonlin,
                                    backward=True, reslice=True)
            save = io.save if fix.label else io.savef
            save(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped


def _almost_identity(aff):
    """Return True if an affine is almost the identity matrix"""
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


def warp_one_image(image, target, shape=None, affine=None, nonlin=None,
                   backward=False, reslice=False):
    """Returns the warped image, with channel dimension last

    Parameters
    ----------
    image : objects.Image
        Image to warp
    target : (D+1, D+1) tensor
        Orientation matrix of the target space
    shape : list[int]
        Spatial shape of the target space
    affine : objects.AffineModel, optional
        An affine transform (fitted using `PairwiseRegister`)
    nonlin : objects.NonlinModel, optional
        A non-linear transform (fitted using `PairwiseRegister`)
    backward : bool, default=False
        Whether to apply the transform backward (moving to fixed)
    reslice : bool, default=False
        Whether to reslice to the target grid, or keep the input grid.

    Returns
    -------
    warped : (*spatial, C) tensor
        Warped image

    """
    # build transform
    aff_right = target
    aff_left = spatial.affine_inv(image.affine)
    aff = None
    if affine:
        # exp = affine.iexp if backward else affine.exp
        exp = affine.exp
        aff = exp(recompute=False, cache_result=True)
        if backward:
            aff = spatial.affine_inv(aff)
    if nonlin:
        if affine:
            if affine.position[0].lower() in ('ms' if backward else 'fs'):
                aff_right = spatial.affine_matmul(aff, aff_right)
            if affine.position[0].lower() in ('fs' if backward else 'ms'):
                aff_left = spatial.affine_matmul(aff_left, aff)
        exp = nonlin.iexp if backward else nonlin.exp
        phi = exp(recompute=False, cache_result=True)
        aff_left = spatial.affine_matmul(aff_left, nonlin.affine)
        aff_right = spatial.affine_lmdiv(nonlin.affine, aff_right)
        if _almost_identity(aff_right) and nonlin.shape == shape:
            phi = nonlin.add_identity(phi)
        else:
            tmp = spatial.affine_grid(aff_right.to(phi), shape)
            phi = regutils.smart_pull_grid(phi, tmp).add_(tmp)
            del tmp
        if not _almost_identity(aff_left):
            phi = spatial.affine_matvec(aff_left.to(phi), phi)
    else:
        # no nonlin: single affine even if position == 'symmetric'
        if reslice:
            if affine.position[0].lower():
                aff = aff.matmul(aff)
            aff = spatial.affine_matmul(aff, aff_right)
            aff = spatial.affine_matmul(aff_left, aff)
            phi = spatial.affine_grid(aff, shape)
        else:
            phi = None

    # warp image
    if phi is not None:
        warped = image.pull(phi)
    else:
        warped = image.dat

    # write to disk
    if len(warped) == 1:
        warped = warped[0]
    else:
        warped = utils.movedim(warped, 0, -1)
    return warped
