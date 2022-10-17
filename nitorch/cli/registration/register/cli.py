from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from .parser import parser, help
from nitorch.tools.registration import (pairwise, losses, optim,
                                        utils as regutils, objects)
from nitorch import io, spatial
from nitorch.core import utils, py, dtypes
import torch
import sys
import os
import json
import warnings
import copy
import math as pymath


def cli(args=None):
    f"""Command-line interface for `register`
    
    {help[1]}
    
    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help[1])
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['register'] = cli


def _cli(args):
    """Command-line interface for `register` without exception handling"""
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help[options.help])
        return

    if options.verbose > 3:
        print(options)
        print('')

    _main(options)


def _map_image(fnames, dim=None):
    """
    Load a N-D image from disk.
    Returns:
        image : (C, *spatial) MappedTensor
        affine: (D+1, D+1) tensor
    """
    affine = None
    imgs = []
    for fname in fnames:
        img = io.map(fname)
        if affine is None:
            affine = img.affine
        if dim is None:
            dim = img.affine.shape[-1] - 1
        # img = img.fdata(rand=True, device=device)
        if img.dim > dim:
            img = img.movedim(-1, 0)
        else:
            img = img[None]
        img = img.unsqueeze(-1, dim + 1 - img.dim)
        if img.dim > dim + 1:
            raise ValueError(f'Don\'t know how to deal with an image of '
                             f'shape {tuple(img.shape)}')
        imgs.append(img)
        del img
    imgs = io.cat(imgs, dim=0)
    return imgs, affine


def _load_image(fnames, dim=None, device=None, label=False, missing=0):
    """
    Load a N-D image from disk
    Returns:
        dat : (C, *spatial) MappedTensor
        mask : (C, *spatial) MappedTensor
        affine: (D+1, D+1) tensor
    """
    dat, affine = _map_image(fnames, dim)
    if label:
        dtype = dat.dtype
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        dtype = dtypes.as_torch(dtype, upcast=True)
        dat0 = dat.data(device=device, dtype=dtype)[0]  # assume single channel
        if label is True:
            label = dat0.unique(sorted=True)
            label = label[label != 0].tolist()
        dat = torch.zeros([len(label), *dat0.shape], device=device)
        for i, l in enumerate(label):
            dat[i] = dat0 == l
        mask = None
    else:
        dat = dat.fdata(device=device, rand=True, missing=missing)
        mask = torch.isfinite(dat)
        mask = mask.all(dim=0)
        dat.masked_fill_(~mask, 0)
    affine = affine.to(dat.device, torch.float32)
    return dat, mask, affine


def _rescale_image(dat, quantiles):
    """Rescale an image between (0, 1) based on two quantiles"""
    dim = dat.dim() - 1
    if not isinstance(quantiles, (list, tuple)):
        quantiles = [quantiles]
    if len(quantiles) == 0:
        mn = 0
        mx = 95
    elif len(quantiles) == 1:
        mn = 0
        mx = quantiles[0]
    else:
        mn, mx = quantiles
    mx = mx / 100
    mn, mx = utils.quantile(dat, (mn, mx), dim=range(-dim, 0),
                            keepdim=True, bins=1024).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn)
    return dat


def _discretize_image(dat, nbins=256):
    """
    Discretize an image into a number of bins
    Input : (C, *spatial) tensor[float]
    Returns: (C, *spatial) tensor[long]
    """
    dim = dat.dim() - 1
    mn, mx = utils.quantile(dat, (0.0005, 0.9995), dim=range(-dim, 0), keepdim=True).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn).clamp_(0, 1).mul_(nbins-1)
    dat = dat.long()
    return dat


def _soft_quantize_image(dat, nbins=16):
    """
    Discretize an image into a number of bins
    Input : (1, *spatial) tensor[float]
    Returns: (C, *spatial) tensor[long]
    """
    dim = dat.dim() - 1
    dat = dat[0]
    mn, mx = utils.quantile(dat, (0.0005, 0.9995), dim=range(-dim, 0), keepdim=True).unbind(-1)
    dat = dat.sub_(mn).div_(mx - mn).clamp_(0, 1).mul_(nbins)
    centers = torch.linspace(0, nbins, nbins+1, **utils.backend(dat))
    centers = (centers[1:] + centers[:-1]) / 2
    centers = centers.flip(0)
    centers = centers[(Ellipsis,) + (None,) * dim]
    dat = (centers - dat).square().mul_(-2.355**2).exp_()
    dat /= dat.sum(0, keepdims=True)
    return dat


def _make_image(option, dim=None, device=None):
    """
    Load an image and build a Gaussian pyramid (if requireD)
    Returns: ImagePyramid
    """
    dat, mask, affine = _load_image(option.files, dim=dim, device=device,
                                    label=option.label, missing=option.missing)
    dim = dat.dim() - 1
    if option.mask:
        mask1 = mask
        mask, _, _ = _load_image([option.mask], dim=dim, device=device,
                                 label=option.label)
        if mask.shape[-dim:] != dat.shape[-dim:]:
            raise ValueError('Mask should have the same shape as the image. '
                             f'Got {mask.shape[-dim:]} and {dat.shape[-dim:]}')
        if mask1 is not None:
            mask = mask * mask1
        del mask1
    if option.world:  # overwrite orientation matrix
        affine = io.transforms.map(option.world).fdata().squeeze()
    for transform in (option.affine or []):
        transform = io.transforms.map(transform).fdata().squeeze()
        affine = spatial.affine_lmdiv(transform, affine)
    if not option.discretize and any(option.rescale):
        dat = _rescale_image(dat, option.rescale)
    if option.pad:
        pad = option.pad
        if isinstance(pad[-1], str):
            *pad, unit = pad
        else:
            unit = 'vox'
        if unit == 'mm':
            voxel_size = spatial.voxel_size(affine)
            pad = torch.as_tensor(pad, **utils.backend(voxel_size))
            pad = pad / voxel_size
            pad = pad.floor().int().tolist()
        else:
            pad = [int(p) for p in pad]
        pad = py.make_list(pad, dim)
        if any(pad):
            affine, _ = spatial.affine_pad(affine, dat.shape[-dim:], pad, side='both')
            dat = utils.pad(dat, pad, side='both', mode=option.bound)
            if mask is not None:
                mask = utils.pad(mask, pad, side='both', mode=option.bound)
    if option.fwhm:
        fwhm = option.fwhm
        if isinstance(fwhm[-1], str):
            *fwhm, unit = fwhm
        else:
            unit = 'vox'
        if unit == 'mm':
            voxel_size = spatial.voxel_size(affine)
            fwhm = torch.as_tensor(fwhm, **utils.backend(voxel_size))
            fwhm = fwhm / voxel_size
        dat = spatial.smooth(dat, dim=dim, fwhm=fwhm, bound=option.bound)
    image = objects.ImagePyramid(
        dat,
        levels=option.pyramid,
        affine=affine,
        dim=dim,
        bound=option.bound,
        mask=mask,
        extrapolate=option.extrapolate,
        method=option.pyramid_method
    )
    if getattr(option, 'soft_quantize', False) and len(image[0].dat) == 1:
        for level in image:
            level.preview = level.dat
            level.dat = _soft_quantize_image(level.dat, option.soft_quantize)
    elif not option.label and option.discretize:
        for level in image:
            level.preview = level.dat
            level.dat = _discretize_image(level.dat, option.discretize)
    return image


def _almost_identity(aff):
    """Return True if an affine is almost the identity matrix"""
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


def _warp_image(option, affine=None, nonlin=None, dim=None, device=None, odir=None):
    """Warp and save the moving and fixed images from a loss object"""

    if not (option.mov.output or option.mov.resliced or
            option.fix.output or option.fix.resliced):
        return

    fix, fix_affine = _map_image(option.fix.files, dim=dim)
    mov, mov_affine = _map_image(option.mov.files, dim=dim)
    fix_affine = fix_affine.float()
    mov_affine = mov_affine.float()
    dim = dim or (fix.dim - 1)

    if option.fix.world:  # overwrite orientation matrix
        fix_affine = io.transforms.map(option.fix.world).fdata().squeeze()
    for transform in (option.fix.affine or []):
        transform = io.transforms.map(transform).fdata().squeeze()
        fix_affine = spatial.affine_lmdiv(transform, fix_affine)

    if option.mov.world:  # overwrite orientation matrix
        mov_affine = io.transforms.map(option.mov.world).fdata().squeeze()
    for transform in (option.mov.affine or []):
        transform = io.transforms.map(transform).fdata().squeeze()
        mov_affine = spatial.affine_lmdiv(transform, mov_affine)

    # moving
    if option.mov.output or option.mov.resliced:
        ifname = option.mov.files[0]
        idir, base, ext = py.fileparts(ifname)
        odir_mov = odir or idir or '.'

        image = objects.Image(mov.fdata(rand=True, device=device), dim=dim,
                              affine=mov_affine, bound=option.mov.bound,
                              extrapolate=option.mov.extrapolate)

        if option.mov.output:
            target_affine = mov_affine
            target_shape = image.shape
            if affine and affine.position[0].lower() in 'ms':
                aff = affine.exp(recompute=True, cache_result=True)
                target_affine = spatial.affine_lmdiv(aff, target_affine)

            fname = option.mov.output.format(dir=odir_mov, base=base, sep=os.path.sep, ext=ext)
            print(f'Minimal reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

        if option.mov.resliced:
            target_affine = fix_affine
            target_shape = fix.shape[1:]

            fname = option.mov.resliced.format(dir=odir_mov, base=base, sep=os.path.sep, ext=ext)
            print(f'Full reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin, reslice=True)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

    # fixed
    if option.fix.output or option.fix.resliced:
        ifname = option.fix.files[0]
        idir, base, ext = py.fileparts(ifname)
        odir_fix = odir or idir or '.'

        image = objects.Image(fix.fdata(rand=True, device=device), dim=dim,
                              affine=fix_affine, bound=option.fix.bound,
                              extrapolate=option.fix.extrapolate)

        if option.fix.output:
            target_affine = fix_affine
            target_shape = image.shape
            if affine and affine.position[0].lower() in 'fs':
                aff = affine.exp(recompute=True, cache_result=True)
                target_affine = spatial.affine_matmul(aff, target_affine)

            fname = option.fix.output.format(dir=odir_fix, base=base, sep=os.path.sep, ext=ext)
            print(f'Minimal reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin, backward=True)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped

        if option.fix.resliced:
            target_affine = mov_affine
            target_shape = mov.shape[1:]

            fname = option.fix.resliced.format(dir=odir_fix, base=base, sep=os.path.sep, ext=ext)
            print(f'Full reslice: {ifname} -> {fname} ...', end=' ')
            warped = _warp_image1(image, target_affine, target_shape,
                                  affine=affine, nonlin=nonlin,
                                  backward=True, reslice=True)
            io.savef(warped, fname, like=ifname, affine=target_affine)
            print('done.')
            del warped


def _warp_image1(image, target, shape=None, affine=None, nonlin=None,
                 backward=False, reslice=False):
    """Returns the warped image, with channel dimension last"""
    # build transform
    aff_right = target
    aff_left = spatial.affine_inv(image.affine)
    aff = None
    if affine:
        # exp = affine.iexp if backward else affine.exp
        exp = affine.exp
        aff = exp(recompute=True, cache_result=True)
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
            tmp = spatial.affine_grid(aff_right, shape)
            phi = regutils.smart_pull_grid(phi, tmp).add_(tmp)
            del tmp
        if not _almost_identity(aff_left):
            phi = spatial.affine_matvec(aff_left, phi)
    else:
        # no nonlin: single affine even if position == 'symmetric'
        if reslice:
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


def setup_device(device='cpu', ndevice=0):
    if device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device, ndevice = 'cpu', None
    if device == 'cpu':
        device = torch.device('cpu')
        if ndevice:
            torch.set_num_threads(ndevice)
    else:
        assert device == 'gpu'
        if ndevice is not None:
            device = torch.device(f'cuda:{ndevice}')
        else:
            device = torch.device('cuda')
    return device


def _get_loss(loss, dim):
    """Instantiate the correct loss object based on a loss name"""
    if loss.name == 'mi':
        lossobj = losses.MI(bins=loss.bins, norm=loss.norm,
                            spline=loss.order, fwhm=loss.fwhm, dim=dim)
    elif loss.name == 'ent':
        lossobj = losses.Entropy(bins=loss.bins, spline=loss.order,
                                 fwhm=loss.fwhm, dim=dim)
    elif loss.name == 'mse':
        lossobj = losses.MSE(lam=loss.weight, dim=dim)
    elif loss.name == 'mad':
        lossobj = losses.MAD(lam=loss.weight, dim=dim)
    elif loss.name == 'tuk':
        lossobj = losses.Tukey(lam=loss.weight, dim=dim)
    elif loss.name == 'cc':
        lossobj = losses.CC(dim=dim)
    elif loss.name == 'lcc':
        lossobj = losses.LCC(patch=loss.patch, dim=dim, stride=loss.stride,
                             mode=loss.kernel)
    elif loss.name == 'gmm':
        lossobj = losses.GMMH(bins=loss.bins, dim=dim,
                              max_iter=loss.max_iter)
    elif loss.name == 'lgmm':
        lossobj = losses.LGMMH(bins=loss.bins, dim=dim,
                               max_iter=loss.max_iter,
                               patch=loss.patch,
                               stride=loss.stride,
                               mode=loss.kernel)
    elif loss.name == 'cat':
        lossobj = losses.Cat(dim=dim, log=False)
        # lossobj = losses.AutoCat()
    elif loss.name == 'dice':
        lossobj = losses.Dice(weighted=loss.weight, log=False)
    elif loss.name == 'prod':
        lossobj = losses.ProdLoss(dim=dim)
    elif loss.name == 'normprod':
        lossobj = losses.NormProdLoss(dim=dim)
    elif loss.name == 'sqz':
        lossobj = losses.SqueezedProdLoss(dim=dim, lam=loss.weight)
    elif loss.name == 'emmi':
        fwhm = None
        if not loss.fix.label:
            fwhm = loss.fix.discretize // 64
        lossobj = losses.EMMI(dim=dim, fwhm=fwhm)
    elif loss.name == 'extra':
        # Not a proper loss, we just want to warp these images at the end
        lossobj = None
    else:
        raise ValueError(loss.name)
    if loss.slicewise is not False:
        lossobj = losses.SliceWiseLoss(lossobj, loss.slicewise)
    return lossobj


def _ras_to_layout(x, affine):
    """Guess layout (e.g. "RAS") from an affine matrix"""
    layout = spatial.affine_to_layout(affine)
    ras_to_layout = layout[..., 0]
    return [x[i] for i in ras_to_layout]


def _patch(patch, affine, shape, level):
    """Compute the patch size in voxels"""
    dim = affine.shape[-1] - 1
    patch = py.make_list(patch)
    unit = 'pct'
    if isinstance(patch[-1], str):
        *patch, unit = patch
    patch = py.make_list(patch, dim)
    unit = unit.lower()
    if unit[0] == 'v':  # voxels
        patch = [float(p) / 2**level for p in patch]
    elif unit in ('m', 'mm', 'cm', 'um'):  # assume RAS orientation
        factor = (1e-3 if unit == 'um' else
                  1e1 if unit == 'cm' else
                  1e3 if unit == 'm' else
                  1)
        affine_ras = spatial.affine_reorient(affine, layout='RAS')
        vx_ras = spatial.voxel_size(affine_ras).tolist()
        patch = [factor * p / v for p, v in zip(patch, vx_ras)]
        patch = _ras_to_layout(patch, affine)
    elif unit[0] in 'p%':    # percentage of shape
        patch = [0.01 * p * s for p, s in zip(patch, shape)]
    else:
        raise ValueError('Unknown patch unit:', unit)

    # round down to zero small patch sizes
    patch = [0 if p < 1e-3 else p for p in patch]
    return patch


def _pyramid_levels1(vx, shape, opt):
    def is_valid(vx, shape):
        if opt.min_size and any(s < opt.min_size for s in shape):
            return False
        if opt.max_size and any(s > opt.max_size for s in shape):
            return False
        if opt.min_vx and any(v < opt.min_vx for v in vx):
            return False
        if opt.max_vx and any(v > opt.max_vx for v in vx):
            return False
        return True

    def is_max(vx, shape):
        if opt.min_size and any(s < opt.min_size for s in shape):
            return True
        if opt.max_vx and any(v > opt.max_vx for v in vx):
            return True
        if all(s == 1 for s in shape):
            return True
        return False

    full_pyramid = [(0, vx, shape)]
    valid_pyramid = []
    while True:
        level0, vx0, shape0 = full_pyramid[-1]
        if is_max(vx0, shape0):
            break
        if is_valid(vx0, shape0):
            valid_pyramid.append(full_pyramid[-1])
        vx1 = [v*2 for v in vx0]
        shape1 = [int(pymath.ceil(s/2)) for s in shape0]
        full_pyramid.append((level0+1, vx1, shape1))

    if not valid_pyramid:
        raise ValueError(f'Image with shape {shape} and voxel size {vx} '
                         f'does not fit in the pyramid.')
    return valid_pyramid


def _pyramid_levels(vxs, shapes, opt):
    """
    Map global pyramid levels to per-image pyramid levels.

    The idea is that we're trying to match resolutions as well as possible
    across images at each global pyramid level. So we may be matching
    the 3rd pyramid level of an image with the 5th pyramid level of another
    image.
    """
    dim = len(shapes[0])
    # first: compute approximate voxel size and shape at each level
    pyramids = [
        _pyramid_levels1(vx, shape, opt)
        for vx, shape in zip(vxs, shapes)
    ]
    # NOTE: pyramids = [[(level, vx, shape), ...], ...]

    # second: match voxel sizes across images
    vx0 = min([py.prod(pyramid[0][1]) for pyramid in pyramids])
    vx0 = pymath.log2(vx0**(1/dim))
    level_offsets = []
    for pyramid in pyramids:
        level, vx, shape = pyramid[0]
        vx1 = pymath.log2(py.prod(vx)**(1/dim))
        level_offsets.append(round(vx1 - vx0))

    # third: keep only levels that overlap across images
    max_level = min(o + len(p) for o, p in zip(level_offsets, pyramids))
    pyramids = [p[:max_level-o] for o, p in zip(level_offsets, pyramids)]
    if any(len(p) == 0 for p in pyramids):
        raise ValueError(f'Some images do not overlap in the pyramid.')

    # fourth: compute pyramid index of each image at each level
    select_levels = []
    for level in opt.levels:
        if isinstance(level, int):
            select_levels.append(level)
        else:
            select_levels.extend(list(level))

    map_levels = []
    for pyramid, offset in zip(pyramids, level_offsets):
        map_levels1 = [pyramid[0][0]] * offset
        for pyramid_level in pyramid:
            map_levels1.append(pyramid_level[0])
        if select_levels:
            map_levels1 = [map_levels1[l] for l in select_levels
                           if l < len(map_levels1)]
        map_levels.append(map_levels1)

    return map_levels


def _prepare_pyramid_levels(losses, opt, dim=None):
    """
    For each loss, compute the pyramid levels of `fix` and `mov` that
    must be computed.
    """
    if opt.name == 'none':
        return [{'fix': None, 'mov': None}] * len(losses)

    vxs = []
    shapes = []
    for loss in losses:
        # fixed image
        dat, affine = _map_image(loss.fix.files, dim)
        vx = dat.voxel_size[0].tolist()
        shape = dat.shape[:dim]
        if loss.fix.pad:
            pad = py.make_list(loss.fix.pad, len(shape))
            shape = [s + 2*p for s, p in zip(shape, pad)]
        vxs.append(vx)
        shapes.append(shape)
        # moving image
        dat, affine = _map_image(loss.mov.files, dim)
        vx = dat.voxel_size[0].tolist()
        shape = dat.shape[:dim]
        if loss.mov.pad:
            pad = py.make_list(loss.mov.pad, len(shape))
            shape = [s + 2*p for s, p in zip(shape, pad)]
        vxs.append(vx)
        shapes.append(shape)

    levels = _pyramid_levels(vxs, shapes, opt)
    levels = [{'fix': fix, 'mov': mov}
              for fix, mov in zip(levels[::2], levels[1::2])]
    return levels


def _sequential_pyramid(loss_list):
    fixs = []
    movs = []
    for loss in loss_list:
        fixs.append(loss.fixed)
        movs.append(loss.moving)
        loss.fixed = None
        loss.moving = None

    pyramid = []
    for i in range(len(fixs[0])):
        level = []
        for loss, fix, mov in zip(loss_list, fixs, movs):
            loss_level = copy.deepcopy(loss)
            loss_level.fixed = fix[i]
            loss_level.moving = mov[i]
            if hasattr(loss.loss, 'patch'):
                dim = fix[i].affine.shape[-1] - 1
                shape = fix[i].shape[-dim:]
                loss_level.loss.patch = _patch(
                    loss_level.loss.patch, fix[i].affine, shape, i)
            level.append(loss_level)
        pyramid.append(level)
    pyramid = pyramid[::-1]
    return pyramid


def _concurrent_pyramid(loss_list):
    fixs = []
    movs = []
    for loss in loss_list:
        fixs.append(loss.fixed)
        movs.append(loss.moving)
        loss.fixed = None
        loss.moving = None

    level = []
    for i in range(len(fixs[0])):
        for loss, fix, mov in zip(loss_list, fixs, movs):
            loss_level = copy.deepcopy(loss)
            loss_level.fixed = fix[i]
            loss_level.moving = mov[i]
            if hasattr(loss_level.loss, 'patch'):
                dim = fix[i].affine.shape[-1] - 1
                shape = fix[i].shape[-dim:]
                loss_level.loss.patch = _patch(
                    loss_level.loss.patch, fix[i].affine, shape, i)
            level.append(loss_level)
    return [level]


def _do_register(loss_list, affine, nonlin,
                 affine_optim, nonlin_optim, options):
    dim = 3
    line_size = 89 if nonlin else 74

    if options.pyramid.concurrent:
        loss_list = _concurrent_pyramid(loss_list)
    else:
        loss_list = _sequential_pyramid(loss_list)

    # ------------------------------------------------------------------
    #       INITIAL PROGRESSIVE AFFINE
    # ------------------------------------------------------------------
    if len(affine) > 1:
        print('-' * line_size)
        print(f'   PROGRESSIVE INITIALIZATION')
        print('-' * line_size)
        affines = affine
        affine_prev = None
        for i, affine in enumerate(affines):
            line_pad = line_size - len(affine.basis_name) - 5
            if affine_prev:
                n = len(affine_prev.dat.dat)
                affine.set_dat(dim=dim, device=affine_prev.dat.dat.device)
                affine.dat.dat[:n] = affine_prev.dat.dat
            if i == len(affines) - 1:
                break
            print(f'--- {affine.basis_name} ', end='')
            print('-' * max(0, line_pad))
            affine_optim.reset_state()
            register = pairwise.PairwiseRegister(loss_list[0], affine, None, affine_optim,
                                                 verbose=options.verbose,
                                                 framerate=options.framerate)
            register.fit()
            affine_prev = affine
    elif len(affine) == 1:
        affine = affine[0]

    # ------------------------------------------------------------------
    #       BUILD JOINT OPTIMIZER
    # ------------------------------------------------------------------

    if nonlin:
        if affine.dat and affine.dat.dat is not None:
            affine.dat.dat /= 2  # take the matrix square root
        if affine:
            if options.optim.name == 'sequential':
                joptim = optim.InterleavedOptimIterator(
                    [affine_optim, nonlin_optim], max_iter=1, tol=0)
            else:
                joptim = optim.InterleavedOptimIterator(
                    [affine_optim, nonlin_optim],
                    max_iter=options.optim.max_iter,
                    tol=options.optim.tolerance)
        else:
            joptim = nonlin_optim
    else:
        joptim = affine_optim

    # ------------------------------------------------------------------
    #       CONCURRENT PYRAMID
    # ------------------------------------------------------------------
    if options.pyramid.concurrent:
        joptim.reset_state()
        if nonlin and hasattr(nonlin_optim, 'factor'):
            nonlin_optim.factor /= py.prod(nonlin.shape)
        register = pairwise.PairwiseRegister(loss_list, affine, nonlin, joptim,
                                             verbose=options.verbose,
                                             framerate=options.framerate)
        register.fit()

    # ------------------------------------------------------------------
    #       SEQUENTIAL PYRAMID
    # ------------------------------------------------------------------
    else:
        n_level = nb_levels = len(loss_list)
        factor = None
        if nonlin and hasattr(nonlin_optim, 'factor'):
            factor = nonlin_optim.factor
        if nonlin and n_level > 1:
            vel_shape = nonlin.shape
            nonlin = nonlin.downsample_(2**n_level)
        while loss_list:
            loss_level = loss_list.pop(0)
            n_level -= 1
            print('-' * line_size)
            print(f'   PYRAMID LEVEL {n_level}')
            print('-' * line_size)
            if nonlin and nb_levels > 1:
                if n_level == 0:
                    nonlin.upsample_(shape=vel_shape, interpolation=3)
                else:
                    nonlin.upsample_()
            if nonlin and factor is not None:
                nonlin_optim.factor = factor / py.prod(nonlin.shape)
            joptim.reset_state()
            register = pairwise.PairwiseRegister(loss_level, affine, nonlin, joptim,
                                                 verbose=options.verbose,
                                                 framerate=options.framerate)
            register.fit()


def _build_losses(options, pyramids, device):
    dim = 3

    image_dict = {}
    loss_list = []
    for loss, pyramid in zip(options.loss, pyramids):
        lossobj = _get_loss(loss, dim)
        if not lossobj:
            # not a proper loss
            continue
        if not loss.fix.rescale[-1]:
            loss.fix.rescale = (0, 0)
        if not loss.mov.rescale[-1]:
            loss.mov.rescale = (0, 0)
        if loss.name in ('cat', 'dice'):
            loss.fix.rescale = (0, 0)
            loss.mov.rescale = (0, 0)
        if loss.name == 'emmi':
            loss.mov.rescale = (0, 0)
            loss.fix.discretize = loss.fix.discretize or 256
            loss.mov.soft_quantize = loss.mov.discretize or 16
            loss.mov.missing = []
        loss.fix.pyramid = pyramid['fix']
        loss.mov.pyramid = pyramid['mov']
        loss.fix.pyramid_method = options.pyramid.name
        loss.mov.pyramid_method = options.pyramid.name
        fix = _make_image(loss.fix, dim=options.dim, device=device)
        mov = _make_image(loss.mov, dim=options.dim, device=device)
        image_dict[loss.fix.name or loss.fix.files[0]] = fix
        image_dict[loss.mov.name or loss.mov.files[0]] = mov

        # Forward loss
        factor = loss.factor / (2 if loss.symmetric else 1)
        lossobj = objects.Similarity(lossobj, mov, fix, factor=factor)
        loss_list.append(lossobj)

        # Backward loss
        if loss.symmetric:
            lossobj = _get_loss(loss, dim)
            if loss.name != 'emmi':
                lossobj = objects.Similarity(
                    lossobj, fix, mov, factor=factor, backward=True)
            else:
                loss.fix, loss.mov = loss.mov, loss.fix
                loss.mov.rescale = (0, 0)
                loss.fix.discretize = loss.fix.discretize or 256
                loss.mov.soft_quantize = loss.mov.discretize or 16
                lossobj = objects.Similarity(
                    lossobj, mov, fix, factor=factor, backward=True)
            loss_list.append(lossobj)

    return loss_list, image_dict


def _build_affine(options, can_use_2nd_order, ref_affine=None):
    affine = []
    affine_optim = None
    if options.affine:
        if options.affine.is2d is not False:
            make_affine = lambda name: objects.Affine2dModel(
                name, options.affine.is2d, factor=options.affine.factor,
                ref_affine=ref_affine, position=options.affine.position)
        else:
            make_affine = lambda name: objects.AffineModel(
                name, options.affine.factor, position=options.affine.position)
        name = options.affine.name
        while name:
            affine = [make_affine(name), *affine]
            if not options.affine.progressive:
                break
            if name == 'affine':
                name = 'similitude'
            elif name == 'similitude':
                name = 'rigid'
            else:
                name = ''
        max_iter = options.affine.optim.max_iter
        if not max_iter:
            if options.nonlin and options.optim.name == 'interleaved':
                max_iter = 50
            else:
                max_iter = 100
        if options.affine.optim.name == 'unset':
            if can_use_2nd_order:
                options.affine.optim.name = 'gn'
            else:
                options.affine.optim.name = 'lbfgs'
        if options.affine.optim.name == 'gd':
            affine_optim = optim.GradientDescent(lr=options.affine.optim.lr)
        elif options.affine.optim.name == 'cg':
            affine_optim = optim.ConjugateGradientDescent(
                lr=options.affine.optim.lr,
                beta=options.affine.optim.beta)
        elif options.affine.optim.name == 'mom':
            affine_optim = optim.Momentum(lr=options.affine.optim.lr,
                                          momentum=options.affine.optim.momentum)
        elif options.affine.optim.name == 'nes':
            affine_optim = optim.Nesterov(lr=options.affine.optim.lr,
                                          momentum=options.affine.optim.momentum,
                                          auto_restart=options.affine.optim.restart)
        elif options.affine.optim.name == 'ogm':
            affine_optim = optim.OGM(lr=options.affine.optim.lr,
                                     momentum=options.affine.optim.momentum,
                                     relax=options.affine.optim.relax,
                                     auto_restart=options.affine.optim.restart)
        elif options.affine.optim.name == 'gn':
            affine_optim = optim.GaussNewton(lr=options.affine.optim.lr,
                                             marquardt=getattr(
                                                 options.affine.optim,
                                                 'marquardt', None))
        elif options.affine.optim.name == 'lbfgs':
            affine_optim = optim.LBFGS(lr=options.affine.optim.lr,
                                       history=getattr(options.affine.optim,
                                                       'history', 100))
            # TODO: tolerance?
        elif options.affine.optim.name == 'pow':
            affine_optim = optim.Powell(lr=options.affine.optim.lr)
        else:
            raise ValueError(options.affine.optim.name)
        if options.affine.optim.line_search \
                and not isinstance(affine_optim, (optim.Powell, optim.LBFGS)):
            affine_optim.search = options.affine.optim.line_search
        affine_optim.iter = optim.OptimIterator(
            max_iter=max_iter, tol=options.affine.optim.tolerance,
            stop=options.affine.optim.crit)

    return affine, affine_optim


def _build_nonlin(options, can_use_2nd_order, affine, image_dict, ref_affine=None):
    dim = 3
    device = next(iter(image_dict.values())).dat.device

    nonlin = None
    nonlin_optim = None
    if options.nonlin:
        # build mean space
        vx = options.nonlin.voxel_size
        if isinstance(vx[-1], str):
            *vx, vx_unit = vx
        else:
            vx_unit = 'mm'
        pad = options.nonlin.pad
        if isinstance(pad[-1], str):
            *pad, pad_unit = pad
        else:
            pad_unit = '%'
        vx = py.make_list(vx, dim)
        pad = py.make_list(pad, dim)
        space = objects.MeanSpace(
            [image_dict[key] for key in (options.nonlin.fov or image_dict)],
            voxel_size=vx, vx_unit=vx_unit, pad=pad, pad_unit=pad_unit)
        prm = dict(absolute=options.nonlin.absolute,
                   membrane=options.nonlin.membrane,
                   bending=options.nonlin.bending,
                   lame=options.nonlin.lame)

        vel = objects.Displacement(space.shape, affine=space.affine, dim=dim,
                                   device=device)
        Model = objects.NonLinModel.subclass(options.nonlin.name)
        nonlin = Model(dat=vel, factor=options.nonlin.factor,
                       penalty=prm, steps=getattr(options.nonlin, 'steps', None))
        if options.nonlin.is2d is not False:
            nonlin = objects.Nonlin2dModel(
                nonlin, options.nonlin.is2d, ref_affine=ref_affine)

        max_iter = options.nonlin.optim.max_iter
        if not max_iter:
            if affine and options.optim.name == 'interleaved':
                max_iter = 10
            else:
                max_iter = 50
        if options.nonlin.optim.name == 'unset':
            if can_use_2nd_order:
                options.nonlin.optim.name = 'gn'
            else:
                options.nonlin.optim.name = 'lbfgs'
        if options.nonlin.optim.name == 'gd':
            nonlin_optim = optim.GradientDescent(lr=options.nonlin.optim.lr)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'cg':
            nonlin_optim = optim.ConjugateGradientDescent(lr=options.nonlin.optim.lr,
                                                          beta=options.nonlin.optim.beta)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'mom':
            nonlin_optim = optim.Momentum(lr=options.nonlin.optim.lr,
                                          momentum=options.nonlin.optim.momentum)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'nes':
            nonlin_optim = optim.Nesterov(lr=options.nonlin.optim.lr,
                                          momentum=options.nonlin.optim.momentum,
                                          auto_restart=options.nonlin.optim.restart)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'ogm':
            nonlin_optim = optim.OGM(lr=options.nonlin.optim.lr,
                                     momentum=options.nonlin.optim.momentum,
                                     relax=options.nonlin.optim.relax,
                                     auto_restart=options.nonlin.optim.restart)
            nonlin_optim.preconditioner = nonlin.greens_apply
        elif options.nonlin.optim.name == 'gn':
            marquardt = getattr(options.nonlin.optim, 'marquardt', None)
            sub_iter = getattr(options.nonlin.optim, 'sub_iter', None)
            if not sub_iter:
                if options.nonlin.optim.fmg:
                    sub_iter = 2
                else:
                    sub_iter = 16
            prm = {'factor': nonlin.factor,
                   'voxel_size': nonlin.voxel_size,
                   **nonlin.penalty}
            if getattr(options.nonlin.optim, 'solver', 'cg') == 'cg':
                nonlin_optim = optim.GridCG(
                    lr=options.nonlin.optim.lr,
                    marquardt=marquardt,
                    max_iter=sub_iter,
                    **prm)
            elif getattr(options.nonlin.optim, 'solver') == 'relax':
                nonlin_optim = optim.GridRelax(lr=options.nonlin.optim.lr,
                                               marquardt=marquardt,
                                               max_iter=sub_iter,
                                               **prm)
            else:
                raise ValueError(getattr(options.nonlin.optim, 'solver'))
        elif options.nonlin.optim.name == 'lbfgs':
            nonlin_optim = optim.LBFGS(
                lr=options.nonlin.optim.lr,
                history=getattr(options.nonlin.optim, 'history'))
            nonlin_optim.preconditioner = nonlin.greens_apply
            # TODO: tolerance?
        else:
            raise ValueError(options.nonlin.optim.name)
        if options.nonlin.optim.line_search:
            nonlin_optim.search = options.nonlin.optim.line_search
        nonlin_optim.iter = optim.OptimIterator(
            max_iter=max_iter, tol=options.nonlin.optim.tolerance)

    return nonlin, nonlin_optim


def _main(options):
    device = setup_device(*options.device)
    dim = 3

    # ------------------------------------------------------------------
    #                       COMPUTE PYRAMID
    # ------------------------------------------------------------------
    pyramids = _prepare_pyramid_levels(options.loss, options.pyramid, dim)

    # ------------------------------------------------------------------
    #                       BUILD LOSSES
    # ------------------------------------------------------------------
    loss_list, image_dict = _build_losses(options, pyramids, device)

    can_use_2nd_order = all(loss.loss.order >= 2 for loss in loss_list)

    # ------------------------------------------------------------------
    #                           BUILD AFFINE
    # ------------------------------------------------------------------
    affine, affine_optim = _build_affine(options, can_use_2nd_order,
                                         loss_list[0].fixed.affine)

    # ------------------------------------------------------------------
    #                           BUILD DENSE
    # ------------------------------------------------------------------
    nonlin, nonlin_optim = _build_nonlin(options, can_use_2nd_order,
                                         affine, image_dict,
                                         loss_list[0].fixed.affine)

    if not affine and not nonlin:
        raise ValueError('At least one of @affine or @nonlin must be used.')

    # ------------------------------------------------------------------
    #                           BACKEND STUFF
    # ------------------------------------------------------------------
    if options.verbose > 1:
        import matplotlib
        matplotlib.use('TkAgg')

    # local losses may benefit from selecting the best conv
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    #                      PERFORM REGISTRATION
    # ------------------------------------------------------------------
    _do_register(loss_list, affine, nonlin,
                 affine_optim, nonlin_optim, options)

    # ------------------------------------------------------------------
    #                           WRITE RESULTS
    # ------------------------------------------------------------------
    if affine:
        affine = affine[-1]

    if affine and options.affine.output:
        odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'
        fname = options.affine.output.format(dir=odir, sep=os.path.sep,
                                             name=options.affine.name)
        print('Affine ->', fname)
        aff = affine.exp(cache_result=True, recompute=True)
        io.transforms.savef(aff.cpu(), fname, type=1)  # 1 = RAS_TO_RAS
    if nonlin and options.nonlin.output:
        odir = options.odir or py.fileparts(options.loss[0].fix.files[0])[0] or '.'
        fname = options.nonlin.output.format(dir=odir, sep=os.path.sep,
                                             name=options.nonlin.name)
        io.savef(nonlin.dat.dat, fname, affine=nonlin.affine)
        if isinstance(nonlin, objects.ShootModel):
            nldir, nlbase, _ = py.fileparts(fname)
            fname = os.path.join(nldir, nlbase + '.json')
            with open(fname, 'w') as f:
                prm = dict(nonlin.penalty)
                prm['factor'] = nonlin.factor / py.prod(nonlin.shape)
                json.dump(prm, f)
        print('Nonlin ->', fname)
    for loss in options.loss:
        _warp_image(loss, affine=affine, nonlin=nonlin,
                    dim=dim, device=device, odir=options.odir)

