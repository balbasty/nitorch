__all__ = ['pyramid_levels', 'valid_pyramid_levels',
           'sequential_pyramid', 'concurrent_pyramid',
           'compute_patch_size']

from nitorch.core.py import make_list, prod
from nitorch import spatial
from .objects import SumSimilarity, Similarity
from .losses import SliceWiseLoss
import copy
import math as pymath


def pyramid_levels(vxs, shapes, levels, **opt):
    """
    Map global pyramid levels to per-image pyramid levels.

    The idea is that we're trying to match resolutions as well as possible
    across images at each global pyramid level. So we may be matching
    the 3rd pyramid level of an image with the 5th pyramid level of another
    image.

    Parameters
    ----------
    vxs : list[list[float]]
        All voxel sizes
    shapes : list[list[int]]
        All shapes
    levels : list[int]
        Global pyramid levels that should be computed
    min_size, max_size : int, optional
        Minimum and maximum acceptable size, in voxels
    min_vx, max_vx : float
        Minimum and maximum acceptable voxel size, in mm

    Returns
    -------
    levels : list[list[int]]
        The inner loop corresponds to global levels effectively used
        during registration. The outer loop contains the image-specific
        pyramid level that corresponds to this global level, for each image.

    """
    dim = len(shapes[0])
    # first: compute approximate voxel size and shape at each level
    pyramids = [
        valid_pyramid_levels(vx, shape, **opt)
        for vx, shape in zip(vxs, shapes)
    ]
    # NOTE: pyramids = [[(level, vx, shape), ...], ...]

    # second: match voxel sizes across images
    vx0 = min([prod(pyramid[0][1]) for pyramid in pyramids])
    vx0 = pymath.log2(vx0**(1/dim))
    level_offsets = []
    for pyramid in pyramids:
        level, vx, shape = pyramid[0]
        vx1 = pymath.log2(prod(vx)**(1/dim))
        level_offsets.append(round(vx1 - vx0))

    # third: keep only levels that overlap across images
    max_level = min(o + len(p) for o, p in zip(level_offsets, pyramids))
    pyramids = [p[:max_level-o] for o, p in zip(level_offsets, pyramids)]
    if any(len(p) == 0 for p in pyramids):
        raise ValueError(f'Some images do not overlap in the pyramid.')

    # fourth: compute pyramid index of each image at each level
    select_levels = []
    levels = make_list(levels)
    for level in levels:
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


def valid_pyramid_levels(vx, shape, min_size=None, max_size=None,
                         min_vx=None, max_vx=None, **opt):
    """Compute the shape and voxel size of all valid pyramid levels

    Parameters
    ----------
    vx : list[float]
        Voxel size at level 0
    shape : list[int[
        Shape at level 0
    min_size, max_size : int, optional
        Minimum and maximum acceptable size, in voxels
    min_vx, max_vx : float
        Minimum and maximum acceptable voxel size, in mm

    Returns
    -------
    valid_pyramid : list[(level, vx, shape)]
        Valid pyramid levels and their properties

    """
    def is_valid(vx, shape):
        if min_size and any(s < min_size for s in shape):
            return False
        if max_size and any(s > max_size for s in shape):
            return False
        if min_vx and any(v < min_vx for v in vx):
            return False
        if max_vx and any(v > max_vx for v in vx):
            return False
        return True

    def is_max(vx, shape):
        if min_size and any(s < min_size for s in shape):
            return True
        if max_vx and any(v > max_vx for v in vx):
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


def sequential_pyramid(loss):
    """Transform a loss (or losses) into a pyramid of losses, whose
    levels are fitted sequentially.

    Parameters
    ----------
    loss : objects.Similarity
        A Similarity object, where the images are ImagePyramid objects.

    Returns
    -------
    loss : list[objects.Similarity]
        A pyramid of losses, where the images in each loss are Image objects.

    """
    class EndOfPyramid(Exception):
        pass

    def get_level(loss, level):
        moving = fixed = None
        if hasattr(loss, 'fixed'):
            fixed, loss.fixed = loss.fixed, None
        if hasattr(loss, 'moving'):
            moving, loss.moving = loss.moving, None
        new_loss = copy.deepcopy(loss)
        if hasattr(new_loss, 'fixed'):
            loss.fixed = fixed
        if hasattr(new_loss, 'moving'):
            loss.moving = moving
        if hasattr(new_loss, 'fixed'):
            if level >= len(fixed):
                raise EndOfPyramid
            new_loss.fixed = fixed[level]
        if hasattr(new_loss, 'moving'):
            if level >= len(moving):
                raise EndOfPyramid
            new_loss.moving = moving[level]
        if hasattr(loss, 'loss') and hasattr(loss.loss, 'patch'):
            dim = new_loss.fixed.affine.shape[-1] - 1
            shape = new_loss.fixed.shape[-dim:]
            new_loss.loss.patch = compute_patch_size(
                new_loss.loss.patch, new_loss.fixed.affine, shape, level)
        if hasattr(loss, 'loss') and isinstance(loss.loss, SliceWiseLoss):
            slice_loss = loss.loss
            tmp_loss = Similarity(loss.loss.base_loss, loss.moving, loss.fixed)
            tmp_loss = get_level(tmp_loss, level)
            slice_loss = SliceWiseLoss(tmp_loss.loss, slice_loss.axis)
            new_loss = Similarity(slice_loss, tmp_loss.moving, tmp_loss.fixed)
        if isinstance(new_loss, SumSimilarity):
            for i in range(len(new_loss)):
                new_loss[i] = get_level(new_loss[i], level)
        return new_loss

    pyramid = []
    level = 0
    while True:
        try:
            pyramid.append(get_level(loss, level))
            level += 1
        except EndOfPyramid:
            break
    pyramid = pyramid[::-1]
    return pyramid


def concurrent_pyramid(loss):
    """Transform a loss into a pyramid of losses, whose
    levels are fitted concurrently.

    Parameters
    ----------
    loss : objects.Similarity
        A Similarity object, where the images are ImagePyramid objects.

    Returns
    -------
    loss : objects.SumSimilarity
        A Similarity object, where the images are Image objects.

    """
    def get_level(loss, level):
        new_loss = copy.copy(loss)
        if hasattr(new_loss, 'fixed'):
            new_loss.fixed = new_loss.fixed[level]
        if hasattr(new_loss, 'moving'):
            new_loss.moving = new_loss.moving[level]
        if hasattr(loss, 'loss') and hasattr(loss.loss, 'patch'):
            dim = new_loss.fixed.affine.shape[-1] - 1
            shape = new_loss.fixed.shape[-dim:]
            new_loss.loss.patch = compute_patch_size(
                new_loss.loss.patch, new_loss.fixed.affine, shape, level)
        if hasattr(new_loss, '__len__'):
            for i in range(len(new_loss)):
                new_loss[i] = get_level(new_loss[i], level)
        return new_loss

    pyramid = []
    level = 0
    while True:
        try:
            pyramid.append(get_level(loss, level))
            level += 1
        except Exception:
            break
    pyramid = SumSimilarity.sum(pyramid) / (level + 1)
    return pyramid


def ras_to_layout(x, affine):
    """
    Guess layout from an affine matrix and reorder a "RAS" input list
    so that it matches this layout.
    """
    layout = spatial.affine_to_layout(affine)
    ras_to_layout = layout[..., 0]
    return [x[i] for i in ras_to_layout]


def compute_patch_size(patch, affine, shape, level):
    """Compute the patch size in voxels"""
    dim = affine.shape[-1] - 1
    patch = make_list(patch)
    unit = 'pct'
    if isinstance(patch[-1], str):
        *patch, unit = patch
    patch = make_list(patch, dim)
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
        patch = ras_to_layout(patch, affine)
    elif unit[0] in 'p%':    # percentage of shape
        patch = [0.01 * p * s for p, s in zip(patch, shape)]
    else:
        raise ValueError('Unknown patch unit:', unit)

    # round down to zero small patch sizes
    patch = [0 if p < 1e-3 else p for p in patch]
    return patch
