from nitorch import spatial, io
from nitorch.core import py, utils
import torch
import os


def _crop_to_param(aff0, aff, shape):
    dim = aff0.shape[-1] - 1
    shape = shape[:dim]
    layout0 = spatial.affine_to_layout(aff0)
    layout = spatial.affine_to_layout(aff)
    if (layout0 != layout).any():
        raise ValueError('Input and Ref do not have the same layout: '
                         f'{spatial.volume_layout_to_name(layout0)} vs '
                         f'{spatial.volume_layout_to_name(layout)}.')
    size = shape
    layout = None
    unit = 'vox'

    center = torch.as_tensor(shape, dtype=torch.float).sub_(1).mul_(0.5)
    like_aff = spatial.affine_lmdiv(aff0, aff)
    center = spatial.affine_matvec(like_aff, center)

    return size, center, unit, layout


def crop(inp, size=None, center=None, space='vx', like=None, bbox=False,
         output=None, transform=None):
    """Crop a ND volume, while preserving the orientation matrices.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    size : [sequence of] int, optional
        Size of the patch to extract.
        Its unit and axes are defined by `units` and `layout`.
    center : [sequence of] int, optional
        Coordinate of the center of the patch.
        Its unit and axes are defined by `units` and `layout`.
        By default, the center of the FOV is used.
    space : [sequence of] {'vox', 'ras'}, default='vox'
        The space in which the `size` and `center` parameters are expressed.
    bbox : bool or float, default=False
        Crop at the bounding box of `inp > threshold`.
            If `bbox` is a float, it is the threshold to use.
            If `bbox` is `True`, the threshold is 0.
    like : str or (tensor, tensor), optional
        Reference patch.
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    output : [sequence of] str, optional
        Output filename(s).
        If the input is not a path, the unstacked data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.{i}{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file,
        `i` is the coordinate (starting at 1) of the slice.
    transform : [sequence of] str, optional
        Input or output filename(s) of the corresponding transforms.
        Not written by default.
        If a transform is provided and all other parameters
        (i.e., `size` and `like`) are None, the transform is considered
        as an input transform to apply.

    Returns
    -------
    output : list[str or (tensor, tensor)]
        If the input is a path, the output paths are returned.
        Else, the unstacked data and orientation matrices are returned.

    """
    dir = ''
    base = ''
    ext = ''
    fname = None
    transform_in = False
    use_bbox = bool(bbox or isinstance(bbox, float))

    # --- Open input ---
    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        inp = (f.data(numpy=True) if use_bbox else f, f.affine)
        if output is None:
            output = '{dir}{sep}{base}.crop{ext}'
        dir, base, ext = py.fileparts(fname)
    dat, aff0 = inp
    dim = aff0.shape[-1] - 1
    shape0 = dat.shape[:dim]
    layout0 = spatial.affine_to_layout(aff0)

    # save input space in case we reorient later
    aff00 = aff0
    shape00 = shape0

    if bool(size) + bool(like) + bool(bbox or isinstance(bbox, float)) > 1:
        raise ValueError('Can only use one of `size`, `like` and `bbox`.')

    # --- Open reference and compute size/center ---
    if like:
        like_is_file = isinstance(like, str)
        if like_is_file:
            f = io.volumes.map(like)
            like = (f.shape, f.affine)
        like_shape, like_aff = like
        like_layout = spatial.affine_to_layout(like_aff)
        if (layout0 != like_layout).any():
            aff0, dat = spatial.affine_reorient(aff0, dat, like_layout)
            shape0 = dat.shape[:dim]
        if torch.is_tensor(like_shape):
            like_shape = like_shape.shape
        size, center, unit, layout = _crop_to_param(aff0, like_aff, like_shape)
        space = 'vox'

    elif bbox or isinstance(bbox, float):
        if bbox is True:
            bbox = 0.
        mask = torch.as_tensor(dat > bbox)
        while mask.dim() > 3:
            mask = mask.any(dim=-1)
        mins = []
        maxs = []
        for d in range(dim):
            n = mask.shape[d]
            idx = utils.movedim(mask, d, 0).reshape([n, -1]).any(-1).nonzero(as_tuple=False)
            mins.append(idx.min())
            maxs.append(idx.max())
        mins = utils.as_tensor(mins)
        maxs = utils.as_tensor(maxs)
        size = maxs + 1 - mins
        center = (maxs + 1 + mins).float()/2
        space = 'vox'
        del mask

    # --- Open transformation file and compute size/center ---
    elif not size:
        if not transform:
            raise ValueError('At least one of size/like/transform must '
                             'be provided')
        transform_in = True
        t = io.transforms.map(transform)
        if not isinstance(t, io.transforms.LinearTransformArray):
            raise TypeError('Expected an LTA file')
        like_aff, like_shape = t.destination_space()
        size, center, unit, layout = _crop_to_param(aff0, like_aff, like_shape)

    # --- use center of the FOV ---
    if not torch.is_tensor(center) and not center:
        center = torch.as_tensor(shape0[:dim], dtype=torch.float)
        center = center.sub_(1).mul_(0.5)

    # --- convert size/center to voxels ---
    size = utils.make_vector(size, dim, dtype=torch.long)
    center = utils.make_vector(center, dim, dtype=torch.float)
    space_size, space_center = py.make_list(space, 2)
    if space_center.lower() == 'ras':
        center = spatial.affine_matvec(spatial.affine_inv(aff0), center)
    if space_size.lower() == 'ras':
        perm = spatial.affine_to_layout(aff0)[:, 0]
        size = size[perm.long()]
        size = size / spatial.voxel_size(aff0)

    # --- compute first/last ---
    center = center.float()
    size = (size.ceil() if size.dtype.is_floating_point else size).long()
    first = center - size.float().sub_(1).mul_(0.5)
    first = first.round().long()
    last = (first + size).tolist()
    first = [max(f, 0) for f in first.tolist()]
    last = [min(l, s) for l, s in zip(last, shape0[:dim])]
    verb = 'Cropping patch ['
    verb += ', '.join([f'{f}:{l}' for f, l in zip(first, last)])
    verb += f'] from volume with shape {shape0[:dim]}'
    print(verb)
    slicer = tuple(slice(f, l) for f, l in zip(first, last))

    # --- do crop ---
    if use_bbox:
        dat = dat.numpy()
    dat = dat[slicer]
    if not torch.is_tensor(dat):
        dat = dat.data(numpy=True)
    aff, _ = spatial.affine_sub(aff0, shape0[:dim], slicer)
    shape = dat.shape[:dim]

    if output:
        if is_file:
            output = output.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep)
            io.volumes.save(dat, output, like=fname, affine=aff)
        else:
            output = output.format(sep=os.path.sep)
            io.volumes.save(dat, output, affine=aff)

    if transform and not transform_in:
        if is_file:
            transform = transform.format(dir=dir or '.', base=base, ext=ext,
                                         sep=os.path.sep)
        else:
            transform = transform.format(sep=os.path.sep)
        trf = io.transforms.LinearTransformArray(transform, 'w')
        trf.set_source_space(aff00, shape00)
        trf.set_destination_space(aff, shape)
        trf.set_metadata({'src': {'filename': fname},
                          'dst': {'filename': output},
                          'type': 1})  # RAS_TO_RAS
        trf.set_fdata(torch.eye(4))
        trf.save()

    if is_file:
        return output
    else:
        return dat, aff
