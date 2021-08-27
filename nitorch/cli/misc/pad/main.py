from nitorch import spatial, io
from nitorch.core import py, utils, dtypes
import torch
import os


def _pad_to_param(aff0, shape0, aff, shape):
    dim = aff0.shape[-1] - 1
    shape = shape[:dim]
    shape0 = shape0[:dim]
    layout0 = spatial.affine_to_layout(aff0)
    layout = spatial.affine_to_layout(aff)
    if (layout0 != layout).any():
        raise ValueError('Input and Ref do not have the same layout: '
                         f'{spatial.volume_layout_to_name(layout0)} vs '
                         f'{spatial.volume_layout_to_name(layout)}.')
    layout = None
    unit = 'vox'

    shape = torch.as_tensor(shape, dtype=torch.float)
    shape0 = torch.as_tensor(shape0, dtype=torch.float)
    center = shape * 0.5
    like_aff = spatial.affine_lmdiv(aff0, aff)
    center = spatial.affine_matvec(like_aff, center)
    center0 = shape0 * 0.5
    alpha = 2 * (center - center0)
    beta = shape - shape0
    size_lower = (0.5 * (beta - alpha)).ceil().long().tolist()
    size_upper = (0.5 * (beta + alpha)).floor().long().tolist()
    size = list(zip(size_lower, size_upper))

    return size, unit, layout


def pad(inp, size=None, space='vx', like=None, bound=0.,
        output=None, transform=None):
    """Pad a ND volume, while preserving the orientation matrices.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    size : [sequence of] int, optional
        Amount of padding to perform.
    space : [sequence of] {'vox', 'ras'}, default='vox'
        The space in which the `size` and `center` parameters are expressed.
    like : str or (tensor, tensor), optional
        Reference patch.
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    bound : {'replicate', 'dct', 'dct2', 'dft'} or Number, default=0
        Boundary condition.
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

    # --- Open input ---
    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        dt = dtypes.dtype(f.dtype).torch_upcast
        inp = (f.data(dtype=dt), f.affine)
        if output is None:
            output = '{dir}{sep}{base}.pad{ext}'
        dir, base, ext = py.fileparts(fname)
    dat, aff0 = inp
    dim = aff0.shape[-1] - 1
    shape0 = dat.shape[:dim]

    if bool(size) + bool(like) > 1:
        raise ValueError('Can only use one of `size` and `like``.')

    # --- Open reference and compute size/center ---
    if like:
        like_is_file = isinstance(like, str)
        if like_is_file:
            f = io.volumes.map(like)
            like = (f.shape, f.affine)
        like_shape, like_aff = like
        if torch.is_tensor(like_shape):
            like_shape = like_shape.shape
        size, unit, layout = _pad_to_param(aff0, shape0, like_aff, like_shape)

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
        size, unit, layout = _pad_to_param(aff0, shape0, like_aff, like_shape)

    # --- prepare size ---
    size = py.make_list(size, dim)
    size = [py.make_tuple(s, 2) for s in size]
    size = [val for pair in size for val in pair]

    # --- convert size to voxels ---
    size = utils.make_vector(size, dtype=torch.long)
    if space.lower() == 'ras':
        perm = spatial.affine_to_layout(aff0)[:, 0]
        size = size[perm.long()]
        size = size / spatial.voxel_size(aff0)
        size = size.ceil().long().tolist()
    size = size.tolist()

    verb = f'Padding volume with shape {list(shape0[:dim])} by ['
    verb += ', '.join([f'({f}, {l})' for f, l in zip(size[::2], size[1::2])])
    verb += f'] voxels'
    print(verb)

    # --- do crop ---
    aff, _ = spatial.affine_pad(aff0, shape0[:dim], size)
    size += [0] * (2*(dat.dim() - dim))
    mode = bound if isinstance(bound, str) else 'constant'
    dat = utils.pad(dat, size, mode=mode, value=bound)
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
        trf.set_source_space(aff0, shape0)
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
