from nitorch import spatial, io
from nitorch.core import py, utils
import torch
import os


def orient(inp, layout=None, voxel_size=None, center=None, like=None, output=None):
    """Overwrite the orientation matrix

    Parameters
    ----------
    inp : str or (tuple, tensor)
        Either a path to a volume file or a tuple `(shape, affine)`, where
        the first element contains the volume shape and the second contains
        the orientation matrix.
    layout : str or layout-like, default=None (= preserve)
        Target orientation.
    voxel_size : [sequence of] float, default=None (= preserve)
        Target voxel size.
    center : [sequence of] float, default=None (= preserve)
        World coordinate of the center of the field of view.
    like : str or (tuple, tensor)
        Either a path to a volume file or a tuple `(shape, affine)`, where
        the first element contains the volume shape and the second contains
        the orientation matrix.
    output : str, optional
        Output filename.
        If the input is not a path, the reoriented data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.{layout}{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file.

    Returns
    -------
    output : str or (tuple, tensor)
        If the input is a path, the output path is returned.
        Else, the new shape and orientation matrix are returned.

    """
    dir = ''
    base = ''
    ext = ''
    fname = ''

    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        dim = f.affine.shape[-1] - 1
        inp = (f.shape[:dim], f.affine)
        if output is None:
            output = '{dir}{sep}{base}.{layout}{ext}'
        dir, base, ext = py.fileparts(fname)

    like_is_file = isinstance(like, str) and like
    if like_is_file:
        f = io.volumes.map(like)
        dim = f.affine.shape[-1] - 1
        like = (f.shape[:dim], f.affine)

    shape, aff0 = inp
    dim = aff0.shape[-1] - 1
    if like:
        shape_like, aff_like = like
    else:
        shape_like, aff_like = (shape, aff0)

    if voxel_size is None:
        voxel_size = spatial.voxel_size(aff_like)
    voxel_size = utils.make_vector(voxel_size, dim)

    if not layout:
        layout = spatial.affine_to_layout(aff_like)
    layout = spatial.volume_layout(layout)

    if not center:
        center = torch.as_tensor(shape_like, dtype=torch.float) * 0.5
        center = spatial.affine_matvec(aff_like, center)
    center = utils.make_vector(center, dim)

    aff = spatial.affine_default(shape, voxel_size=voxel_size, layout=layout,
                                 center=center, dtype=torch.double)

    if output:
        dat = io.volumes.load(fname, numpy=True)
        layout = spatial.volume_layout_to_name(layout)
        if is_file:
            output = output.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep, layout=layout)
            io.volumes.save(dat, output, like=fname, affine=aff)
        else:
            output = output.format(sep=os.path.sep, layout=layout)
            io.volumes.save(dat, output, affine=aff)

    if is_file:
        return output
    else:
        return shape, aff
