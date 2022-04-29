from nitorch import spatial, io
from nitorch.core import py, utils
import torch
import os


def orient(inp, affine=None, layout=None, voxel_size=None, center=None,
           like=None, output=None, output_transform=None):
    """Overwrite the orientation matrix

    Parameters
    ----------
    inp : str or (tuple, tensor)
        Either a path to a volume file or a tuple `(shape, affine)`, where
        the first element contains the volume shape and the second contains
        the orientation matrix.
    affine : {'self', 'like'} or (4, 4) tensor_like, default='like'
        Target affine matrix
    layout : {'self', 'like'} or layout-like, default='like'
        Target orientation.
    voxel_size : {'self', 'like'} or [sequence of] float, default='like'
        Target voxel size.
    center : {'self', 'like'} or [sequence of] float, default='like'
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
    output_transform : str, optional
        Filename of output transform.
        If the input is not a path, the reoriented data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}_to_{layout}.lta', where `dir` and `base`
        are the directory and base name of the input file.

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
        if output_transform is None:
            output_transform = '{dir}{sep}{base}_to_{layout}.lta'
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

    if voxel_size in (None, 'like') or len(voxel_size) == 0:
        voxel_size = spatial.voxel_size(aff_like)
    elif voxel_size == 'self':
        voxel_size = spatial.voxel_size(aff0)
    elif voxel_size == 'standard':
        voxel_size = 1.
    voxel_size = utils.make_vector(voxel_size, dim)

    if not layout or layout == 'like':
        layout = spatial.affine_to_layout(aff_like)
    elif layout == 'self':
        layout = spatial.affine_to_layout(aff0)
    elif layout == 'standard':
        layout = 'RAS'
    layout = spatial.volume_layout(layout)

    if center in (None, 'like') or len(center) == 0:
        center = (torch.as_tensor(shape_like, dtype=torch.float) - 1) * 0.5
        center = spatial.affine_matvec(aff_like, center)
    elif center == 'self':
        center = (torch.as_tensor(shape, dtype=torch.float) - 1) * 0.5
        center = spatial.affine_matvec(aff0, center)
    elif center == 'standard':
        center = 0.
    center = utils.make_vector(center, dim)

    if affine in (None, 'like') or len(affine) == 0:
        affine = aff_like
    elif affine == 'self':
        affine = aff0
    elif affine == 'standard':
        affine = torch.eye(dim+1, dim+1)
    affine = torch.as_tensor(affine, dtype=torch.float)
    if affine.numel() == dim*(dim+1):
        affine = spatial.affine_make_rect(affine.reshape(dim, dim+1))
    elif affine.numel() == (dim+1)**2:
        affine = affine.reshape(dim+1, dim+1)
    else:
        raise ValueError(f'Input affine should have {dim*(dim+1)} or '
                         f'{(dim+1)**2} element but got {affine.numel()}.')

    affine = spatial.affine_modify(affine, shape, voxel_size=voxel_size,
                                   layout=layout, center=center)
    affine = affine.double()

    if output:
        dat = io.volumes.load(fname, numpy=True)
        layout = spatial.volume_layout_to_name(layout)
        if is_file:
            output = output.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep, layout=layout)
            io.volumes.save(dat, output, like=fname, affine=affine)
        else:
            output = output.format(sep=os.path.sep, layout=layout)
            io.volumes.save(dat, output, affine=affine)

    if output_transform:
        transform = spatial.affine_rmdiv(affine, aff0)
        output_transform = output_transform.format(
            dir=dir or '.', base=base, sep=os.path.sep, layout=layout)
        io.transforms.savef(transform.cpu(), output_transform, type=2)

    if is_file:
        return output
    else:
        return shape, affine
