from nitorch import spatial, io
from nitorch.core import py, utils
import torch
import os
import math


def extract_patches(inp, size=64, stride=None, output=None, transform=None):
    """Extracgt patches from a 3D volume.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    size : [sequence of] int, default=64
        Patch size.
    stride : [sequence of] int, default=size
        Stride between patches.
    output : [sequence of] str, optional
        Output filename(s).
        If the input is not a path, the unstacked data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.{i}_{j}_{k}{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file,
        `i` is the coordinate (starting at 1) of the slice.
    transform : [sequence of] str, optional
        Output filename(s) of the corresponding transforms.
        Not written by default.

    Returns
    -------
    output : list[str] or (tensor, tensor)
        If the input is a path, the output paths are returned.
        Else, the unfolded data and orientation matrices are returned.
            Data will have shape (nx, ny, nz, *size, *channels).
            Affines will have shape (nx, ny, nz, 4, 4).

    """
    dir = ''
    base = ''
    ext = ''
    fname = ''

    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        inp = (f.fdata(), f.affine)
        if output is None:
            output = '{dir}{sep}{base}.{i}_{j}_{k}{ext}'
        dir, base, ext = py.fileparts(fname)

    dat, aff0 = inp

    shape = dat.shape[:3]
    size = py.make_list(size, 3)
    stride = py.make_list(stride, 3)
    stride = [st or sz for st, sz in zip(stride, size)]

    dat = utils.movedim(dat, [0, 1, 2], [-3, -2, -1])
    dat = utils.unfold(dat, size, stride)
    dat = utils.movedim(dat, [-6, -5, -4, -3, -2, -1], [0, 1, 2, 3, 4, 5])

    aff = aff0.new_empty(dat.shape[:3] + aff0.shape)
    for i in range(dat.shape[0]):
        for j in range(dat.shape[1]):
            for k in range(dat.shape[2]):
                index = (i, j, k)
                sub = [slice(st*idx, st*idx + sz)
                       for st, sz, idx in zip(stride, size, index)]
                aff[i, j, k], _ = spatial.affine_sub(aff0, shape, tuple(sub))

    formatted_output = []
    if output:
        output = py.make_list(output, py.prod(dat.shape[:3]))
        formatted_output = []
        for i in range(dat.shape[0]):
            for j in range(dat.shape[1]):
                for k in range(dat.shape[2]):
                    out1 = output.pop(0)
                    if is_file:
                        out1 = out1.format(dir=dir or '.', base=base, ext=ext,
                                           sep=os.path.sep, i=i+1, j=j+1, k=k+1)
                        io.volumes.savef(dat[i, j, k], out1, like=fname,
                                         affine=aff[i, j, k])
                    else:
                        out1 = out1.format(sep=os.path.sep, i=i, j=j, k=k)
                        io.volumes.savef(dat[i, j, k], out1, affine=aff[i, j, k])
                    formatted_output.append(out1)

    if transform:
        transform = py.make_list(transform, py.prod(dat.shape[:3]))
        for i in range(dat.shape[0]):
            for j in range(dat.shape[1]):
                for k in range(dat.shape[2]):
                    trf1 = transform.pop(0)
                    if is_file:
                        trf1 = trf1.format(dir=dir or '.', base=base, ext=ext,
                                           sep=os.path.sep, i=i+1, j=j+1, k=k+1)
                    else:
                        trf1 = trf1.format(sep=os.path.sep, i=i+1, j=j+1, k=k+1)
                    io.transforms.savef(torch.eye(4), trf1,
                                        source=aff0, target=aff[i, j, k])

    if is_file:
        return formatted_output
    else:
        return dat, aff
