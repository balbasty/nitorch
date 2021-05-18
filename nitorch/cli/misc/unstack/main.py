from nitorch import spatial, io
from nitorch.core import py
import torch
import os


def unstack(inp, dim=-1, output=None, transform=None):
    """Unstack a ND volume, while preserving the orientation matrices.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    dim : int, default=-1
        Dimension along which to unstack.
    output : [sequence of] str, optional
        Output filename(s).
        If the input is not a path, the unstacked data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.{i}{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file,
        `i` is the coordinate (starting at 1) of the slice.
    transform : [sequence of] str, optional
        Output filename(s) of the corresponding transforms.
        Not written by default.

    Returns
    -------
    output : list[str or (tensor, tensor)]
        If the input is a path, the output paths are returned.
        Else, the unstacked data and orientation matrices are returned.

    """
    dir = ''
    base = ''
    ext = ''
    fname = ''

    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        inp = (f.data(), f.affine)
        if output is None:
            output = '{dir}{sep}{base}.{i}{ext}'
        dir, base, ext = py.fileparts(fname)

    dat, aff0 = inp
    ndim = aff0.shape[-1] - 1
    if dim > ndim:
        # we didn't touch the spatial dimensions
        aff = [aff0.clone() for _ in range(len(dat))]
    else:
        aff = []
        slicer = [slice(None) for _ in range(ndim)]
        shape = dat.shape[:ndim]
        for z in range(dat.shape[dim]):
            slicer[dim] = slice(z,z+1)
            aff1, _ = spatial.affine_sub(aff0, shape, tuple(slicer))
            aff.append(aff1)
    dat = torch.unbind(dat, dim)
    dat = [d.unsqueeze(dim) for d in dat]
    dat = list(zip(dat, aff))

    formatted_output = []
    if output:
        output = py.make_list(output, len(dat))
        formatted_output = []
        for i, ((dat1, aff1), out1) in enumerate(zip(dat, output)):
            if is_file:
                out1 = out1.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep, i=i+1)
                io.volumes.save(dat1, out1, like=fname, affine=aff1)
            else:
                out1 = out1.format(sep=os.path.sep, i=i+1)
                io.volumes.save(dat1, out1, affine=aff1)
            formatted_output.append(out1)

    if transform:
        transform = py.make_list(transform, len(dat))
        for i, ((_, aff1), trf1) in enumerate(zip(dat, transform)):
            if is_file:
                trf1 = trf1.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep, i=i+1)
            else:
                trf1 = trf1.format(sep=os.path.sep, i=i+1)
            io.transforms.savef(torch.eye(4), trf1, source=aff0, target=aff1)

    if is_file:
        return formatted_output
    else:
        return dat, aff
