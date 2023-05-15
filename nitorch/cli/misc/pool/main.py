from nitorch import spatial, io
from nitorch.core import py, utils
import os


def pool(inp, window=3, stride=None, padding=0, bound='dct2',
         method='mean', dim=3, output=None, device=None):
    """Pool a ND volume, while preserving the orientation matrices.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    window : [sequence of] int, default=3
        Window size
    stride : [sequence of] int, optional
        Stride between output elements.
        By default, it is the same as `window`.
    padding : [sequence of] int or 'same', optional
        Padding on both sides.
    method : {'mean', 'sum', 'min', 'max', 'median'}, default='mean'
        Pooling function.
    dim : int, default=3
        Number of spatial dimensions.
    output : [sequence of] str, optional
        Output filename(s).
        If the input is not a path, the unstacked data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.pool{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file,
        `i` is the coordinate (starting at 1) of the slice.

    Returns
    -------
    output : str or (tensor, tensor)
        If the input is a path, the output path is returned.
        Else, the pooled data and orientation matrix are returned.

    """
    dir = ''
    base = ''
    ext = ''
    fname = ''

    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        inp = (f.fdata(device=device), f.affine)
        if output is None:
            output = '{dir}{sep}{base}.pool{ext}'
        dir, base, ext = py.fileparts(fname)

    dat, aff0 = inp
    dat = dat.to(device)
    dim = dim or aff0.shape[-1] - 1

    # `pool` needs the spatial dimensions at the end
    spatial_in = dat.shape[:dim]
    batch = dat.shape[dim:]
    dat = dat.reshape([*spatial_in, -1])
    dat = utils.movedim(dat, -1, 0)
    dat, aff = spatial.pool(
        dim, dat,
        kernel_size=window,
        stride=stride,
        padding=padding,
        bound=bound,
        reduction=method,
        affine=aff0
    )
    dat = utils.movedim(dat, 0, -1)
    dat = dat.reshape([*dat.shape[:dim], *batch])

    if output:
        if is_file:
            output = output.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep)
            io.volumes.save(dat, output, like=fname, affine=aff)
        else:
            output = output.format(sep=os.path.sep)
            io.volumes.save(dat, output, affine=aff)

    if is_file:
        return output
    else:
        return dat, aff
