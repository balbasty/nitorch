from nitorch import spatial, io
from nitorch.core import py
import torch
import os


def vexp(inp, type='displacement', unit='voxel', inverse=False,
         bound='dft', steps=8, device=None, output=None):
    """Exponentiate a stationary velocity fields.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    type : {'displacement', 'transformation'}, default='displacement'
        Whether to return a displacement field (coord-to-shift) or a
        transformation field (coord-to-coord).
    unit : {'voxel', 'mm'}, default='voxel'
        Whether to return displacement/coordinates in voxel or in mm.
        If mm, the input orientation matrix is used to convert voxels to mm.
    inverse : bool, default=False
        Whether to return the inverse field.
    bound : str, default='dft'
        Boundary conditions.
    steps : int, default=8
        Number of scaling and squaring steps.
    device : str, optional
        Device to use.
    output : str, optional
        Output filename(s).
        If the input is not a path, the unstacked data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.vexp{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file.

    Returns
    -------
    output : str or (tensor, tensor)
        If the input is a path, the output path is returned.
        Else, the output tensor and orientation matrix are returned.

    """
    dir = ''
    base = ''
    ext = ''
    fname = None

    # --- Open input ---
    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        inp = (f.fdata(device=device), f.affine)
        if output is None:
            output = '{dir}{sep}{base}.vexp{ext}'
        dir, base, ext = py.fileparts(fname)
    else:
        if torch.is_tensor(inp):
            inp = (inp.clone(), spatial.affine_default(shape=inp.shape[:3]))
    dat, aff = inp
    dat = dat.to(device=device)
    aff = aff.to(device=device)

    # exponentiate
    displacement = (type.lower()[0] == 'd')
    dat = spatial.exp(dat[None], inverse=inverse, steps=steps, bound=bound,
                      displacement=displacement)[0]
    if unit == 'mm':
        # if type.lower()[0] == 'd':
        #     vx = spatial.voxel_size(aff)
        #     dat *= vx
        # else:
        dat = spatial.affine_matvec(aff, dat)

    if output:
        if is_file:
            output = output.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep)
            io.volumes.save(dat, output, like=fname, affine=aff.cpu())
        else:
            output = output.format(sep=os.path.sep)
            io.volumes.save(dat, output, affine=aff.cpu())

    if is_file:
        return output
    else:
        return dat, aff
