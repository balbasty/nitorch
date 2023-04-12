from nitorch import spatial, io
from nitorch.core import py, dtypes
import torch
import os


def convert(inp, meta=None, dtype=None, casting='unsafe', format=None, output=None):
    """Convert a volume.

    Parameters
    ----------
    inp : str
        A path to a volume file.
    meta : sequence of (key, value)
        List of metadata fields to set.
    dtype : str or dtype, optional
        Output data type
    casting : {'unsafe', 'rescale', 'rescale_zero'}, default='unsafe'
        Casting method
    format : {'nii', 'nii.gz', 'mgh', 'mgz'}, optional
        Output format

    """

    meta = dict(meta or {})
    if dtype:
        meta['dtype'] = dtype
    fname = inp
    f = io.volumes.map(fname)
    d = f.data(numpy=True)

    dir, base, ext = py.fileparts(fname)
    if format:
        ext = format
        if ext == 'nifti':
            ext = 'nii'
        if ext[0] != '.':
            ext = '.' + ext
    output = output or '{dir}{sep}{base}{ext}'
    output = output.format(dir=dir or '.', sep=os.sep, base=base, ext=ext)

    odtype = meta.get('dtype', None) or f.dtype
    if ext in ('.mgh', '.mgz'):
        from nibabel.freesurfer.mghformat import _dtdefs
        odtype = dtypes.dtype(odtype)
        mgh_dtypes = [dtypes.dtype(dt[2]) for dt in _dtdefs]
        for mgh_dtype in mgh_dtypes:
            if odtype <= mgh_dtype:
                odtype = mgh_dtype
                break
        odtype = odtype.numpy
        meta['dtype'] = odtype

    if output.endswith(('.nii', '.nii.gz', '.mgh', '.mgz')):
        while d.ndim < 3:
            d = d[..., None]

    io.save(d, output, like=f, casting=casting, **meta)

