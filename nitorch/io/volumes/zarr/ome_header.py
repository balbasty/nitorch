"""Derive NIfTI-equivalent header metadata from plain OME-Zarr metadata.

Used when a Zarr store has no embedded NIfTI header (a plain OME-Zarr store,
version 0.4 or 0.5). Mirrors the reference `nifti-zarr-py` implementation's
`default_nifti_header()`/`_ome2affine()` (see
`specs/003-nifti-zarr-support/research.md` §3) so that the derived affine and
voxel size match what that implementation itself would produce (SC-006).
"""

import numpy as np
from nibabel.nifti1 import Nifti1Header
from nibabel.nifti2 import Nifti2Header

#: Per-unit scale factor to convert a spatial unit to millimeters.
_SPACE_TO_MM = {
    'angstrom': 1e-7, 'nanometer': 1e-6, 'nm': 1e-6,
    'micrometer': 1e-3, 'um': 1e-3, 'micron': 1e-3,
    'millimeter': 1.0, 'mm': 1.0,
    'centimeter': 1e1, 'cm': 1e1,
    'meter': 1e3, 'm': 1e3,
}

#: Per-unit scale factor to convert a temporal unit to seconds.
_TIME_TO_S = {
    'attosecond': 1e-18, 'femtosecond': 1e-15, 'picosecond': 1e-12,
    'nanosecond': 1e-9, 'microsecond': 1e-6, 'millisecond': 1e-3, 'ms': 1e-3,
    'second': 1.0, 's': 1.0, 'sec': 1.0,
    'minute': 60.0, 'hour': 3600.0, 'day': 86400.0,
}


def _convert_unit(value, unit):
    """Convert an OME-Zarr axis value to millimeters (space) or seconds
    (time). Returns `value` unchanged if `unit` is not recognized.
    """
    unit = (unit or '').lower()
    if unit in _SPACE_TO_MM:
        return value * _SPACE_TO_MM[unit]
    if unit in _TIME_TO_S:
        return value * _TIME_TO_S[unit]
    return value


def _ome2affine(ome, level=0):
    """Build a 4x4 voxel-to-world affine from OME-Zarr axes/transformations.

    Mirrors `nifti-zarr-py`'s `_ome2affine`: only the `x`/`y`/`z` axes
    contribute to the affine (channel/time axes are ignored here, matching
    the reference implementation).
    """
    axes = ome[0]['axes']
    names = [axis['name'] for axis in axes]
    units = [axis.get('unit') for axis in axes]

    scales, offsets = None, None
    for transform in ome[0]['datasets'][level]['coordinateTransformations']:
        if transform['type'] == 'scale':
            scales = transform['scale']
        elif transform['type'] == 'translation':
            offsets = transform['translation']
    scales = scales or [1.0] * len(names)
    offsets = offsets or [0.0] * len(names)

    scales = [_convert_unit(s, u) for s, u in zip(scales, units)]
    offsets = [_convert_unit(o, u) for o, u in zip(offsets, units)]
    scale_by_name = dict(zip(names, scales))
    offset_by_name = dict(zip(names, offsets))

    affine = np.eye(4)
    affine[range(3), range(3)] = [scale_by_name.get(n, 1.0) for n in 'xyz']
    affine[:3, -1] = [offset_by_name.get(n, 0.0) for n in 'xyz']
    return affine


def axis_order(raw_ndim, ome):
    """Return the raw on-disk axis names, in the store's own order.

    Mirrors `nifti-zarr-py`'s `_zarr2nii.py` axis-order defaulting: when no
    OME axes metadata is present (a nifti-zarr store with an embedded
    header but no multiscale metadata), the nifti-zarr spec's storage
    convention is assumed -- axes in the reverse of standard NIfTI order.
    """
    if ome:
        return [axis['name'] for axis in ome[0]['axes']]
    return list(('x', 'y', 'z', 'c', 't')[:raw_ndim][::-1])


def ome_permutation(names):
    """Permutation mapping the raw on-disk axis order (`names`) onto
    standard NIfTI x/y/z/t/c order.

    Mirrors `nifti-zarr-py`'s own axis-permutation logic in `_zarr2nii.py`
    exactly (`perm[i] = names.index(name)` for each NIfTI axis name in
    turn). Correct whenever every axis in `names` is one of x/y/z/t/c and
    every declared axis has a corresponding raw dimension -- true for the
    common case where OME axes metadata (or the nifti-zarr fallback order)
    covers exactly the store's own raw dimensions (see `ome_shape`, which
    can add synthetic size-1 t/c axes with no raw counterpart -- callers
    should verify the returned permutation's length against the raw and
    declared dimensionality before using it).
    """
    return tuple(names.index(name) for name in 'xyztc' if name in names)


def ome_shape(array_shape, ome):
    """Map an OME-Zarr array's raw shape onto standard x/y/z/t/c ordering."""
    names = [axis['name'] for axis in ome[0]['axes']]
    shape_by_name = {name: array_shape[i] for i, name in enumerate(names)}
    shape = [shape_by_name.get(name, 1) for name in 'xyztc']
    # trim trailing singleton axes the store doesn't actually have, from
    # the least-significant (c, then t, then z, ...) end, matching the
    # reference implementation
    for name in reversed('xyztc'):
        if name not in names and len(shape) > 3:
            shape = shape[:-1]
        else:
            break
    return shape


def derive_header(array, ome, level=0):
    """Derive a NIfTI-equivalent header from an OME-Zarr array's own metadata.

    Parameters
    ----------
    array : zarr.Array
        The (finest, by convention) OME-Zarr array.
    ome : list or None
        OME-Zarr `multiscales` metadata (`ome[0]["axes"]`/`["datasets"]`), or
        `None` if the store carries no OME metadata at all.
    level : int, default=0
        Which dataset entry's `coordinateTransformations` to use.

    Returns
    -------
    header : nibabel.nifti1.Nifti1Header or nibabel.nifti2.Nifti2Header

    """
    NiftiHeader = Nifti2Header if any(d > 2 ** 15 for d in array.shape) else Nifti1Header
    header = NiftiHeader()

    if ome:
        affine = _ome2affine(ome, level)
        shape = ome_shape(array.shape, ome)
    else:
        affine = np.eye(4)
        shape = list(array.shape)

    header.set_data_shape(shape)
    header.set_data_dtype(array.dtype)
    header.set_qform(affine)
    header.set_sform(affine)
    header.set_xyzt_units('mm', 'sec')
    return header
