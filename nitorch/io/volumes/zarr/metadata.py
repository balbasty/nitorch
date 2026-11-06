"""Embedded NIfTI header <-> metadata conversion for nifti-zarr stores.

A nifti-zarr store embeds the raw NIfTI header as a byte array named
``"nifti"`` inside its Zarr group (see the nifti-zarr spec and the reference
``nifti-zarr-py`` implementation's ``_zarr2nii.py``). This module parses those
raw bytes into a real ``nibabel`` header object; the generic
header-object -> metadata-dict conversion is reused as-is from
``nitorch.io.volumes.babel.metadata.header_to_metadata`` (both the embedded
header here and the OME-derived header in ``ome_header.py`` produce ordinary
``nibabel`` header objects, so the same conversion applies to both).
"""

import io as _io

import numpy as np
from nibabel.nifti1 import Nifti1Header
from nibabel.nifti2 import Nifti2Header

#: Fixed on-disk sizes of the NIfTI-1 and NIfTI-2 binary headers.
_NIFTI1_SIZE = 348
_NIFTI2_SIZE = 540


def read_embedded_header(root):
    """Read and parse the embedded NIfTI header from a nifti-zarr store.

    Parameters
    ----------
    root : zarr.Group or zarr.Array
        The opened root of a (candidate) nifti-zarr store.

    Returns
    -------
    header : nibabel.nifti1.Nifti1Header or nibabel.nifti2.Nifti2Header or None
        `None` if no embedded header array (named `"nifti"`) is present
        (e.g. a plain OME-Zarr store, or a plain Zarr array).

    """
    import zarr
    if not isinstance(root, zarr.Group):
        # a plain zarr.Array cannot contain a child "nifti" header array;
        # `'nifti' in root` would otherwise fall back to elementwise data
        # comparison and raise a confusing ValueError
        return None
    if 'nifti' not in root:
        return None

    raw = np.asarray(root['nifti']).tobytes()
    return _bytes_to_header(raw)


def _bytes_to_header(raw):
    """Parse raw NIfTI header bytes into a nibabel header object."""
    if len(raw) >= _NIFTI2_SIZE:
        try:
            return Nifti2Header.from_fileobj(_io.BytesIO(raw), check=False)
        except Exception:
            pass
    return Nifti1Header.from_fileobj(_io.BytesIO(raw[:_NIFTI1_SIZE]), check=False)
