"""Compatibility layer between nitorch and different versions of nibabel.

Everything that depends on the nibabel version -- or on nibabel
internals that may change without notice -- should live here, so that
the rest of the code base does not need to care about which nibabel
version is installed.
"""
import numpy as np
import nibabel as nib


def _parse_version(version):
    """Parse the leading (major, minor, patch) integers of a version string"""
    parsed = []
    for part in version.split('.')[:3]:
        digits = ''
        for char in part:
            if not char.isdigit():
                break
            digits += char
        parsed.append(int(digits or 0))
    parsed += [0] * (3 - len(parsed))
    return tuple(parsed)


NIBABEL_VERSION = _parse_version(nib.__version__)


# ----------------------------------------------------------------------
#   Public nibabel API (re-exported for convenience)
# ----------------------------------------------------------------------

from nibabel.spatialimages import SpatialImage                  # noqa: E402
from nibabel.imageclasses import all_image_classes              # noqa: E402
from nibabel.filebasedimages import ImageFileError              # noqa: E402
from nibabel.volumeutils import array_from_file, array_to_file  # noqa: E402
from nibabel.fileslice import (                                 # noqa: E402
    fileslice, threshold_heuristic, is_fancy, fill_slicer,
    canonical_slicers, read_segments, predict_shape, slicers2segments)


# ----------------------------------------------------------------------
#   Private nibabel API (with fallbacks in case they get renamed/removed)
# ----------------------------------------------------------------------

try:
    from nibabel.volumeutils import _is_compressed_fobj as is_compressed_fobj
except ImportError:
    import bz2
    import gzip
    from nibabel import volumeutils as _volumeutils
    _COMPRESSED_FILE_LIKES = getattr(
        _volumeutils, 'COMPRESSED_FILE_LIKES', (gzip.GzipFile, bz2.BZ2File))

    def is_compressed_fobj(fobj):
        """Return True if fobj represents a compressed data file-like object"""
        return isinstance(fobj, _COMPRESSED_FILE_LIKES)

try:
    from nibabel.fileslice import _NullLock
except ImportError:
    class _NullLock:
        """No-op object that can be used in place of a ``threading.Lock``"""
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

try:
    from nibabel.fileslice import _positive_slice
except ImportError:
    def _positive_slice(slicer):
        """Return full slice `slicer` enforcing positive step size
        `slicer` assumed full in the sense of :func:`fill_slicer`
        """
        start, stop, step = slicer.start, slicer.stop, slicer.step
        if step > 0:
            return slicer
        if stop is None:
            stop = -1
        gap = stop - start
        n = gap / step
        n = int(n) - 1 if int(n) == n else int(n)
        end = start + n * step
        return slice(end, start + 1, -step)


# ----------------------------------------------------------------------
#   Behaviour that changed across nibabel versions
# ----------------------------------------------------------------------


def disk_header(image):
    """Return the header of an image, as it is stored on disk.

    nibabel < 4 kept a private copy of the on-disk header in the array
    proxy (`image.dataobj._header`). It was removed in nibabel 4.0.
    `image.header` cannot be used as-is instead, because nibabel resets
    its scaling (`scl_slope`/`scl_inter`) when an image is loaded -- the
    scaling is only kept in the array proxy.

    When the private header is not available, the scaling stored in the
    array proxy is restored into `image.header` (in-place, so that later
    modifications of the returned header are seen by later calls) and
    `image.header` is returned.

    Parameters
    ----------
    image : nibabel.SpatialImage

    Returns
    -------
    header : nibabel.Header

    """
    dataobj = image.dataobj
    header = getattr(dataobj, '_header', None)
    if header is not None:
        return header
    header = image.header
    if hasattr(header, 'set_slope_inter') and hasattr(dataobj, 'slope'):
        header.set_slope_inter(dataobj.slope, dataobj.inter)
    return header


def mgh_dtypes():
    """Return the data types supported by the MGH format.

    The order of the data types in nibabel's (private) table changed in
    nibabel 5.0 (where uint16 was also added), so we sort them ourselves:
    integers first (narrowest first), then floating point types.

    Returns
    -------
    dtypes : list[np.dtype]
        Native-endian data types, sorted from "smallest" to "largest".

    """
    from nibabel.freesurfer.mghformat import _dtdefs
    # each entry is (code, label, on-disk dtype, ...)
    dtypes = [np.dtype(dt[2]).newbyteorder('=') for dt in _dtdefs]
    return sorted(set(dtypes),
                  key=lambda dt: (dt.kind == 'f', dt.itemsize, dt.kind == 'i'))


def set_quaternion_tolerance(tolerance):
    """Set the tolerance used to check that NIfTI quaternions are unit.

    nibabel reconstructs the first component of the qform quaternion
    from the three stored ones, which requires that `1 - |bcd|**2 >= 0`.
    Because of finite precision, this is sometimes slightly negative.

    nibabel < 5 raised if `1 - |bcd|**2 < threshold` (with a negative
    threshold), whereas nibabel >= 5 raises if `1 - |bcd|**2 < 0` and
    `|1 - |bcd|**2| >= |threshold|`.

    Parameters
    ----------
    tolerance : float
        Positive tolerance on `1 - |bcd|**2`.

    """
    tolerance = abs(tolerance)
    if NIBABEL_VERSION < (5,):
        tolerance = -tolerance
    nib.Nifti1Header.quaternion_threshold = tolerance
