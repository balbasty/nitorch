"""Check which optional (io) modules are available."""


# Numpy
try:
    import numpy
except ImportError:
    numpy = None
    
# Nibabel
try:
    import nibabel
except ImportError:
    nibabel = None
except Exception as e:
    # Old versions of nibabel do not import with recent versions of numpy
    # (e.g., nibabel < 5.2 with numpy >= 2 raises an AttributeError).
    # Do not let this break the import of nitorch.
    import warnings
    try:
        from importlib.metadata import version as _version
        _nibabel_version = _version('nibabel')
    except Exception:
        _nibabel_version = 'unknown'
    warnings.warn(
        f'nibabel (version {_nibabel_version}) could not be imported, '
        f'probably because it is not compatible with the installed version '
        f'of numpy (version {getattr(numpy, "__version__", "unknown")}). '
        f'Upgrade nibabel (e.g., nibabel >= 5.2 for numpy >= 2) to enable '
        f'nibabel-based formats (NIfTI, MGH, ...). '
        f'Original error: {type(e).__name__}: {e}',
        RuntimeWarning)
    nibabel = None

# TiffFile
try:
    import tifffile
except ImportError:
    tifffile = None

# FreeSurfer
try:
    import freesurfer
except ImportError:
    freesurfer = None

