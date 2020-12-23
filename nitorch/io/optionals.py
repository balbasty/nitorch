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

