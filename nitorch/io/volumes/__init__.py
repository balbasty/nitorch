from . import loadsave
from . import mapping
from . import readers
from . import writers

from .mapping import MappedArray, CatArray, stack, cat
from .loadsave import map, load, loadf, save, savef

# Optional imports
from .. import optionals

if optionals.nibabel:
    from .babel import BabelArray

if optionals.tifffile:
    from .tiff import TiffArray
