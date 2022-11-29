from . import loadsave
from . import mappedarray
from . import readers
from . import writers

from .mappedarray import MappedArray
from .catarray import MappedArray, CatArray, stack, cat
from .loadsave import map, load, loadf, save, savef

# Optional imports
from .. import optionals

if optionals.nibabel:
    from .babel import BabelArray

if optionals.tifffile:
    from .tiff import TiffArray
