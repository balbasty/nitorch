"""I/O utilities"""

from . import mapping
from . import optionals
from . import readers
from . import dtype
from . import metadata

from .mapping import MappedArray, CatArray
from .readers import map, read, fread

if optionals.nibabel:
    from . import babel
    from .babel import BabelArray
