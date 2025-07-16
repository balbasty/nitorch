from . import loadsave
from . import mapping
from . import readers
from . import writers

from .mapping import MappedStreamlines
from .loadsave import map, load, loadf, save, savef

# Import implementations

from .. import optionals

if optionals.nibabel:
    from .trk import TrkStreamlines
