import torch as _torch  # Necessary for linking extensions

# TODO:
# . check compatible cuda versions between torch and nitorch
#   (see torchvision.extension)


from . import cli
from . import core
from . import io
from . import nn
from . import plot
from . import spatial
from . import tools
from . import vb

from . import _version
__version__ = _version.get_versions()['version']

from ._C import COMPILED_BACKEND as compiled_backend
