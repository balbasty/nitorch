import torch as _torch  # Necessary for linking extensions

# The legacy jit executor (used by torchscript) was the default
# until v1.6 (included), but sometimes sets `requires_grad = True`
# on new variables even though it should not, making TS code
# virtually unusable. Here, we force the use of the 
# profiling executor (default from v1.7).
try:
    from torch._C import _jit_set_profiling_executor
    _jit_set_profiling_executor(True)
except ImportError:
    from warnings import warn
    warn('Could not use profiling executor. Parts of nitorch '
         '(e.g. `nitorch register`) may break.', RuntimeWarning)


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
from . import uniseg

from . import _version
__version__ = _version.get_versions()['version']

from ._C import COMPILED_BACKEND as compiled_backend
