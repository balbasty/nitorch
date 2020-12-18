import torch
import copy


class OptionBase:
    def __init__(self, *args, **kwargs):
        for attr in self.keys():
            setattr(self, attr, copy.deepcopy(getattr(self, attr)))
        if len(args) == 1 and isinstance(args[0], OptionBase):
            for key, val in args[0].iteritems():
                if key in self.keys():
                    setattr(self, key, val)
        elif len(args) != 0:
            raise TypeError(f'Expected at most one argument, got {len(args)}')

        for key, val in kwargs.items():
            if key not in self.keys():
                raise KeyError(key)
            setattr(self, key, val)

    def keys(self):
        return [key for key in dir(self)
                if not key.startswith('_')
                and key not in ('keys', 'copy', 'items')]

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __eq__(self, other):
        other = type(self)(other)
        if type(self) != type(other):
            return False
        for attr in self.keys():
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return self._str()

    def _str(self, level=0):
        as_str = ''
        key_length = str(max(len(key) for key in self.keys()))
        pad = ' ' * level
        for key, val in self.items():
            vals = ''
            if isinstance(val, OptionBase):
                vals = val._str(level + int(key_length) + 3) + '\n'
                val = type(val).__name__
            as_str += pad + ('{:' + key_length + 's} : {}\n').format(key, val)
            as_str += vals
        return as_str[:-1]

    def __repr__(self):
        return dict(self).__repr__()

    
class ReconOptions(OptionBase):
    """Options for the reconstruction space"""
    space: int or str = 'mean'              # Orientation of the recon space
    layout: int or str or None = None       # ?
    fov: str = 'bb'                         # Field-of-view of the recon space
    crop: float = 0                         # Crop size if fov == 'bb'


class RegularizationOptions(OptionBase):
    """Options for the regularization"""
    norm: str = 'jtv'                       # Norm to optimize: {'jtv', 'tv', 'tkh', None}
    factor: float or list = 32.             # Regularization factor


class OptimOptions(OptionBase):
    """Options for the optimizer(s)"""
    max_iter_gn: int = 1                   # Number of Gauss-Newton iterations
    max_iter_cg: int = 10                  # Number of Conjugate Gradient iteration
    max_iter_rls: int = 10                 # Number of Reweighted LS iterations
    max_iter_ls: int = 0                   # Number of Line Search iterations
    tolerance_gn: float = 1e-7
    tolerance_cg: float = 1e-5
    tolerance_rls: float = 1e-7


class BackendOptions(OptionBase):
    """Dtype and device on which to perform computations"""
    dtype: torch.dtype = torch.get_default_dtype()  # Data type used for computations
    device: str or torch.device = None              # Device ('cpu' / 'cuda')


class PreprocOptions(OptionBase):
    """Which preprocessing steps should be performed"""
    register: bool = True               # Rigid registration between series


class Options(OptionBase):
    """Encapsulate all options"""
    preproc: PreprocOptions = PreprocOptions()
    recon: ReconOptions = ReconOptions()
    optim: OptimOptions = OptimOptions()
    backend: BackendOptions = BackendOptions()
    regularization: RegularizationOptions = RegularizationOptions()
    verbose : int or bool = 1
