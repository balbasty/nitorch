import torch
from nitorch.core.options import Option


class ReconOptions(Option):
    """Options for the reconstruction space"""
    space: int or str = 0                   # Recon space: 
    affine: torch.tensor or None = None     # Recon orientation matrix (default: from space)
    layout: int or str or None = None       # Force output RAS layout (default: same as affine)
    fov: tuple or str = None                # Field-of-view of the recon space: shape or 'bb' (default: from space)
    crop: float = 0                         # Crop size (in pct) if fov == 'bb'


class PenaltyOptions(Option):
    """Options for the regularization"""
    norm: str = 'jtv'                       # Norm to optimize: {'jtv', 'tv', 'tkh', None}
    factor: float or list or dict = 10      # Regularization factor


class OptimOptions(Option):
    """Options for the optimizer(s)"""
    nb_levels: int = 5                     # Number of pyramid levels
    max_iter_gn: int = 5                   # Number of Gauss-Newton iterations
    max_iter_cg: int = 32                  # Number of Conjugate Gradient iteration
    max_iter_rls: int = 10                 # Number of Reweighted LS iterations
    tolerance: float = 1e-4                # Tolerance for early stopping
    tolerance_cg: float = 1e-4
    solver = 'cg'                          # Linear solver: {'fmg', 'cg'}


class BackendOptions(Option):
    """Dtype and device on which to perform computations"""
    dtype: torch.dtype = torch.get_default_dtype()  # Data type used for computations
    device: str or torch.device = None              # Device ('cpu' / 'cuda')


class PreprocOptions(Option):
    """Which preprocessing steps should be performed"""
    register: bool = True               # Rigid registration between series


class GREEQOptions(Option):
    """Encapsulate all options"""
    likelihood: str = 'chi'           # Likelihood model: {'gauss', 'chi'}
    preproc: PreprocOptions = PreprocOptions()
    recon: ReconOptions = ReconOptions()
    optim: OptimOptions = OptimOptions()
    backend: BackendOptions = BackendOptions()
    penalty: PenaltyOptions = PenaltyOptions()
    verbose: int or bool = 1
    uncertainty: bool = False            # Whether to return uncertainty maps (posterior variance)


class VFAOptions(Option):
    """Encapsulate all options"""
    preproc: PreprocOptions = PreprocOptions()
    recon: ReconOptions = ReconOptions()
    backend: BackendOptions = BackendOptions()
    verbose: int or bool = 1
    rational: bool = False
