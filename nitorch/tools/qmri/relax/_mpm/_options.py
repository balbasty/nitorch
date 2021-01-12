import torch
from nitorch.core.options import Option


class ReconOptions(Option):
    """Options for the reconstruction space"""
    space: int or str = 'mean'              # Orientation of the recon space
    layout: int or str or None = None       # ?
    fov: str = 'bb'                         # Field-of-view of the recon space
    crop: float = 0                         # Crop size if fov == 'bb'


class PenaltyOptions(Option):
    """Options for the regularization"""
    norm: str = 'jtv'                       # Norm to optimize: {'jtv', 'tv', 'tkh', None}
    factor: float or list or dict \
        = dict(r1=10, pd=10, r2s=2, mt=2)   # Regularization factor


class OptimOptions(Option):
    """Options for the optimizer(s)"""
    nb_levels: int = 1                     # Number of pyramid leveks
    max_iter_gn: int = 5                   # Number of Gauss-Newton iterations
    max_iter_cg: int = 32                  # Number of Conjugate Gradient iteration
    max_iter_rls: int = 10                 # Number of Reweighted LS iterations
    tolerance_gn: float = 1e-5             # Tolerance for early stopping
    tolerance_cg: float = 1e-3
    tolerance_rls: float = 1e-5


class BackendOptions(Option):
    """Dtype and device on which to perform computations"""
    dtype: torch.dtype = torch.get_default_dtype()  # Data type used for computations
    device: str or torch.device = None              # Device ('cpu' / 'cuda')


class PreprocOptions(Option):
    """Which preprocessing steps should be performed"""
    register: bool = True               # Rigid registration between series


class GREEQOptions(Option):
    """Encapsulate all options"""
    preproc: PreprocOptions = PreprocOptions()
    recon: ReconOptions = ReconOptions()
    optim: OptimOptions = OptimOptions()
    backend: BackendOptions = BackendOptions()
    penalty: PenaltyOptions = PenaltyOptions()
    verbose: int or bool = 1


class VFAOptions(Option):
    """Encapsulate all options"""
    preproc: PreprocOptions = PreprocOptions()
    recon: ReconOptions = ReconOptions()
    backend: BackendOptions = BackendOptions()
    verbose: int or bool = 1
