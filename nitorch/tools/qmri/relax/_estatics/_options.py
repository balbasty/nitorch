import torch
from nitorch.core.options import Option


class ReconOptions(Option):
    """Options for the reconstruction space"""
    space: int or str = 0                   # Recon space: 
    affine: torch.tensor or None = None     # Recon orientation matrix (default: from space)
    layout: int or str or None = None       # Force output RAS layout (default: same as affine)
    fov: tuple or str = None                # Field-of-view of the recon space: shape or 'bb' (default: from space)
    crop: float = 0                         # Crop size (in pct) if fov == 'bb'


class RegularizationOptions(Option):
    """Options for the regularization"""
    norm: str = 'jtv'                       # Norm to optimize: {'jtv', 'tv', 'tkh', None}
    factor: float or list = [1e3, 1e0]      # Regularization factor


class OptimOptions(Option):
    """Options for the optimizer(s)"""
    max_iter_gn: int = 3                   # Number of Gauss-Newton iterations
    max_iter_cg: int = 32                  # Number of Conjugate Gradient iteration
    max_iter_rls: int = 10                 # Number of Reweighted LS iterations
    tolerance_gn: float = 1e-5
    tolerance_cg: float = 1e-5
    tolerance_rls: float = 1e-5


class BackendOptions(Option):
    """Dtype and device on which to perform computations"""
    dtype: torch.dtype = torch.get_default_dtype()  # Data type used for computations
    device: str or torch.device = None              # Device ('cpu' / 'cuda')


class PreprocOptions(Option):
    """Which preprocessing steps should be performed"""
    register: bool = True               # Rigid registration between series


class ESTATICSOptions(Option):
    """Encapsulate all options"""
    model: str = 'nonlin'               # Model type: {'loglin', 'nonlin'}
    preproc: PreprocOptions = PreprocOptions()
    recon: ReconOptions = ReconOptions()
    optim: OptimOptions = OptimOptions()
    backend: BackendOptions = BackendOptions()
    regularization: RegularizationOptions = RegularizationOptions()
    verbose: int or bool = 1
