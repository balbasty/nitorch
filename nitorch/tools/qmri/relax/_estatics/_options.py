import torch
from nitorch.core.options import Option
from nitorch.core import py


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
    factor: float or list = [1, 5e-4]       # Regularization factor


class DistortionOption(Option):
    """Options for the regularization or distortion fields"""
    enable: bool = False                    # enable distortion correction
    model: str = 'svf'                      # {'smalldef', 'svf', 'shoot'}
    steps: int = 8                          # Number of integration steps
    factor: float = 1                       # Global regularization factor
    absolute: float = 0                     # Penalty on absolute displacements
    membrane: float = 0                     # Penalty on 1st derivatives
    bending: float = 1                      # Penalty on 2nd derivatives
    te_scaling: bool or str = None          # {None or False, 'pre' or True, 'post'}


class OptimOptions(Option):
    """Options for the optimizer(s)"""
    max_iter_gn: int = 3                   # Number of Gauss-Newton iterations
    max_iter_cg: int = 2                   # Number of Conjugate Gradient iteration
    max_iter_rls: int = 10                 # Number of Reweighted LS iterations
    tolerance_gn: float = 1e-5             # Tolerance for GN early stopping
    tolerance_cg: float = 0                # Tolerance for CG early stopping (advised to be zero now that we use FMG)
    tolerance_rls: float = 1e-5            # Tolerance for RLS early stopping


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
    distortion: DistortionOption = DistortionOption()
    verbose: int or bool = 1
    plot: bool = False
    uncertainty: bool = False  # Whether to return uncertainty maps (posterior variance)

    __protected_fields__ = (*Option.__protected_fields__, 'cleanup_')

    def cleanup_(self):
        # Make all values consistent so that the main code is readable
        if self.distortion.te_scaling is True:
            self.distortion.te_scaling = 'pre'
        if not self.distortion.te_scaling:
            self.distortion.te_scaling = ''
        self.distortion.te_scaling = self.distortion.te_scaling.lower()
        if not self.regularization.norm:
            self.regularization.norm = ''
        self.regularization.norm = self.regularization.norm.lower()
        if self.regularization.norm == 'none':
            self.regularization.norm = ''
        self.regularization.factor = py.ensure_list(self.regularization.factor, 2)
        return self

