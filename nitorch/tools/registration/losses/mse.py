from nitorch.core import utils, py
from .base import OptimizationLoss
import torch


def preproc_lam(lam, dim):
    if lam is None:
        lam = 1
    elif lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    return lam


def sumweights(weights, spatial, keepdim=True):
    dim = len(spatial)
    if torch.is_tensor(weights):
        return weights.sum(list(range(-dim, 0)), keepdim=keepdim)
    else:
        nvox = py.prod(spatial)
        if weights:
            nvox *= weights
        return nvox


#: Default prior expected variance, as a fraction of the (weighted)
#: per-channel intensity variance of the fixed image. 1e-3 corresponds to a
#: residual standard deviation of ~3% of the image's ("near-perfect"
#: alignment): it caps the precision on (nearly) identical images while
#: barely biasing ordinary estimates.
PRIOR_FRACTION = 1e-3
#: Default prior degrees of freedom (pseudo-voxels). A few voxels' worth:
#: enough to dominate when S -> 0 on tiny masks, negligible (relative
#: weight nu0/(nu0+N)) on any realistic image.
PRIOR_DOF = 4


def weighted_precision(moving, fixed, weights=None, dim=None, keepdim=False,
                       prior_fraction=PRIOR_FRACTION, prior_dof=PRIOR_DOF,
                       prior_variance=None):
    """Estimate the (weighted) error precision (= inverse variance)

    A conjugate Gamma prior is placed on the precision, parameterised by
    the prior expected variance s0^2 and the prior degrees of freedom nu0::

        lam ~ Gamma(shape = nu0/2, rate = nu0*s0^2/2)
        posterior: Gamma(nu0/2 + N/2, nu0*s0^2/2 + S/2),
                   with S = sum(w * r^2) and N = sum(w)
        estimate:  lam = (nu0 + N) / (nu0*s0^2 + S),
                   i.e. 1/lam = (nu0*s0^2 + N*sigma^2) / (nu0 + N)
                   with sigma^2 = S/N

    i.e. the (posterior-mean) estimate's inverse is a weighted average of
    the prior variance and the sample variance, with weights nu0 and N.
    All quantities are per channel. Unless given, s0^2 is estimated from
    the intensities: `s0^2 = prior_fraction * var(fixed)`, with `var` the
    (weighted) intensity variance within the mask (the second moment if
    the image is constant there; machine epsilon if it is all zeros).

    With nu0 = 0 this is the ML estimate N/S. Identical images (S = 0)
    give lam = (nu0 + N) / (nu0 * s0^2): finite, and equal to 1/s0^2
    when N << nu0 and (1 + N/nu0) / s0^2 in general.

    Parameters
    ----------
    moving : (..., *spatial) tensor
    fixed : (... *spatial) tensor
    weights : (..., *spatial) tensor, optional
    dim : int, default=`fixed.dim()-1`
    keepdim
    prior_fraction : float, default=PRIOR_FRACTION
        s0^2 as a fraction of the intensity variance (if `prior_variance`
        is not given).
    prior_dof : float, default=PRIOR_DOF
        nu0, prior degrees of freedom (pseudo-voxels). 0 disables the prior.
    prior_variance : float or (K|1,) tensor, optional
        s0^2, prior expected variance (per channel).

    Returns
    -------
    lam : (...) tensor or (..., *ones) tensor
        Precision. If `keepdim`, the spatial dimensions are preserved.

    """
    dim = dim or fixed.dim() - 1
    dims = list(range(-dim, 0))
    residuals = (moving - fixed).square_()
    if weights is not None:
        residuals.mul_(weights)
        nvox = weights.sum(dim=dims, keepdim=True)
    else:
        nvox = py.prod(residuals.shape[-dim:])
    ssq = residuals.sum(dim=dims, keepdim=True)
    if prior_dof:
        if prior_variance is None:
            s0 = _prior_variance(fixed, weights, dim, prior_fraction)
        else:
            s0 = utils.unsqueeze(torch.as_tensor(
                prior_variance, dtype=ssq.dtype, device=ssq.device), -1, dim)
        lam = (prior_dof + nvox) / (ssq + prior_dof * s0)
    else:
        lam = nvox / ssq
    if not keepdim:
        lam = lam.reshape(lam.shape[:-dim])
    return lam


def _prior_variance(fixed, weights, dim, fraction):
    """`max(fraction * var(fixed), eps)`, spatial dims kept as ones"""
    dims = list(range(-dim, 0))
    if weights is not None:
        wsum = weights.sum(dim=dims, keepdim=True)
        mean = (fixed * weights).sum(dim=dims, keepdim=True).div_(wsum)
        var = (fixed - mean).square_().mul_(weights)
        var = var.sum(dim=dims, keepdim=True).div_(wsum)
    else:
        var = fixed.var(dim=dims, unbiased=False, keepdim=True)
    if not fraction:
        return var.mul_(0)
    # constant image (within the mask): fall back to its second moment
    eps = torch.finfo(var.dtype).eps
    if weights is not None:
        msq = (fixed.square() * weights).sum(dim=dims, keepdim=True).div_(wsum)
    else:
        msq = fixed.square().mean(dim=dims, keepdim=True)
    var = torch.where(var > eps * msq, var, msq)
    return var.mul_(fraction).clamp_min_(eps)


def mse(moving, fixed, lam=1, dim=None, grad=True, hess=True, mask=None):
    """Mean-squared error loss for optimisation-based registration.

    (A factor 1/2 is included, and the loss is averaged across voxels,
    but not across channels or batches)

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Gaussian noise precision (or IRLS weights)
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return Hessian

    Returns
    -------
    ll : () tensor
        Negative log-likelihood
    g : (..., K, *spatial) tensor, optional
        Gradient with respect to the moving imaged
    h : (..., K, *spatial) tensor, optional
        (Diagonal) Hessian with respect to the moving image

    """
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    mask = mask.to(fixed.device) if mask is not None else 1
    dim = dim or (fixed.dim() - 1)
    lam = preproc_lam(lam, dim)
    nvox = sumweights(mask, fixed.shape[-dim:])

    ll = moving - fixed
    ll = ll.square() if moving.requires_grad else ll.square_()
    ll = 0.5 * ll.mul_(mask).mul_(lam).div_(nvox).sum()

    out = [ll]
    if grad:
        g = (moving - fixed).mul_(mask).mul_(lam).div_(nvox)
        out.append(g)
    if hess:
        h = lam * mask / nvox
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


class MSE(OptimizationLoss):
    """Mean-squared error"""

    order = 2  # Hessian defined

    def __init__(self, lam=None, dim=None, prior_fraction=PRIOR_FRACTION,
                 prior_dof=PRIOR_DOF, prior_variance=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision. If None, MAP-estimated (per channel) before each
            step, under a Gamma prior on the precision
            (see `weighted_precision`).
        prior_fraction : float, default=1e-3
            Prior expected variance s0^2, as a fraction of the fixed
            image's intensity variance (within the mask, per channel).
        prior_dof : float, default=4
            Prior degrees of freedom nu0 (pseudo-voxels).
        prior_variance : float or (K|1,) tensor_like, optional
            Prior expected variance s0^2 (overrides `prior_fraction`).
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim
        self.prior_fraction = prior_fraction
        self.prior_dof = prior_dof
        self.prior_variance = prior_variance

    def compute_lam(self, lam, moving, fixed, mask, dim):
        dim = dim or fixed.dim() - 1
        lll = 0
        if lam is None:
            nvox = sumweights(mask, fixed.shape[-dim:], keepdim=True)
            nvoxsum = nvox.sum() if torch.is_tensor(nvox) else nvox
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask, keepdim=True,
                                     prior_fraction=self.prior_fraction,
                                     prior_dof=self.prior_dof,
                                     prior_variance=self.prior_variance)
            lll = -0.5 * lam.log().mul(nvox).sum().div(nvoxsum)
            for _ in range(dim):
                lam = lam.squeeze(-1)
        return lam, lll

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        lam, lll = self.compute_lam(lam, moving, fixed, mask, dim)
        llx = mse(moving, fixed, dim=dim, lam=lam, mask=mask,
                  grad=False, hess=False)
        return llx + lll

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        lam, lll = self.compute_lam(lam, moving, fixed, mask, dim)
        llx, g = mse(moving, fixed, dim=dim, lam=lam, mask=mask,
                     hess=False)
        return llx + lll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient
        hess : ([B], K, *spatial) tensor
            Diagonal Hessian

        """
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        lam, lll = self.compute_lam(lam, moving, fixed, mask, dim)
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, mask=mask)
        return llx + lll, g, h