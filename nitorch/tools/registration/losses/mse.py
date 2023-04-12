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


def weighted_precision(moving, fixed, weights=None, dim=None, keepdim=False):
    """Estimate the (weighted) error precision (= inverse variance)

    Parameters
    ----------
    moving : (..., *spatial) tensor
    fixed : (... *spatial) tensor
    weights : (..., *spatial) tensor, optional
    dim : int, default=`fixed.dim()-1`
    keepdim

    Returns
    -------
    lam : (...) tensor or (..., *ones) tensor
        Precision. If `keepdim`, the spatial dimensions are preserved.

    """
    dim = dim or fixed.dim() - 1
    residuals = (moving - fixed).square_()
    if weights is not None:
        residuals.mul_(weights)
        lam = residuals.sum(dim=list(range(-dim, 0)), keepdim=keepdim)
        lam = lam.div_(weights.sum(dim=list(range(-dim, 0)), keepdim=keepdim))
    else:
        lam = residuals.mean(dim=list(range(-dim, 0)), keepdim=keepdim)
    lam = lam.reciprocal_()  # variance to precision
    return lam


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

    def __init__(self, lam=None, dim=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision. If None, ML-estimated before each step.
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim

    def compute_lam(self, lam, moving, fixed, mask, dim):
        dim = dim or fixed.dim() - 1
        lll = 0
        if lam is None:
            nvox = sumweights(mask, fixed.shape[-dim:], keepdim=True)
            nvoxsum = nvox.sum() if torch.is_tensor(nvox) else nvox
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask, keepdim=True)
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