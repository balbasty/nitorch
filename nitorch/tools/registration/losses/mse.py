from nitorch.core import utils, py
from .base import OptimizationLoss


def weighted_precision(moving, fixed, weights=None, dim=None):
    """Estimate the (weighted) error precision (= inverse variance)

    Parameters
    ----------
    moving : ([*batch], *spatial) tensor
    fixed : ([*batch], *spatial) tensor
    weights : ([*batch], *spatial) tensor, optional
    dim : int, default=`fixed.dim()-1`

    Returns
    -------
    lam : ([*batch])) tensor
        Precision

    """
    dim = dim or fixed.dim() - 1
    residuals = (moving - fixed).square_()
    if weights is not None:
        residuals.mul_(weights)
        lam = residuals.sum(dim=list(range(-dim, 0)))
        lam = lam.div_(weights.sum(dim=list(range(-dim, 0))))
    else:
        lam = residuals.mean(dim=list(range(-dim, 0)))
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
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    nvox = py.prod(fixed.shape[-dim:])

    if moving.requires_grad:
        ll = moving - fixed
        if mask is not None:
            ll = ll.mul_(mask)
        ll = ll.square().mul_(lam).sum() / (2*nvox)
    else:
        ll = moving - fixed
        if mask is not None:
            ll = ll.mul_(mask)
        ll = ll.square_().mul_(lam).sum() / (2*nvox)

    out = [ll]
    if grad:
        g = moving - fixed
        if mask is not None:
            g = g.mul_(mask)
        g = g.mul_(lam).div_(nvox)
        out.append(g)
    if hess:
        h = lam/nvox
        if mask is not None:
            h = mask * h
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
            Precision
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim

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
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False,
                  mask=mask)
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
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g = mse(moving, fixed, dim=dim, lam=lam, hess=False, **kwargs)
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
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, mask=mask, **kwargs)
        return llx + lll, g, h