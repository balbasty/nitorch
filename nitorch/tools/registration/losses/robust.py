from nitorch.core import utils, py
from .base import OptimizationLoss
from .mse import weighted_precision, mse


def irls_laplace_reweight(moving, fixed, lam=1, joint=False, eps=1e-5,
                          dim=None, mask=None):
    """Update iteratively reweighted least-squares weights for l1

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Inverse-squared scale parameter of the Laplace distribution.
        (equivalent to Gaussian noise precision)
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions

    Returns
    -------
    weights : (..., K|1, *spatial) tensor
        IRLS weights

    """
    if lam is None:
        lam = 1
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    weights = (moving - fixed).square_().mul_(lam)
    if mask is not None:
        weights = weights.mul_(mask)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    weights = weights.sqrt_().clamp_min_(eps).reciprocal_()
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, 0)
    return weights


def irls_tukey_reweight(moving, fixed, lam=1, c=4.685, joint=False, dim=None,
                        mask=None):
    """Update iteratively reweighted least-squares weights for Tukey's biweight

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Equivalent to Gaussian noise precision
        (used to standardize the residuals)
    c  : float, default=4.685
        Tukey's threshold.
        Approximately equal to a number of standard deviations above
        which the loss is capped.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions

    Returns
    -------
    weights : (..., K|1, *spatial) tensor
        IRLS weights

    """
    if lam is None:
        lam = 1
    c = c * c
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    weights = (moving - fixed).square_().mul_(lam)
    if mask is not None:
        weights = weights.mul_(mask)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    zeromsk = weights > c
    weights = weights.div_(-c).add_(1).square()
    weights[zeromsk].zero_()
    return weights


class MAD(OptimizationLoss):
    """Median absolute deviation (using IRLS)"""

    order = 2  # Hessian defined

    def __init__(self, lam=None, joint=False, dim=None):
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
        self.joint = joint

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
        joint = kwargs.pop('joint', self.joint)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint,
                                        dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False,
                  **kwargs)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll

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
        joint = kwargs.pop('joint', self.joint)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint,
                                        dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=False)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g

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
        joint = kwargs.pop('joint', self.joint)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint,
                                        dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=True)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g, h


class Tukey(OptimizationLoss):
    """Tukey's biweight loss (using IRLS)"""

    order = 2  # Hessian defined

    def __init__(self, lam=None, c=4.685, joint=False, dim=None):
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
        self.joint = joint
        self.c = c

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
        joint = kwargs.pop('joint', self.joint)
        c = kwargs.pop('c', self.c)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint,
                                      dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll

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

        """
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        c = kwargs.pop('c', self.c)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint,
                                      dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=False)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g

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
        joint = kwargs.pop('joint', self.joint)
        c = kwargs.pop('c', self.c)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint,
                                      dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=True)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g, h