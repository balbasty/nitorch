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


class RobustLoss(OptimizationLoss):

    order = 2               # Hessian defined
    reweight = None         # Reweighting function
    reweight_param = []     # Additional parameters of reweight function

    def __init__(self, lam=None, joint=False, dim=None, cache=True):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like, optional
            Precision. If None, estimated by IRLS.
        joint : bool, default=False
            Joint sparsity across channels
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        cache : bool, default=True
            Cache estimated precision between calls.
        """
        super().__init__()
        self.lam = lam
        self.compute_lam = lam is None
        self.dim = dim
        self.joint = joint
        self.cache = cache

    def irls(self, moving, fixed, lam, mask, joint, dim, **kwargs):
        nvox = py.prod(fixed.shape[-dim:])
        compute_lam = lam is None or self.compute_lam
        # --- Fixed lam -> no reweighting ------------------------------
        lll = llw = 0
        if not compute_lam:
            weights = self.reweight(moving, fixed, lam=lam, joint=joint,
                                    dim=dim, mask=mask, **kwargs)
            llw = weights[weights > 1e-9].reciprocal_().sum().div_(2 * nvox)
            lam = lam * weights
            return lll, llw, lam

        # --- Estimated lam -> IRLS loop -------------------------------
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
        lll = llw = float('inf')
        for n_iter in range(32):
            lll_prev = lll
            weights = self.reweight(moving, fixed, lam=lam, joint=joint,
                                    dim=dim, mask=mask, **kwargs)
            lam = weighted_precision(moving, fixed, dim=dim, weights=weights)
            lam /= weights.mean(list(range(-dim, 0)))
            lll = -0.5 * lam.log().sum()
            llw = weights[weights > 1e-9].reciprocal_().sum().div_(2 * nvox)
            if abs(lll_prev - lll) < 1e-4:
                break
        if self.cache:
            self.lam = lam
        lam = lam * weights
        return lll, llw, lam

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
        prm = {k: kwargs.pop(k, getattr(self, k, None))
               for k in self.reweight_param}
        dim = dim or (fixed.dim() - 1)

        lll, llw, lam = self.irls(moving, fixed, lam, mask, joint, dim, **prm)

        kwargs['lam'] = lam
        kwargs['dim'] = dim

        llx = mse(moving, fixed, grad=False, hess=False, **kwargs)
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
        prm = {k: kwargs.pop(k, getattr(self, k, None))
               for k in self.reweight_param}
        dim = dim or (fixed.dim() - 1)

        lll, llw, lam = self.irls(moving, fixed, lam, mask, joint, dim, **prm)

        kwargs['lam'] = lam
        kwargs['dim'] = dim

        llx, g = mse(moving, fixed, grad=True, hess=False, **kwargs)
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
        prm = {k: kwargs.pop(k, getattr(self, k, None))
               for k in self.reweight_param}
        dim = dim or (fixed.dim() - 1)

        lll, llw, lam = self.irls(moving, fixed, lam, mask, joint, dim, **prm)

        kwargs['lam'] = lam
        kwargs['dim'] = dim

        llx, g, h = mse(moving, fixed, grad=True, hess=True, **kwargs)
        return llx + llw + lll, g, h

    def clear_state(self):
        if self.compute_lam:
            self.lam = None

    def get_state(self):
        return dict(lam=self.lam)

    def set_state(self, state):
        self.lam = state.get('lam', self.lam)


class MAD(RobustLoss):
    """Median absolute deviation (using IRLS)"""

    reweight = staticmethod(irls_laplace_reweight)


class Tukey(RobustLoss):
    """Tukey's biweight loss (using IRLS)"""

    reweight = staticmethod(irls_tukey_reweight)
    reweight_param = ['c']

    def __init__(self, lam=None, c=4.685, joint=False, dim=None, cache=True):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like, optional
            Precision. If None, estimated by IRLS.
        c : float, default=4.685
            Tukey parameter
        joint : bool, default=False
            Joint sparsity across channels
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        cache : bool, default=True
            Cache estimated precision between calls.
        """
        super().__init__(lam, joint, dim, cache)
        self.c = c
