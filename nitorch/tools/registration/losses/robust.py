from nitorch.core import utils, py
from .base import OptimizationLoss
from .mse import weighted_precision, mse
import torch
from functools import lru_cache


def preproc_lam(lam, dim):
    if lam is None:
        lam = 1
    elif torch.is_tensor(lam) and lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    return lam


def mask_(x, mask):
    if mask is not None:
        x = x.masked_fill_(mask == 0, 0)
    return x


def mul_(x, mask):
    if mask is not None:
        x = x.mul_(mask)
    return x


def mask_inf_(x):
    # same as x[~isfinite(x)] = 0, without the boolean mask
    return torch.nan_to_num_(x, nan=0., posinf=0., neginf=0.)


def laplace_weights_(s2, eps=1e-5):
    """IRLS weights for l1 from squared standardized residuals (in place)"""
    return s2.sqrt_().clamp_min_(eps).reciprocal_()


def tukey_weights_(s2, c=4.685):
    """IRLS weights for Tukey from squared standardized residuals (in place)

    (1 - s²/c²)² if s² <= c² else 0. Clamping 1 - s²/c² at zero before
    squaring gives the same values as masking, without a boolean mask.
    """
    return s2.div_(-c * c).add_(1).clamp_min_(0).square_()


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
    dim = dim or (fixed.dim() - 1)
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    fixed, moving, lam, mask = utils.to_max_device(fixed, moving, lam, mask)
    lam = preproc_lam(lam, dim)

    weights = (moving - fixed).square_().mul_(lam)
    weights = mul_(weights, mask)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    weights = weights.sqrt_().clamp_min_(eps).reciprocal_()
    weights = mask_(weights, mask)
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
    c = c * c

    dim = dim or (fixed.dim() - 1)
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    fixed, moving, lam, mask = utils.to_max_device(fixed, moving, lam, mask)
    lam = preproc_lam(lam, dim)

    weights = (moving - fixed).square_().mul_(lam)
    weights = mul_(weights, mask)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    zeromsk = weights > c
    weights = weights.div_(-c).add_(1).square()
    weights = weights.masked_fill_(zeromsk, 0)
    weights = mask_(weights, mask)
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

    @staticmethod
    def weights_(s2, **prm):
        """IRLS weights w(s²) = argmin_w ½ w s² + φ(w), in place in s²"""
        raise NotImplementedError

    @staticmethod
    def dual(weights, out=None, **prm):
        """Half-quadratic dual term φ(w), such that
        ρ(s) = min_w ½ w s² + φ(w), with s² = lam * r²"""
        raise NotImplementedError

    @staticmethod
    def kappa(**prm):
        """Target value of mean(w s²) at the estimated precision

        The precision lam is the M-estimate that solves
        mean(w(s) s²) = κ, which minimises ½ lam mean(w r²) - ½ κ log lam.
        κ = E[w(s) s²] for s ~ N(0, 1) makes lam consistent with the
        precision of Gaussian noise."""
        return 1.

    def irls(self, moving, fixed, lam, mask, joint, dim, **kwargs):
        """Compute IRLS weights (and estimate the precision)

        With s² = lam * r², the loss is
            F(lam, w) = mean[½ lam w r² + φ(w)] - ½ κ Σₖ log lamₖ
        where the log term is only present when lam is estimated.
        Minimising over w gives ρ(s) = min_w ½ w s² + φ(w) (see `reweight`
        and `dual`), and minimising over lam gives lam = κ / mean(w r²).
        Both are block minimisations of the same F, so that the returned
        value is F at the final (lam, w) and its derivative w.r.t. the
        moving image is mean(lam w r) (the returned weighted precision
        times the residuals), which is what `mse` computes.

        Returns
        -------
        lll : log-precision term (-½ κ Σ log lam)
        llw : dual term mean(φ(w))
        lam : weighted precision (lam * w)
        """
        dim = dim or fixed.dim() - 1
        for key in self.reweight_param:
            if kwargs.get(key, None) is None:
                kwargs[key] = getattr(self, key)
        if mask is not None:
            nvox = mask.sum(list(range(-dim, 0)), keepdim=True)
        else:
            mask = 1
            nvox = py.prod(fixed.shape[-dim:])

        def dual(weights, out=None):
            llw = mask_inf_(self.dual(weights, out=out, **kwargs))
            if torch.is_tensor(mask):
                llw = llw.mul_(mask)
            return llw.div_(nvox).sum()

        compute_lam = lam is None or self.compute_lam
        # --- Fixed lam -> no reweighting ------------------------------
        lll = 0
        if not compute_lam:
            weights = self.reweight(moving, fixed, lam=lam, joint=joint,
                                    dim=dim, mask=mask, **kwargs)
            llw = dual(weights)
            lam = preproc_lam(lam, dim)
            # (weights may be needed by autograd: no in-place here)
            lam = lam * weights
            return lll, llw, lam

        # --- Estimated lam -> IRLS loop -------------------------------
        # The squared residuals r² do not depend on (lam, w): compute
        # them once, and reweight in place from s² = lam * r².
        # Buffers (allocated at the first iteration, then reused):
        #   weights : s², then w, shape of `self.reweight`'s output
        #   tmp     : lam * r² (joint) or w * r², shape of r²
        kappa = self.kappa(**kwargs)
        fixed_, moving_ = utils.to_max_backend(fixed, moving)
        res2 = (moving_ - fixed_).square_()
        res2 = mul_(res2, mask if torch.is_tensor(mask) else None)
        del fixed_, moving_
        spatial = list(range(-dim, 0))
        zero = (mask == 0) if torch.is_tensor(mask) else None
        weights = tmp = None

        if lam is None:
            # = weighted_precision(moving, fixed, mask), from r²
            if torch.is_tensor(mask):
                lam = res2.sum(spatial, keepdim=True).div_(nvox)
            else:
                lam = res2.mean(spatial, keepdim=True)
            lam = lam.reciprocal_()
        lam = preproc_lam(lam, dim)

        def reweight(lam):
            # same as `self.reweight(moving, fixed, lam, mask=mask)`
            nonlocal weights, tmp
            if joint:
                tmp = torch.mul(res2, lam, out=tmp)
                weights = torch.sum(tmp, dim=-dim-1, keepdim=True, out=weights)
            else:
                weights = torch.mul(res2, lam, out=weights)
            weights = self.weights_(weights, **kwargs)
            if zero is not None:
                weights = weights.masked_fill_(zero, 0)
            return weights

        for n_iter in range(64):
            lam_prev = lam
            weights = reweight(lam)
            tmp = torch.mul(weights, res2, out=tmp)
            lam = tmp.sum(spatial, keepdim=True)
            lam = lam.reciprocal_().mul_(nvox).mul_(kappa)
            if (lam / lam_prev).log().abs().max() < 1e-6:
                break
        # final weights: optimal for the final precision, so that the
        # value, the weights and the precision are consistent
        weights = reweight(lam)
        del res2
        lll = -0.5 * kappa * lam.log().sum()
        llw = dual(weights, out=tmp if tmp.shape == weights.shape else None)
        if self.cache:
            self.lam = lam

        # weighted precision lam * w, in a buffer that is free by now
        shape = torch.broadcast_shapes(lam.shape, weights.shape)
        out = (weights if shape == weights.shape else
               tmp if shape == tmp.shape else None)
        lam = torch.mul(weights, lam, out=out)
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
    weights_ = staticmethod(laplace_weights_)

    @staticmethod
    def dual(weights, out=None, **prm):
        # ρ(s) = |s| = min_w ½ w s² + 1 / (2 w)
        return torch.reciprocal(weights, out=out).mul_(0.5)

    @staticmethod
    def kappa(**prm):
        # κ = 1: F is the negative log-likelihood of a Laplace
        # distribution, mean(√lam |r|) - ½ log lam, and lam is its
        # maximum-likelihood estimate 1 / mean(|r|)²
        return 1.


@lru_cache
def _tukey_kappa(c):
    """E[w(s) s²] for s ~ N(0, 1), with w(s) = (1 - s²/c²)² if |s| < c"""
    s = torch.linspace(-c, c, 20001, dtype=torch.double)
    f = s.square() * (1 - (s / c).square()).square()
    f = f * (-0.5 * s.square()).exp() / (2 * torch.pi) ** 0.5
    return torch.trapezoid(f, s).item()


class Tukey(RobustLoss):
    """Tukey's biweight loss (using IRLS)"""

    reweight = staticmethod(irls_tukey_reweight)
    weights_ = staticmethod(tukey_weights_)
    reweight_param = ['c']

    @staticmethod
    def dual(weights, out=None, c=4.685, **prm):
        # ρ(s) = c²/6 (1 - (1 - s²/c²)³) if |s| < c else c²/6
        #      = min_w ½ w s² + c²/6 (1 - 3w + 2 w^(3/2))
        # (1 - 3w) is built in `out`; only w^(3/2) needs a temporary
        out = torch.mul(weights, -3, out=out).add_(1)
        return out.add_(weights.pow(1.5), alpha=2).mul_(c * c / 6)

    @staticmethod
    def kappa(c=4.685, **prm):
        # Tukey's ρ is bounded, so it is not a log-likelihood and
        # κ = 1 would bias lam. κ = E[w s²] under Gaussian noise
        # (≈ 0.76 for c = 4.685) makes lam consistent at the Gaussian.
        return _tukey_kappa(float(c))

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
