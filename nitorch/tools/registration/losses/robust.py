from nitorch.core import utils, py
from nitorch.core.math import nan_to_num_
from .base import OptimizationLoss
from .mse import weighted_precision, mse
import torch
import math
import warnings
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
    return nan_to_num_(x, nan=0., posinf=0., neginf=0.)


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
    _last_lam = None        # Precision estimated by the last evaluation

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
    def kappa(nb_channels=1, **prm):
        """Target value of mean(w sₖ²) at the estimated precision

        The precision lamₖ is the M-estimate that solves
        mean(w sₖ²) = κ, which minimises ½ lamₖ mean(w rₖ²) - ½ κ log lamₖ.
        κ = E[w sₖ²] for s ~ N(0, I) makes lam consistent with the
        precision of Gaussian noise.

        With joint weights, w = w(S²) with S² = Σₖ sₖ² over the
        `nb_channels` = C channels, which is χ²-distributed with C
        degrees of freedom under Gaussian noise, so that
        κ(C) = E[w(S²) sₖ²] = E[w(S²) S²] / C. Otherwise, C = 1."""
        return 1.

    def reweight_params(self, joint=False, nb_channels=1):
        """Default parameters of `reweight` (e.g. Tukey's `c`), which may
        depend on the number of channels that are jointly reweighted"""
        return {key: getattr(self, key) for key in self.reweight_param}

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
        # joint weights depend on the squared norm across the C channels
        nb_channels = (utils.expanded_shape(moving.shape, fixed.shape)
                       [-dim-1] if joint else 1)
        default = self.reweight_params(joint, nb_channels)
        for key in self.reweight_param:
            if kwargs.get(key, None) is None:
                kwargs[key] = default[key]
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
        # --- Fixed lam -> no IRLS loop --------------------------------
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
        fixed_, moving_ = utils.to_max_backend(fixed, moving)
        res2 = (moving_ - fixed_).square_()
        res2 = mul_(res2, mask if torch.is_tensor(mask) else None)
        del fixed_, moving_
        kappa = self.kappa(nb_channels=nb_channels, **kwargs)
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
        self._last_lam = lam

        # weighted precision lam * w, in a buffer that is free by now
        shape = utils.expanded_shape(lam.shape, weights.shape)
        out = (weights if shape == weights.shape else
               tmp if shape == tmp.shape else None)
        lam = torch.mul(weights, lam, out=out)
        return lll, llw, lam

    def weighted_lam(self, moving, fixed, mask=None, dim=None):
        """Weighted precision lam * w (e.g., for display)

        With an estimated lam, the precision estimated by the last
        evaluation of the loss (`_last_lam`) is reused, so that this
        costs one reweighting instead of an IRLS loop. Otherwise (fixed
        lam, or nothing estimated yet), this is the output of `irls`.
        In both cases, the state of the loss (cached lam) is unchanged.

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image
        mask : ([B], K|1, *spatial) tensor, optional
            Mask or weights
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions

        Returns
        -------
        lam : ([B], K|1, *spatial) tensor
            Weighted precision
        """
        dim = dim or self.dim or (fixed.dim() - 1)
        lam = self._last_lam
        if lam is None or not self.compute_lam:
            # (without updating the cached lam, so that displaying the
            # weights does not change the next evaluation of the loss)
            state = self.lam, self._last_lam
            try:
                return self.irls(moving, fixed, self.lam, mask, self.joint,
                                 dim)[-1]
            finally:
                self.lam, self._last_lam = state
        nb_channels = (utils.expanded_shape(moving.shape, fixed.shape)
                       [-dim-1] if self.joint else 1)
        prm = self.reweight_params(self.joint, nb_channels)
        weights = self.reweight(moving, fixed, lam=lam, joint=self.joint,
                                dim=dim, mask=1 if mask is None else mask,
                                **prm)
        return torch.mul(weights, lam)

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
        prm = {k: kwargs.pop(k, None) for k in self.reweight_param}
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
        prm = {k: kwargs.pop(k, None) for k in self.reweight_param}
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
        prm = {k: kwargs.pop(k, None) for k in self.reweight_param}
        dim = dim or (fixed.dim() - 1)

        lll, llw, lam = self.irls(moving, fixed, lam, mask, joint, dim, **prm)

        kwargs['lam'] = lam
        kwargs['dim'] = dim

        llx, g, h = mse(moving, fixed, grad=True, hess=True, **kwargs)
        return llx + llw + lll, g, h

    def clear_state(self):
        if self.compute_lam:
            self.lam = None
        self._last_lam = None

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
    def kappa(nb_channels=1, **prm):
        # κ = 1: F is the negative log-likelihood of a Laplace
        # distribution, mean(√lam |r|) - ½ log lam, and lam is its
        # maximum-likelihood estimate 1 / mean(|r|)²
        #
        # Joint weights: ρ(s) = ‖s‖ is the negative log-likelihood of
        # the C-variate density ∝ Πₖ √lamₖ exp(-‖s‖) (the normaliser of
        # exp(-‖s‖) does not depend on lam), so κ = 1 is still the
        # maximum-likelihood estimate, for any C. Under that density,
        # E[w sₖ²] = E[‖s‖] / C = 1 (‖s‖ ~ Gamma(C, 1)), so lam is
        # consistent for its own noise model. Like for C = 1, it is not
        # a Gaussian precision: on Gaussian noise, lam is (C / E[χ_C])²
        # times the precision (π/2 for C = 1).
        return 1.


@lru_cache
def _tukey_kappa(c):
    """E[w(s) s²] for s ~ N(0, 1), with w(s) = (1 - s²/c²)² if |s| < c"""
    s = torch.linspace(-c, c, 20001, dtype=torch.double)
    f = s.square() * (1 - (s / c).square()).square()
    f = f * (-0.5 * s.square()).exp() / (2 * torch.pi) ** 0.5
    return torch.trapezoid(f, s).item()


def _gammainc_half(n, x):
    """Regularised lower incomplete gamma function P(n/2, x), integer n ≥ 1

    P(½, x) = erf(√x), P(1, x) = 1 - exp(-x), and
    P(a + 1, x) = P(a, x) - x^a exp(-x) / Γ(a + 1)."""
    if x <= 0:
        return 0.
    a, p = (0.5, math.erf(math.sqrt(x))) if n % 2 else (1., -math.expm1(-x))
    while a < n / 2:
        p -= math.exp(a * math.log(x) - x - math.lgamma(a + 1))
        a += 1
    return p


@lru_cache
def _tukey_kappa_joint(c, nb_channels):
    """E[w(S²) S²] / C for S² ~ χ²(C), with w(S²) = (1 - S²/c²)² if S² < c²

    This is E[w(S²) sₖ²] for s ~ N(0, I_C) and S² = ‖s‖², by symmetry.
    With E[Xᵐ 1(X < a)] = 2ᵐ Γ(C/2 + m) / Γ(C/2) P(C/2 + m, a/2) for
    X ~ χ²(C), and w(X) X = X - 2 X²/c² + X³/c⁴:
        κ(C) = P(C/2 + 1, c²/2)
             - 2 (C + 2) / c² P(C/2 + 2, c²/2)
             + (C + 2) (C + 4) / c⁴ P(C/2 + 3, c²/2)
    For C = 1 it matches `_tukey_kappa` (to about 1e-16).

    The Gaussian precision is a stable fixed point of the IRLS (a local
    minimum of the objective in lam) if d/da E[w(aX) aX] > 0 at a = 1,
    i.e., E[X (1 - X/c²) (1 - 3X/c²) 1(X < c²)] > 0. Otherwise (e.g.,
    C ≥ 7 for c = 4.685), lam drifts away from it, so we warn.
    """
    C, c2 = nb_channels, c * c
    x = c2 / 2
    p1, p2, p3 = (_gammainc_half(C + 2 * m, x) for m in (1, 2, 3))
    q2, q3 = (C + 2) / c2, (C + 2) * (C + 4) / (c2 * c2)
    if p1 - 4 * q2 * p2 + 3 * q3 * p3 <= 0:
        warnings.warn(f'Tukey with joint=True and {C} channels: c={c} is '
                      f'too small for the precision to be estimated '
                      f'(it may diverge). Use a larger c.')
    return p1 - 2 * q2 * p2 + q3 * p3


def _tukey_mean_ws2(a, c, nb_channels):
    """E[w(aX) aX] / C for X ~ χ²(C): the expected mean(w sₖ²) when lam
    is `a` times the precision of Gaussian noise (κ(C) at a = 1)"""
    C, c2 = nb_channels, c * c
    x = c2 / (2 * a)
    p1, p2, p3 = (_gammainc_half(C + 2 * m, x) for m in (1, 2, 3))
    return (a * p1 - 2 * a * a * (C + 2) / c2 * p2
            + a ** 3 * (C + 2) * (C + 4) / (c2 * c2) * p3)


def _tukey_breakdown(c, nb_channels):
    """Largest fraction of zero-weight voxels (gross outliers) for which
    the IRLS precision has a finite fixed point, with Gaussian inliers

    With a fraction ε of outliers, mean(w sₖ²) = (1 - ε) g(a), with
    g(a) = `_tukey_mean_ws2(a)`, so that lam = a × precision exists iff
    (1 - ε) max_a g(a) ≥ κ = g(1): ε* = 1 - g(1) / max_a g(a).
    (g rises then falls with a: golden-section search on log a.)
    """
    def f(t):
        return _tukey_mean_ws2(math.exp(t), c, nb_channels)
    r = (math.sqrt(5) - 1) / 2
    lo, hi = math.log(1e-3), math.log(1e3)
    t1, t2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = f(t1), f(t2)
    for _ in range(80):
        if f1 < f2:
            lo, t1, f1 = t1, t2, f2
            t2 = lo + r * (hi - lo)
            f2 = f(t2)
        else:
            hi, t2, f2 = t2, t1, f1
            t1 = hi - r * (hi - lo)
            f1 = f(t1)
    return 1 - _tukey_mean_ws2(1., c, nb_channels) / max(f1, f2)


@lru_cache
def _tukey_c(nb_channels, c=4.685):
    """Tukey's c for joint weights across C channels

    c(C) is such that the breakdown of the estimated precision
    (`_tukey_breakdown`) is that of c with a single channel (48.7% for
    c = 4.685), so that the IRLS is as robust for any C, and the
    Gaussian precision is a stable fixed point. c(1) = c, and
    c(C) ≈ 3.3 √C for large C (√C is the typical norm of the noise).
    """
    if nb_channels == 1:
        return c
    target = _tukey_breakdown(c, 1)
    # the breakdown increases with c, and is smaller than target at c
    lo, hi = c, c + 6 * math.sqrt(nb_channels)
    for _ in range(64):
        mid = (lo + hi) / 2
        if _tukey_breakdown(mid, nb_channels) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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
    def kappa(c=4.685, nb_channels=1, **prm):
        # Tukey's ρ is bounded, so it is not a log-likelihood and
        # κ = 1 would bias lam. κ = E[w s²] under Gaussian noise
        # (≈ 0.76 for c = 4.685) makes lam consistent at the Gaussian.
        # With joint weights, w = w(S²) with S² ~ χ²(C), and
        # κ(C) = E[w(S²) S²] / C. With the default c, c = c(C) (see
        # `reweight_params`): κ ≈ 0.786, 0.798, 0.808, 0.814, 0.816 for
        # C = 2, 3, 5, 8, 10. (With a fixed c = 4.685 instead, it would
        # be 0.685, 0.617, 0.493, and the Gaussian precision would not
        # be a stable fixed point of the IRLS for C ≥ 7.)
        if nb_channels == 1:
            return _tukey_kappa(float(c))
        return _tukey_kappa_joint(float(c), int(nb_channels))

    def __init__(self, lam=None, c=None, joint=False, dim=None, cache=True):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like, optional
            Precision. If None, estimated by IRLS.
        c : float, default=4.685
            Tukey parameter, in units of standardised residual.
            If not provided (default) and `joint=True`, the threshold on
            the norm of the C standardised residuals is c(C), for which
            the estimated precision is as robust as with 4.685 and a
            single channel (same breakdown, 49%):
            c(C) = 4.685, 5.83, 6.75, 8.25, 10.05, 11.09 for C = 1, 2, 3,
            5, 8, 10 (`scale_c`). A value that is provided is used as is
            (a threshold on the joint norm).
        joint : bool, default=False
            Joint sparsity across channels
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        cache : bool, default=True
            Cache estimated precision between calls.
        """
        super().__init__(lam, joint, dim, cache)
        self.scale_c = c is None
        self.c = 4.685 if c is None else c

    def reweight_params(self, joint=False, nb_channels=1):
        # With joint weights, ‖s‖² ~ χ²(C) under Gaussian noise, so a
        # threshold c meant for one channel gets closer and closer to the
        # typical norm √C: the fraction of outliers that the estimated
        # lam tolerates drops (49%, 31%, 18%, 3% for C = 1, 2, 3, 5 with
        # c = 4.685), and for C ≥ 7 the Gaussian precision is not even a
        # stable fixed point of the IRLS. c(C) keeps the breakdown of
        # c = 4.685 at C = 1 (49%), for any C.
        c = self.c
        if self.scale_c and joint and nb_channels > 1:
            c = _tukey_c(int(nb_channels), float(c))
        return dict(c=c)
