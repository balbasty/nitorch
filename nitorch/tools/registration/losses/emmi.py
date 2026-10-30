r"""
EM Mutual Information. This loss has (mostly) the same behaviour as
`spm_maff8` (see also `nitorch.tools.registration.affine_tpm`, which
uses the same model with a TPM as moving image).

Notation (matching the code)
----------------------------
. Let 𝒙 ∈ ⟦1,J⟧ᴺ be the *observed* label map (the fixed image).
  Typically, an image is discretized into J bins, and 𝑥ₙ indexes
  the bin in which voxel n falls. But it could also be any segmentation.
. Let 𝒚 ∈ ⟦1,K⟧ᴺ be a *hidden* label map: the K classes of the moving
  image (with a potentially different labelling scheme: J != K).
. Let 𝜇 ∈ ℝᴺᴷ be the (deformed) moving image, i.e., the prior
  probability to see class k in voxel n: p(𝑦ₙ = 𝑘) = 𝜇ₙₖ.
. Let 𝑯 ∈ ℝᴶᴷ link the two labellings (`prior`, of shape (..., J, K)
  in the code):  p(𝑥ₙ = 𝑗 | 𝑦ₙ = 𝑘) ∝ 𝐻ⱼₖ.

Objective
---------
. We want to find the parameters {𝑯, 𝜇} that maximize the marginal
  likelihood:
      p(𝒙; 𝑯, 𝜇) = ∏ₙ ∑ₖ p(𝑥ₙ | 𝑦ₙ = 𝑘; 𝑯) p(𝑦ₙ = 𝑘; 𝜇)
  where we use ";" to separate random variables from parameters, i.e.,
      𝓛(𝑯, 𝜇) = ∑ₙ ln ∑ₖ 𝐻[𝑥ₙ, 𝑘] 𝜇ₙₖ
. In order to maximize 𝑯, we use an EM algorithm:
      q(𝑦ₙ = 𝑘 | 𝑥ₙ) ∝ 𝐻[𝑥ₙ, 𝑘] 𝜇ₙₖ                             (E-step)
      p(𝑥 = 𝑗, 𝑦 = 𝑘) ∝ ∑ₙ 𝛿(𝑥ₙ = 𝑗) q(𝑦ₙ = 𝑘 | 𝑥ₙ)               (M-step)
      𝐻ⱼₖ = p(𝑥 = 𝑗, 𝑦 = 𝑘) / (p(𝑥 = 𝑗) p(𝑦 = 𝑘))
. 𝑯 is therefore the "normalized" joint histogram p(x,y) / (p(x) p(y))
  = p(x|y) / p(x), not the conditional p(x|y). The factor 1/p(𝑥ₙ)
  cancels in the E-step, and only shifts 𝓛 by -∑ₙ ln p(𝑥ₙ), a constant
  that depends on the fixed image histogram only. With this
  normalization, when 𝜇 is one-hot (and the histogram is not smoothed),
  𝓛/N is the mutual information between 𝒙 and 𝒚.
. If `fwhm` is set, the joint histogram is smoothed along the fixed-bin
  axis (j) after each M-step. The M-step is then not an exact maximizer
  of 𝓛, and EM is not guaranteed to be monotonic: it stops when the
  absolute change of 𝓛 is below `tolerance * N`.
. With 𝑯 fixed, we differentiate 𝓛 with respect to 𝜇. If 𝑯 were the
  exact maximizer 𝑯*(𝜇), this would be the gradient of the profile
  log-likelihood 𝓛(𝑯*(𝜇), 𝜇) (envelope theorem). Because EM is
  stopped early (and the histogram may be smoothed), this is only
  approximate.
. If we do not have an observed hard segmentation, but a (fixed) separable
  distribution qₙⱼ = q(𝑥ₙ = 𝑗) over 𝒙, we maximize the expected
  log-likelihood instead:
      𝓛 = 𝔼_q[ln p(𝒙)] = ∑ₙⱼ qₙⱼ ln ∑ₖ 𝐻ⱼₖ 𝜇ₙₖ
  EM then treats each pair (n, j) as an observation with weight qₙⱼ:
      rₙⱼₖ ∝ 𝐻ⱼₖ 𝜇ₙₖ                                           (E-step)
      p(𝑥 = 𝑗, 𝑦 = 𝑘) ∝ ∑ₙ qₙⱼ rₙⱼₖ                              (M-step)

Sign and scale conventions
--------------------------
. `emmi` (and the `EMMI` loss) return the value to *minimize*:
  -𝓛/N, evaluated at the final 𝑯, where N is the total voxel weight,
  together with its gradient and Hessian with respect to 𝜇 (𝑯 fixed).
. `affine_tpm.em_prior` returns +𝓛 (summed over voxels, to maximize),
  with the same 𝑯 and the same EM.
"""
from nitorch.core import utils, linalg
from nitorch import spatial
from .base import OptimizationLoss
import torch


def emmi(moving, fixed, *args, **kwargs):
    """Expectation-Maximization Mutual Information

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
    fixed : (..., J|1, *spatial) tensor
        Moving probabilities 𝜇 (K classes)
    fixed : (..., J|1, *spatial) tensor
        Fixed image: discretized into J bins (integer, one channel)
        or a soft (float) distribution over J bins.
    dim : int, default=`moving.dim()-1`
    prior : (..., J, K) tensor
        Initial normalized joint histogram H = p(x,y)/(p(x)p(y))
    fwhm : float, optional
        Smoothing of the joint histogram along the fixed-bin axis.
        Default: no smoothing.
    max_iter : int, default=32
        Maximum number of EM iterations (0 = H fixed).
    tolerance : float, default=1e-5
        Stop EM when the absolute change of the log-likelihood is
        less than `tolerance * N`.
    weights : tensor (..., 1, *spatial) tensor, optional

    Returns
    -------
    ll : () tensor
        Negative average log-likelihood -𝓛/N, at the final H.
    grad : (..., K, *spatial) tensor, if `grad`
        Gradient of -𝓛/N with respect to 𝜇 (H fixed)
    hess : (..., K*(K+1)//2, *spatial) tensor, if `hess`
        Hessian of -𝓛/N with respect to 𝜇 (H fixed), packed: K diagonal
        elements, then the (k, l), k < l, elements in row-major order.
    prior : (..., J, K) tensor, if `return_prior`
        Final H

    """
    if fixed.dtype.is_floating_point:
        return emmi_soft(moving, fixed, *args, **kwargs)
    else:
        return emmi_hard(moving, fixed, *args, **kwargs)


def emmi_prepare(moving, fixed, weights, prior, dim):

    if not fixed.dtype.is_floating_point:
        # torch requires indexing tensors to be long :(
        fixed = fixed.long()

    # --- Get shape ----------------------------------------------------
    if prior is not None:
        *_, J, K = prior.shape
        # TODO: detect if moving and/or fixed have an implicit class
    else:
        K = moving.shape[-dim-1]
        if fixed.dtype.is_floating_point:
            J = fixed.shape[-dim-1]
        else:
            J = fixed.max() + 1

    # --- Compute batch size -------------------------------------------
    # make sure all tensors have the same batch dimensions
    # (using zero-strides so no spurious allocation)
    shapes = [moving.shape[:-dim-1], fixed.shape[:-dim-1]]
    if prior is not None:
        shapes += [prior.shape[:-2]]
    batch = utils.expanded_shape(*shapes)

    # --- initialize prior ---------------------------------------------
    if prior is None:
        prior = moving.new_ones([*batch, J, K])
        prior /= prior.sum([-1, -2], keepdim=True)
        prior /= prior.sum(dim=-2, keepdim=True) * prior.sum(dim=-1, keepdim=True)
        # prior /= prior.sum(dim=-2, keepdim=True)  # conditional X | Z
    else:
        prior = prior.clone()

    # --- create mask/weights ------------------------------------------
    if weights is None:
        weights = 1
    weights = weights * (moving != 0).any(-dim-1, keepdim=True)
    weights *= torch.isfinite(moving).all(-dim-1, keepdim=True)
    if fixed.dtype.is_floating_point:
        weights *= (fixed != 0).any(-dim-1, keepdim=True)
        weights *= torch.isfinite(fixed).all(-dim-1, keepdim=True)

    # --- flatten spatial dimensions + expand batch --------------------
    shape = moving.shape[-dim:]
    moving = moving.reshape([*moving.shape[:-dim], -1])
    moving = moving.expand([*batch, *moving.shape[-2:]])     # [*batch, K, N]
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])
    fixed = fixed.expand([*batch, *fixed.shape[-2:]])       # [*batch, J|1, N]
    weights = weights.reshape([*weights.shape[:-dim], -1])
    weights = weights.expand([*batch, *weights.shape[-2:]])   # [*batch, 1, N]

    return moving, fixed, weights, prior, shape


def em_converged(ll, ll_prev, tolerance, nvox):
    """EM stopping test.

    With a smoothed histogram, EM is not monotonic, so a decrease of the
    log-likelihood does not mean that it has converged: stop only when
    the absolute change is small (or the value is not finite).
    """
    if not torch.isfinite(ll).all():
        return True
    return bool(((ll - ll_prev).abs() <= tolerance * nvox).all())


def emmi_hard(moving, fixed, dim=None, prior=None, fwhm=None, max_iter=32,
              weights=None, grad=True, hess=True, return_prior=False,
              tolerance=1e-5):
    # ------------------------------------------------------------------
    #           PREPARATION
    # ------------------------------------------------------------------
    tiny = 1e-16
    dim = dim or (moving.dim() - 1)
    moving, fixed, weights, prior, shape = emmi_prepare(
        moving, fixed, weights, prior, dim)

    *batch, J, K = prior.shape
    Nb = len(batch)
    N = moving.shape[-1]
    Nm = weights.sum()

    # ------------------------------------------------------------------
    #           EM LOOP
    # ------------------------------------------------------------------
    ll = torch.as_tensor(-float('inf'), dtype=torch.double)
    z = moving.new_empty([*batch, K, N])
    prior0 = torch.empty_like(prior)
    for n_iter in range(max_iter):
        ll_prev = ll
        # --------------------------------------------------------------
        # E-step
        # ------
        # estimate responsibilities of each moving cluster for each
        # fixed voxel using Bayes' rule:
        #   p(z[n] == k | x[n] == j[n]) ∝ p(x[n] == j[n] | z[n] == k) p(z[n] == k)
        #
        # . j[n] is the discretized fixed image
        # . p(z[n] == k) is the moving template
        # . p(x[n] == j[n] | z[n] == k) is the conditional prior evaluated at (j[n], k)
        # --------------------------------------------------------------
        z = sample_prior(prior, fixed, z)
        z *= moving

        # --------------------------------------------------------------
        # compute log-likelihood (log_sum of the posterior)
        # ll = Σ_n log p(x[n] == j[n])
        #    = Σ_n log Σ_k p(x[n] == j[n] | z[n] == k)  p(z[n] == k)
        #    = Σ_n log Σ_k p(z[n] == k | x[n] == j) + constant{\z}
        # --------------------------------------------------------------
        ll = z.sum(-2, keepdim=True)
        ll = add_tiny_(ll, Nb)
        z /= ll
        ll = ll.log_().mul_(weights).sum(dtype=torch.double)

        z *= weights

        # --------------------------------------------------------------
        # M-step
        # ------
        # estimate joint prior by maximizing Q = E_{Z;H,mu}[ln p(X, Z; H)]
        # => H_jk = p(x == j, z == k) ∝ Σ_n p(z[n] == k | x[n] == j) 𝛿(x[n] == j)
        # --------------------------------------------------------------
        prior0 = scatter_prior(prior0, fixed, z)
        prior.copy_(prior0).add_(tiny)
        # make it a joint distribution
        prior /= add_tiny_(prior.sum(dim=[-1, -2], keepdim=True), Nb)

        if fwhm:
            # smooth "prior" for the prior
            prior = prior.transpose(-1, -2)
            prior = spatial.smooth(prior, dim=1, basis=0, fwhm=fwhm, bound='replicate')
            prior = prior.transpose(-1, -2)

        # prior /= prior.sum(dim=[-1, -2], keepdim=True)
        # MI-like normalization
        prior /= add_tiny_(prior.sum(dim=-1, keepdim=True) * prior.sum(dim=-2, keepdim=True), Nb)
        if em_converged(ll, ll_prev, tolerance, Nm):
            break

    # ------------------------------------------------------------------
    #           OBJECTIVE
    # ------------------------------------------------------------------
    # The objective is the (negative, averaged) marginal log-likelihood
    # evaluated at the final prior H = p(x,y) / (p(x) p(y)):
    #   𝓛 = -1/N Σₙ ln Σₖ H[x[n], k] μ[n, k]
    # This is the function whose gradient and Hessian are returned below.
    # When μ is a hard (one-hot) map (and fwhm=0), 𝓛 = -MI[x, y].
    #
    # The previous implementation returned -Σ_{jk} p(j,k) log H[j,k],
    # the MI of the expected joint histogram, which is a different
    # function of μ: its finite differences did not match the gradient
    # (not even in sign), and Gauss-Newton stopped early.
    g = sample_prior(prior, fixed)                  # H[x[n], k]
    norm = (g * moving).sum(-2, keepdim=True)       # Σₖ H[x[n], k] μ[n, k]
    norm = add_tiny_(norm, Nb)
    ll = -(norm.log() * weights).sum(dtype=torch.double) / Nm
    out = [ll]

    # ------------------------------------------------------------------
    #           GRADIENTS
    # ------------------------------------------------------------------
    # The prior H is kept fixed (it maximises 𝓛 w.r.t. H, so its own
    # variation does not contribute to the first-order derivative).
    # With aₙₖ = H[x[n], k] / Σⱼ H[x[n], j] μ[n, j]:
    #   ∂𝓛/∂μ[n, k] = -aₙₖ / N
    #   ∂²𝓛/∂μ[n, k]∂μ[n, l] = aₙₖ aₙₗ / N    (exact, positive semi-definite)
    if grad or hess:

        g /= norm
        if hess:
            h = sym_outer(g, -2)

        if grad:
            g *= weights
            g /= -Nm
            # g.neg_()
            g = g.reshape([*g.shape[:-1], *shape])
            out.append(g)
        if hess:
            h *= weights
            h /= Nm
            h = h.reshape([*h.shape[:-1], *shape])
            out.append(h)

    if return_prior:
        out.append(prior)

    return out[0] if len(out) == 1 else tuple(out)


def emmi_soft(moving, fixed, dim=None, prior=None, fwhm=None, max_iter=32,
              weights=None, grad=True, hess=True, return_prior=False,
              tolerance=1e-5):
    # The fixed image is a (fixed) separable distribution q over x, and
    # the objective is the expected marginal log-likelihood
    #   𝓛 = -1/N Σₙ Σⱼ q[n, j] ln Σₖ H[j, k] μ[n, k]
    # H is estimated by EM (a weighted mixture: each pair (n, j) is an
    # observation with weight q[n, j]), so that the gradient and Hessian
    # with respect to μ (with H fixed) are those of the returned value.
    # ------------------------------------------------------------------
    #           PREPARATION
    # ------------------------------------------------------------------
    tiny = 1e-16
    dim = dim or (moving.dim() - 1)
    moving, fixed, weights, prior, shape = emmi_prepare(
        moving, fixed, weights, prior, dim)

    *batch, J, K = prior.shape
    Nb = len(batch)
    Nm = weights.sum()
    fixed = fixed * weights                         # [*batch, J, N]

    def marginal(prior):
        # Σₖ H[j, k] μ[n, k]  -> [*batch, J, N]
        return add_tiny_(torch.matmul(prior, moving), Nb)

    # ------------------------------------------------------------------
    #           EM LOOP
    # ------------------------------------------------------------------
    ll = torch.as_tensor(-float('inf'), dtype=torch.double)
    for n_iter in range(max_iter):
        ll_prev = ll
        # E-step: responsibilities r[n, j, k] ∝ H[j, k] μ[n, k]
        # (only their weighted sums over n are needed)
        norm = marginal(prior)
        ll = (fixed * norm.log()).sum(dtype=torch.double)
        resp = fixed / norm                         # q[n, j] / Σₖ H μ
        # M-step: N p(j, k) = Σₙ q[n, j] r[n, j, k]
        prior0 = prior * torch.matmul(resp, moving.transpose(-1, -2))
        prior = prior0.add_(tiny)
        # make it a joint distribution
        prior /= add_tiny_(prior.sum(dim=[-1, -2], keepdim=True), Nb)
        if fwhm:
            # smooth "prior" for the prior
            prior = prior.transpose(-1, -2)
            prior = spatial.smooth(prior, dim=1, basis=0, fwhm=fwhm, bound='replicate')
            prior = prior.transpose(-1, -2)
        # MI-like normalization
        prior /= add_tiny_(prior.sum(dim=-1, keepdim=True) * prior.sum(dim=-2, keepdim=True), Nb)
        if em_converged(ll, ll_prev, tolerance, Nm):
            break

    # ------------------------------------------------------------------
    #           OBJECTIVE
    # ------------------------------------------------------------------
    norm = marginal(prior)
    ll = -(fixed * norm.log()).sum(dtype=torch.double) / Nm
    out = [ll]

    # ------------------------------------------------------------------
    #           GRADIENTS
    # ------------------------------------------------------------------
    #   ∂𝓛/∂μ[n, k] = -1/N Σⱼ q[n, j] H[j, k] / Σₗ H[j, l] μ[n, l]
    #   ∂²𝓛/∂μ[n, k]∂μ[n, l] = 1/N Σⱼ q[n, j] H[j, k] H[j, l] / (Σ H μ)²
    if grad or hess:
        resp = fixed / norm                         # [*batch, J, N]
        if grad:
            g = torch.matmul(prior.transpose(-1, -2), resp)
            g /= -Nm
            g = g.reshape([*g.shape[:-1], *shape])
            out.append(g)
        if hess:
            resp = resp / norm
            h = moving.new_zeros([*batch, K*(K+1)//2, resp.shape[-1]])
            h[..., :K, :] = torch.matmul(prior.square().transpose(-1, -2), resp)
            c = K
            for k in range(K):
                for kk in range(k + 1, K):
                    pp = prior[..., :, k] * prior[..., :, kk]   # [*batch, J]
                    h[..., c, :] = torch.matmul(pp.unsqueeze(-2), resp).squeeze(-2)
                    c += 1
            h /= Nm
            h = h.reshape([*h.shape[:-1], *shape])
            out.append(h)

    if return_prior:
        out.append(prior)

    return out[0] if len(out) == 1 else tuple(out)


def sym_outer(x, dim=-1, out=None):
    """Compute the symmetric outer product of a vector"""
    K = x.shape[dim]
    K2 = K*(K+1)//2
    if out is None:
        shape = list(x.shape)
        shape[dim] = K2
        out = x.new_empty(shape)
    x = x.movedim(dim, 0)
    out = out.movedim(dim, 0)

    torch.mul(x[:K], x[:K], out=out[:K])
    c = K
    for k in range(K):
        for kk in range(k + 1, K):
            torch.mul(x[k], x[kk], out=out[c])
            c += 1

    out = out.movedim(0, dim)
    return out


def sample_prior(prior, fixed, out=None):
    """
    prior : (*B, J, K)
    fixed : (*B, 1|J, N)
    out : (*B, K, N)
    """
    fn = (sample_prior_soft if fixed.dtype.is_floating_point else
          sample_prior_hard)
    return fn(prior, fixed, out)


def sample_prior_hard(prior, fixed, out=None):
    """ out = prior[fixed,:]
    prior : (*B, J, K)
    fixed : (*B, 1, N)
    out : (*B, K, N)
    """
    *B, J, K = prior.shape
    *_, _, N = fixed.shape
    prior = prior.unsqueeze(-1).expand([*B, J, K, N])
    fixed = fixed.unsqueeze(-2).expand([*B, 1, K, N])
    if out is not None:
        out = out.unsqueeze(-3)
    out = torch.gather(prior, -3, fixed, out=out).squeeze(-3)
    return out


def sample_prior_soft(prior, fixed, out=None):
    r""" out = \sum_j prior[j] * fixed[j]
    prior : (*B, J, K)
    fixed : (*B, J, N)
    out : (*B, K, N)
    """
    prior = prior.transpose(-1, -2).unsqueeze(-2)  # [*B, K, 1, J]
    fixed = fixed.transpose(-1, -2).unsqueeze(-3)  # [*B, 1, N, J]
    out = linalg.dot(prior, fixed, out=out)
    return out


def scatter_prior(prior, fixed, z):
    """
    prior : (*B, J, K)
    fixed : (*B, 1|J, N)
    z : (*B, K, N)
    """
    fn = (scatter_prior_soft if fixed.dtype.is_floating_point else
          scatter_prior_hard)
    return fn(prior, fixed, z)


def scatter_prior_hard(prior, fixed, z):
    """
    prior : (*B, J, K)
    fixed : (*B, 1, N)
    z : (*B, K, N)
    """
    *B, J, K = prior.shape
    *_, _, N = fixed.shape
    fixed = fixed.expand([*B, K, N])
    prior.zero_()
    prior.transpose(-1, -2).scatter_add_(-1, fixed, z)
    return prior


def scatter_prior_soft(prior, fixed, z):
    """
    prior : (*B, J, K)
    fixed : (*B, J, N)
    z : (*B, K, N)
    """
    z = z.unsqueeze(-3)             # [*B, 1, K, N]
    fixed = fixed.unsqueeze(-2)     # [*B, J, 1, N]
    prior = linalg.dot(z, fixed, out=prior)
    return prior


def spatial_sum(x, dim, keepdim=False):
    dims = list(range(-dim, 0))
    return x.sum(dims, keepdim=keepdim)


def make_finite_(x):
    x = x.masked_fill_(torch.isfinite(x).bitwise_not_(), 0)
    return x


def add_tiny_(x, n=0, eps=1e-12):
    if n:
        # one value per batch element
        tiny = x.abs().amax(dim=list(range(n, x.dim())), keepdim=True)
    else:
        tiny = x.abs().max()
    x.add_(tiny, alpha=eps)
    return x


class EMMI(OptimizationLoss):
    """EM Mutual Information"""

    order = 2

    def __init__(self, dim=None, fwhm=None, max_iter=32, cache=True,
                 tolerance=1e-5):
        """
        Parameters
        ----------
        dim : int, optional
        fwhm : float, optional
            Smoothing of the joint histogram along the fixed-bin axis
        max_iter : int, default=32
            Maximum number of EM iterations per call
        cache : bool, default=True
            Warm-start EM from the previous joint histogram
        tolerance : float, default=1e-5
            Stop EM when the absolute change of the log-likelihood is
            less than `tolerance * N`
        """
        super().__init__()
        self.dim = dim
        self.fwhm = fwhm
        self.max_iter = max_iter
        self.cache = cache
        self.tolerance = tolerance

    def loss(self, moving, fixed, **kwargs):
        dim = kwargs.pop('dim', self.dim)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        max_iter = kwargs.pop('max_iter', self.max_iter)
        kwargs.setdefault('tolerance', self.tolerance)
        mask = kwargs.pop('mask', None)
        prior = getattr(self, '_prior', None) if self.cache else None
        ll, prior = emmi(moving, fixed, grad=False, hess=False, weights=mask,
                         dim=dim, fwhm=fwhm, max_iter=max_iter,
                         prior=prior, return_prior=True, **kwargs)
        if self.cache:
            self._prior = prior
        return ll

    def loss_grad(self, moving, fixed, **kwargs):
        dim = kwargs.pop('dim', self.dim)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        max_iter = kwargs.pop('max_iter', self.max_iter)
        kwargs.setdefault('tolerance', self.tolerance)
        mask = kwargs.pop('mask', None)
        prior = getattr(self, '_prior', None) if self.cache else None
        ll, g, prior = emmi(moving, fixed, grad=True, hess=False, weights=mask,
                            dim=dim, fwhm=fwhm, max_iter=max_iter,
                            prior=prior, return_prior=True, **kwargs)
        if self.cache:
            self._prior = prior
        return ll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        dim = kwargs.pop('dim', self.dim)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        max_iter = kwargs.pop('max_iter', self.max_iter)
        kwargs.setdefault('tolerance', self.tolerance)
        mask = kwargs.pop('mask', None)
        prior = getattr(self, '_prior', None) if self.cache else None
        ll, g, h, prior = emmi(moving, fixed, grad=True, hess=True, weights=mask,
                               dim=dim, fwhm=fwhm, max_iter=max_iter,
                               prior=prior, return_prior=True, **kwargs)
        if self.cache:
            self._prior = prior
        return ll, g, h

    def clear_state(self):
        self._prior = None

    def get_state(self):
        return dict(prior=getattr(self, '_prior', None))

    def set_state(self, state):
        self._prior = state.get('prior', None)
