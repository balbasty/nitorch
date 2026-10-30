r"""
This loss has the same behaviour as `spm_maff8`
(see also `nitorch.tools.registration.affine_tpm`, which uses the same
model and the same EM, with a TPM as moving image).

General idea
------------
. Let 𝒙 ∈ ⟦1,K⟧ᴺ be an *observed* label map.
  Typically, an image is discretized into K bins, and 𝑥ₙ indexes
  the bin in which voxel n falls. But it could also be any segmentation.
. Let 𝒚 ∈ ⟦1,J⟧ᴺ be a *hidden* label map (with a potentially different
  labelling scheme: J != K), such that p(𝑥ₙ = 𝑖 | 𝑦ₙ = 𝑗) = p(𝑥 = 𝑖) 𝐻ᵢⱼ
  (see below for the normalization of 𝑯).
. Let 𝜇 ∈ ℝᴺᴶ be a (deformable) prior probability to see
  label j in voxel n, and 𝜙 ∈ ℝᴺᴰ be the corresponding
  deformation field, such that: p(𝑦ₙ = 𝑗) = (𝜇 ∘ 𝜙)ₙⱼ
  This loss receives the deformed probabilities 𝜇 ∘ 𝜙 (the moving image).
. We want to find the parameters {𝐻, 𝜙} that maximize the marginal
  likelihood:
      p(𝒙; 𝑯, 𝜇, 𝜙) = ∏ₙ ∑ⱼ p(𝑥ₙ | 𝑦ₙ = 𝑗; 𝑯) p(𝑦ₙ; 𝜇, 𝜙)
  where we use ";" to separate random variables from parameters.
. Note that 𝜇 is assumed constant in this problem: only H and 𝜙 are
  optimized.
. In order to maximize H, we use an EM algorithm:
  We introduce an approximate parameter H and compute the approximate
  posterior distribution:
      q(𝑦ₙ = 𝑗 | 𝑥ₙ = 𝑖) ∝ 𝐻ᵢⱼ (𝜇 ∘ 𝜙)ₙⱼ                   (E-step)
  We then compute the expected value of the joint log-likelihood with respect
  to q, and maximize it with respect to H:
      p(𝑥 = 𝑖, 𝑦 = 𝑗) ∝ ∑ₙ 𝛿(𝑥ₙ = 𝑖) q(𝑦ₙ = 𝑗)              (M-step)
      𝐻ᵢⱼ = p(𝑥 = 𝑖, 𝑦 = 𝑗) / (p(𝑥 = 𝑖) p(𝑦 = 𝑗))
. 𝑯 is therefore the "normalized" joint histogram p(x,y) / (p(x) p(y))
  = p(x|y) / p(x), not the conditional p(x|y). The factor 1/p(𝑥 = 𝑖)
  does not depend on j, so it cancels in the E-step. In 𝓛 (below) it
  only adds the constant -∑ₙ ln p(𝑥ₙ), which depends on the fixed image
  histogram only. With this normalization, when 𝜇 ∘ 𝜙 is one-hot (and
  the histogram is not smoothed), 𝓛/N is the mutual information
  between 𝒙 and 𝒚.
. If `fwhm` is set, the joint histogram is smoothed along the fixed-bin
  axis (i) after each M-step. The M-step is then not an exact maximizer
  of 𝓛, and EM is not guaranteed to be monotonic: it stops when the
  absolute change of 𝓛 is below `tolerance * N`.
. With H fixed, in order to maximize 𝜙, we go back to the
  marginal log-likelihood:
      𝓛 = ln p(𝒙) = ∑ₙ ln ∑ⱼ 𝐻{[𝑥ₙ],𝑗} (𝜇 ∘ 𝜙)ₙⱼ
  which we differentiate with respect to 𝜇 ∘ 𝜙 (and, by the chain rule,
  𝜙). If 𝑯 were the exact maximizer 𝑯*(𝜙), this would be the gradient
  of the profile log-likelihood 𝓛(𝑯*(𝜙), 𝜙) (envelope theorem).
  Because EM is stopped early (and the histogram may be smoothed), this
  is only approximate.
. If we do not have an observed hard segmentation, but a (fixed) separable
  distribution q over x, we can maximize the expected log-likelihood instead:
      𝓛 = 𝔼_q[ln p(𝒙)] = ∑ₙᵢ q(𝑥ₙ = 𝑖) ln ∑ⱼ 𝐻ᵢⱼ (𝜇 ∘ 𝜙)ₙⱼ
  EM then treats each pair (n, i) as an observation with weight
  q(𝑥ₙ = 𝑖):
      rₙᵢⱼ ∝ 𝐻ᵢⱼ (𝜇 ∘ 𝜙)ₙⱼ                                 (E-step)
      p(𝑥 = 𝑖, 𝑦 = 𝑗) ∝ ∑ₙ q(𝑥ₙ = 𝑖) rₙᵢⱼ                    (M-step)

Sign and scale conventions
--------------------------
. `emmi` (and the `EMMI` loss) return the value to *minimize*:
  -𝓛/N, evaluated at the final 𝑯, where N is the total voxel weight,
  together with its gradient and Hessian with respect to 𝜇 ∘ 𝜙
  (𝑯 fixed).
. `affine_tpm.em_prior` returns +𝓛 (summed over voxels, to maximize),
  with the same 𝑯 and the same EM.

Mapping to the code
-------------------
. `prior[..., i, j]` is 𝐻ᵢⱼ. Its shape is (..., J, K) where, in the
  code, J is the number of fixed bins and K the number of moving
  classes (i.e., the K and J of this docstring, swapped). Shapes in the
  function docstrings below use the code's J and K.
. `fixed` holds 𝒙 (or q) and `moving[..., j, :]` holds (𝜇 ∘ 𝜙)ₙⱼ.
  Below, μ[n, j] is short for (𝜇 ∘ 𝜙)ₙⱼ.
"""
from nitorch.core import utils, linalg
from nitorch import spatial
from .base import OptimizationLoss
import torch


def emmi(moving, fixed, *args, **kwargs):
    """Expectation-Maximization Mutual Information

    Shapes use the code's sizes: J fixed bins and K moving classes
    (the K and J of the module docstring, respectively).

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Deformed moving probabilities 𝜇 ∘ 𝜙 (K classes)
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
        Gradient of -𝓛/N with respect to 𝜇 ∘ 𝜙 (H fixed)
    hess : (..., K*(K+1)//2, *spatial) tensor, if `hess`
        Hessian of -𝓛/N with respect to 𝜇 ∘ 𝜙 (H fixed), packed: the K
        diagonal elements (j, j), then the (j, l), j < l, elements in
        row-major order.
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
    # The value is the negative, averaged marginal log-likelihood,
    # evaluated at the final prior H[i, j] = p(x=i, y=j) / (p(x=i) p(y=j)):
    #   -𝓛/N = -1/N Σₙ ln Σⱼ H[x[n], j] μ[n, j]
    # This is the function whose gradient and Hessian are returned below.
    # When μ is a hard (one-hot) map (and fwhm=0), -𝓛/N = -MI[x, y].
    #
    # The previous implementation returned -Σᵢⱼ p(x=i, y=j) ln H[i, j],
    # the MI of the expected joint histogram, which is a different
    # function of μ: its finite differences did not match the gradient
    # (not even in sign), and Gauss-Newton stopped early.
    g = sample_prior(prior, fixed)                  # H[x[n], j]
    norm = (g * moving).sum(-2, keepdim=True)       # Σⱼ H[x[n], j] μ[n, j]
    norm = add_tiny_(norm, Nb)
    ll = -(norm.log() * weights).sum(dtype=torch.double) / Nm
    out = [ll]

    # ------------------------------------------------------------------
    #           GRADIENTS
    # ------------------------------------------------------------------
    # The prior H is kept fixed (it (approximately) maximises 𝓛 w.r.t. H,
    # so its own variation does not contribute to the first-order
    # derivative). With aₙⱼ = H[x[n], j] / Σₗ H[x[n], l] μ[n, l]:
    #   ∂(-𝓛/N)/∂μ[n, j] = -aₙⱼ / N
    #   ∂²(-𝓛/N)/∂μ[n, j]∂μ[n, l] = aₙⱼ aₙₗ / N   (exact, positive semi-definite)
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
    # the value is the negative, averaged expected log-likelihood
    #   -𝓛/N = -1/N Σₙ Σᵢ q[n, i] ln Σⱼ H[i, j] μ[n, j]
    # H is estimated by EM (a weighted mixture: each pair (n, i) is an
    # observation with weight q[n, i]), so that the gradient and Hessian
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
        # Σⱼ H[i, j] μ[n, j]  -> [*batch, J, N]
        return add_tiny_(torch.matmul(prior, moving), Nb)

    # ------------------------------------------------------------------
    #           EM LOOP
    # ------------------------------------------------------------------
    ll = torch.as_tensor(-float('inf'), dtype=torch.double)
    for n_iter in range(max_iter):
        ll_prev = ll
        # E-step: responsibilities r[n, i, j] ∝ H[i, j] μ[n, j]
        # (only their weighted sums over n are needed)
        norm = marginal(prior)
        ll = (fixed * norm.log()).sum(dtype=torch.double)
        resp = fixed / norm                         # q[n, i] / Σⱼ H μ
        # M-step: N p(x=i, y=j) = Σₙ q[n, i] r[n, i, j]
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
    #   ∂(-𝓛/N)/∂μ[n, j] = -1/N Σᵢ q[n, i] H[i, j] / Σₗ H[i, l] μ[n, l]
    #   ∂²(-𝓛/N)/∂μ[n, j]∂μ[n, l] = 1/N Σᵢ q[n, i] H[i, j] H[i, l] / (Σ H μ)²
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
