r"""
This loss has the same behaviour as `spm_maff8`

General idea
------------
. Let 𝒙 ∈ ⟦1,K⟧ᴺ be an *observed* label map.
  Typically, an image is discretized into K bins, and 𝑥ₙ indexes
  the bin in which voxel n falls. But it could also be any segmentation.
. Let 𝒚 ∈ ⟦1,J⟧ᴺ be a *hidden* label map (with a potentially different
  labelling scheme: J != K), such that p(𝑥ₙ = 𝑖 | 𝑦ₙ = 𝑗) = 𝐻ᵢⱼ
. Let 𝜇 ∈ ℝᴺᴶ be a (deformable) prior probability to see
  label j in voxel n, and 𝜙 ∈ ℝᴺᴰ be the corresponding
  deformation field, such that: p(𝑦ₙ = 𝑗) = (𝜇 ∘ 𝜙)ₙⱼ
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
      𝐻ᵢⱼ ∝ ∑ₙ (𝑥ₙ = 𝑖) q(𝑦ₙ = 𝑗)                          (M-step)
. With H fixed, in order to maximize 𝜙, we go back to the
  marginal log-likelihood:
      𝓛 = ln p(𝒙) = ∑ₙ ln ∑ⱼ 𝐻{[𝑥ₙ],𝑗} (𝜇 ∘ 𝜙)ₙⱼ
  which we differentiate with respect to 𝜙.
. If we do not have an observed hard segmentation, but a (fixed) separable
  distribution q over x, we can maximize the expected log-likelihood instead:
      𝓛 = 𝔼_q[ln p(𝒙)] = ∑ₙᵢ q(𝑥ₙ = 𝑖) ln ∑ⱼ 𝐻ᵢⱼ (𝜇 ∘ 𝜙)ₙⱼ
"""
from nitorch.core import utils, linalg, math
from nitorch import spatial
from .base import OptimizationLoss
import torch


def emmi(moving, fixed, *args, **kwargs):
    """Expectation-Maximization Mutual Information

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
    fixed : (..., J|1, *spatial) tensor
    dim : int, default=`moving.dim()-1`
    prior : (..., J, K) tensor
    fwhm : float, optional
    max_iter : int, default=32
    weights : tensor (..., 1, *spatial) tensor, optional

    Returns
    -------
    ll : () tensor
    grad : (..., K, *spatial) tensor, if `grad`
    hess : (..., K, *spatial) tensor, if `hess`
    prior : (..., J, K) tensor, if `return_prior`

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


def emmi_hard(moving, fixed, dim=None, prior=None, fwhm=None, max_iter=32,
              weights=None, grad=True, hess=True, return_prior=False):
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
    ll = -float('inf')
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
        ll = ll.log_().mul_(weights).sum([-1, -2], dtype=torch.double)

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
        if ll - ll_prev < 1e-5 * Nm:
            break

    # compute mutual information (times number of observations)
    # > prior contains p(x,y)/(p(x) p(y))
    # > prior0 contains N * p(x,y)
    # >> 1/N \sum_{j,k} prior0[j,k] * log(prior[j,k])
    #    = \sum_{x,y} p(x,y) * (log p(x,y) - log p(x) - log p(y)
    #    = \sum_{x,y} p(x,y) * log p(x,y)
    #       - \sum_{xy} p(x,y) log p(x)
    #       - \sum_{xy} p(x,y) log p(y)
    #    = \sum_{x,y} p(x,y) * log p(x,y)
    #       - \sum_{x} p(x) log p(x)
    #       - \sum_{y} p(y) log p(y)
    #    = -H[x,y] + H[x] + H[y]
    #    = MI[x, y]
    ll = -(prior0 * add_tiny_(prior, Nb).log()).sum() / Nm
    out = [ll]

    # ------------------------------------------------------------------
    #           GRADIENTS
    # ------------------------------------------------------------------
    # compute gradients
    # Keeping only terms that depend on y, the mutual information is H[y]-H[x,y]
    # The objective function is \sum_n E[y_n]
    # > ll = \sum_n log p(x[n] == j[n], h)
    #      = \sum_n log \sum_k p(x[n] == j[n] | z[n] == k, h) p(z[n] == k)
    if grad or hess:

        g = sample_prior(prior, fixed)
        norm = linalg.dot(g.transpose(-1, -2), moving.transpose(-1, -2))
        norm = add_tiny_(norm, Nb).unsqueeze(-2).reciprocal_()
        g *= norm
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
              weights=None, grad=True, hess=True, return_prior=False):
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
    ll = -float('inf')
    z = moving.new_empty([*batch, K, N])
    prior0 = torch.empty_like(prior)
    for n_iter in range(max_iter):
        ll_prev = ll
        # --------------------------------------------------------------
        # E-step
        # --------------------------------------------------------------
        z = sample_prior(prior.log(), fixed, z)
        z += moving.log()
        z, ll = math.softmax_lse(z, -2, lse=True, weights=weights)

        # --------------------------------------------------------------
        # M-step
        # ------
        # estimate joint prior by maximizing Q = E_{Z;H,mu}[ln p(X, Z; H)]
        # => H_jk = p(x == j, z == k) ∝ Σ_n p(z[n] == k | x[n] == j) 𝛿(x[n] == j)
        # --------------------------------------------------------------
        z *= weights
        prior0 = scatter_prior(prior0, fixed, z)
        prior.copy_(prior0).add_(tiny)
        # make it a joint distribution
        prior /= add_tiny_(prior.sum(dim=[-1, -2], keepdim=True), Nb)
        if fwhm:
            # smooth "prior" for the prior
            prior = prior.transpose(-1, -2)
            prior = spatial.smooth(prior, dim=1, basis=0, fwhm=fwhm, bound='replicate')
            prior = prior.transpose(-1, -2)
        # MI-like normalization
        prior /= prior.sum(dim=-1, keepdim=True) * prior.sum(dim=-2, keepdim=True)
        if ll - ll_prev < 1e-5 * Nm:
            break

    # compute mutual information (times number of observations)
    # > prior contains p(x,y)/(p(x) p(y))
    # > prior0 contains N * p(x,y)
    # >> 1/N Σ_{j,k} prior0[j,k] * log(prior[j,k])
    #    = Σ_{x,y} p(x,y) * (log p(x,y) - log p(x) - log p(y)
    #    = Σ_{x,y} p(x,y) * log p(x,y)
    #       - Σ_{xy} p(x,y) log p(x)
    #       - Σ_{xy} p(x,y) log p(y)
    #    = Σ_{x,y} p(x,y) * log p(x,y)
    #       - Σ_{x} p(x) log p(x)
    #       - Σ_{y} p(y) log p(y)
    #    = -H[x,y] + H[x] + H[y]
    #    = MI[x, y]
    ll = -(prior0 * prior.log()).sum() / Nm
    out = [ll]

    # ------------------------------------------------------------------
    #           GRADIENTS
    # ------------------------------------------------------------------
    # compute gradients
    # Keeping only terms that depend on y, the mutual information is H[y]-H[x,y]
    # The objective function is \sum_n E[y_n]
    # > ll = Σ_n log p(x[n] == j[n], h)
    #      = Σ_nj \sum_j q(x[n] == j) log \sum_k p(x[n] == j | z[n] == k, h) p(z[n] == k)
    if grad or hess:

        norm = linalg.dot(prior.transpose(-1, -2).unsqueeze(-1),
                          moving.transpose(-1, -2).unsqueeze(-3))
        norm = norm.add_(tiny).reciprocal_()
        g = sample_prior(prior, fixed * norm)

        if hess:
            norm = norm.square_().mul_(fixed).unsqueeze(-1)
            h = moving.new_zeros([*g.shape[:-2], K*(K+1)//2, N])
            for j in range(J):
                h[..., :K, :] += prior[..., j, :K, None].square() * norm
                c = K
                for k in range(K):
                    for kk in range(k + 1, K):
                        h[..., c, :] += (prior[..., j, k, None] *
                                         prior[..., j, kk, None] * norm)
                        c += 1

        if grad:
            g *= weights
            g.neg_()
            g = g.reshape([*g.shape[:-1], *shape])
            out.append(g)
        if hess:
            h *= weights
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
    """ out = \sum_j prior[j] * fixed[j]
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
        tiny = x.abs().max(range(n), keepdim=True)
    else:
        tiny = x.abs().max()
    x.add_(tiny, alpha=eps)
    return x


class EMMI(OptimizationLoss):
    """EM Mutual Information"""

    order = 2

    def __init__(self, dim=None, fwhm=None, max_iter=32, cache=True):
        super().__init__()
        self.dim = dim
        self.fwhm = fwhm
        self.max_iter = max_iter
        self.cache = cache

    def loss(self, moving, fixed, **kwargs):
        dim = kwargs.pop('dim', self.dim)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        max_iter = kwargs.pop('max_iter', self.max_iter)
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
