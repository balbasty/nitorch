from nitorch.core import utils, linalg
from nitorch import spatial
from .base import OptimizationLoss
import torch

# =============================================================================
# This loss has the same behaviour as `spm_maff8`
#
# General idea
# ------------
# . Let x \in [1..K]^{N} be an *observed* label map.
#   Typically, an image is discretized into K bins, and x_n indexes
#   the bin in which voxel n falls. But it could also be any segmentation.
# . Let y \in [1..J]^N be a *hidden* label map (with a potentially different
#   labelling scheme: J != K), such that p(x_n = i | y_n = j) = H_{ij}
# . Let mu \in R^{NJ} be a (deformable) prior probability to see
#   label j in voxel n, and phi \in R^{ND} be the corresponding
#   deformation field, such that: p(y_n = j) = (mu o phi)_{nj}
# . We want to find the parameters {H, phi} that maximize the marginal
#   likelihood:
#       p(x; H, mu, phi) = \prod_n \sum_j p(x_n | y_n = j; H) p(y_n; mu, phi)
#   where we use ";" to separate random variables from parameters.
# . Note that mu is assumed constant in this problem: only H and phi are
#   optimized.
# . In order to maximize H, we use an EM algorithm:
#   We introduce an approximate parameter H and compute the approximate
#   posterior distribution:
#       q(y_n = j | x_n = i) \propto H_{ij} (mu o phi)_{nj}            (E-step)
#   We then compute the expected value of the joint log-likelihood with respect
#   to q, and maximize it with respect to H:
#       H_ij \propto \sum_n (x_n = i) q(y_n = j)                       (M-step)
# . With H fixed, in order to maximize phi, we go back to the
#   marginal log-likelihood:
#       L = ln p(x) = \sum_n ln \sum_{j} H_{[x_n]j} (mu o phi)_{nj}
#   which we differentiate with respect to phi.
# . If we do not have an observed hard segmentation, but a (fixed) separable
#   distribution q over x, we can maximize the expected log-likelihood instead:
#       L = E_q[ln p(x)] = \sum_{ni} q(x_n = i) ln \sum_{j} H_{ij} (mu o phi)_{nj}
# =============================================================================


def emi(moving, fixed, dim=None, prior=None, fwhm=None, max_iter=32,
        weights=None, grad=True, hess=True, return_prior=False):
    """

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
    fixed : (..., J|1, *spatial) tensor
    dim : int, default=`moving.dim()-1`
    prior : (..., J, K) tensor
    fwhm : float, optional
    max_iter : int, default=32
    weights : tensor, optional

    Returns
    -------
    ll : () tensor
    grad : (..., K, *spatial) tensor, if `grad`
    hess : (..., K, *spatial) tensor, if `hess`
    prior : (..., J, K) tensor, if `return_prior`

    """
    tiny = 1e-16

    dim = dim or (moving.dim() - 1)

    if not fixed.dtype.is_floating_point:
        # torch requires indexing tensors to be long :(
        fixed = fixed.long()

    if prior is not None:
        J = prior.shape[-2]
        K = prior.shape[-1]
        # TODO: detect if moving and/or fixed have an implicit class
    else:
        K = moving.shape[-dim-1]
        if fixed.dtype.is_floating_point:
            J = fixed.shape[-dim-1]
        else:
            J = fixed.max() + 1

    # make sure all tensors have the same batch dimensions
    # (using zero-strides so no spurious allocation)
    shapes = [moving.shape[:-dim-1], fixed.shape[:-dim-1]]
    if prior is not None:
        shapes += [prior.shape[:-2]]
    batch = utils.expanded_shape(*shapes)

    # initialize normalized histogram
    #   `prior` contains the "normalized" joint histogram p[x,y] / (p[x] p[y])
    #    However, it is used as the conditional histogram p[x|y] during
    #    the E-step. This is because the normalized histogram is equal to
    #    the conditional histogram up to a constant term that does not
    #    depend on x, and therefore disappears after normalization of the
    #    responsibilities.
    if prior is None:
        prior = moving.new_ones([*batch, J, K])
    else:
        prior0 = prior.clone()
        prior0 /= prior0.sum([-1, -2], keepdim=True)
        prior0 /= prior0.sum(-1, keepdim=True) * prior0.sum(-2, keepdim=True)
        prior = moving.new_ones([*batch, J, K])
        prior[...] = prior0

    # mask weights
    moving = moving.clone()
    moving += (moving == 0).all(-dim-1, keepdim=True) / K
    weights = moving.new_ones([*batch, 1, *moving.shape[-dim:]])
    weights0 = weights
    # weights = (moving == 0).all(-dim-1, keepdim=True).bitwise_not_()
    # if weights0 is not None:
    #     if weights0.dtype is torch.bool:
    #         weights = weights.bitwise_and_(weights0)
    #     else:
    #         weights = weights.to(moving.dtype).mul_(weights0)

    # flatten spatial dimensions
    shape = moving.shape[-dim:]
    moving = moving.reshape([*moving.shape[:-dim], -1])
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])
    weights = weights.reshape([*weights.shape[:-dim], -1])
    N = moving.shape[-1]
    Nm = weights.sum()

    # infer conditional prior by EM
    ll = -float('inf')
    for n_iter in range(max_iter):
        ll_prev = ll
        # estimate responsibilities of each moving cluster for each fixed voxel
        # -> Baye's rule: p(z[n] == k | x[n] == j[n]) \propto p(x[n] == j[n] | z[n] == k) p(z[n] == k)
        # . j[n] is the discretized fixed image
        # . p(z[n] == k) is the moving template
        # . p(x[n] == j[n] | z[n] == k) is the conditional prior evaluted in (j[n], k)
        if not fixed.dtype.is_floating_point:
            z = moving.new_empty([*batch, K, N])
            for k in range(K):
                prior1 = prior[..., k, None].expand([*batch, J, N])
                z[..., k, :] = prior1.gather(-2, fixed) * moving[..., k, :]
        else:
            z = 0
            for j in range(J):
                z += prior[..., j, :, None] * fixed[..., j, None, :] * moving
        # compute log-likelihood (log_sum of the posterior)
        # ll = \sum_n log p(x[n] == j[n])
        #    = \sum_n log \sum_k p(x[n] == j[n] | z[n] == k)  p(z[n] == k)
        #    = \sum_n log \sum_k p(z[n] == k | x[n] == j) + constant
        ll = z.sum(-2, keepdim=True) + tiny
        z /= ll
        ll = ll.log_().mul_(weights).sum(dtype=torch.double)
        z *= weights
        # estimate joint prior
        # p(x == j, z == k) \propto \sum_n p(z[n] == k | x[n] == j) delta(j[n] == j)
        prior0 = torch.zeros_like(prior)
        if not fixed.dtype.is_floating_point:
            tmp = fixed.expand([*batch, K, N])
            prior0.transpose(-1, -2).scatter_add_(-1, tmp, z)
        else:
            for j in range(J):
                prior0[..., j, :] += linalg.dot(z, fixed[..., j, :])
        prior = prior0.clone()
        prior.add_(tiny)
        # make it a joint distribution
        prior /= prior.sum(dim=[-1, -2], keepdim=True).add_(tiny)
        if fwhm:
            # smooth "prior" for the prior
            prior = prior.transpose(-1, -2)
            prior = spatial.smooth(prior, dim=1, basis=0, fwhm=fwhm, bound='replicate')
            prior = prior.transpose(-1, -2)
        # MI-like normalization
        prior /= prior.sum(dim=-1, keepdim=True) * prior.sum(dim=-2, keepdim=True)
        # prior /= prior.sum(dim=[-1, -2], keepdim=True)
        print('em', ll.item(), ll_prev, ((ll - ll_prev)/Nm).item())
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
    ll = -(prior0 * prior.log()).sum() / (prior0.sum() * Nm)  # we want an average metric
    out = [ll]

    # compute gradients
    # Keeping only terms that depend on y, the mutual information is H[y]-H[x,y]
    # The objective function is \sum_n E[y_n]
    # > ll = \sum_n log p(x[n] == j[n], h)
    #      = \sum_n log \sum_k p(x[n] == j[n] | z[n] == k, h) p(z[n] == k)
    if grad or hess:
        g = torch.zeros_like(moving)
        if hess:
            h = moving.new_zeros([*moving.shape[:-2], K*(K+1)//2, N])
        if not fixed.dtype.is_floating_point:
            norm = 0
            for k in range(K):
                prior1 = prior[..., k]
                prior1 = prior1.unsqueeze(-1).expand([*batch, J, N])
                prior1 = prior1.gather(-2, fixed)
                norm += prior1 * moving[..., k, None, :]
                g[..., k, :] = prior1
            norm = norm.add_(tiny).reciprocal_()
            g *= norm
            if hess:
                for k in range(K):
                    h[..., k, :] = g[..., k, :].square()
                    c = K
                    for kk in range(k+1, K):
                        h[..., c, :] = g[..., k, :] * g[..., kk, :]
                        c += 1
        else:
            for j in range(J):
                norm = 0
                tmp = torch.zeros_like(g)
                for k in range(K):
                    prior1 = prior[..., j, k, None]
                    norm += prior1 * moving[..., k, :]
                    tmp[..., k, :] = prior1
                tmp /= norm.add_(tiny)
                g += tmp * fixed[..., j, None, :]
                if hess:
                    h[..., :K, :] += tmp.square() * fixed[..., j, None, :]
                    for k in range(K):
                        c = K
                        for kk in range(k+1, K):
                            h[..., c, :] += tmp[..., k, :] * tmp[..., kk, :] \
                                            * fixed[..., j, :]
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


class EMI(OptimizationLoss):
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
        ll, prior = emi(moving, fixed, grad=False, hess=False, weights=mask,
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
        ll, g, prior = emi(moving, fixed, grad=True, hess=False, weights=mask,
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
        ll, g, h, prior = emi(moving, fixed, grad=True, hess=True, weights=mask,
                              dim=dim, fwhm=fwhm, max_iter=max_iter,
                              prior=prior, return_prior=True, **kwargs)
        if self.cache:
            self._prior = prior
        return ll, g, h