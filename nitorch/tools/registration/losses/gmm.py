"""Experimental losses based on [local] mixtures of Gaussians."""

from nitorch.core import utils, py
import math as pymath
import torch
from nitorch.tools.registration.losses import OptimizationLoss
from nitorch.tools.registration.losses.utils_gmm2 import fit_gmm2, fit_lgmm2, sumspatial, Fwd, Bwd
Tensor = torch.Tensor
pyutils = py


def _quickmi(moving, fixed, bins=64, dim=None):
    import matplotlib.pyplot as plt

    dim = dim or (fixed.dim() - 1)

    def normal(res, lam):
        p = res.square().mul_(-lam * 0.5)
        p += 0.5 * (lam.log() - pymath.log(2*pymath.pi))
        p.exp_()
        return p

    centroids = (0.5 + torch.arange(bins))/bins
    deltam = (moving.max() - moving.min())
    deltaf = (fixed.max() - fixed.min())
    mcentroids = moving.min() + centroids * deltam
    fcentroids = fixed.min() + centroids * deltaf
    lamm = (2.355 * bins / deltam) ** 2
    lamf = (2.355 * bins / deltaf) ** 2
    mhist = normal(moving - mcentroids, lamm)
    fhist = normal(fixed - fcentroids, lamf)
    jhist = (mhist[..., None] * fhist[..., None, :]).mean(list(range(-dim-2, -2)))
    mhist = mhist.mean(list(range(-dim-1, -1)))
    fhist = fhist.mean(list(range(-dim-1, -1)))
    # jhist = jhist/jhist.sum(-1, True)
    # fhist = fhist/fhist.sum(-1, True)
    # mhist = mhist/mhist.sum(-1, True)
    plt.imshow(jhist.clamp_min(1e-5).log()[0])
    plt.clim(-10, 0)
    plt.colorbar()
    plt.title('Parzen')
    plt.show()
    hmf = -(jhist*jhist.clamp_min(1e-5).log()).mean([-1, -2]) * deltam * deltaf
    hm = -(mhist*mhist.clamp_min(1e-5).log()).mean(-1) * deltam
    hf = -(fhist*fhist.clamp_min(1e-5).log()).mean(-1) * deltaf
    # hmf = -jhist.log().mean(-1)
    # hm = -mhist.log().mean(-1)
    # hf = -fhist.log().mean(-1)

    return hm + hf - hmf


def _plot_gmm(moving, fixed, prm, bins=64):
    import matplotlib.pyplot as plt

    prior, mmean, fmean, mvar, fvar, cov = prm

    det = mvar * fvar - cov * cov
    idet = det.clamp_min(1e-5).reciprocal_()
    mprec = fvar * idet
    fprec = mvar * idet
    icov = -cov * idet

    def logp(m, f):
        m = m - mmean
        f = f - fmean
        p = m * m * mprec + f * f * fprec + 2 * m * f * icov
        p = -0.5 * (p + det.clamp_min(1e-5).log() + 2 * pymath.log(2*pymath.pi))
        p = p.exp_()
        p *= prior
        p = p.sum(-1)
        p = p.log_()
        return p

    def logm(m):
        m = m - mmean
        p = m * m / mvar
        p = -0.5 * (p + mvar.clamp_min(1e-5).log() + pymath.log(2*pymath.pi))
        p = p.exp_()
        p *= prior
        p = p.sum(-1)
        p = p.log_()
        return p

    def logf(f):
        f = f - fmean
        p = f * f / fvar
        p = -0.5 * (p + fvar.clamp_min(1e-5).log() + pymath.log(2*pymath.pi))
        p = p.exp_()
        p *= prior
        p = p.sum(-1)
        p = p.log_()
        return p

    centroids = (0.5 + torch.arange(bins))/bins
    deltam = (moving.max() - moving.min())
    deltaf = (fixed.max() - fixed.min())
    mcentroids = moving.min() + centroids * deltam
    fcentroids = fixed.min() + centroids * deltaf
    mcentroids = mcentroids[:, None, None]
    fcentroids = fcentroids[None, :, None]
    jhist = logp(mcentroids, fcentroids)
    mhist = logm(mcentroids[..., :, 0, :])
    fhist = logf(fcentroids[..., 0, :, :])

    # mhist = jhist.mean(-1) * deltaf
    # fhist = jhist.mean(-2) * deltam
    hmf = -(jhist*jhist.clamp_min(1e-5).log()).mean([-1, -2]) * deltam * deltaf
    hm = -(mhist*mhist.clamp_min(1e-5).log()).mean(-1) * deltam
    hf = -(fhist*fhist.clamp_min(1e-5).log()).mean(-1) * deltaf
    # print(hm + hf - hmf)

    # jhist = jhist/jhist.sum(-1, True)
    # fhist = fhist/fhist.sum(-1, True)
    # mhist = mhist/mhist.sum(-1, True)
    plt.imshow(jhist[0, 0])
    plt.clim(-10, 0)
    plt.colorbar()
    plt.title('fit')
    plt.show()


def lgmmh(moving, fixed, dim=None, bins=3, patch=7, stride=1,
          grad=True, hess=True, kernel='g', max_iter=128,
          theta=None, return_theta=False):

    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]

    if not isinstance(patch, (list, tuple)):
        patch = [patch]
    patch = list(patch)
    if not isinstance(stride, (list, tuple)):
        stride = [stride]
    stride = [s or 0 for s in stride]

    fwd = Fwd(patch, stride, dim, kernel)
    bwd = Bwd(patch, stride, dim, kernel, shape)

    gmmfit = fit_lgmm2(moving, fixed, bins, max_iter, dim,
                       patch=patch, stride=stride, mode=kernel, theta=theta)

    # drop unused variables
    get = gmmfit.get if return_theta else gmmfit.pop
    pop = gmmfit.pop

    z = pop('resp')
    moving_mean = get('xmean')
    fixed_mean = get('ymean')
    moving_var = get('xvar')
    fixed_var = get('yvar')
    corr = get('corr')
    prior = get('prior')
    out = [pop('nll')]
    nvox = py.prod(z.shape[-dim:])

    moving = moving.unsqueeze(-dim-1)
    fixed = fixed.unsqueeze(-dim-1)

    if grad:
        z0 = fwd(z, None).clamp_min_(1e-10)

        # gradient of the GMM entropy
        # L = 0.5 * \sum_k pi_k log|\Sigma_k| + cte
        @torch.jit.script
        def make_grad(bwd: Bwd, z, z0, moving, fixed, moving_mean, fixed_mean,
                      moving_var, fixed_var, corr, prior) -> Tensor:
            cov = corr * (moving_var * fixed_var).sqrt()
            idet = moving_var * fixed_var * (1 - corr * corr)
            idet = prior / idet
            # gradient of determinant + chain rule of log
            g = moving * bwd(fixed_var * idet, z, z0) - fixed * bwd(cov * idet, z, z0)
            g -= bwd((moving_mean * fixed_var - fixed_mean * cov) * idet, z, z0)
            g = g.sum(-bwd.dim-1)
            return g

        g = make_grad(bwd, z, z0, moving, fixed, moving_mean, fixed_mean,
                      moving_var, fixed_var, corr, prior)
        g.div_(nvox)
        out.append(g)

        if hess:
            # # hessian of (1 - corr^2)
            # imoving_var = moving_var.reciprocal()
            # corr2 = corr * corr
            # h = corr2 * imoving_var
            # # chain rule (with Fisher's scoring)
            # h /= 1 - corr2
            # # hessian of log(moving_var)
            # h += imoving_var
            # # weight by proportion and sum
            # h = h * (z * prior)
            # h = h.sum(-1)

            @torch.jit.script
            def make_hess(bwd: Bwd, z, z0, moving_var, corr, prior) -> Tensor:
                h = (1 - z0) * prior / (moving_var * (1 - corr * corr))
                h = bwd(h, z, z0)
                h = h.sum(-bwd.dim-1)
                return h

            h = make_hess(bwd, z, z0, moving_var, corr, prior)
            h.div_(nvox)
            out.append(h)

    if return_theta:
        out.append(gmmfit)

    return out[0] if len(out) == 1 else tuple(out)


def gmmh(moving, fixed, dim=None, bins=6, max_iter=128, mask=None,
         grad=True, hess=True, theta=None, return_theta=False):
    """Entropy estimated by Gaussian Mixture Modeling

    Parameters
    ----------
    moving : (..., *spatial) tensor
        Moving image
    fixed : (..., *spatial) tensor
        Fixed image
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions
    bins : int, default=6
        Number of clusters in the mixture
    max_iter : int, default=25
        Number of EM iterations
    mask : tensor, optional
    grad : bool, default=True
    hess : bool, default=True

    Returns
    -------
    l : () tensor
        Negative log likelihood
    g : (..., *spatial) tensor, if `grad`
        Gradient
    h : (..., *spatial) tensor, if `hess`
        Hessian
    theta : dict, if `return_theta`
        GMM parameters
    """
    # TODO: multivariate GMM?
    #   -> must differentiate ln|C| where C is the correlation matrix

    dim = dim or (fixed.dim() - 1)
    gmmfit = fit_gmm2(moving, fixed, bins, max_iter,
                      dim=dim, mask=mask, theta=theta)

    # drop unused variables
    get = gmmfit.get if return_theta else gmmfit.pop
    pop = gmmfit.pop

    z = pop('resp')
    if mask is not None:
        z *= mask.unsqueeze(-1)
    z = z.div_(sumspatial(z, dim))
    prior = get('prior')
    moving_mean = get('xmean')
    fixed_mean = get('ymean')
    moving_var = get('xvar')
    fixed_var = get('yvar')
    corr = get('corr')
    out = [pop('nll')]

    moving = moving[..., None]
    fixed = fixed[..., None]

    if grad:

        @torch.jit.script
        def make_grad(moving, fixed, moving_mean, fixed_mean, moving_var, fixed_var, corr, prior, z):
            # compute normalized values
            moving_std = moving_var.sqrt()
            fixed_std = fixed_var.sqrt()
            moving = (moving - moving_mean) / moving_std
            fixed = (fixed - fixed_mean) / fixed_std
            # gradient of (1 - corr^2)
            g = corr * (moving * corr - fixed) / moving_std
            # gradient of log (chain rule)
            g /= 1 - corr * corr
            # gradient of log(moving_var)
            g += moving / moving_std
            # weight by proportion and sum
            g *= (z * prior)
            g = g.sum(-1)
            return g

        g = make_grad(moving, fixed, moving_mean, fixed_mean,
                      moving_var, fixed_var, corr, prior, z)
        out.append(g)

        if hess:

            @torch.jit.script
            def make_hess(moving_var, corr, prior, z):
                # hessian of (1 - corr^2)
                imoving_var = moving_var.reciprocal()
                corr2 = corr * corr
                h = corr2 * imoving_var
                # chain rule (with Fisher's scoring)
                h /= 1 - corr2
                # hessian of log(moving_var)
                h += imoving_var
                # weight by proportion and sum
                h = h * (z * prior)
                h = h.sum(-1)
                return h

            h = make_hess(moving_var, corr, prior, z)
            out.append(h)

    if return_theta:
        out.append(gmmfit)

    return out[0] if len(out) == 1 else tuple(out)


class GMMH(OptimizationLoss):
    """Gaussian Mixture Entropy"""

    order = 2

    def __init__(self, dim=None, bins=3, max_iter=128, cache=True):
        """

        Parameters
        ----------
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.max_iter = max_iter
        self.cache = cache
        self.theta = None

    def loss(self, moving, fixed, **kwargs):
        """Compute the squared LCC loss

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss

        """
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs.setdefault('theta', self.theta)
        kwargs['return_theta'] = self.cache
        ll = gmmh(moving, fixed, **kwargs, grad=False, hess=False)
        if self.cache:
            ll, self.theta = ll
        return ll

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the squared LCC loss

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image

        """
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs.setdefault('theta', self.theta)
        kwargs['return_theta'] = self.cache
        ll, g, *theta = gmmh(moving, fixed, **kwargs, grad=True, hess=False)
        if self.cache:
            self.theta = theta[0]
        return ll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the squared LCC loss

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image
        h : (..., K*(K+1)//2, *spatial) tensor, optional
            Hessian with respect to the moving image.
            Its spatial dimensions are singleton when `acceleration == 0`.

        """
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs.setdefault('theta', self.theta)
        kwargs['return_theta'] = self.cache
        ll, g, h, *theta = gmmh(moving, fixed, **kwargs, grad=True, hess=True)
        if self.cache:
            self.theta = theta[0]
        return ll, g, h

    def clear_state(self):
        self.theta = None

    def get_state(self):
        return dict(theta=self.theta)

    def set_state(self, state):
        self.theta = state.get('theta', None)


class LGMMH(OptimizationLoss):
    """Gaussian Mixture Entropy"""

    order = 2

    def __init__(self, dim=None, bins=3, patch=20, stride=1, kernel='g',
                 max_iter=128, cache=True):
        """

        Parameters
        ----------
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.patch = patch
        self.stride = stride
        self.kernel = kernel
        self.max_iter = max_iter
        self.cache = cache
        self.theta = None

    def loss(self, moving, fixed, **kwargs):
        """Compute the squared LCC loss

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss

        """
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('patch', self.patch)
        kwargs.setdefault('stride', self.stride)
        kwargs.setdefault('kernel', self.kernel)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs['return_theta'] = self.cache
        kwargs.pop('mask', None)
        ll = lgmmh(moving, fixed, **kwargs, grad=False, hess=False)
        if self.cache:
            ll, self.theta = ll
        return ll

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the squared LCC loss

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image

        """
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('patch', self.patch)
        kwargs.setdefault('stride', self.stride)
        kwargs.setdefault('kernel', self.kernel)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs['return_theta'] = self.cache
        kwargs.pop('mask', None)
        ll, g, *theta = lgmmh(moving, fixed, **kwargs, grad=True, hess=False)
        if self.cache:
            self.theta = theta[0]
        return ll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the squared LCC loss

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image
        h : (..., K*(K+1)//2, *spatial) tensor, optional
            Hessian with respect to the moving image.
            Its spatial dimensions are singleton when `acceleration == 0`.

        """
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('patch', self.patch)
        kwargs.setdefault('stride', self.stride)
        kwargs.setdefault('kernel', self.kernel)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs['return_theta'] = self.cache
        kwargs.pop('mask', None)
        ll, g, h, *theta = lgmmh(moving, fixed, **kwargs, grad=True, hess=True)
        if self.cache:
            self.theta = theta[0]
        return ll, g, h

    def clear_state(self):
        self.theta = None

    def get_state(self):
        return dict(theta=self.theta)

    def set_state(self, state):
        self.theta = state.get('theta', None)
