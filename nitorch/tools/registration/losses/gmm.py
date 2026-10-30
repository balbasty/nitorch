"""Experimental losses based on [local] mixtures of Gaussians."""

from nitorch.core import utils, py
import math as pymath
import torch
from nitorch.tools.registration.losses import OptimizationLoss
from nitorch.tools.registration.losses.utils_gmm2 import fit_gmm2, fit_lgmm2, Bwd
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


_measures = ('cond', 'mi', 'joint')


def _check_measure(measure):
    measure = (measure or 'cond').lower()
    if measure.startswith('cond'):
        measure = 'cond'
    if measure not in _measures:
        raise ValueError(f'Unknown measure {measure}. '
                         f'Expected one of {_measures}.')
    return measure


def lgmmh(moving, fixed, dim=None, bins=6, patch=20, stride=1,
          grad=True, hess=True, kernel='g', max_iter=128, measure='cond',
          theta=None, return_theta=False):
    """Entropy estimated by a Local Gaussian Mixture Model

    The parameters of the joint mixture vary smoothly in space: they are
    fitted within (Gaussian or square) patches, and the responsibility of
    each voxel combines the Gaussians of all patches it belongs to.

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
    patch : [list of] float, default=20
        Patch size (FWHM if gaussian, full width if square), in voxels
    stride : [list of] int, default=1
        Stride between patches
    kernel : {'g', 'c'}, default='g'
        Patch kernel: gaussian or constant (square)
    max_iter : int, default=128
        Number of EM iterations
    measure : {'cond', 'mi', 'joint'}, default='cond'
        See `gmmh`.
    grad : bool, default=True
    hess : bool, default=True

    Returns
    -------
    l : () tensor
        Loss (average across voxels)
    g : (..., *spatial) tensor, if `grad`
        Gradient
    h : (..., *spatial) tensor, if `hess`
        Hessian
    theta : dict, if `return_theta`
        GMM parameters
    """
    measure = _check_measure(measure)
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]

    if not isinstance(patch, (list, tuple)):
        patch = [patch]
    patch = list(map(float, patch))
    if not isinstance(stride, (list, tuple)):
        stride = [stride]
    stride = [s or 0 for s in stride]

    bwd = Bwd(patch, stride, dim, kernel, shape)

    gmmfit = fit_lgmm2(moving, fixed, bins, max_iter, dim,
                       patch=patch, stride=stride, mode=kernel, theta=theta)

    # drop unused variables
    get = gmmfit.get if return_theta else gmmfit.pop
    pop = gmmfit.pop

    z = pop('resp')                     # (..., K, *spatial)
    xmean = get('xmean')                # (..., K, *patches)
    ymean = get('ymean')
    xvar = get('xvar')
    yvar = get('yvar')
    corr = get('corr')
    prior = get('prior')
    ll = pop('nll')
    # the value is averaged across (batch and spatial) voxels
    nvox = z.numel() // z.shape[-dim-1]

    moving = moving.unsqueeze(-dim-1)
    fixed = fixed.unsqueeze(-dim-1)

    # Value: the responsibilities of a voxel combine (geometric average)
    # the Gaussians of all patches it belongs to, i.e., quantities that
    # depend on the Gaussian parameters are "pushed" (adjoint of the
    # local mean) from patches to voxels.
    if measure != 'joint':
        ixvar = xvar.reciprocal()
        stats = torch.stack([ixvar, xmean * ixvar,
                             xmean * xmean * ixvar + xvar.log()
                             - 2 * prior.log()])
        ixvar, ixmean, cst = bwd(stats, None, None).unbind(0)
        lx = moving * moving * ixvar - 2 * moving * ixmean + cst
        lx *= -0.5
        ll = ll + torch.logsumexp(lx, -dim-1).mean()
        if measure == 'mi':
            iyvar = yvar.reciprocal()
            stats = torch.stack([iyvar, ymean * iyvar,
                                 ymean * ymean * iyvar + yvar.log()
                                 - 2 * prior.log()])
            iyvar, iymean, cst = bwd(stats, None, None).unbind(0)
            ly = fixed * fixed * iyvar - 2 * fixed * iymean + cst
            ly *= -0.5
            ll = ll + torch.logsumexp(ly, -dim-1).mean()
    out = [ll]

    if grad:
        # gradient of the joint negative log-likelihood (at fixed parameters)
        cov = corr * (xvar * yvar).sqrt()
        idet = (xvar * yvar - cov * cov).reciprocal()
        xprec = yvar * idet
        icov = -cov * idet
        stats = torch.stack([xprec, icov, xprec * xmean + icov * ymean])
        xprec, icov, xmeanprec = bwd(stats, None, None).unbind(0)
        g = moving * xprec + fixed * icov - xmeanprec
        g = (z * g).sum(-dim-1)
        if measure != 'joint':
            # minus the gradient of the marginal negative log-likelihood
            w = torch.softmax(lx, -dim-1)
            g -= (w * (moving * ixvar - ixmean)).sum(-dim-1)
        g /= nvox
        out.append(g)

        if hess:
            # Gauss-Newton: precision of each Gaussian, combined by the
            # responsibilities. The marginal term removes the part of the
            # precision that does not depend on the fixed image:
            #   [Sigma^{-1}]_xx - 1/xvar = corr^2 / (xvar * (1 - corr^2))
            if measure == 'joint':
                h = xprec
            else:
                h = bwd(cov * cov * idet / xvar, None, None)
            h = (z * h).sum(-dim-1)
            h /= nvox
            out.append(h)

    if return_theta:
        out.append(gmmfit)

    return out[0] if len(out) == 1 else tuple(out)


def gmmh(moving, fixed, dim=None, bins=6, max_iter=128, mask=None,
         measure='cond', grad=True, hess=True, theta=None,
         return_theta=False):
    """Entropy estimated by Gaussian Mixture Modeling

    A 2D Gaussian mixture is fitted (by EM) to the joint intensity
    distribution of the moving and fixed images, and used to estimate
    an (upper bound of an) entropy.

    The joint entropy H[moving, fixed] (`measure='joint'`) is *not* a good
    registration objective on its own: it can also be decreased by
    making the warped moving image more uniform (e.g., by shrinking it
    or expanding homogeneous regions), irrespective of the fixed image.
    The conditional entropy H[fixed | moving] = H[moving, fixed] - H[moving]
    (`measure='cond'`, default) is the negative mutual information, up to
    the constant H[fixed]. With one cluster, it reduces to
    0.5 * log(1 - corr^2) (+ cte), i.e., it is equivalent to the
    (global) correlation coefficient.

    The marginal entropies are estimated with the marginals of the
    fitted joint mixture. Constant terms (log 2*pi) are omitted, so that
    the 'mi' value is exactly the negative mutual information of the
    fitted mixture.

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
    max_iter : int, default=128
        Number of EM iterations
    mask : tensor, optional
    measure : {'cond', 'mi', 'joint'}, default='cond'
        - 'cond':  Conditional entropy H[fixed | moving]
        - 'mi':    Negative mutual information:
                   H[moving, fixed] - H[moving] - H[fixed].
                   Same gradient as 'cond'.
        - 'joint': Joint entropy H[moving, fixed]
    grad : bool, default=True
    hess : bool, default=True

    Returns
    -------
    l : () tensor
        Loss (average across voxels)
    g : (..., *spatial) tensor, if `grad`
        Gradient
    h : (..., *spatial) tensor, if `hess`
        Hessian
    theta : dict, if `return_theta`
        GMM parameters
    """
    # TODO: multivariate GMM?
    #   -> must differentiate ln|C| where C is the correlation matrix

    measure = _check_measure(measure)
    dim = dim or (fixed.dim() - 1)
    gmmfit = fit_gmm2(moving, fixed, bins, max_iter,
                      dim=dim, mask=mask, theta=theta)

    # drop unused variables
    get = gmmfit.get if return_theta else gmmfit.pop
    pop = gmmfit.pop

    z = pop('resp')
    prior = get('prior')
    moving_mean = get('xmean')
    fixed_mean = get('ymean')
    moving_var = get('xvar')
    fixed_var = get('yvar')
    corr = get('corr')
    ll = pop('nll')
    nvox = pop('nvox')  # normalisation used for the value

    moving = moving[..., None]
    fixed = fixed[..., None]
    if mask is not None:
        mask = mask[..., None]

    def marginal(x, mean, var):
        # log-likelihood of a 1D marginal, averaged over voxels
        lx = prior.log() - 0.5 * var.log() - 0.5 * (x - mean).square() / var
        lpx = torch.logsumexp(lx, -1, keepdim=True)
        if mask is not None:
            lpx = lpx * mask
        return lx, lpx.sum() / nvox

    if measure != 'joint':
        lx, lpx = marginal(moving, moving_mean, moving_var)
        ll = ll + lpx
        if measure == 'mi':
            ll = ll + marginal(fixed, fixed_mean, fixed_var)[1]
    out = [ll]

    if grad:
        if mask is not None:
            z = z * mask

        # gradient of the joint negative log-likelihood (at fixed parameters):
        # [Sigma^{-1} (v - mu)]_moving
        #   = (moving - corr * fixed) / (moving_std * (1 - corr^2)),
        # with normalized values (v - mu) / std
        moving_std = moving_var.sqrt()
        nmoving = (moving - moving_mean) / moving_std
        nfixed = (fixed - fixed_mean) / fixed_var.sqrt()
        g = (nmoving - corr * nfixed) / (moving_std * (1 - corr * corr))
        g = (z * g).sum(-1)
        if measure != 'joint':
            # minus the gradient of the marginal negative log-likelihood
            w = torch.softmax(lx, -1)
            if mask is not None:
                w = w * mask
            g -= (w * nmoving / moving_std).sum(-1)
        g /= nvox
        out.append(g)

        if hess:
            # Gauss-Newton: precision of each Gaussian, weighted by the
            # responsibilities. The marginal term removes the part of the
            # precision that does not depend on the fixed image:
            #   [Sigma^{-1}]_xx - 1/xvar = corr^2 / (xvar * (1 - corr^2))
            corr2 = corr * corr
            h = (1 - corr2) * moving_var
            h = (corr2 if measure != 'joint' else 1) / h
            h = (z * h).sum(-1)
            h /= nvox
            out.append(h)

    if return_theta:
        out.append(gmmfit)

    return out[0] if len(out) == 1 else tuple(out)


class GMMH(OptimizationLoss):
    """Entropy estimated by Gaussian Mixture Modeling"""

    order = 2

    def __init__(self, dim=None, bins=6, max_iter=128, measure='cond',
                 cache=True):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        bins : int, default=6
            Number of clusters in the mixture
        max_iter : int, default=128
            Maximum number of EM iterations
        measure : {'cond', 'mi', 'joint'}, default='cond'
            Conditional entropy H[fixed | moving], negative mutual
            information, or joint entropy H[moving, fixed].
            See `gmmh`.
        cache : bool, default=True
            Warm-start EM with the parameters of the previous call
        """
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.max_iter = max_iter
        self.measure = _check_measure(measure)
        self.cache = cache
        self.theta = None

    def _call(self, moving, fixed, grad, hess, **kwargs):
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs.setdefault('measure', self.measure)
        kwargs.setdefault('theta', self.theta)
        kwargs['return_theta'] = self.cache
        out = gmmh(moving, fixed, **kwargs, grad=grad, hess=hess)
        if self.cache:
            *out, self.theta = out
            out = out[0] if len(out) == 1 else tuple(out)
        return out

    def loss(self, moving, fixed, **kwargs):
        """Compute the GMM entropy

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
        return self._call(moving, fixed, False, False, **kwargs)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the GMM entropy and its gradient

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
        return self._call(moving, fixed, True, False, **kwargs)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the GMM entropy, its gradient and Hessian

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
        h : (..., K, *spatial) tensor, optional
            (Diagonal) Hessian with respect to the moving image.

        """
        return self._call(moving, fixed, True, True, **kwargs)

    def clear_state(self):
        self.theta = None

    def get_state(self):
        return dict(theta=self.theta)

    def set_state(self, state):
        self.theta = state.get('theta', None)


class LGMMH(OptimizationLoss):
    """Entropy estimated by a Local Gaussian Mixture Model"""

    order = 2

    def __init__(self, dim=None, bins=6, patch=20, stride=1, kernel='g',
                 max_iter=128, measure='cond', cache=True):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        bins : int, default=6
            Number of clusters in the mixture
        patch : [list of] float, default=20
            Patch size, in voxels
        stride : [list of] int, default=1
            Stride between patches
        kernel : {'g', 'c'}, default='g'
            Patch kernel: gaussian or constant (square)
        max_iter : int, default=128
            Maximum number of EM iterations
        measure : {'cond', 'mi', 'joint'}, default='cond'
            Conditional entropy H[fixed | moving], negative mutual
            information, or joint entropy H[moving, fixed].
            See `gmmh`.
        cache : bool, default=True
            Keep the parameters of the last fit
        """
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.patch = patch
        self.stride = stride
        self.kernel = kernel
        self.max_iter = max_iter
        self.measure = _check_measure(measure)
        self.cache = cache
        self.theta = None

    def _call(self, moving, fixed, grad, hess, **kwargs):
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('bins', self.bins)
        kwargs.setdefault('patch', self.patch)
        kwargs.setdefault('stride', self.stride)
        kwargs.setdefault('kernel', self.kernel)
        kwargs.setdefault('max_iter', self.max_iter)
        kwargs.setdefault('measure', self.measure)
        kwargs['return_theta'] = self.cache
        kwargs.pop('mask', None)  # TODO: masked local fit
        out = lgmmh(moving, fixed, **kwargs, grad=grad, hess=hess)
        if self.cache:
            *out, self.theta = out
            out = out[0] if len(out) == 1 else tuple(out)
        return out

    def loss(self, moving, fixed, **kwargs):
        """Compute the local GMM entropy

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
        return self._call(moving, fixed, False, False, **kwargs)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the local GMM entropy and its gradient

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
        return self._call(moving, fixed, True, False, **kwargs)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the local GMM entropy, its gradient and Hessian

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
        h : (..., K, *spatial) tensor, optional
            (Diagonal) Hessian with respect to the moving image.

        """
        return self._call(moving, fixed, True, True, **kwargs)

    def clear_state(self):
        self.theta = None

    def get_state(self):
        return dict(theta=self.theta)

    def set_state(self, state):
        self.theta = state.get('theta', None)
