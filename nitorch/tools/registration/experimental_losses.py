import matplotlib.pyplot as plt

from nitorch.core import utils, py, math, constants, linalg, kernels
from nitorch.core.math import softmax, _softmax_bwd
import math as pymath
import torch
from torch.nn import functional as F
from .losses import OptimizationLoss, local_mean
pyutils = py


def fit_gmm2(x, y, bins=3, max_iter=20, dim=None, z=None, theta=None,
             update='em', regularization=1e-3):
    """Fit a 2D Gaussian mixture

    Parameters
    ----------
    x : (..., *spatial) tensor
        Moving image (first dimension)
    y : (..., *spatial) tensor
        Fixed image (first dimension)
    bins : int, default=3
        Number of clusters
    max_iter : int, default=20
        Maximum number of iterations (only if 'em')
    dim : int, default=`fixed.dim()-1`
        Number of spatial dimensions
    z : (..., *spatial, bins) tensor, default=`1/bins`
        Initial responsibilities
    theta : dict(ymean, xmean, yvar, xvar, corr, prior), optional
        Initial parameters.
    update : {'e', 'm', 'em'}, default='em'
        - 'e' requires `theta`
        - 'm' requires `z`
        - 'em' doesn't need anything
    regularization : float, default=1e-3
        Loading of the diagonal of the covariances.
        Acts like a small Wishart prior.

    Returns
    -------
    dict:
        z : (..., *spatial, bins) tensor
            Voxel-wise responsibilties
        xmean : (..., *1, bins) tensor
            Mean of the moving (1st dimension) image
        ymean : (..., *1, bins) tensor
            Mean of the fixed (1st dimension) image
        xvar : (..., *1, bins) tensor
            Variance of the moving (1st dimension) image
        yvar : (..., *1, bins) tensor
            Variance of the fixed (1st dimension) image
        corr : (..., *1, bins) tensor
            Correlation coefficient
        prior : (..., *1, bins) tensor
            Proportion of each class
        logidet : (..., *1, bins) tensor
            Pre-computed log-determinant of the inverse covariance
        hz : (..., *1, 1) tensor
            Pre-computed entropy of the responsibilites: sum(z*log(z))



    """
    # Fit a 2D Gaussian mixture model
    dim = dim or (y.dim() - 1)
    nvox = py.prod(y.shape[-dim:])
    eps = 1e-10

    x = x[..., None]
    y = y[..., None]

    def e_step(xmean, ymean, xvar, yvar, cov, idet, logprior):
        xnorm = (x - xmean)
        ynorm = (y - ymean)
        z = (xnorm * xnorm) * yvar
        z += (ynorm * ynorm) * xvar
        z -= (xnorm * ynorm) * (2*cov)
        z *= idet
        del xnorm, ynorm
        z -= idet.log()
        z *= -0.5
        z += logprior
        z = softmax(z, dim=-1)
        return z

    sumspatial = lambda x: x.sum(list(range(-dim - 1, -1)), keepdim=True, dtype=torch.double).float()

    def suffstat(z):
        mom0 = sumspatial(z)
        z.div_(mom0)
        xmom1 = sumspatial(x * z)
        ymom1 = sumspatial(y * z)
        xmom2 = sumspatial(x.square() * z)
        ymom2 = sumspatial(y.square() * z)
        xymom2 = sumspatial((x * y) * z)
        return mom0, xmom1, ymom1, xmom2, ymom2, xymom2

    def m_step(z):
        mom0, xmom1, ymom1, xmom2, ymom2, xymom2 = suffstat(z)

        prior = mom0.div_(nvox)
        xmean = xmom1
        ymean = ymom1
        xvar = xmom2.addcmul_(xmean, xmean, value=-1)
        yvar = ymom2.addcmul_(ymean, ymean, value=-1)
        cov = xymom2.addcmul_(xmean, ymean, value=-1)

        # regularization
        alpha = regularization
        xvar = xvar.add_(alpha).div_(1+alpha)
        yvar = yvar.add_(alpha).div_(1+alpha)
        cov = cov.div_(1+alpha)

        # invert
        det = xvar * yvar - cov * cov
        idet = det.clamp_min(1e-30).reciprocal_()
        return xmean, ymean, xvar, yvar, cov, idet, prior

    if theta:
        xmean = theta.pop('xmean')
        ymean = theta.pop('ymean')
        xvar = theta.pop('xvar')
        yvar = theta.pop('yvar')
        corr = theta.pop('corr')
        prior = theta.pop('prior')
    elif z is not None:
        xmean, ymean, xvar, yvar, cov, idet, prior = m_step(z)
        corr = (xvar * yvar).reciprocal_().mul_(cov)
    else:
        quantiles = torch.arange(bins+1)/bins
        xmean = utils.quantile(x[..., 0], quantiles, dim=range(-dim, 0), keepdim=True)
        ymean = utils.quantile(y[..., 0], quantiles, dim=range(-dim, 0), keepdim=True)
        xvar = (xmean[..., 1:] - xmean[..., :-1]).div_(2.355).square_()
        yvar = (ymean[..., 1:] - ymean[..., :-1]).div_(2.355).square_()
        xmean = (xmean[..., 1:] + xmean[..., :-1]).div_(2)
        ymean = (ymean[..., 1:] + ymean[..., :-1]).div_(2)
        corr = torch.zeros_like(yvar)
        prior = y.new_full([bins], 1 / bins)

    cov = corr * yvar.sqrt() * xvar.sqrt()
    det = xvar * yvar - cov * cov
    idet = det.clamp_min(1e-5).reciprocal_()
    logprior = prior.clamp_min(1e-5).log_()
    # TODO: include trace(REG\ICOV) and log|ICOV| bits that correspond
    #       to the Wishart prior.

    if update.lower() == 'e':
        z = e_step(xmean, ymean, xvar, yvar, cov, idet, logprior)
        hz = -sumspatial(z.clamp_min(eps).log_().mul_(z))
    elif update.lower() == 'em':
        ell_prev = None
        for nit in range(max_iter):
            z = e_step(xmean, ymean, xvar, yvar, cov, idet, logprior)
            hz = -sumspatial(z.clamp_min(eps).log_().mul_(z))
            xmean, ymean, xvar, yvar, cov, idet, prior = m_step(z)
            logidet = idet.clamp_min(1e-5).log_()
            logprior = prior.clamp_min(1e-5).log_()

            # negative log-likelihood (upper bound)
            ll = ((logidet * 0.5 + logprior) * prior).sum()
            ll += hz.sum() / nvox
            ll = -ll
            # ell = ll.neg().exp()
            # print(ll.item())
            # if ell_prev is not None:
            #     gain = (ell - ell_prev) / ell_prev
            #     ell_prev = ell
            #     if gain < 1e-5:
            #         break

    corr = (xvar*yvar).sqrt_().reciprocal_().mul_(cov)
    output = dict(
        z=z, prior=prior, moving_mean=xmean, fixed_mean=ymean, ll=ll,
        moving_var=xvar, fixed_var=yvar, corr=corr, hz=hz, idet=idet,
    )
    return output


def _quickmi(moving, fixed, bins=64, dim=None):

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


def gmmi(moving, fixed, dim=None, bins=3, max_iter=20,
         grad=True, hess=True):

    dim = dim or (fixed.dim() - 1)
    n = py.prod(fixed.shape[-dim:])
    z, prior, moving_mean, fixed_mean, moving_var, fixed_var, cov, hz = \
        fit_gmm2(moving, fixed, bins, max_iter, dim=dim)

    prm = prior, moving_mean, fixed_mean, moving_var, fixed_var, cov
    # _plot_gmm(moving, fixed, prm)

    moving = moving[..., None]
    fixed = fixed[..., None]
    # _quickmi(moving, fixed, dim=dim)

    std = (moving_var*fixed_var).sqrt_()
    corr = cov / std
    # mi = corr.square().neg_().add_(1).clamp_min_(1e-5)
    mi = (1 - corr.square())

    moving = (moving - moving_mean) / moving_var.sqrt()
    fixed = (fixed - fixed_mean) / fixed_var.sqrt()

    if grad:
        # g = 2 * (z / mi) * corr * (moving * corr - fixed) / (moving_var.sqrt())
        # g *= prior
        # g = g.sum(-1)

        g = (corr * moving - fixed)
        g *= corr / (mi * moving_var.sqrt())
        g *= z
        g *= prior
        g = g.sum(-1)
        g *= 2

    if hess:
        # approximate hessian
        # h = 2 * (z / mi) * (corr.square() / moving_var)
        # h *= prior
        # h = h.sum(-1)

        h = corr.square() / (mi * moving_var)
        h = h * z
        h *= prior
        h = h.sum(-1)
        h *= 2

    # weighted sum
    mi = mi.log_().mul_(0.5)
    # print('intra cluster', (mi*prior).sum().item())
    # print('extra cluster', (prior.log()*prior).sum().item())
    mi += prior.log()
    mi *= prior
    mi = mi.sum(-1)
    mi += hz

    mi = mi.sum()
    # print(-mi, _quickmi(moving, fixed, dim=dim).sum())
    return (mi, g, h) if hess else (mi, g) if grad else mi


def lgmm(moving, fixed, dim=None, bins=3, patch=7, stride=1,
         grad=True, hess=True):

    fixed, moving = utils.to_max_backend(fixed, moving)

    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]

    def fwd(x, r=None):
        if r is not None:
            x = x*r
        return local_mean(x, patch, stride, dim=dim, mode='g')

    def bwd(x, r=None, r0=None):
        if r0 is not None:
            x = x/r0
        x = local_mean(x, patch, stride, dim=dim, mode='g',
                       backward=True, shape=shape)
        if r is not None:
            x.mul_(r)
        return x

    # initialize parameters globally
    resp, prior, *_ = fit_gmm2(moving, fixed, bins, dim=dim)
    resp = utils.fast_movedim(resp, -1, -dim-2)
    prior = utils.fast_movedim(prior, -1, -dim-2)

    # update responsibilities by EM
    # resp = torch.rand([*fixed.shape[:-dim-1], bins, 1, *shape], **utils.backend(fixed))
    # resp /= resp.sum(-dim-2, keepdim=True)
    fixed = fixed.unsqueeze(-dim-1)
    moving = moving.unsqueeze(-dim-1)
    # prior = None

    # for k in range(bins):
    #     plt.subplot(1, bins, k+1)
    #     plt.imshow(resp[k, 0, ..., resp.shape[-1]//2])
    #     plt.colorbar()
    # plt.suptitle('resp')
    # plt.show()

    for nit in range(8):

        # M-step: update Gaussian parameters (pull suffstat)
        # I stack everything in a single tensor to batch convolutions

        # if prior is None:
        #     prior = torch.full_like(norm, 1/bins)
        resp0 = fwd(resp)
        suffstat = fwd(torch.stack([moving, moving.square(),
                                    fixed, fixed.square(),
                                    moving*fixed]), resp)
        suffstat /= resp0
        moving_mean, moving_var, fixed_mean, fixed_var, cov = suffstat
        moving_var.addcmul_(moving_mean, moving_mean, value=-1).clamp_min_(0)
        fixed_var.addcmul_(fixed_mean, fixed_mean, value=-1).clamp_min_(0)
        cov.addcmul_(moving_mean, fixed_mean, value=-1)

        if py.prod(prior.shape[-dim:]) == 1:
            prior = prior.expand([*prior.shape[:-dim], *suffstat.shape[-dim:]])

        # for k in range(bins):
        #     plt.subplot(1, bins, k+1)
        #     plt.imshow(moving_mean[k, 0, ..., resp.shape[-1]//2])
        #     plt.colorbar()
        # plt.suptitle('moving_mean')
        # plt.show()
        # for k in range(bins):
        #     plt.subplot(1, bins, k+1)
        #     plt.imshow(fixed_mean[k, 0, ..., resp.shape[-1]//2])
        #     plt.colorbar()
        # plt.suptitle('fixed_mean')
        # plt.show()

        # for k in range(bins):
        #     plt.subplot(1, bins, k+1)
        #     plt.imshow(moving_mean[k, 0, ..., resp.shape[-1]//2])
        #     plt.colorbar()
        # plt.suptitle('moving_mean')
        # plt.show()
        # for k in range(bins):
        #     plt.subplot(1, bins, k+1)
        #     plt.imshow(fixed_mean[k, 0, ..., resp.shape[-1]//2])
        #     plt.colorbar()
        # plt.suptitle('fixed_mean')
        # plt.show()
        # for k in range(bins):
        #     plt.subplot(1, bins, k+1)
        #     plt.imshow(corr[k, 0, ..., resp.shape[-1]//2])
        #     plt.colorbar()
        # plt.suptitle('corr')
        # plt.show()

        det = (moving_var * fixed_var).addcmul_(cov, cov, value=-1).clamp_min_(1e-5)
        idet = det.reciprocal()
        fixed_var.mul_(idet)   # = moving_prec
        moving_var.mul_(idet)  # = fixed_prec
        cov.neg_().mul_(idet)  # = off-diagonal of precision
        det.log_()             # log determinant of covariance matrix

        # push suffstat (icov, icov*mean, mean*icov*mean + logdetcov)
        # we have enough room in `suffstat` to store icov and icov*mean
        # -> we can allocate an additional volume for the scalar bit at
        #    the cost of an extra convolution (but saves a new stack)
        scalar = moving_mean.square() * fixed_var \
               + fixed_mean.square() * moving_var \
               + 2 * moving_mean * fixed_mean * cov
        scalar += det - 2 * prior.log()
        tmp_moving = (fixed_var * moving_mean).addcmul_(cov, fixed_mean)  # (A * mu)[moving]
        tmp_fixed = (moving_var * fixed_mean).addcmul_(cov, moving_mean)  # (A * mu)[fixed]                                   # -log|A|
        moving_mean.copy_(tmp_moving)
        fixed_mean.copy_(tmp_fixed)
        scalar = bwd(scalar)
        suffstat = bwd(suffstat)
        Amu_moving, fixed_prec, Amu_fixed, moving_prec, icov = suffstat

        # E-step: update responsibilities
        resp = moving.square() * moving_prec + fixed.square() * fixed_prec
        resp += 2 * moving * fixed * icov
        resp -= 2 * (moving * Amu_moving + fixed * Amu_fixed)
        resp += scalar
        resp *= -0.5
        resp = F.softmax(resp, -dim-2)

        # M-step (2): update proportions
        prior = fwd(resp)
        norm = prior.sum(-dim-2, keepdim=True).clamp_min_(1e-5).reciprocal_()
        prior.mul_(norm).clamp_min_(1e-5)

        # for k in range(bins):
        #     plt.subplot(1, bins, k+1)
        #     plt.imshow(resp[k, 0, ..., resp.shape[-1]//2])
        #     plt.colorbar()
        # plt.suptitle('resp')
        # plt.show()

    #
    # for k in range(bins):
    #     plt.subplot(1, bins, k+1)
    #     plt.imshow(tmpf[k, 0, ..., tmpf.shape[-1]//2])
    #     plt.colorbar()
    # plt.suptitle('fixed mean')
    # plt.show()
    #
    # for k in range(bins):
    #     plt.subplot(1, bins, k+1)
    #     plt.imshow(tmpm[k, 0, ..., tmpm.shape[-1]//2])
    #     plt.colorbar()
    # plt.suptitle('moving mean')
    # plt.show()

    # Compute final estimate of Gaussian parameters
    resp0 = fwd(resp)
    suffstat = fwd(torch.stack([moving, moving.square(),
                                fixed, fixed.square(),
                                moving * fixed]), resp)
    suffstat /= resp0
    moving_mean, moving_var, fixed_mean, fixed_var, cov = suffstat
    moving_var.addcmul_(moving_mean, moving_mean, value=-1).clamp_min_(0)
    fixed_var.addcmul_(fixed_mean, fixed_mean, value=-1).clamp_min_(0)
    cov.addcmul_(moving_mean, fixed_mean, value=-1)
    istd = (fixed_var * moving_var).sqrt_().clamp_min_(1e-3).reciprocal_()
    imoving_var = moving_var.clamp_min_(1e-5).reciprocal_()
    corr = cov.mul_(istd)

    # Compute mutual information
    #
    #   We follow Leiva-Murillo & Antonio Artes-Rodriguez (ICA 2004)
    #   and approximate the entropy of the mixture by the sum of the
    #   entropy of each cluster weighted by mixing proportion.
    #
    # MI within each cluster is -log(1 - c**2), where c is the
    # correlation coefficient of the Gaussian (log is applied later).
    # (here, we multiply by -1 to get something to minimize)
    mi = corr.square().neg_().add_(1).clamp_min_(1e-5)

    resp0 = fwd(resp)

    if grad or hess:
        chain = mi.reciprocal()
        chain = chain.mul_(corr).mul_(prior)
        h = bwd(corr * imoving_var * chain, resp, resp0)

        if grad:
            # first compute the gradient wrt each correlation coefficient
            # (same as in LCC, expect that we multiply by resp in backward)
            g = fixed_mean * istd - moving_mean * corr * imoving_var
            g = bwd(g * chain, resp, resp0)
            g -= fixed * bwd(istd * chain, resp, resp0)
            g += moving * h

            # weighted sum
            g = g.sum(-dim-2)
            g.mul_(2)
            # print(g.min(), g.max(), g.median())

        if hess:
            h = h.sum(-dim-2)
            h.mul_(2)
            # print(h.min(), h.max(), h.median())

    # plt.imshow(g[0, ..., g.shape[-1]//2])
    # plt.colorbar()
    # plt.show()

    # weighted sum
    mi = mi.log_().mul_(0.5)
    mi += prior.log()
    mi *= prior
    mi = mi.sum(-dim-2)

    # tmp = mi[0, ..., mi.shape[-1]//2].neg()
    # mn, mx = utils.quantile(tmp, [0.05, 0.95])
    # tmp = tmp.clamp(mn, mx)
    # plt.imshow(tmp)
    # plt.colorbar()
    # plt.show()

    mi = mi.mean(list(range(-dim, 0))).sum()

    return (mi, g, h) if hess else (mi, g) if grad else mi


def gmmh(moving, fixed, dim=None, bins=6, max_iter=25,
         grad=True, hess=True):
    """Entropy estimated by Gaussian Mixture Modeling"""

    dim = dim or (fixed.dim() - 1)
    n = py.prod(fixed.shape[-dim:])
    gmmfit = fit_gmm2(moving, fixed, bins, max_iter, dim=dim)

    z = gmmfit.pop('z')
    prior = gmmfit.pop('prior')
    moving_mean = gmmfit.pop('moving_mean')
    fixed_mean = gmmfit.pop('fixed_mean')
    moving_var = gmmfit.pop('moving_var')
    fixed_var = gmmfit.pop('fixed_var')
    corr = gmmfit.pop('corr')
    hz = gmmfit.pop('hz')
    idet = gmmfit.pop('idet')

    # prm = prior, moving_mean, fixed_mean, moving_var, fixed_var, cov
    # _plot_gmm(moving, fixed, prm)

    moving = moving[..., None]
    fixed = fixed[..., None]
    # _quickmi(moving, fixed, dim=dim)

    # mi = corr.square().neg_().add_(1).clamp_min_(1e-5)
    onemc2 = (1 - corr.square())

    moving_std = moving_var.sqrt()
    fixed_std = fixed_var.sqrt()
    moving = (moving - moving_mean).div_(moving_std)
    fixed = (fixed - fixed_mean).div_(fixed_std)

    if grad:
        # g = 2 * (z / mi) * corr * (moving * corr - fixed) / (moving_var.sqrt())
        # g *= prior
        # g = g.sum(-1)

        # gradient of (1 - corr^2)
        g = 2 * z * corr * (moving * corr - fixed) / moving_std
        # gradient of log (chain rule)
        g /= onemc2
        # gradient of log(moving_var)
        g += 2 * z * moving / moving_std
        g *= 0.5

        # weight by proportion and sum
        g *= prior
        g = g.sum(-1)

        # gradient of z*log(z)
        # gz = z.log().add_(1).div_(n)  # - prior.log().add_(1)
        # gz = _softmax_bwd(z, gz, dim=-1)
        # gz *= (fixed * corr - moving)
        # gz /= moving_std * onemc2
        # gz = gz.sum(-1)
        # g += gz

        # g = (corr * moving - fixed)
        # g *= corr / (mi * moving_var.sqrt())
        # g *= z
        # g *= prior
        # g = g.sum(-1)
        # g *= 2

    if hess:
        # approximate hessian
        # h = 2 * (z / mi) * (corr.square() / moving_var)
        # h *= prior
        # h = h.sum(-1)

        # hessian of (1 - corr^2)
        h = 2 * (corr / moving_std).square() / n
        # "semi" chain rule
        h /= onemc2
        h *= 0.5

        h *= prior
        h = h.sum(-1)

        # h = corr.square() / (mi * moving_var)
        # h = h * z
        # h *= prior
        # h = h.sum(-1)
        # h *= 2

    # weighted sum
    ll = moving_var.log_()
    ll += fixed_var.log_()
    ll += onemc2.log_()
    ll *= 0.5
    ll -= prior.log()
    ll *= prior
    ll = ll.sum()
    ll -= hz.sum() / n

    return (ll, g, h) if hess else (ll, g) if grad else ll


class LGMM(OptimizationLoss):
    """Local mutual information estimated by GMM"""

    def __init__(self, dim=None, bins=3, patch=7, stride=1, lam=1):
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
        self.lam = lam

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        patch = kwargs.get('patch', self.patch)
        stride = kwargs.get('stride', self.stride)
        lam = kwargs.get('lam', self.lam)
        return lgmm(moving, fixed, dim=dim, bins=bins, patch=patch, stride=stride,
                    grad=False, hess=False)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        patch = kwargs.get('patch', self.patch)
        stride = kwargs.get('stride', self.stride)
        lam = kwargs.get('lam', self.lam)
        return lgmm(moving, fixed, dim=dim, bins=bins, patch=patch, stride=stride,
                    hess=False)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        patch = kwargs.get('patch', self.patch)
        stride = kwargs.get('stride', self.stride)
        lam = kwargs.get('lam', self.lam)
        return lgmm(moving, fixed, dim=dim, bins=bins, patch=patch, stride=stride)


class GMMI(OptimizationLoss):
    """Gaussian Mixture Mutual Information"""

    def __init__(self, dim=None, bins=3, lam=1):
        """

        Parameters
        ----------
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.lam = lam

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        lam = kwargs.get('lam', self.lam)
        return gmmi(moving, fixed, dim=dim, bins=bins,
                    grad=False, hess=False)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        lam = kwargs.get('lam', self.lam)
        return gmmi(moving, fixed, dim=dim, bins=bins,
                    hess=False)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        lam = kwargs.get('lam', self.lam)
        return gmmi(moving, fixed, dim=dim, bins=bins)


class GMMH(OptimizationLoss):
    """Gaussian Mixture Entropy"""

    order = 2

    def __init__(self, dim=None, bins=3, lam=1):
        """

        Parameters
        ----------
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.lam = lam

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        lam = kwargs.get('lam', self.lam)
        return gmmh(moving, fixed, dim=dim, bins=bins,
                    grad=False, hess=False)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        lam = kwargs.get('lam', self.lam)
        return gmmh(moving, fixed, dim=dim, bins=bins,
                    hess=False)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        lam = kwargs.get('lam', self.lam)
        return gmmh(moving, fixed, dim=dim, bins=bins)
