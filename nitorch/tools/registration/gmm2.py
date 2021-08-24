"""Utilities to fit 2D Gaussian Mixture Models for registration."""
from nitorch.core import utils, py, math, constants, linalg, kernels
import torch
from torch.nn import functional as F
from .local import local_mean, prod as script_prod
from typing import List, Optional, Dict, Tuple
Tensor = torch.Tensor
pyutils = py


@torch.jit.script
def sumspatial(x: Tensor, ndim: int) -> Tensor:
    """Reduce across spatial dimensions.

    Parameters
    ----------
    x : (..., *spatial, K) tensor
    ndim : int

    Returns
    -------
    x : (..., *1, K) tensor

    """
    dims = [d for d in range(-ndim - 1, -1)]
    return x.sum(dims, keepdim=True, dtype=torch.double).float()


# cannot use TorchScript because of `quantile`.
def init_gmm2(x, y, bins=6, dim=None):
    """Initialize parameters of a 2D GMM by drawing quantiles.

    Parameters
    ----------
    x : (..., *spatial) tensor
        Moving image (first dimension)
    y : (..., *spatial) tensor
        Fixed image (first dimension)
    bins : int, default=3
        Number of clusters
    dim : int, default=`fixed.dim()-1`
        Number of spatial dimensions

    Returns
    -------
    dict:
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
    """
    dim = dim or x.dim() - 1
    quantiles = torch.arange(bins + 1) / bins
    xmean = utils.quantile(x[..., 0], quantiles, dim=range(-dim, 0), keepdim=True)
    ymean = utils.quantile(y[..., 0], quantiles, dim=range(-dim, 0), keepdim=True)
    xvar = (xmean[..., 1:] - xmean[..., :-1]).div_(2.355).square_()
    yvar = (ymean[..., 1:] - ymean[..., :-1]).div_(2.355).square_()
    xmean = (xmean[..., 1:] + xmean[..., :-1]).div_(2)
    ymean = (ymean[..., 1:] + ymean[..., :-1]).div_(2)
    corr = torch.zeros_like(yvar)
    prior = y.new_full([bins], 1 / bins)
    return dict(xmean=xmean, ymean=ymean, xvar=xvar, yvar=yvar, corr=corr, prior=prior)


def fit_gmm2(x: Tensor, y: Tensor, bins: int = 3, max_iter: int = 20,
             dim: Optional[int] = None, z: Optional[Tensor] = None,
             theta: Optional[Dict[str, Tensor]] = None,
             update: str = 'em'):
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
    if z is None and not theta:
        theta = init_gmm2(x, y, bins, dim)
    return _fit_gmm2(x, y, max_iter, dim, z, theta, update)


@torch.jit.script
def e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior):
    x = x - xmean
    y = y - ymean
    z = (x * x) * yvar
    z += (y * y) * xvar
    z -= (x * y) * (2*cov)
    z *= idet
    z -= idet.log()
    z *= -0.5
    z += logprior
    z = F.softmax(z, dim=-1)
    return z


@torch.jit.script
def suffstat(x, y, z, ndim: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # z must be normalized (== divided by mom0)
    mom0 = sumspatial(z, ndim)
    z = z / mom0
    xmom1 = sumspatial(x * z, ndim)
    ymom1 = sumspatial(y * z, ndim)
    xmom2 = sumspatial((x * x) * z, ndim)
    ymom2 = sumspatial((y * y) * z, ndim)
    xymom2 = sumspatial((x * y) * z, ndim)
    return mom0, xmom1, ymom1, xmom2, ymom2, xymom2


@torch.jit.script
def m_step(x, y, z, ndim: int, nvox: int) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    mom0, xmom1, ymom1, xmom2, ymom2, xymom2 = suffstat(x, y, z, ndim)
    prior = mom0.div_(nvox)
    xmean = xmom1
    ymean = ymom1
    xvar = xmom2.addcmul_(xmean, xmean, value=-1)
    yvar = ymom2.addcmul_(ymean, ymean, value=-1)
    cov = xymom2.addcmul_(xmean, ymean, value=-1)

    # regularization
    alpha = 1e-3
    xvar = xvar.add_(alpha).div_(1+alpha)
    yvar = yvar.add_(alpha).div_(1+alpha)
    cov = cov.div_(1+alpha)

    # invert
    det = xvar * yvar - cov * cov
    idet = det.clamp_min(1e-30).reciprocal_()
    return xmean, ymean, xvar, yvar, cov, idet, prior


@torch.jit.script
def em_loop(max_iter: int, x, y, xmean, ymean, xvar, yvar, cov, idet, logprior,
            ndim: int, nvox: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    z = e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior)
    for nit in range(max_iter):
        xmean, ymean, xvar, yvar, cov, idet, prior = m_step(x, y, z, ndim, nvox)
        z = e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior)
        logprior = prior.clamp_min(1e-5).log_()

        # negative log-likelihood (upper bound)
        hz = -sumspatial(z.clamp_min(1e-5).log_().mul_(z), ndim)
        logidet = idet.clamp_min(1e-5).log_()
        ll = ((logidet * 0.5 + logprior) * prior).sum()
        ll += hz.sum() / nvox
        ll = -ll
        print('gmm', ll.item())
    return z, xmean, ymean, xvar, yvar, cov, idet, logprior


@torch.jit.script
def _fit_gmm2(x: Tensor, y: Tensor, max_iter: int = 20,
             dim: Optional[int] = None, z: Optional[Tensor] = None,
             theta: Optional[Dict[str, Tensor]] = None,
             update: str = 'em') -> Dict[str, Tensor]:
    # Fit a 2D Gaussian mixture model
    if dim is None:
        dim = y.dim() - 1
    nvox = script_prod(y.shape[-dim:])
    eps = 1e-10

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)

    if z is not None:
        xmean, ymean, xvar, yvar, cov, idet, prior = m_step(x, y, z, dim, nvox)
        corr = cov / (xvar * yvar)
    elif theta is not None:
        xmean = theta.pop('xmean')
        ymean = theta.pop('ymean')
        xvar = theta.pop('xvar')
        yvar = theta.pop('yvar')
        corr = theta.pop('corr')
        prior = theta.pop('prior')
    else:
        raise ValueError('One of z or theta must be provided')

    cov = corr * yvar.sqrt() * xvar.sqrt()
    det = xvar * yvar - cov * cov
    idet = det.clamp_min(1e-5).reciprocal_()
    logprior = prior.clamp_min(1e-5).log_()

    if update.lower() == 'e':
        z = e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior)
    elif update.lower() == 'em':
        z, xmean, ymean, xvar, yvar, cov, idet, logprior = \
            em_loop(max_iter, x, y, xmean, ymean, xvar, yvar, cov, idet,
                    logprior, dim, nvox)
    elif z is None:
        raise ValueError('update == "m" requires z')

    # negative log-likelihood (upper bound)
    hz = -sumspatial(z.clamp_min(eps).log_().mul_(z), dim)
    corr = (xvar*yvar).sqrt_().reciprocal_().mul_(cov)
    prior = logprior.exp()
    logidet = idet.clamp_min(1e-5).log_()
    ll = ((logidet * 0.5 + logprior) * prior).sum()
    ll += hz.sum() / nvox
    ll = -ll

    output = dict(
        z=z, prior=prior, moving_mean=xmean, fixed_mean=ymean, ll=ll,
        moving_var=xvar, fixed_var=yvar, corr=corr, hz=hz, idet=idet,
    )
    return output


@torch.jit.script
class Fwd:
    def __init__(self, patch: List[int], stride: List[int], dim: int, mode:str):
        self.patch = patch
        self.stride = stride
        self.dim = dim
        self.mode = mode

    def __call__(self, x: Tensor, r: Optional[Tensor] = None):
        if r is not None:
            x = x*r
        return local_mean(x, self.patch, self.stride,
                          dim=self.dim, mode=self.mode)


@torch.jit.script
class Bwd:
    def __init__(self, patch: List[int], stride: List[int], dim:int,
                 mode:str, shape: List[int]):
        self.patch = patch
        self.stride = stride
        self.dim = dim
        self.mode = mode
        self.shape = shape

    def __call__(self, x: Tensor,
                 r: Optional[Tensor] = None,
                 r0: Optional[Tensor] = None):
        if r0 is not None:
            x = x/r0
        x = local_mean(x, self.patch, self.stride, dim=self.dim,
                       mode=self.mode, backward=True, shape=self.shape)
        if r is not None:
            x.mul_(r)
        return x


@torch.jit.script
def make_moments(x, y):
    return torch.stack([x, y, x*x, y*y, x*y])


@torch.jit.script
def m_step_local(fwd: Fwd, z: Tensor, moments: Tensor)\
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    z0 = fwd(z)
    suffstat = fwd(moments, z).div_(z0)
    xmean, ymean, xvar, yvar, cov = suffstat.unbind(0)
    xvar = xvar - xmean * xmean
    yvar = yvar - ymean * ymean
    cov = cov - xmean * ymean
    pi = z0.clamp_min_(1e-5).clamp_max_(1-1e-5)
    pi = pi / pi.sum(-fwd.dim-1, keepdim=True)
    return xmean, ymean, xvar, yvar, cov, pi


@torch.jit.script
def em_loop_local(fwd: Fwd, bwd: Bwd, moments: Tensor, z: Tensor,
                  max_iter: int, dim: int) -> Dict[str, Tensor]:

    x, y, x2, y2, xy = moments.unbind(0)

    nll_prev = torch.zeros([1], dtype=x.dtype, device=x.device)
    nll_max = torch.zeros([1], dtype=x.dtype, device=x.device)
    for nit in range(max_iter):

        # M-step: update Gaussian parameters (pull suffstat)
        hz = -sumspatial(z.clamp_min(1e-10).log_().mul_(z).unsqueeze(-1), dim).squeeze(-1)
        xmean, ymean, xvar, yvar, cov, pi = m_step_local(fwd, z, moments)

        # compute log likelihood
        det = xvar * yvar - cov * cov
        det = det.clamp_min_(1e-5)
        logdet = det.log()
        logpi = pi.log()
        nll = ((-0.5 * logdet + logpi) * pi).sum()
        nll /= script_prod(hz.shape[-dim:])
        nll += hz.sum() / script_prod(hz.shape[-dim:])
        nll = -nll
        if nit > 1:
            gain = abs(nll - nll_prev) / abs(nll_max - nll)
            print('lgmm', nll.item(), gain.item())
            if gain < 1e-8:
                break
        nll_prev = nll
        nll_max = torch.maximum(nll_max, nll)

        # covariance to precision
        idet = det.reciprocal()
        icov = -cov * idet
        xprec = yvar * idet
        yprec = xvar * idet

        # push suffstat (icov, icov*mean, mean*icov*mean + logdetcov)
        mAm = (xmean * xmean * xprec       # \
               + ymean * ymean * yprec     # | mu * A * mu
               + 2 * xmean * ymean * icov  # /
               + logdet - 2 * logpi)       # log|A| - 2 * log pi
        Ax = xprec * xmean + icov * ymean  # (A * mu)[moving]
        Ay = yprec * ymean + icov * xmean  # (A * mu)[fixed]

        suffstat = torch.stack([mAm, Ax, Ay, xprec, yprec, icov])
        suffstat = bwd(suffstat)
        mAm, Ax, Ay, xprec, yprec, icov = suffstat.unbind(0)

        # E-step: update responsibilities
        z = x2 * xprec + y2 * yprec
        z += 2 * x * y * icov
        z -= 2 * (x * Ax + y * Ay)
        z += mAm
        z *= -0.5
        z = F.softmax(z, -dim-1)

    # final M-step
    hz = -sumspatial(z.clamp_min(1e-10).log_().mul_(z).unsqueeze(-1), dim).squeeze(-1)
    xmean, ymean, xvar, yvar, cov, pi = m_step_local(fwd, z, moments)

    corr = cov / (xvar * yvar).sqrt_().clamp_min_(1e-5)
    det = xvar * yvar - cov * cov
    det = det.clamp_min_(1e-5)
    logdet = det.log()
    logpi = pi.log()
    nll = ((-0.5 * logdet + logpi) * pi).sum()
    nll /= script_prod(hz.shape[-dim:])
    nll += hz.sum() / script_prod(hz.shape[-dim:])
    nll = -nll

    output = dict(
        resp=z, prior=pi, xmean=xmean, ymean=ymean, nll=nll,
        xvar=xvar, yvar=yvar, corr=corr, resp_entropy=hz,
    )
    return output


def fit_lgmm2(x, y, bins=6, max_iter=20, dim=None,
              patch=None, stride=None, mode='g'):

    dim = dim or (x.dim() - 1)
    shape = x.shape[-dim:]
    patch = patch or [20]
    stride = stride or [1]

    # initialize parameters globally
    gmmfit = fit_gmm2(x, y, bins, dim=dim)
    z = gmmfit.pop('z')
    del gmmfit
    z = utils.fast_movedim(z, -1, -dim-1)  # [nmom(=1), k, *spatial]

    fwd = Fwd(patch, stride, dim, mode)
    bwd = Bwd(patch, stride, dim, mode, shape)
    moments = make_moments(x, y)
    moments = moments.unsqueeze(-dim-1)  # [nmom, k(=1), *spatial]
    gmmfit = em_loop_local(fwd, bwd, moments, z, max_iter, dim)
    return gmmfit
