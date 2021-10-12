"""Utilities to fit 2D Gaussian Mixture Models for registration.

The main entry points are:
- fit_gmm2(x: Tensor, y: Tensor, bins: int = 3, max_iter: int = 20, ...)
- fit_lgmm2(x: Tensor, y: Tensor, bins: int = 3, max_iter: int = 20,
            patch: int|List[int] = 20,  stride: int|List[int] = 1,
            mode: str = 'g', ...)
"""
from nitorch.core import utils, py
import torch
from torch.nn import functional as F
from nitorch.tools.registration.losses.utils_local import local_mean, prod as script_prod
from typing import List, Optional, Dict, Tuple
Tensor = torch.Tensor
pyutils = py


if utils.torch_version('>=', (1, 7)):
    _maximum = torch.maximum
else:
    _maximum = torch.max


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
def init_gmm2(x, y, bins=6, dim=None, mask=None):
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
    mask : (..., *spatial) tensor, optional
        Mask or weights

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
    if mask:
        mask = mask[..., 0]
    dim = dim or x.dim() - 1
    quantiles = torch.arange(bins + 1, dtype=x.dtype, device=x.device).div_(bins)
    xmean = utils.quantile(x[..., 0], quantiles, dim=range(-dim, 0), keepdim=True, mask=mask)
    ymean = utils.quantile(y[..., 0], quantiles, dim=range(-dim, 0), keepdim=True, mask=mask)
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
             update: str = 'em', mask: Optional[Tensor] = None):
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
        theta = init_gmm2(x, y, bins, dim, mask)
    return _fit_gmm2(x, y, max_iter, dim, z, theta, update, mask)


@torch.jit.script
def e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior, mask: Optional[Tensor] = None):
    x = x - xmean
    y = y - ymean
    lz = (x * x) * yvar
    lz += (y * y) * xvar
    lz -= (x * y) * (2*cov)
    lz *= idet
    lz -= idet.log()
    lz *= -0.5
    lz += logprior
    z = F.softmax(lz, dim=-1)
    lz = F.log_softmax(lz, dim=-1)
    if mask is not None:
        lz *= mask
    hz = -lz.flatten().dot(z.flatten())
    # numerical regularization
    # z = z.clamp_min_(1e-30)
    # z /= z.sum(dim=1, keepdim=True)
    return z, hz


@torch.jit.script
def suffstat(x, y, z, ndim: int, mask: Optional[Tensor] = None):
    # z must be normalized (== divided by mom0)
    if mask is not None:
        z = z * mask
    mom0 = sumspatial(z, ndim)
    z = z / mom0
    xmom1 = sumspatial(x * z, ndim)
    ymom1 = sumspatial(y * z, ndim)
    xmom2 = sumspatial((x * x) * z, ndim)
    ymom2 = sumspatial((y * y) * z, ndim)
    xymom2 = sumspatial((x * y) * z, ndim)
    return mom0, xmom1, ymom1, xmom2, ymom2, xymom2


@torch.jit.script
def load_cov(xvar, yvar, cov, alpha: float = 1e-10):
    # regularization
    #   to add the minimum amount of regularization possible, I compute
    #   the smallest eigenvalue of the covariance matrix and if it is negative
    #   or smaller than `alpha`, I add enough regularization to make it
    #   equal to `alpha`.

    det = xvar * yvar - cov * cov
    tr = xvar + yvar
    delta = tr * tr - 4 * det
    delta = delta.clamp_min_(0)
    lam = (tr - delta.sqrt()) / 2  # smallest eigenvalue
    lam = (alpha - lam).clamp_min_(0)
    # if (lam > 0).any():
    #     plam: List[float] = lam.flatten().tolist()
    #     print('eps', plam)
    xvar = (xvar + lam) / (1 + lam)
    yvar = (yvar + lam) / (1 + lam)
    cov = cov / (1 + lam)

    return xvar, yvar, cov


@torch.jit.script
def m_step(x, y, z, ndim: int, alpha: float = 1e-10,
           mask: Optional[Tensor] = None):
    nvox = script_prod(x.shape[-ndim-1:-1])
    mom0, xmom1, ymom1, xmom2, ymom2, xymom2 = suffstat(x, y, z, ndim, mask=mask)
    if mask is None:
        prior = mom0.div_(nvox)
    else:
        prior = mom0.div_(sumspatial(mask, ndim))
    xmean = xmom1
    ymean = ymom1
    xvar = xmom2.addcmul_(xmean, xmean, value=-1)
    yvar = ymom2.addcmul_(ymean, ymean, value=-1)
    cov = xymom2.addcmul_(xmean, ymean, value=-1)

    xvar, yvar, cov = load_cov(xvar, yvar, cov, alpha)

    # invert
    det = xvar * yvar - cov * cov
    idet = det.reciprocal_()

    return xmean, ymean, xvar, yvar, cov, idet, prior


@torch.jit.script
def em_loop(max_iter: int, x, y, xmean, ymean, xvar, yvar, cov, idet, logprior,
            ndim: int, mask: Optional[Tensor] = None):
    # print('')
    nvox = script_prod(x.shape[-ndim-1:-1])
    nmsk: Optional[Tensor] = sumspatial(mask, ndim) if mask is not None else None
    z, hz = e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior)
    ll_max = torch.zeros([1], dtype=x.dtype, device=x.device)
    ll_prev = torch.zeros([1], dtype=x.dtype, device=x.device)
    for nit in range(max_iter):
        xmean, ymean, xvar, yvar, cov, idet, prior = m_step(x, y, z, ndim, mask=mask)
        z, hz = e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior, mask=mask)
        logprior = prior.log()

        # negative log-likelihood (upper bound)
        logidet = idet.log()
        ll = ((logidet * 0.5 + logprior) * prior).sum()
        if nmsk is not None:
            ll += hz / nmsk
        else:
            ll += hz / nvox
        ll = -ll
        if nit == 0:
            # print('gmm', nit, ll.item())
            ll_max = ll
            ll_prev = ll
            continue
        gain = ((ll_prev - ll) / (ll_max - ll)).abs()
        # print('gmm', nit, ll.item(), gain.item())
        if gain < 1e-5:
            break
        ll_prev = ll
        ll_max = _maximum(ll_max, ll)
    # print('')
    return z, xmean, ymean, xvar, yvar, cov, idet, logprior, hz


@torch.jit.script
def _fit_gmm2(x: Tensor, y: Tensor, max_iter: int = 20,
             dim: Optional[int] = None, z: Optional[Tensor] = None,
             theta: Optional[Dict[str, Tensor]] = None,
             update: str = 'em', mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    # Fit a 2D Gaussian mixture model
    if dim is None:
        dim = y.dim() - 1
    nvox = script_prod(y.shape[-dim:])

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    nmsk: Optional[Tensor] = None
    if mask is not None:
        mask = mask.unsqueeze(-1)
        nmsk = sumspatial(mask, dim)

    if z is not None:
        xmean, ymean, xvar, yvar, cov, idet, prior = \
            m_step(x, y, z, dim, mask=mask)
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
    xvar, yvar, cov = load_cov(xvar, yvar, cov)
    det = xvar * yvar - cov * cov
    idet = det.reciprocal_()
    logprior = prior.log_()

    if update.lower() == 'e':
        z, hz = e_step(x, y, xmean, ymean, xvar, yvar, cov, idet, logprior, mask=mask)
    elif update.lower() == 'em':
        z, xmean, ymean, xvar, yvar, cov, idet, logprior, hz = \
            em_loop(max_iter, x, y, xmean, ymean, xvar, yvar, cov, idet,
                    logprior, dim, mask=mask)
    elif z is None:
        raise ValueError('update == "m" requires z')
    else:
        hz = z.log()
        if mask is not None:
            hz *= mask
        hz = -hz.flatten().dot(z.flatten())

    # negative log-likelihood (upper bound)
    corr = cov / (xvar*yvar).sqrt()
    prior = logprior.exp()
    logidet = idet.log()
    ll = ((logidet * 0.5 + logprior) * prior).sum()
    if nmsk is not None:
        ll += hz / nmsk
    else:
        ll += hz / nvox
    ll = -ll

    output: Dict[str, Tensor] = {
        'z': z, 'prior': prior, 'moving_mean': xmean, 'fixed_mean': ymean,
        'll': ll, 'moving_var': xvar, 'fixed_var': yvar, 'corr': corr,
        'hz': hz, 'idet': idet,
    }
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
    z0 = fwd(z, None).clamp_min(1e-10)
    suffstat = fwd(moments, z).div_(z0)
    xmean, ymean, xvar, yvar, cov = suffstat.unbind(0)
    xvar = xvar - xmean * xmean
    yvar = yvar - ymean * ymean
    cov = cov - xmean * ymean
    pi = z0 / z0.sum(-fwd.dim-1, keepdim=True)
    return xmean, ymean, xvar, yvar, cov, pi


@torch.jit.script
def e_step_local(bwd: Bwd, x, y, xmean, ymean, xvar, yvar, cov, prior):

    # covariance to precision
    det = xvar*yvar - cov*cov
    idet = det.reciprocal()
    icov = -cov * idet
    xprec = yvar * idet
    yprec = xvar * idet

    # push suffstat (icov, icov*mean, mean*icov*mean + logdetcov)
    mAm = (xmean * xmean * xprec  # \
           + ymean * ymean * yprec  # | mu * A * mu
           + 2 * xmean * ymean * icov  # /
           + det.log() - 2 * prior.log())  # log|A| - 2 * log pi
    Ax = xprec * xmean + icov * ymean  # (A * mu)[moving]
    Ay = yprec * ymean + icov * xmean  # (A * mu)[fixed]

    suffstat = torch.stack([mAm, Ax, Ay, xprec, yprec, icov])
    suffstat = bwd(suffstat, None, None)
    mAm, Ax, Ay, xprec, yprec, icov = suffstat.unbind(0)

    # E-step: update responsibilities
    lz = (x*x) * xprec + (y*y) * yprec
    lz += 2 * x * y * icov
    lz -= 2 * (x * Ax + y * Ay)
    lz += mAm
    lz *= -0.5
    z = F.softmax(lz, -bwd.dim - 1)
    lz = F.log_softmax(lz, -bwd.dim - 1)
    hz = -lz.flatten().dot(z.flatten())
    return z, hz


@torch.jit.script
def em_loop_local(fwd: Fwd, bwd: Bwd, moments: Tensor, z: Tensor,
                  max_iter: int, dim: int) -> Dict[str, Tensor]:

    x, y, x2, y2, xy = moments.unbind(0)

    nll_prev = torch.zeros([1], dtype=x.dtype, device=x.device)
    nll_max = torch.zeros([1], dtype=x.dtype, device=x.device)
    hz = -z.log().flatten().dot(z.flatten())
    for nit in range(max_iter):

        # M-step: update Gaussian parameters (pull suffstat)
        xmean, ymean, xvar, yvar, cov, pi = m_step_local(fwd, z, moments)
        xvar, yvar, cov = load_cov(xvar, yvar, cov, alpha=1e-6)

        # compute log likelihood
        det = xvar * yvar - cov * cov
        logdet = det.log()
        logpi = pi.log()
        nll = ((-0.5 * logdet + logpi) * pi).sum()
        nll /= script_prod(hz.shape[-dim:])
        nll += hz / script_prod(hz.shape[-dim:])
        nll = -nll
        if nit > 1:
            gain = (nll - nll_prev).abs() / (nll_max - nll).abs()
            # print('lgmm', nit, nll.item(), gain.item())
            if gain < 1e-6:
                break
            nll_prev = nll
            nll_max = _maximum(nll_max, nll)
        else:
            # print('lgmm', nit, nll.item())
            nll_prev = nll
            nll_max = nll

        # covariance to precision
        z, hz = e_step_local(bwd, x, y, xmean, ymean, xvar, yvar, cov, pi)

    # final M-step
    xmean, ymean, xvar, yvar, cov, pi = m_step_local(fwd, z, moments)
    xvar, yvar, cov = load_cov(xvar, yvar, cov, alpha=1e-6)

    corr = cov / (xvar * yvar).sqrt()
    det = xvar * yvar - cov * cov
    logdet = det.log()
    logpi = pi.log()
    nll = ((-0.5 * logdet + logpi) * pi).sum()
    nll /= script_prod(hz.shape[-dim:])
    nll += hz.sum() / script_prod(hz.shape[-dim:])
    nll = -nll

    output: Dict[str, Tensor] = {
        'resp': z, 'prior': pi, 'xmean': xmean, 'ymean': ymean, 'nll': nll,
        'xvar': xvar, 'yvar': yvar, 'corr': corr, 'resp_entropy': hz,
    }
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
