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
from typing import List, Optional, Dict, Tuple, Union
Tensor = torch.Tensor
pyutils = py


_maximum = getattr(torch, 'maximum', torch.max)


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
    return x.sum(dims, keepdim=True, dtype=torch.double).to(x.dtype)


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
    if mask is not None:
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
        resp : (..., *spatial, bins) tensor
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
        idet : (..., *1, bins) tensor
            Pre-computed log-determinant of the inverse covariance
        resp_entropy : (..., *1, 1) tensor
            Pre-computed entropy of the responsibilites: sum(z*log(z))
    """
    if z is None and not theta:
        theta = init_gmm2(x, y, bins, dim, mask)
    return _fit_gmm2(x, y, max_iter, dim, z, theta, update, mask)


@torch.jit.script
def get_det(xvar, yvar, cov):
    return xvar * yvar - cov * cov


@torch.jit.script
def e_step(x, y, xmean, ymean, xvar, yvar, cov, prior,
           mask: Optional[Tensor] = None):
    """

    Parameters
    ----------
    x : (..., *spatial, 1)
    y : (..., *spatial, 1)
    xmean : (..., *ones, K)
    ymean : (..., *ones, K)
    xvar : (..., *ones, K)
    yvar : (..., *ones, K)
    cov : (..., *ones, K)
    prior : (..., *ones, K)
    mask : (..., *spatial)

    Returns
    -------
    nll, z

    """
    idet = get_det(xvar, yvar, cov).reciprocal()
    x = x - xmean
    y = y - ymean
    lz = (x * x) * yvar
    lz += (y * y) * xvar
    lz -= (x * y) * (2*cov)
    lz *= idet
    lz -= idet.log()
    lz *= -0.5
    lz += prior.log()
    z = F.softmax(lz, dim=-1)
    ll = torch.logsumexp(lz, dim=-1)
    if mask is not None:
        ll *= mask.squeeze(-1)
    ll = ll.sum()
    return -ll, z


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

    return xmean, ymean, xvar, yvar, cov, prior


@torch.jit.script
def sum_weights(x, y, mask: Optional[Tensor], ndim: int):
    nvox = script_prod(x.shape[-ndim-1:-1])
    nbatch = max(script_prod(x.shape[:-ndim-1]),
                 script_prod(y.shape[:-ndim-1]))
    nmsk: float = float(nbatch*nvox)
    if mask is not None:
        nmsk = float(mask.sum().item())
        if (script_prod(x.shape[:-mask.dim()])
                or script_prod(y.shape[:-mask.dim()])):
            nmsk *= max(script_prod(x.shape[:-mask.dim()]),
                        script_prod(y.shape[:-mask.dim()]))
    return nmsk


@torch.jit.script
def em_loop(max_iter: int, x, y, xmean, ymean, xvar, yvar, cov, prior,
            ndim: int, mask: Optional[Tensor] = None):
    # print('')
    nmsk = sum_weights(x, y, mask, ndim)

    nll, z = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior)
    nll /= nmsk
    nll_max = nll_prev = nll
    for nit in range(max_iter):
        xmean, ymean, xvar, yvar, cov, prior = m_step(x, y, z, ndim, mask=mask)
        nll, z = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior, mask=mask)
        nll /= nmsk

        gain = ((nll_prev - nll) / (nll_max - nll)).abs()
        # print('gmm', nit, nll.item(), gain.item())
        if gain < 1e-5:
            break
        nll_prev = nll
        nll_max = _maximum(nll_max, nll)
    # print('')
    return nll, z, xmean, ymean, xvar, yvar, cov, prior


@torch.jit.script
def _fit_gmm2(x: Tensor, y: Tensor, max_iter: int = 20,
              dim: Optional[int] = None, z: Optional[Tensor] = None,
              theta: Optional[Dict[str, Tensor]] = None,
              update: str = 'em', mask: Optional[Tensor] = None
              ) -> Dict[str, Tensor]:
    # Fit a 2D Gaussian mixture model
    if dim is None:
        dim = y.dim() - 1

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    if mask is not None:
        mask = mask.unsqueeze(-1)
    nmsk = sum_weights(x, y, mask, dim)

    if z is not None:
        xmean, ymean, xvar, yvar, cov, prior = m_step(x, y, z, dim, mask=mask)
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

    if update.lower() == 'e':
        nll, z = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior, mask=mask)
        nll /= nmsk
    elif update.lower() == 'em':
        nll, z, xmean, ymean, xvar, yvar, cov, prior = \
            em_loop(max_iter, x, y, xmean, ymean, xvar, yvar, cov, prior,
                    dim, mask=mask)
        nll /= nmsk
    elif z is None:
        raise ValueError('update == "m" requires z')
    else:
        nll, _ = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior, mask=mask)

    # negative log-likelihood (upper bound)
    corr = cov / (xvar*yvar).sqrt()

    output: Dict[str, Tensor] = {
        'resp': z, 'prior': prior, 'xmean': xmean, 'ymean': ymean,
        'nll': nll, 'xvar': xvar, 'yvar': yvar, 'corr': corr,
    }
    return output


@torch.jit.script
class Fwd:
    def __init__(self, patch: List[float], stride: List[int], dim: int, mode:str):
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
    def __init__(self, patch: List[float], stride: List[int], dim:int,
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
    ll = torch.logsumexp(lz, -bwd.dim - 1).mean()
    return -ll, z


@torch.jit.script
def em_loop_local(fwd: Fwd, bwd: Bwd, moments: Tensor, max_iter: int,
                  z: Optional[Tensor] = None,
                  theta: Optional[Dict[str, Tensor]] = None
                  ) -> Dict[str, Tensor]:

    x, y, x2, y2, xy = moments.unbind(0)

    if theta is not None:
        xmean = theta.pop('xmean')
        ymean = theta.pop('ymean')
        xvar = theta.pop('xvar')
        yvar = theta.pop('yvar')
        cov = theta.pop('corr') * yvar.sqrt() * xvar.sqrt()
        pi = theta.pop('prior')
        # initial E-step
        nll, z = e_step_local(bwd, x, y, xmean, ymean, xvar, yvar, cov, pi)
    elif z is None:
        raise ValueError('theta or z should be provided')
    else:
        nll = torch.full([1], float('inf'), dtype=x.dtype, device=x.device)
        xmean = ymean = xvar = yvar = cov = pi = torch.tensor([0])

    nll_prev = nll_max = nll
    for nit in range(max_iter):

        # M-step: update Gaussian parameters
        xmean, ymean, xvar, yvar, cov, pi = m_step_local(fwd, z, moments)
        xvar, yvar, cov = load_cov(xvar, yvar, cov, alpha=1e-6)

        # E-step
        nll, z = e_step_local(bwd, x, y, xmean, ymean, xvar, yvar, cov, pi)

        gain = (nll - nll_prev).abs() / (nll_max - nll).abs()
        # print('lgmm', nit, nll.item(), gain.item())
        if gain < 1e-6:
            break
        nll_prev = nll
        nll_max = _maximum(nll_max, nll)

    corr = cov / (xvar * yvar).sqrt()

    output: Dict[str, Tensor] = {
        'resp': z, 'prior': pi, 'xmean': xmean, 'ymean': ymean, 'nll': nll,
        'xvar': xvar, 'yvar': yvar, 'corr': corr,
    }
    return output


def fit_lgmm2(x, y, bins=6, max_iter=20, dim=None,
              patch=None, stride=None, mode='g', theta=None):

    dim = dim or (x.dim() - 1)
    shape = x.shape[-dim:]
    patch = list(map(float, py.ensure_list(patch or [20])))
    stride = py.ensure_list(stride or [1])

    # initialize parameters globally
    z = None
    if theta is None:
        gmmfit = fit_gmm2(x, y, bins, dim=dim, theta=theta)
        z = gmmfit.pop('resp')
        del gmmfit
        z = utils.fast_movedim(z, -1, -dim-1)  # [nmom(=1), k, *spatial]

    fwd = Fwd(patch, stride, dim, mode)
    bwd = Bwd(patch, stride, dim, mode, shape)
    moments = make_moments(x, y)
    moments = moments.unsqueeze(-dim-1)  # [nmom, k(=1), *spatial]
    gmmfit = em_loop_local(fwd, bwd, moments, max_iter, z=z, theta=theta)
    return gmmfit
