"""Utilities to fit 2D Gaussian Mixture Models for registration.

The main entry points are:
- fit_gmm2(x: Tensor, y: Tensor, bins: int = 6, max_iter: int = 20, ...)
- fit_lgmm2(x: Tensor, y: Tensor, bins: int = 6, max_iter: int = 20,
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
def _masked_quantile(x, q, dim, mask):
    """Quantiles over the last `dim` axes, using only voxels where mask > 0.
    Returns a (*batch, *ones, K) tensor."""
    batch = torch.broadcast_shapes(x.shape[:-dim], mask.shape[:-dim])
    spatial = x.shape[-dim:]
    x = x.expand([*batch, *spatial]).reshape([-1, py.prod(spatial)])
    mask = mask.expand([*batch, *spatial]).reshape([-1, py.prod(spatial)])
    out = torch.stack([torch.quantile(x1[m1 > 0], q)
                       for x1, m1 in zip(x, mask)])
    return out.reshape([*batch, *([1] * dim), len(q)])


def init_gmm2(x, y, bins=6, dim=None, mask=None):
    """Initialize parameters of a 2D GMM by drawing quantiles.

    Parameters
    ----------
    x : (..., *spatial) tensor
        Moving image (first dimension)
    y : (..., *spatial) tensor
        Fixed image (first dimension)
    bins : int, default=6
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
    # NOTE: `x`, `y` and `mask` are (..., *spatial) -- without the trailing
    # class dimension. This used to index `[..., 0]`, which (since the only
    # caller passes un-expanded tensors) dropped the last *spatial* axis:
    # quantiles were computed on a single slice, and a mask that was zero
    # on that slice gave NaN parameters.
    dim = dim or y.dim() - 1
    quantiles = torch.arange(bins + 1, dtype=x.dtype, device=x.device).div_(bins)
    if mask is None:
        xmean = utils.quantile(x, quantiles, dim=range(-dim, 0), keepdim=True)
        ymean = utils.quantile(y, quantiles, dim=range(-dim, 0), keepdim=True)
    else:
        # `utils.quantile(..., mask=)` lets masked-out values leak into the
        # extreme quantiles (and returns NaN for q=0 when the smallest
        # values are masked out), so compute quantiles on in-mask voxels.
        xmean = _masked_quantile(x, quantiles, dim, mask)
        ymean = _masked_quantile(y, quantiles, dim, mask)
    xvar = (xmean[..., 1:] - xmean[..., :-1]).div_(2.355).square_()
    yvar = (ymean[..., 1:] - ymean[..., :-1]).div_(2.355).square_()
    xmean = (xmean[..., 1:] + xmean[..., :-1]).div_(2)
    ymean = (ymean[..., 1:] + ymean[..., :-1]).div_(2)
    corr = torch.zeros_like(yvar)
    prior = y.new_full([bins], 1 / bins)
    return dict(xmean=xmean, ymean=ymean, xvar=xvar, yvar=yvar, corr=corr, prior=prior)


def fit_gmm2(x: Tensor, y: Tensor, bins: int = 6, max_iter: int = 20,
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
    bins : int, default=6
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
        nll : () tensor
            Negative log-likelihood, averaged across voxels
        nvox : () tensor
            Number of (masked) voxels used to average the nll
    """
    if z is None and not theta:
        theta = init_gmm2(x, y, bins, dim, mask)
    return _fit_gmm2(x, y, max_iter, dim, z, theta, update, mask)


@torch.jit.script
def get_det(xvar, yvar, cov):
    return xvar * yvar - cov * cov


@torch.jit.script
def softmax_lse_(lz, dim: int, out: Optional[Tensor] = None
                 ) -> Tuple[Tensor, Tensor]:
    """Softmax and log-sum-exp, with a single exponentiation.

    `lz` is overwritten (unless `out` is provided, in which case the
    softmax is written into `out`).

    Returns
    -------
    z : softmax(lz, dim)
    lse : logsumexp(lz, dim, keepdim=True)
    """
    mx = lz.amax(dim, keepdim=True)
    if out is None:
        z = lz.sub_(mx).exp_()
    else:
        z = torch.sub(lz, mx, out=out).exp_()
    sz = z.sum(dim, keepdim=True)
    z = z.div_(sz)
    return z, sz.log_().add_(mx)


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
    # log N(v | mu, Sigma) + log pi
    #   = -0.5 * [yvar * dx^2 + xvar * dy^2 - 2 * cov * dx * dy] / det
    #     -0.5 * log(det) + log(pi)
    # Computed in place to limit the number of (*spatial, K) buffers.
    idet = get_det(xvar, yvar, cov).reciprocal()
    dx = x - xmean
    dy = y - ymean
    dxy = dx * dy
    lz = dx.square_().mul_(yvar * idet * -0.5)
    lz.addcmul_(dy.square_(), xvar * idet, value=-0.5)
    lz.addcmul_(dxy, cov * idet)
    lz.add_(prior.log() + 0.5 * idet.log())
    # normalize (in place)
    z, ll = softmax_lse_(lz, -1)
    if mask is not None:
        ll = ll * mask
    ll = ll.sum()
    return -ll, z


@torch.jit.script
def make_features(x, y, ndim: int, mask: Optional[Tensor] = None) -> Tensor:
    """Features whose (weighted) sums are the sufficient statistics.

    Parameters
    ----------
    x, y : (..., *spatial, 1) tensor
    mask : (..., *spatial, 1) tensor, optional

    Returns
    -------
    f : (..., N, 6) tensor
        [1, x, y, x*x, y*y, x*y], multiplied by the mask
    """
    xy = torch.broadcast_tensors(x, y)
    x, y = xy[0], xy[1]
    f = torch.cat([torch.ones_like(x), x, y, x * x, y * y, x * y], -1)
    if mask is not None:
        f = f * mask
    batch: List[int] = f.shape[:-ndim-1]
    return f.reshape(batch + [-1, 6])


@torch.jit.script
def sum_outer(f, z, chunk: int = 4096) -> Tensor:
    """Sum of outer products across voxels: `sum_n f[n, :, None] * z[n]`

    A single (batched) matrix product, accumulated in double precision
    across chunks of voxels.

    Parameters
    ----------
    f : (..., N, F) tensor
    z : (..., N, K) tensor

    Returns
    -------
    s : (..., F, K) double tensor
    """
    n = f.shape[-2]
    nchunk = n // chunk
    nfull = nchunk * chunk
    fbatch: List[int] = f.shape[:-2]
    zbatch: List[int] = z.shape[:-2]
    nf, nk = f.shape[-1], z.shape[-1]
    s = torch.matmul(
        f[..., :nfull, :].reshape(fbatch + [nchunk, chunk, nf]).transpose(-1, -2),
        z[..., :nfull, :].reshape(zbatch + [nchunk, chunk, nk]),
    ).sum(-3, dtype=torch.double)
    if nfull < n:
        s += torch.matmul(f[..., nfull:, :].transpose(-1, -2),
                          z[..., nfull:, :]).double()
    return s


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
def m_step(f, z, ndim: int, alpha: float = 1e-10, prior_weight: float = 1e-8):
    """

    Parameters
    ----------
    f : (..., N, 6) tensor
        Features, from `make_features`
    z : (..., *spatial, K) tensor
        Responsibilities
    ndim : int
        Number of spatial dimensions
    alpha : float
        Minimum eigenvalue of the covariance matrices
    prior_weight : float
        Weight of a (weak) prior that pulls each class towards the
        statistics of the pooled classes (i.e., of the whole image), as
        a fraction of the pooled statistics. It keeps classes that become
        empty well defined (they revert to the global statistics with a
        tiny proportion) instead of 0/0 = NaN, which happens in single
        precision when responsibilities underflow (e.g., more classes
        than tissues).

    Returns
    -------
    xmean, ymean, xvar, yvar, cov, prior : (..., *ones, K) tensor
    """
    zbatch: List[int] = z.shape[:-ndim-1]
    z2 = z.reshape(zbatch + [f.shape[-2], z.shape[-1]])
    # (tiny) statistics are processed in double precision
    s = sum_outer(f, z2)
    s = s + prior_weight * s.sum(-1, keepdim=True)
    for _ in range(ndim):
        s = s.unsqueeze(-2)
    mom0, xmom1, ymom1, xmom2, ymom2, xymom2 = s.unbind(-ndim-2)
    prior = mom0 / mom0.sum(-1, keepdim=True)
    xmean = xmom1 / mom0
    ymean = ymom1 / mom0
    xvar = xmom2 / mom0 - xmean * xmean
    yvar = ymom2 / mom0 - ymean * ymean
    cov = xymom2 / mom0 - xmean * ymean

    xvar, yvar, cov = load_cov(xvar, yvar, cov, alpha)

    dtype = z.dtype
    return (xmean.to(dtype), ymean.to(dtype), xvar.to(dtype),
            yvar.to(dtype), cov.to(dtype), prior.to(dtype))


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
            ndim: int, mask: Optional[Tensor] = None,
            f: Optional[Tensor] = None):
    # print('')
    nmsk = sum_weights(x, y, mask, ndim)
    if f is None:
        f = make_features(x, y, ndim, mask)

    nll, z = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior, mask=mask)
    nll /= nmsk
    nll_max = nll_prev = nll
    for nit in range(max_iter):
        xmean, ymean, xvar, yvar, cov, prior = m_step(f, z, ndim)
        nll, z = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior, mask=mask)
        nll /= nmsk

        # Convergence: relative gain w.r.t. the total improvement so far.
        # - The denominator is guarded: a warm start at a fixed point
        #   (nll == nll_max) used to give 0/0 = NaN, which never satisfies
        #   `gain < tol`, so EM ran all `max_iter` iterations.
        # - Also stop when the absolute change is negligible: on a warm
        #   start the total improvement is itself tiny, so the relative
        #   criterion alone keeps iterating on ~1e-10 changes.
        delta = (nll_prev - nll).abs()
        denom = _maximum((nll_max - nll).abs(), 1e-12 * (1 + nll.abs()))
        gain = delta / denom
        # print("gmm", nit, nll.item(), gain.item())
        if (gain < 1e-5 or delta < 1e-9 * (1 + nll.abs())
                or not bool(torch.isfinite(gain))):
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
    f: Optional[Tensor] = None

    if z is not None:
        f = make_features(x, y, dim, mask)
        xmean, ymean, xvar, yvar, cov, prior = m_step(f, z, dim)
        corr = cov / (xvar * yvar).sqrt()
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
                    dim, mask=mask, f=f)
        # em_loop already returns the per-voxel (normalised) nll
    elif z is None:
        raise ValueError('update == "m" requires z')
    else:
        nll, _ = e_step(x, y, xmean, ymean, xvar, yvar, cov, prior, mask=mask)
        nll /= nmsk

    # negative log-likelihood (upper bound)
    corr = cov / (xvar*yvar).sqrt()

    output: Dict[str, Tensor] = {
        'resp': z, 'prior': prior, 'xmean': xmean, 'ymean': ymean,
        'nll': nll, 'xvar': xvar, 'yvar': yvar, 'corr': corr,
        'nvox': torch.tensor(nmsk, dtype=nll.dtype, device=nll.device),
    }
    return output


@torch.jit.script
class Fwd:
    def __init__(self, patch: List[float], stride: List[int], dim: int,
                 mode: str, cache: Optional[Dict[str, Tensor]] = None):
        self.patch = patch
        self.stride = stride
        self.dim = dim
        self.mode = mode
        # cache of the normalization map (local mean of ones)
        cache1: Dict[str, Tensor] = {}
        if cache is not None:
            cache1 = cache
        self.cache = cache1

    def __call__(self, x: Tensor, r: Optional[Tensor] = None):
        if r is not None:
            x = x*r
        return local_mean(x, self.patch, self.stride,
                          dim=self.dim, mode=self.mode, cache=self.cache)


@torch.jit.script
class Bwd:
    def __init__(self, patch: List[float], stride: List[int], dim:int,
                 mode:str, shape: List[int],
                 cache: Optional[Dict[str, Tensor]] = None):
        self.patch = patch
        self.stride = stride
        self.dim = dim
        self.mode = mode
        self.shape = shape
        # cache of the normalization map (local mean of ones)
        cache1: Dict[str, Tensor] = {}
        if cache is not None:
            cache1 = cache
        self.cache = cache1

    def __call__(self, x: Tensor,
                 r: Optional[Tensor] = None,
                 r0: Optional[Tensor] = None):
        if r0 is not None:
            x = x/r0
        x = local_mean(x, self.patch, self.stride, dim=self.dim,
                       mode=self.mode, backward=True, shape=self.shape,
                       cache=self.cache)
        if r is not None:
            x.mul_(r)
        return x


@torch.jit.script
def make_moments(x, y):
    return torch.stack([x, y, x*x, y*y, x*y])


@torch.jit.script
def m_step_local(fwd: Fwd, buf: Tensor, moments: Tensor,
                 prior_weight: float = 1e-8)\
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    buf : (6, ..., K, *spatial) tensor
        On input, `buf[0]` contains the responsibilities.
        `buf[1:]` is overwritten with `moments * buf[0]`.
    moments : (5, ..., 1, *spatial) tensor
    """
    # a single local mean of [z, z*moments]
    torch.mul(moments, buf[0], out=buf[1:])
    stats = fwd(buf, None)
    # weak prior towards the pooled local statistics (see `m_step`)
    stats = stats.add_(stats.sum(-fwd.dim-1, keepdim=True), alpha=prior_weight)
    z0 = stats[0].clamp_min_(1e-10)
    xmean, ymean, xvar, yvar, cov = stats[1:].div_(z0).unbind(0)
    xvar = xvar.addcmul_(xmean, xmean, value=-1)
    yvar = yvar.addcmul_(ymean, ymean, value=-1)
    cov = cov.addcmul_(xmean, ymean, value=-1)
    pi = z0 / z0.sum(-fwd.dim-1, keepdim=True)
    return xmean, ymean, xvar, yvar, cov, pi


@torch.jit.script
def e_step_local(bwd: Bwd, moments: Tensor, xmean, ymean, xvar, yvar,
                 cov, prior, out: Optional[Tensor] = None):
    """
    moments : (5, ..., 1, *spatial) tensor
        [x, y, x*x, y*y, x*y]
    out : (..., K, *spatial) tensor, optional
        Output buffer for the responsibilities
    """
    x, y, x2, y2, xy = moments.unbind(0)

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

    # E-step: update responsibilities (in place)
    lz = mAm.addcmul_(xprec, x2).addcmul_(yprec, y2).addcmul_(icov, xy, value=2)
    lz = lz.addcmul_(Ax, x, value=-2).addcmul_(Ay, y, value=-2).mul_(-0.5)
    z, ll = softmax_lse_(lz, -bwd.dim - 1, out)
    return -ll.mean(), z


@torch.jit.script
def em_loop_local(fwd: Fwd, bwd: Bwd, moments: Tensor, max_iter: int,
                  z: Optional[Tensor] = None,
                  theta: Optional[Dict[str, Tensor]] = None
                  ) -> Dict[str, Tensor]:

    if theta is not None:
        xmean = theta.pop('xmean')
        ymean = theta.pop('ymean')
        xvar = theta.pop('xvar')
        yvar = theta.pop('yvar')
        cov = theta.pop('corr') * yvar.sqrt() * xvar.sqrt()
        pi = theta.pop('prior')
        # initial E-step
        nll, z = e_step_local(bwd, moments, xmean, ymean, xvar, yvar, cov, pi)
    elif z is None:
        raise ValueError('theta or z should be provided')
    else:
        nll = torch.full([1], float('inf'), dtype=moments.dtype,
                         device=moments.device)
        xmean = ymean = xvar = yvar = cov = pi = torch.tensor([0])

    # Persistent buffer: [z, z * moments], reused across iterations.
    # The E-step writes the responsibilities into buf[0] and the M-step
    # writes the weighted moments into buf[1:].
    buf = torch.empty([6] + z.shape, dtype=z.dtype, device=z.device)
    buf[0].copy_(z)
    z = buf[0]

    nll_prev = nll_max = nll
    for nit in range(max_iter):

        # M-step: update Gaussian parameters
        xmean, ymean, xvar, yvar, cov, pi = m_step_local(fwd, buf, moments)
        xvar, yvar, cov = load_cov(xvar, yvar, cov, alpha=1e-6)

        # E-step
        nll, z = e_step_local(bwd, moments, xmean, ymean, xvar, yvar, cov,
                              pi, out=buf[0])

        # Same guarded criterion as `em_loop`. When EM starts from
        # responsibilities, nll_max is +inf: inf/inf = NaN at the first
        # iteration and finite/inf = 0 at the second (spurious stop), so
        # skip the check until a finite reference exists.
        delta = (nll - nll_prev).abs()
        if not bool(torch.isfinite(nll_max)):
            nll_prev = nll
            nll_max = nll
            continue
        denom = _maximum((nll_max - nll).abs(), 1e-12 * (1 + nll.abs()))
        gain = delta / denom
        # print('lgmm', nit, nll.item(), gain.item())
        if (gain < 1e-6 or delta < 1e-9 * (1 + nll.abs())
                or not bool(torch.isfinite(gain))):
            break
        nll_prev = nll
        nll_max = _maximum(nll_max, nll)

    corr = cov / (xvar * yvar).sqrt()
    z = z.clone()  # release the persistent buffer

    output: Dict[str, Tensor] = {
        'resp': z, 'prior': pi, 'xmean': xmean, 'ymean': ymean, 'nll': nll,
        'xvar': xvar, 'yvar': yvar, 'corr': corr,
    }
    return output


def fit_lgmm2(x, y, bins=6, max_iter=20, dim=None,
              patch=None, stride=None, mode='g', theta=None, cache=None):

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

    cache = {} if cache is None else cache
    fwd = Fwd(patch, stride, dim, mode, cache)
    bwd = Bwd(patch, stride, dim, mode, shape, cache)
    moments = make_moments(x, y)
    moments = moments.unsqueeze(-dim-1)  # [nmom, k(=1), *spatial]
    gmmfit = em_loop_local(fwd, bwd, moments, max_iter, z=z, theta=theta)
    return gmmfit
