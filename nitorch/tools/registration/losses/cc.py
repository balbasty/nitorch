from nitorch.core import utils, py
import torch
from .utils_local import local_mean
from .base import OptimizationLoss


def cc(moving, fixed, dim=None, grad=True, hess=True, mask=None):
    """Squared Pearson's correlation coefficient loss

        1 - (E[(x - mu_x)'(y - mu_y)]/(s_x * s_y)) ** 2

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image with K channels.
    fixed : (..., K, *spatial) tensor
        Fixed image with K channels.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    grad : bool, default=True
        Compute an return gradient
    hess : bool, default=True
        Compute and return approximate Hessian

    Returns
    -------
    ll : () tensor

    """
    moving, fixed = utils.to_max_backend(moving, fixed)
    moving = moving.clone()
    fixed = fixed.clone()
    dim = dim or (fixed.dim() - 1)
    dims = list(range(-dim, 0))

    if mask is not None:
        mask = mask.to(fixed.device)
        mean = lambda x: (x*mask).sum(dim=dims, keepdim=True).div_(mask.sum(dim=dims, keepdim=True))
    else:
        mean = lambda x: x.mean(dim=dims, keepdim=True)

    n = py.prod(fixed.shape[-dim:])
    moving -= mean(moving)
    fixed -= mean(fixed)
    sigm = mean(moving.square()).sqrt_()
    sigf = mean(fixed.square()).sqrt_()
    moving = moving.div_(sigm)
    fixed = fixed.div_(sigf)

    corr = mean(moving*fixed)
    corr2 = 1 - corr.square()

    out = []
    if grad:
        g = 2 * corr * (moving * corr - fixed) / (n * sigm)
        g /= corr2  # chain rule for log
        if mask is not None:
            g = g.mul_(mask)
        out.append(g)

    if hess:
        # approximate hessian
        h = 2 * (corr / sigm).square() / n
        h /= corr2  # chain rule for log
        if mask is not None:
            h = h * mask
        out.append(h)

    # return stuff
    corr = corr2.log_().sum()
    out = [corr, *out]
    return tuple(out) if len(out) > 1 else out[0]


def _suffstat(fn, x, y):
    """Compute convolutional sufficient statistics

    Parameters
    ----------
    fn : callable
    x : tensor
    y : tensor

    Returns
    -------
    [fn(1), fn(x), fn(y), fn(x*x), fn(y*y), fn(x*y)]

    """

    square_ = lambda x: x.square() if x.requires_grad else x.square_()
    mul_ = lambda x, y: x.mul(y) if y.requires_grad else x.mul_(y)

    mom = x.new_empty([6, *x.shape])
    mom[0] = 1
    mom[1] = x
    mom[2] = y
    mom[3] = x
    # mom[3] = square_(mom[3])
    mom[3].square_()
    mom[4] = y
    # mom[4] = square_(mom[4])
    mom[4].square_()
    mom[5] = x
    # mom[5] = mul_(mom[5], y)
    mom[5].mul_(y)

    mom = fn(mom)
    return mom


def lcc(moving, fixed, dim=None, patch=20, stride=1, lam=1, mode='g',
        grad=True, hess=True, mask=None):
    """Local correlation coefficient (squared)

    This function implements a squared version of Cachier and
    Pennec's local correlation coefficient, so that anti-correlations
    are not penalized.

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image with K channels.
    fixed : (..., K, *spatial) tensor
        Fixed image with K channels.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    patch : int, default=5
        Patch size
    lam : float or ([B], K|1, [*spatial]) tensor_like, default=1
        Precision of the NCC distribution
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return approximate Hessian

    Returns
    -------
    ll : () tensor

    References
    ----------
    ..[1] "3D Non-Rigid Registration by Gradient Descent on a Gaussian-
           Windowed Similarity Measure using Convolutions"
          Pascal Cachier, Xavier Pennec
          MMBIA (2000)

    """
    if moving.requires_grad:
        sqrt_ = torch.sqrt
        square_ = torch.square
        mul_ = torch.mul
        div_ = torch.div
    else:
        sqrt_ = torch.sqrt_
        square_ = torch.square_
        mul_ = lambda x, y: x.mul_(y)
        div_ = lambda x, y: x.div_(y)

    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]
    if mask is not None:
        mask = mask.to(**utils.backend(fixed))

    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)

    if not isinstance(patch, (list, tuple)):
        patch = [patch]
    patch = list(patch)
    if not isinstance(stride, (list, tuple)):
        stride = [stride]
    stride = [s or 0 for s in stride]
    fwd = lambda x: local_mean(x, patch, stride, dim=dim, mode=mode, mask=mask)
    bwd = lambda x: local_mean(x, patch, stride, dim=dim, mode=mode, mask=mask,
                               backward=True, shape=shape)

    # compute ncc within each patch
    mom0, moving_mean, fixed_mean, moving_std, fixed_std, corr = \
        _suffstat(fwd, moving, fixed)
    mom0 = mom0.div_(mom0.sum(list(range(-dim, 0)), keepdim=True))
    moving_std = sqrt_(moving_std.addcmul_(moving_mean, moving_mean, value=-1))
    fixed_std = sqrt_(fixed_std.addcmul_(fixed_mean, fixed_mean, value=-1))
    moving_std.clamp_min_(1e-5)
    fixed_std.clamp_min_(1e-5)
    std2 = moving_std * fixed_std
    corr = div_(corr.addcmul_(moving_mean, fixed_mean, value=-1), std2)

    out = []
    if grad or hess:
        fixed_mean = div_(fixed_mean, fixed_std)
        moving_mean = div_(moving_mean, moving_std)

        h = bwd(square_(corr / moving_std).mul_(mom0).mul_(lam))

        if grad:
            # g = G' * (corr.*(corr.*xmean./xstd - ymean./ystd)./xstd)
            #   - x .* (G' * (corr./ xstd).^2)
            #   + y .* (G' * (corr ./ (xstd.*ystd)))
            # g = -2 * g
            g = fixed_mean.addcmul_(corr, moving_mean, value=-1)
            g = mul_(g, corr / moving_std).mul_(mom0).mul_(lam)
            g = bwd(g)
            g = g.addcmul_(h, moving)
            g = g.addcmul_(bwd((corr / std2).mul_(mom0).mul_(lam)), fixed, value=-1)
            g = g.mul_(2)
            out.append(g)

        if hess:
            # h = 2 * (G' * (corr./ xstd).^2)
            h = h.mul_(2)
            out.append(h)

    # return stuff
    corr = square_(corr).neg_().add_(1).mul_(mom0).mul_(lam)
    corr = corr.sum()
    out = [corr, *out]
    return tuple(out) if len(out) > 1 else out[0]


class CC(OptimizationLoss):
    """Pearson's correlation coefficient (squared)"""

    order = 2  # Hessian defined

    def __init__(self, dim=None):
        """

        Parameters
        ----------
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.dim = dim

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

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
        dim = kwargs.pop('dim', self.dim)
        return cc(moving, fixed, dim=dim, grad=False, hess=False, **kwargs)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

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
        dim = kwargs.pop('dim', self.dim)
        return cc(moving, fixed, dim=dim, hess=False, **kwargs)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

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
        dim = kwargs.pop('dim', self.dim)
        return cc(moving, fixed, dim=dim, **kwargs)


class LCC(OptimizationLoss):
    """Local correlation coefficient"""

    order = 2  # Hessian defined

    def __init__(self, dim=None, patch=20, stride=1, lam=1, mode='g'):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.stride = stride
        self.lam = lam
        self.mode = mode

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
        dim = kwargs.pop('dim', self.dim)
        patch = kwargs.pop('patch', self.patch)
        stride = kwargs.pop('stride', self.stride)
        lam = kwargs.pop('lam', self.lam)
        mode = kwargs.pop('mode', self.mode)
        return lcc(moving, fixed, dim=dim, patch=patch, stride=stride,
                   lam=lam, mode=mode, grad=False, hess=False, **kwargs)

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
        dim = kwargs.pop('dim', self.dim)
        patch = kwargs.pop('patch', self.patch)
        stride = kwargs.pop('stride', self.stride)
        lam = kwargs.pop('lam', self.lam)
        mode = kwargs.pop('mode', self.mode)
        return lcc(moving, fixed, dim=dim, patch=patch, stride=stride,
                   lam=lam, mode=mode, hess=False, **kwargs)

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
        dim = kwargs.pop('dim', self.dim)
        patch = kwargs.pop('patch', self.patch)
        stride = kwargs.pop('stride', self.stride)
        lam = kwargs.pop('lam', self.lam)
        mode = kwargs.pop('mode', self.mode)
        return lcc(moving, fixed, dim=dim, patch=patch, stride=stride,
                   lam=lam, mode=mode, **kwargs)


# aliases
ncc = cc
lncc = lcc
NCC = CC
LNCC = LCC