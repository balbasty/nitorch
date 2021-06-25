"""
This file implements losses that are typically used for registration
Each of these functions can return analytical gradients and (approximate)
Hessians to be used in optimization-based algorithms (although the
objective function is differentiable and autograd can be used as well).

These function are implemented in functional form (mse, nmi, cat, ...),
but OO wrappers are also provided for ease of use (MSE, NMI, Cat, ...).

Currently, the following losses are implemented:
- mse : mean squared error
- cat : categorical cross entropy
- ncc : normalized cross-correlation
- nmi : normalized mutual information
"""
from nitorch.core import utils, py, math, constants, linalg
import math as pymath
import torch
from .utils import JointHist
pyutils = py


def make_loss(loss, dim=None):
    """Instantiate loss object from string.
    Does nothing if `loss` is already a loss object.

    Parameters
    ----------
    loss : {'mse', 'mad', 'tukey', 'cat', 'ncc', 'nmi'} or OptimizationLoss
        'mse' : Means Squared Error (l2 loss)
        'mad' : Median Absolute Deviation (l1 loss, using IRLS)
        'tukey' : Tukey's biweight (~ truncated l2 loss)
        'cat' : Categorical cross-entropy
        'ncc' : Normalized Cross Correlation (zero-normalized version)
        'nmi' : Normalized Mutual Information (studholme's normalization)
    dim : int, optional
        Number of spatial dimensions

    Returns
    -------
    loss : OptimizationLoss

    """
    loss = (MSE(dim=dim) if loss == 'mse' else
            MAD(dim=dim) if loss == 'mad' else
            Tukey(dim=dim) if loss == 'tukey' else
            Cat(dim=dim) if loss == 'cat' else
            NCC(dim=dim) if loss == 'ncc' else
            NMI(dim=dim) if loss == 'nmi' else
            loss)
    if isinstance(loss, str):
        raise ValueError(f'Unknown loss {loss}')
    return loss


def irls_laplace_reweight(moving, fixed, lam=1, joint=False, eps=1e-5, dim=None):
    """Update iteratively reweighted least-squares weights for l1

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Inverse-squared scale parameter of the Laplace distribution.
        (equivalent to Gaussian noise precision)
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions

    Returns
    -------
    weights : (..., K|1, *spatial) tensor
        IRLS weights

    """
    if lam is None:
        lam = 1
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    weights = (moving - fixed).square_().mul_(lam)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    weights = weights.sqrt_().clamp_min_(eps).reciprocal_()
    return weights


def weighted_precision(moving, fixed, weights=None, dim=None):
    dim = dim or fixed.dim() - 1
    residuals = (moving - fixed).square_()
    if weights is not None:
        residuals.mul_(weights)
    lam = residuals.mean(dim=list(range(-dim, 0)))
    lam = lam.reciprocal_()  # variance to precision
    return lam


def irls_tukey_reweight(moving, fixed, lam=1, c=4.685, joint=False, dim=None):
    """Update iteratively reweighted least-squares weights for Tukey's biweight

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Equivalent to Gaussian noise precision
        (used to standardize the residuals)
    c  : float, default=4.685
        Tukey's threshold.
        Approximately equal to a number of standard deviations above
        which the loss is capped.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions

    Returns
    -------
    weights : (..., K|1, *spatial) tensor
        IRLS weights

    """
    if lam is None:
        lam = 1
    c = c * c
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    residuals = (moving - fixed).square_().mul_(lam)
    if joint:
        residuals = residuals.sum(dim=-dim-1, keepdims=True)
    weights = torch.zeros_like(residuals)
    mask = residuals <= c
    weights[mask] = (1 - weights[mask]/c).square()
    return weights


def mse(moving, fixed, lam=1, dim=None, grad=True, hess=True):
    """Mean-squared error loss for optimisation-based registration.

    (A factor 1/2 is included, and the loss is averaged across voxels,
    but not across channels or batches)

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Gaussian noise precision (or IRLS weights)
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return Hessian

    Returns
    -------
    ll : () tensor
        Negative log-likelihood
    g : (..., K, *spatial) tensor, optional
        Gradient with respect to the moving imaged
    h : (..., K, *spatial) tensor, optional
        (Diagonal) Hessian with respect to the moving image

    """
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    nvox = py.prod(fixed.shape[-dim:])

    ll = (moving - fixed).square().mul_(lam).sum() / (2*nvox)
    out = [ll]
    if grad:
        g = (moving - fixed).mul_(lam/nvox)
        out.append(g)
    if hess:
        h = lam/nvox
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def cat(moving, fixed, dim=None, acceleration=0, grad=True, hess=True):
    """Categorical loss for optimisation-based registration.

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image of log-probabilities (pre-softmax).
        The background class should be omitted.
    fixed : (..., K, *spatial) tensor
        Fixed image of probabilities
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    acceleration : (0..1), default=0
        Weight the contributions of the true Hessian and Boehning bound.
        The Hessian is a weighted sum between the Boehning bound and the
        Gauss-Newton Hessian: H = a * H_gn + (1-a) * H_bnd
        The Gauss-Newton Hessian is less stable but allows larger jumps
        than the Boehning Hessian, so increasing `a` can lead to an
        accelerated convergence.
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return Hessian

    Returns
    -------
    ll : () tensor
        Negative log-likelihood
    g : (..., K, *spatial) tensor, optional
        Gradient with respect to the moving image
    h : (..., K*(K+1)//2, *spatial) tensor, optional
        Hessian with respect to the moving image.
        Its spatial dimensions are singleton when `acceleration == 0`.

    """
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    nc = moving.shape[-dim-1]                               # nb classes - bck
    fixed = utils.slice_tensor(fixed, slice(nc), -dim-1)    # remove bkg class
    nvox = py.prod(fixed.shape[-dim:])

    # log likelihood
    ll = moving*fixed
    ll -= math.logsumexp(moving, dim=-dim-1, implicit=True)            # implicit lse
    ll = ll.sum().neg() / nvox
    out = [ll]

    if grad or (hess and acceleration > 0):
        # implicit softmax
        moving = math.softmax(moving, dim=-dim-1, implicit=True)

    # gradient
    if grad:
        g = (moving - fixed).div_(nvox)
        out.append(g)

    # hessian
    if hess:
        # compute true Hessian
        def allocate_h():
            nch = nc*(nc+1)//2
            shape = list(moving.shape)
            shape[-dim-1] = nch
            h = moving.new_empty(shape)
            return h

        h = None
        if acceleration > 0:
            h = allocate_h()
            h_diag = utils.slice_tensor(h, slice(nc), -dim-1)
            h_diag.copy_(moving*(1 - moving))
            # off diagonal elements
            c = 0
            for i in range(nc):
                pi = utils.slice_tensor(moving, i, -dim-1)
                for j in range(i+1, nc):
                    pj = utils.slice_tensor(moving, j, -dim-1)
                    out = utils.slice_tensor(h, nc+c, -dim-1)
                    out.copy_(pi*pj).neg_()
                    c += 1

        # compute Boehning Hessian
        def allocate_hb():
            nch = nc*(nc+1)//2
            h = moving.new_empty(nch)
            return h

        if acceleration < 1:
            hb = allocate_hb()
            hb[:nc] = 1 - 1/(nc+1)
            hb[nc:] = -1/(nc+1)
            hb = utils.unsqueeze(hb, -1, dim)
            hb.div_(2)
            if acceleration > 0:
                hb.mul_(1-acceleration)
                h.mul_(acceleration).add_(hb)
            else:
                h = hb

        out.append(h.div_(nvox))

    return tuple(out) if len(out) > 1 else out[0]


def ncc_hist(moving, fixed, dim=None,
        bins=64, order=3, grad=True, hess=True):
    """Normalized cross-correlation: E[(x - mu_x)'(y - mu_y)]/(s_x * s_y)

    Parameters
    ----------
    moving : tensor
    fixed : tensor
    dim : int, default=`fixed.dim() - 1`
    bins : int, default=64
    order : int, default=3
    grad : bool, default=True
    hess : bool, default=True

    Returns
    -------
    ll : () tensor

    """

    jointhist = JointHist(bins, order=order)

    # compute histogram
    dim = dim or fixed.shape[-1]
    concat = torch.stack([moving, fixed], dim=-1)
    concat = concat.reshape([-1, py.prod(concat.shape[-dim-1:-1]), 2])
    h, min, max = jointhist.forward(concat)
    h = h / h.sum(dim=[-1, -2], keepdim=True)

    minm = min[..., 0]
    maxm = max[..., 0]
    minf = min[..., 1]
    maxf = max[..., 1]
    deltam = (maxm - minm) / bins
    deltaf = (maxf - minf) / bins
    idx = torch.arange(0, bins, **utils.backend(h))

    def moments(h, mn, delta):
        mean = mn + delta * (h * idx).sum(-1)
        var = (mn.square()
               + delta.square() * (h * idx.square()).sum(-1)
               + 2 * mn * delta * (h * idx).sum(-1))
        var -= mean.square()
        return mean, var

    hm = h.sum(-1)
    hm /= hm.sum(-1, keepdim=True)
    meanm, varm = moments(hm, minm, deltam)

    hf = h.sum(-2)
    hf /= hf.sum(-1, keepdim=True)
    meanf, varf = moments(hf, minf, deltaf)

    minm -= meanm
    minf -= meanf
    idx2 = idx[None, :] * idx[:, None]

    l = (minm * minf
         + minf * deltam * (h.sum(-2) * idx).sum(-1)
         + minm * deltaf * (h.sum(-1) * idx).sum(-1)
         + deltam * deltaf * (h * idx2).sum([-1, -2]))
    l /= varm.sqrt() * varf.sqrt()
    l = l.sum()
    return l


def ncc(moving, fixed, dim=None, grad=True, hess=True):
    """Person's correlation (or Zero-normalized cross-correlation) loss

        (1 - E[(x - mu_x)'(y - mu_y)]/(s_x * s_y)) ** 2

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
    moving = moving.clone()
    fixed = fixed.clone()

    dim = dim or (fixed.dim() - 1)
    dims = list(range(-dim, 0))
    n = py.prod(fixed.shape[-dim:])
    moving -= moving.mean(dim=dims, keepdim=True)
    fixed -= fixed.mean(dim=dims, keepdim=True)
    sigm = moving.square().mean(dim=dims, keepdim=True).sqrt_()
    sigf = fixed.square().mean(dim=dims, keepdim=True).sqrt_()
    moving = moving / sigm
    fixed = fixed / sigf

    ll = (moving*fixed).mean(dim=dims, keepdim=True)

    out = []
    if grad or hess:
        g = (moving * ll - fixed) / (n * sigm)

    if hess:
        # true hessian
        # hh = (3 * ncc) * (xnorm*xnorm') - (N*eye(N) - 1) * ncc - (ynorm*xnorm' + xnorm*ynorm');
        # hh = hh / (N.^2 * std(x, 1).^2);
        # something positive definite
        moving = moving.abs_()
        fixed = fixed.abs_()
        summ = moving.sum(dim=dims, keepdim=True)
        sumf = fixed.sum(dim=dims, keepdim=True)
        lla = ll.abs()
        h = moving * 3 * lla * summ
        h += moving * sumf + fixed * summ
        h += (n - 1) * lla
        h /= (n*n * sigm.square())
        h = lla * h + g.abs() * g.abs().sum(dim=dims, keepdim=True)
        # h = g.abs() * g.abs().sum(dim=dims, keepdim=True)
        h.mul_(2)
        out.append(h)

    if grad:
        g *= ll
        g.mul_(2)
        out.append(g)

    # return stuff
    ll = (1 - ll.square()).sum()
    out = [ll, *out]
    return tuple(out) if len(out) > 1 else out[0]


# def pncc(moving, fixed, dim=None, patch=5, link='vm', lam=1, grad=True, hess=True):
#     """Patch-based Pearson's correlation
#
#     Normalized cross correlation is computed within patches, and an l1
#     norm is applied
#
#     Parameters
#     ----------
#     moving : (..., K, *spatial) tensor
#         Moving image with K channels.
#     fixed : (..., K, *spatial) tensor
#         Fixed image with K channels.
#     dim : int, default=`fixed.dim() - 1`
#         Number of spatial dimensions.
#     patch : int, default=5
#         Patch size
#     link : {'vm', 'u', '
#     lam : float or ([B], K|1, [*spatial]) tensor_like, default=1
#         Precision of the Von-Mises link distribution
#     grad : bool, default=True
#         Compute an return gradient
#     hess : bool, default=True
#         Compute and return approximate Hessian
#
#     Returns
#     -------
#     ll : () tensor
#
#     """
#     moving = moving.clone()
#     fixed = fixed.clone()
#
#     dim = dim or (fixed.dim() - 1)
#     dims = list(range(-dim, 0))
#     n = py.prod(fixed.shape[-dim:])
#     moving -= moving.mean(dim=dims, keepdim=True)
#     fixed -= fixed.mean(dim=dims, keepdim=True)
#     sigm = moving.square().mean(dim=dims, keepdim=True).sqrt_()
#     sigf = fixed.square().mean(dim=dims, keepdim=True).sqrt_()
#     moving = moving / sigm
#     fixed = fixed / sigf
#
#     ll = (moving*fixed).mean(dim=dims, keepdim=True)
#
#     out = []
#     if grad or hess:
#         g = (moving * ll - fixed) / (n * sigm)
#
#     if hess:
#         # true hessian
#         # hh = (3 * ncc) * (xnorm*xnorm') - (N*eye(N) - 1) * ncc - (ynorm*xnorm' + xnorm*ynorm');
#         # hh = hh / (N.^2 * std(x, 1).^2);
#         # something positive definite
#         moving = moving.abs_()
#         fixed = fixed.abs_()
#         summ = moving.sum(dim=dims, keepdim=True)
#         sumf = fixed.sum(dim=dims, keepdim=True)
#         lla = ll.abs()
#         h = moving * 3 * lla * summ
#         h += moving * sumf + fixed * summ
#         h += (n - 1) * lla
#         h /= (n*n * sigm.square())
#         h = lla * h + g.abs() * g.abs().sum(dim=dims, keepdim=True)
#         # h = g.abs() * g.abs().sum(dim=dims, keepdim=True)
#         h.mul_(2)
#         out.append(h)
#
#     if grad:
#         g *= ll
#         g.mul_(2)
#         out.append(g)
#
#     # return stuff
#     ll = (1 - ll.square()).sum()
#     out = [ll, *out]
#     return tuple(out) if len(out) > 1 else out[0]


def nmi(moving, fixed, dim=None, bins=64, order=5, fwhm=2, norm='studholme',
        grad=True, hess=True, minmax=False):
    """(Normalized) Mutual Information

    If multi-channel data is provided, the MI between each  pair of
    channels (e.g., (1, 1), (2, 2), ...) is computed -- but *not*
    between all possible pairs (e.g., (1, 1), (1, 2), (1, 3), ...).

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image with K channels.
    fixed : (..., K, *spatial) tensor
        Fixed image with K channels.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    bins : int, default=64
        Number of bins in the joing histogram.
    order : int, default=3
        Order of B-splines encoding the histogram.
    norm : {'studholme', 'arithmetic', None}, default='studholme'
        Normalization method:
        None : mi = H[x] + H[y] - H[xy]
        'arithmetic' : nmi = 0.5 * mi / (H[x] + H[y])
        'studholme' : nmi = mi / H[xy]
    grad : bool, default=True
        Compute an return gradient
    hess : bool, default=True
        Compute and return approximate Hessian

    Returns
    -------
    ll : () tensor
        1 -  NMI
    grad : (..., K, *spatial) tensor
    hess : (..., K, *spatial) tensor

    """

    hist = JointHist(n=bins, order=order, fwhm=fwhm)

    shape = moving.shape
    dim = dim or fixed.dim() - 1
    nvox = pyutils.prod(shape[-dim:])
    moving = moving.reshape([*moving.shape[:-dim], -1])
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])
    idx = torch.stack([moving, fixed], -1)

    if minmax not in (True, False, None):
        mn, mx = minmax
        h, mn, mx = hist.forward(idx, min=mn, max=mx)
    else:
        h, mn, mx = hist.forward(idx)
    h = h.clamp(1e-8)
    h /= nvox

    pxy = h
    px = pxy.sum(-2, keepdim=True)
    py = pxy.sum(-1, keepdim=True)

    hxy = -(pxy * pxy.log()).sum([-1, -2], keepdim=True)
    hx = -(px * px.log()).sum([-1, -2], keepdim=True)
    hy = -(py * py.log()).sum([-1, -2], keepdim=True)

    if norm == 'studholme':
        nmi = (hx + hy) / hxy  # Studholme's NMI + 1
    elif norm == 'arithmetic':
        nmi = -hxy / (hx + hy)  # 1 - Arithmetic normalization
    else:
        nmi = hx + hy - hxy

    out = []
    if grad or hess:
        if norm == 'studholme':
            g0 = ((1 + px.log()) + (1 + py.log())) - nmi * (1 + pxy.log())
            g0 /= hxy
        elif norm == 'arithmetic':
            g0 = nmi * ((1 + px.log()) + (1 + py.log())) + (1 + pxy.log())
            g0 = -g0 / (hx + hy)
        else:
            g0 = ((1 + px.log()) + (1 + py.log())) - (1 + pxy.log())
        if grad:
            g = hist.backward(idx, g0/nvox)[..., 0]
            g = g.reshape(shape)
            out.append(g)

        if hess:
            # This Hessian is for Studholme's normalization only!
            # I need to derive one for Arithmetic/None cases

            # # # True Hessian: not positive definite
            # ones = torch.ones([bins, bins])
            # g0 = g0.flatten(start_dim=-2)
            # pxy = pxy.flatten(start_dim=-2)
            # h = (g0[..., :, None] * (1 + pxy.log()[..., None, :]))
            # h = h + h.transpose(-1, -2)
            # tmp = linalg.kron2(ones, px[..., 0, :].reciprocal().diag_embed())
            # h -= tmp
            # tmp = linalg.kron2(py[..., :, 0].reciprocal().diag_embed(), ones)
            # h -= tmp
            # h += (nmi/pxy).flatten(start_dim=-2).diag_embed()
            # h /= hxy
            # # h.neg_()
            # # h = h.abs().sum(-1)
            # h.diagonal(0, -1, -2).abs()
            # h = h.reshape([*h.shape[:-1], bins, bins])

            h = (2 - nmi) * py.reciprocal() / hxy
            h = h.diag_embed()

            # Approximate Hessian: positive definite
            # I take the positive definite majorizer diag(|H|1) of each
            # term in the sum.
            #   H = H1 + H2 + H2 + H4
            #   => P = diag(|H1|1) + diag(|H1|2) + diag(|H1|3) + diag(|H1|4)
            # ones = torch.ones([bins, bins])
            # g0 = g0.flatten(start_dim=-2)
            # pxy = pxy.flatten(start_dim=-2)
            # # 1) diag(|H1|1)
            # h = (g0[..., :, None] * (1 + pxy.log()[..., None, :]))
            # h = h + h.transpose(-1, -2)
            # h = h.abs_().sum(-1)
            # # 2) diag(|H2|1)
            # tmp = linalg.kron2(ones, px[..., 0, :].reciprocal().diag_embed())
            # tmp = tmp.abs_().sum(-1)
            # h += tmp
            # # 3) diag(|H3|1)
            # tmp = linalg.kron2(py[..., :, 0].reciprocal().diag_embed(), ones)
            # tmp = tmp.abs_().sum(-1)
            # h += tmp
            # # 4) |H4| (already diagonal)
            # h += (nmi/pxy).flatten(start_dim=-2).abs()
            # # denominator
            # h /= hxy.flatten(start_dim=-2).abs()

            # project
            h = hist.backward(idx, h/nvox, hess=True)[..., 0]
            h = h.reshape(shape)
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(g[0, ..., g.shape[-1]//2])
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(h[0, ..., h.shape[-1]//2])
            plt.colorbar()
            plt.show()
            out.append(h)

    nmi = -nmi
    if norm == 'studholme':
        nmi = (2+nmi)
    elif norm in (None, 'none'):
        nmi = nmi/hy
    nmi = nmi.sum()

    out = [nmi, *out]
    if minmax is True:
        out.extend([mn, mx])
    return tuple(out) if len(out) > 1 else out[0]


def mse_hist(moving, fixed, dim=None, bins=32, order=3, grad=True, hess=True, minmax=False):
    """Studholme's Normalized Mutual Information

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
    fixed : (..., K, *spatial) tensor
    dim : int, default=`fixed.dim() - 1`
    bins : int, default=64
    order : int, default=3
    grad : bool, default=True
    hess : bool, default=True

    Returns
    -------
    ll : () tensor
        Negative NMI
    grad : (..., K, *spatial) tensor
    hess : (..., K, *spatial) tensor

    """
    hist = JointHist(bins, order, bound='zero')

    shape = moving.shape
    dim = dim or fixed.dim() - 1
    moving = moving.reshape([*moving.shape[:-dim], -1])
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])
    idx = torch.stack([moving, fixed], -1)

    if minmax not in (True, False, None):
        mn, mx = minmax
        h, mn, mx = hist.forward(idx, min=mn, max=mx)
    else:
        h, mn, mx = hist.forward(idx)
    minm, minf = mn.unbind(-1)
    maxm, maxf = mx.unbind(-1)
    deltam = (maxm-minm)/bins
    deltaf = (maxf-minf)/bins

    val = torch.linspace(0.5, 0.95, bins, **utils.backend(h))
    nodesm = minm[..., None] + val * deltam[..., None]
    nodesf = minf[..., None] + val * deltaf[..., None]
    mse = nodesm[..., :, None] - nodesf[..., None, :]
    mse = mse.square_()

    out = []
    if grad:
        g = hist.backward(idx, mse)[..., 0]
        g = g.reshape(shape)
        out.append(g)

    if hess:
        # # try 1
        # hh = torch.ones_like(h)
        # hh = hist.backward(idx, hh, hess=True)[..., 0]
        # hh = hh.reshape(shape)
        hh = mse.new_full([1] * (dim + 1), 2.25 * (2**(dim+1)))
        hh = hh * (bins / (maxm - minm)).square()
        out.append(hh)

    mse *= h
    mse = mse.sum()
    out = [mse, *out]
    if minmax is True:
        out.extend([mn, mx])
    return tuple(out) if len(out) > 1 else out[0]


class OptimizationLoss:
    """Base class for losses used in 'old school' optimisation-based stuff."""

    def __init__(self):
        """Specify parameters"""
        pass

    def loss(self, *args, **kwargs):
        """Returns the loss (to be minimized) only"""
        raise NotImplementedError

    def loss_grad(self, *args, **kwargs):
        """Returns the loss (to be minimized) and its gradient
        with respect to the *first* argument."""
        raise NotImplementedError

    def loss_grad_hess(self, *args, **kwargs):
        """Returns the loss (to be minimized) and its gradient
        and hessian with respect to the *first* argument.

        In general, we expect a block-diagonal positive-definite
        approximation of the true Hessian (in general, correlations
        between spatial elements -- voxels -- are discarded).
        """
        raise NotImplementedError


class HistBasedOptimizationLoss(OptimizationLoss):
    """Base class for histogram-bases losses"""

    def __init__(self, dim=None, bins=None, order=3, fwhm=2):
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.order = order
        self.fwhm = fwhm

    def autobins(self, image, dim):
        dim = dim or (image.dim() - 1)
        shape = image.shape[-dim:]
        nvox = py.prod(shape)
        bins = 2 ** int(pymath.ceil(pymath.log2(nvox ** (1/4))))
        return bins


class MSE(OptimizationLoss):
    """Mean-squared error"""

    def __init__(self, lam=None, dim=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False)
        return llx + lll

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g = mse(moving, fixed, dim=dim, lam=lam, hess=False)
        return llx + lll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient
        hess : ([B], K, *spatial) tensor
            Diagonal Hessian

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam)
        return llx + lll, g, h


class MAD(OptimizationLoss):
    """Median absolute deviation (using IRLS)"""

    def __init__(self, lam=None, joint=False, dim=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim
        self.joint = joint

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        joint = kwargs.get('joint', self.joint)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint, dim=dim)
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False)
        llw = weights.reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        joint = kwargs.get('joint', self.joint)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint, dim=dim)
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=False)
        llw = weights.reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient
        hess : ([B], K, *spatial) tensor
            Diagonal Hessian

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        joint = kwargs.get('joint', self.joint)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint, dim=dim)
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=True)
        llw = weights.reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g, h


class Tukey(OptimizationLoss):
    """Tukey's biweight loss (using IRLS)"""

    def __init__(self, lam=None, c=4.685, joint=False, dim=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim
        self.joint = joint
        self.c = c

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        joint = kwargs.get('joint', self.joint)
        c = kwargs.get('c', self.c)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint, dim=dim)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False)
        llw = weights.reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        joint = kwargs.get('joint', self.joint)
        c = kwargs.get('c', self.c)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint, dim=dim)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=False)
        llw = weights.reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient
        hess : ([B], K, *spatial) tensor
            Diagonal Hessian

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        joint = kwargs.get('joint', self.joint)
        c = kwargs.get('c', self.c)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint, dim=dim)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=True)
        llw = weights.reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g, h


class MSEHist(HistBasedOptimizationLoss):
    """Mean-squared error"""

    def __init__(self, lam=1, dim=None, bins=None, order=3):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__(dim, bins, order)
        self.lam = lam
        self.minmax = None

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        return mse_hist(moving, fixed, dim=dim, grad=False, hess=False)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim) # lam=lam,
        return mse_hist(moving, fixed, dim=dim, hess=False)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient
        hess : ([B], K, *spatial) tensor
            Diagonal Hessian

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim) # lam=lam,
        if self.minmax is None:
            ll, g, h, *minmax = mse_hist(moving, fixed, dim=dim, minmax=True)
            self.minmax = minmax
            return ll, g, h
        else:
            return mse_hist(moving, fixed, dim=dim, minmax=self.minmax)


class Cat(OptimizationLoss):
    """Categorical cross-entropy"""

    def __init__(self, acceleration=0, dim=None):
        """

        Parameters
        ----------
        acceleration : (0..1) float
            Acceleration
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.acceleration = acceleration
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
        acceleration = kwargs.get('acceleration', self.acceleration)
        dim = kwargs.get('dim', self.dim)
        return cat(moving, fixed, acceleration=acceleration, dim=dim,
                   grad=False, hess=False)

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
        acceleration = kwargs.get('acceleration', self.acceleration)
        dim = kwargs.get('dim', self.dim)
        return cat(moving, fixed, acceleration=acceleration, dim=dim,
                   hess=False)

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
        acceleration = kwargs.get('acceleration', self.acceleration)
        dim = kwargs.get('dim', self.dim)
        return cat(moving, fixed, acceleration=acceleration, dim=dim)


class NCC(OptimizationLoss):
    """Normalized cross-correlation"""

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
        dim = kwargs.get('dim', self.dim)
        return ncc(moving, fixed, dim=dim, grad=False, hess=False)

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
        dim = kwargs.get('dim', self.dim)
        return ncc(moving, fixed, dim=dim, hess=False)

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
        dim = kwargs.get('dim', self.dim)
        return ncc(moving, fixed, dim=dim)


class NMI(HistBasedOptimizationLoss):
    """Normalized cross-correlation"""

    def __init__(self, dim=None, bins=None, order=3, fwhm=2, norm='studholme'):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__(dim, bins, order, fwhm=fwhm)
        self.norm = norm
        self.minmax = False

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        order = kwargs.get('order', self.order)
        norm = kwargs.get('norm', self.norm)
        fwhm = kwargs.get('fwhm', self.fwhm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, *minmax = nmi(moving, fixed, grad=False, hess=False,
                              norm=norm, bins=bins, dim=dim, fwhm=fwhm, minmax=True)
            self.minmax = minmax
            return ll
        else:
            return nmi(moving, fixed, grad=False, hess=False, norm=norm,
                       bins=bins, order=order, dim=dim, fwhm=fwhm, minmax=self.minmax)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        order = kwargs.get('order', self.order)
        norm = kwargs.get('norm', self.norm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, *minmax = nmi(moving, fixed, hess=False, norm=norm,
                                    bins=bins, dim=dim, minmax=True)
            self.minmax = minmax
            return ll, g
        else:
            return nmi(moving, fixed, hess=False, norm=norm,
                       bins=bins, order=order, dim=dim, minmax=self.minmax)

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
        dim = kwargs.get('dim', self.dim)
        bins = kwargs.get('bins', self.bins)
        order = kwargs.get('order', self.order)
        norm = kwargs.get('norm', self.norm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, h, *minmax = nmi(moving, fixed, norm=norm,
                                    bins=bins, dim=dim, minmax=True)
            self.minmax = minmax
            return ll, g, h
        else:
            return nmi(moving, fixed, norm=norm,
                       bins=bins, order=order, dim=dim, minmax=self.minmax)


# WORK IN PROGRESS
# def nmi(fixed, moving, dim=None, bins=64, order=3, limits=None,
#         normalized='studholme'):
#     fixed, moving = utils.to_max_backend(fixed, moving)
#     dim = dim or (fixed.dim() - 1)
#
#     # compute histograms limits
#     if not isinstance(limits, dict):
#         limits = dict(fixed=limits, moving=limits)
#     limits['fixed'] = py.make_list(limits['fixed'], 2)
#     limits['moving'] = py.make_list(limits['moving'], 2)
#     if limits['fixed'][0] is None:
#         limits['fixed'][0] = math.min(fixed, dim=range(-dim, 0), keepdim=True)
#     if limits['fixed'][1] is None:
#         limits['fixed'][1] = math.max(fixed, dim=range(-dim, 0), keepdim=True)
#     if limits['moving'][0] is None:
#         limits['moving'][0] = math.min(moving, dim=range(-dim, 0), keepdim=True)
#     if limits['moving'][1] is None:
#         limits['moving'][1] = math.max(moving, dim=range(-dim, 0), keepdim=True)
#
#     def pnorm(x, dims=-1):
#         """Normalize a tensor so that it's sum across `dims` is one."""
#         dims = py.make_list(dims)
#         x = x.clamp_min_(constants.eps(x.dtype))
#         s = math.sum(x, dim=dims, keepdim=True)
#         return x/s, s
#
#     vmin = (limits['fixed'][0], limits['moving'][0])
#     vmax = (limits['fixed'][1], limits['moving'][1])
#     pxy = utils.histc2(
#         torch.stack([fixed, moving], -1), bins, vmin, vmax,
#         dim=range(-dim-1, -1), order=order, bound='zero')
#
#     # compute probabilities
#     px, sx = pnorm(pxy.sum(dim=-2))  # -> [B, C, nb_bins]
#     py, sy = pnorm(pxy.sum(dim=-1))  # -> [B, C, nb_bins]
#     pxy, sxy = pnorm(pxy, [-1, -2])
#
#     # compute entropies
#     hx = -(px * px.log()).sum(dim=-1)  # -> [B, C]
#     hy = -(py * py.log()).sum(dim=-1)  # -> [B, C]
#     hxy = -(pxy * pxy.log()).sum(dim=[-1, -2])  # -> [B, C]
#
#     # mutual information
#     mi = (hx + hy) - hxy
#     if normalized == 'studholme':
#         mi /= hxy
#     elif normalized == 'arithmetic':
#         mi /= (hx + hy)
#
#     # gradient
#     gxy = pxy.log()
#     gy = py.log()
#     if normalized == 'studholme':
#         gxy = (gxy + (1 + pxy.log()) * mi) / hxy
#         gy /= hxy
#     elif normalized == 'arithmetic':
#         gy = (gy + (1 + py.log()) * mi) / (hx + hy)
#         gxy /= (hx + hy)
#     gy *= 1/sy - py.square()
#     gxy *= 1/sxy - pxy.square()
#     gxy += gy
#     gxy = gxy.sum(dim=-2)
#     gxy = spatial.grid_pull()
#
#     # take negative
#     mi = 1 - mi
#     g = g.neg_()
#     return mi, g, h
