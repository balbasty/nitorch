"""
This file implements losses that are typically used for registration
Each of these functions can return analytical gradients and (approximate)
Hessians to be used in optimization-based algorithms (although the
objective function is differentiable and autograd can be used as well).

These function are implemented in functional form (mse, nmi, cat, ...),
but OO wrappers are also provided for ease of use (MSE, NMI, Cat, ...).

Currently, the following losses are implemented:
- MSE : mean squared error == l2 == Gaussian negative log-likelihood
- MAD : median absolute deviation == l1 == Laplace negative log-likelihood
- Tukey : Tukey's biweight
- Cat : categorical cross entropy
- CC  : correlation coefficient == normalized cross-correlation
- LCC : local correlation coefficient
- NMI : normalized mutual information
"""
from nitorch.core import utils, py, math, constants, linalg
from nitorch.nn.modules.conv import _guess_output_shape
from torch.nn import functional as F
import math as pymath
import torch
from .utils import JointHist
from .local import local_mean as _local_mean
pyutils = py

# TODO:
#   lcc: option to cache values that only depend on fixed?


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
            CC(dim=dim) if loss == 'ncc' else
            NMI(dim=dim) if loss == 'nmi' else
            loss)
    if isinstance(loss, str):
        raise ValueError(f'Unknown loss {loss}')
    return loss


def irls_laplace_reweight(moving, fixed, lam=1, joint=False, eps=1e-5, dim=None,
                          mask=None):
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
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    weights = (moving - fixed).square_().mul_(lam)
    if mask is not None:
        weights = weights.mul_(mask)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    weights = weights.sqrt_().clamp_min_(eps).reciprocal_()
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, 0)
    return weights


def weighted_precision(moving, fixed, weights=None, dim=None):
    dim = dim or fixed.dim() - 1
    residuals = (moving - fixed).square_()
    if weights is not None:
        residuals.mul_(weights)
        lam = residuals.sum(dim=list(range(-dim, 0)))
        lam = lam.div_(weights.sum(dim=list(range(-dim, 0))))
    else:
        lam = residuals.mean(dim=list(range(-dim, 0)))
    lam = lam.reciprocal_()  # variance to precision
    return lam


def irls_tukey_reweight(moving, fixed, lam=1, c=4.685, joint=False, dim=None,
                        mask=None):
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
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    weights = (moving - fixed).square_().mul_(lam)
    if mask is not None:
        weights = weights.mul_(mask)
    if joint:
        weights = weights.sum(dim=-dim-1, keepdims=True)
    zeromsk = weights > c
    weights = weights.div_(-c).add_(1).square()
    weights[zeromsk].zero_()
    return weights


def mse(moving, fixed, lam=1, dim=None, grad=True, hess=True, mask=None):
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
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions
    nvox = py.prod(fixed.shape[-dim:])

    if moving.requires_grad:
        ll = moving - fixed
        if mask is not None:
            ll = ll.mul_(mask)
        ll = ll.square().mul_(lam).sum() / (2*nvox)
    else:
        ll = moving - fixed
        if mask is not None:
            ll = ll.mul_(mask)
        ll = ll.square_().mul_(lam).sum() / (2*nvox)

    out = [ll]
    if grad:
        g = moving - fixed
        if mask is not None:
            g = g.mul_(mask)
        g = g.mul_(lam).div_(nvox)
        out.append(g)
    if hess:
        h = lam/nvox
        if mask is not None:
            h = mask * h
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def cat(moving, fixed, dim=None, acceleration=0, grad=True, hess=True,
        mask=None):
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
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    nc = moving.shape[-dim-1]                               # nb classes - bck
    fixed = utils.slice_tensor(fixed, slice(nc), -dim-1)    # remove bkg class
    nvox = py.prod(fixed.shape[-dim:])

    # log likelihood
    ll = moving*fixed
    ll -= math.logsumexp(moving, dim=-dim-1, implicit=True)  # implicit lse
    if mask is not None:
        ll = ll.mul_(mask)
    ll = ll.sum().neg() / nvox
    out = [ll]

    if grad or (hess and acceleration > 0):
        # implicit softmax
        moving = math.softmax(moving, dim=-dim-1, implicit=True)

    # gradient
    if grad:
        g = (moving - fixed).div_(nvox)
        if mask is not None:
            g = g.mul_(mask)
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

        h = h.div_(nvox)
        if mask is not None:
            h = h.mul_(mask)
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def cat_nolog(moving, fixed, dim=None, grad=True, hess=True, mask=None):
    """Categorical loss for optimisation-based registration.

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image of probabilities (post-softmax).
        The background class should be omitted.
    fixed : (..., K, *spatial) tensor
        Fixed image of probabilities
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
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
    if mask is not None:
        mask = mask.to(fixed.device)
    dim = dim or (fixed.dim() - 1)
    nc = moving.shape[-dim-1]                               # nb classes - bck
    fixed = utils.slice_tensor(fixed, slice(nc), -dim-1)    # remove bkg class
    nvox = py.prod(fixed.shape[-dim:])

    # log likelihood
    logmoving = moving.clamp_min_(1e-5).log_()
    last_fixed = fixed.sum(-dim-1, keepdim=True).neg_().add_(1)
    last_moving = moving.sum(-dim-1, keepdim=True).neg_().add_(1)
    ll = (logmoving*fixed).sum(-dim-1, keepdim=True)
    ll += last_fixed * last_moving.clamp_min_(1e-5).log_()
    if mask is not None:
        ll = ll.mul_(mask)
    ll = ll.sum().neg() / nvox
    out = [ll]

    # gradient
    if grad:
        g = last_fixed/last_moving.clamp_min(1e-5) - fixed/moving.clamp_min(1e-5)
        g = g.div_(nvox)
        if mask is not None:
            g = g.mul_(mask)
        out.append(g)

    # hessian
    if hess:
        h = last_fixed/last_moving.square().clamp_min(1e-5)
        hshape = list(h.shape)
        hshape[-dim-1] = nc*(nc+1)//2
        h = h.expand(hshape).clone()
        diag = utils.slice_tensor(h, range(nc), -dim-1)
        diag += fixed/moving.square().clamp_min(1e-5)
        h = h.div_(nvox)
        if mask is not None:
            h = h.mul_(mask)
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


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

    out = []
    if grad:
        g = 2 * corr * (moving * corr - fixed) / (n * sigm)
        if mask is not None:
            g = g.mul_(mask)
        out.append(g)

    if hess:
        # approximate hessian
        h = 2 * (corr / sigm).square() / n
        if mask is not None:
            h = h * mask
        out.append(h)

    # return stuff
    corr = (1 - corr.square()).sum()
    out = [corr, *out]
    return tuple(out) if len(out) > 1 else out[0]


def _conv(dim):
    return getattr(F, f'conv{dim}d')


def _convt(dim):
    return getattr(F, f'conv_transpose{dim}d')


def local_mean(x, kernel_size=5, stride=1, mode='constant', dim=None,
               backward=False, shape=None, mask=None):
    """Compute a local weighted mean using convolutions.

    Parameters
    ----------
    x : ([batch], channel, *inspatial)
        Input tensor
    kernel_size : [sequence of] int, default=5
    stride : [sequence of] int, default=1
    mode : {'constant', 'gaussian'}, default='constant'
        If 'gaussian', the fwhm is set to kernel_size/3
    dim : int, default=x.dim() - 1
    backward : bool, default=False
        Perform backward pass (transposed conv) instead of forward
    shape : sequence[int], optional
        Only used if 'backward'

    Returns
    -------
    m : ([batch], channel, *outspatial)

    """
    if mask is not None:
        mask = mask.to(x.device, x.dtype)
    dim = dim or x.dim() - 1
    extra_batch = x.dim() > dim + 2
    if extra_batch:
        batch = x.shape[:-dim-1]
        x = x.reshape([-1, *x.shape[-dim-1:]])
    virtual_channel = x.dim() <= dim
    virtual_batch = x.dim() <= dim + 1
    if virtual_channel:
        x = x[None]
    if virtual_batch:
        x = x[None]
    if mask is not None:
        mask = utils.unsqueeze(mask, 0, x.dim() - mask.dim())

    # build kernel
    kernel_size = py.make_list(kernel_size, dim)
    stride = [s or k for s, k in zip(py.make_list(stride, dim), kernel_size)]
    if mode[0].lower() == 'c':
        # constant kernel (classical mean)
        kernel = x.new_ones(kernel_size)
        kernel = kernel[None, None]
    elif mode[0].lower() == 'g':
        # Gaussian kernel (weighted mean)
        fwhm = [k/3 for k in kernel_size]
        sigma2 = [(f/2.355)**2 for f in fwhm]
        kernel = []
        for d in range(dim):
            k = torch.arange(kernel_size[d], **utils.backend(x))
            k -= kernel_size[d]//2
            k = k.square_().div_(-2*sigma2[d]).exp_()
            k = utils.unsqueeze(k, 0, d)
            k = utils.unsqueeze(k, -1, dim-d-1)
            k = k[None, None]
            kernel.append(k)
        norm = py.prod(k).sum()
        for k in kernel:
            k.div_(norm)
    else:
        raise ValueError(f'Unknown mode {mode}')

    # build convolution function and its transpose

    def do_conv(x):
        if isinstance(kernel, list):
            conv = _conv(dim)
            for d, (k, s) in enumerate(zip(kernel, stride)):
                s = [1] * d + [s] + [1] * (dim-d-1)
                x = conv(x, k, stride=s)
        else:
            x = _conv(dim)(x, kernel, stride=stride)
        return x

    if shape is not None:  # estimate output padding
        ishape = [1, 1, *x.shape[-dim:]]
        oshape = _guess_output_shape(ishape, dim, kernel_size,
                                     stride=stride, transposed=True)
        oshape = oshape[2:]
        opad = [s - os for s, os in zip(shape, oshape)]
    else:
        opad = [0] * dim

    def do_convt(x):
        if isinstance(kernel, list):
            conv = _convt(dim)
            for d, (k, s, p) in enumerate(zip(kernel, stride, opad)):
                s = [1] * d + [s] + [1] * (dim - d - 1)
                p = [0] * d + [p] + [0] * (dim - d - 1)
                x = conv(x, k, stride=s, output_padding=p)
        else:
            x = _convt(dim)(x, kernel, stride=stride, output_padding=opad)
        return x

    # conv
    if backward:
        if mask is not None:
            convmask = do_conv(mask).clamp_min_(1e-5)
            x = x / convmask
            del convmask
        x = do_convt(x)
        if mask is not None:
            x = x.mul_(mask)
    else:  # forward pass
        if mask is not None:
            x = x * mask
        x = do_conv(x)
        if mask is not None:
            mask = do_conv(mask).clamp_min_(1e-5)
            x = x.div_(mask)

    if virtual_batch:
        x = x[0]
    if virtual_channel:
        x = x[0]
    if extra_batch:
        x = x.reshape([*batch, *x.shape[1:]])
    return x


def _suffstat(fn, x, y):

    square_ = lambda x: x.square() if x.requires_grad else x.square_()
    mul_ = lambda x, y: x.mul(y) if y.requires_grad else x.mul_(y)

    mom = x.new_empty([5, *x.shape])
    mom[0] = x
    mom[1] = y
    mom[2] = x
    mom[2] = square_(mom[2])
    mom[3] = y
    mom[3] = square_(mom[3])
    mom[4] = x
    mom[4] = mul_(mom[4], y)

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
    fwd = lambda x: _local_mean(x, patch, stride, dim=dim, mode=mode, mask=mask)
    bwd = lambda x: _local_mean(x, patch, stride, dim=dim, mode=mode, mask=mask,
                                backward=True, shape=shape)

    # compute ncc within each patch
    moving_mean, fixed_mean, moving_std, fixed_std, corr = \
        _suffstat(fwd, moving, fixed)
    moving_std = sqrt_(moving_std.addcmul_(moving_mean, moving_mean, value=-1))
    fixed_std = sqrt_(fixed_std.addcmul_(fixed_mean, fixed_mean, value=-1))
    moving_std.clamp_min_(1e-5)
    fixed_std.clamp_min_(1e-5)
    std2 = moving_std * fixed_std
    corr = div_(corr.addcmul_(moving_mean, fixed_mean, value=-1), std2)
    nvox = py.prod(corr.shape[-dim:])

    out = []
    if grad or hess:
        fixed_mean = div_(fixed_mean, fixed_std)
        moving_mean = div_(moving_mean, moving_std)

        h = bwd(square_(corr / moving_std).mul_(lam))

        if grad:
            # g = G' * (corr.*(corr.*xmean./xstd - ymean./ystd)./xstd)
            #   - x .* (G' * (corr./ xstd).^2)
            #   + y .* (G' * (corr ./ (xstd.*ystd)))
            # g = -2 * g
            g = fixed_mean.addcmul_(corr, moving_mean, value=-1)
            g = mul_(g, corr / moving_std).mul_(lam)
            g = bwd(g)
            g = g.addcmul_(h, moving)
            g = g.addcmul_(bwd((corr / std2).mul_(lam)), fixed, value=-1)
            g = g.mul_(2/nvox)
            if mask is not None:
                g = g.mul_(mask)
            out.append(g)

        if hess:
            # h = 2 * (G' * (corr./ xstd).^2)
            h = h.mul_(2/nvox)
            if mask is not None:
                h = h.mul_(mask)
            out.append(h)

    # return stuff
    corr = square_(corr).neg_().add_(1).mul_(lam)
    corr = corr.mean(list(range(-dim, 0))).sum()
    out = [corr, *out]
    return tuple(out) if len(out) > 1 else out[0]


def nmi(moving, fixed, dim=None, bins=64, order=5, fwhm=2, norm='studholme',
        grad=True, hess=True, minmax=False, mask=None):
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
    if mask is not None:
        mask = mask.to(fixed.device)
        mask = mask.reshape([*mask.shape[:-dim], -1])

    if minmax not in (True, False, None):
        mn, mx = minmax
        h, mn, mx = hist.forward(idx, min=mn, max=mx, mask=mask)
    else:
        h, mn, mx = hist.forward(idx, mask=mask)
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
            g = hist.backward(idx, g0/nvox, mask=mask)[..., 0]
            g = g.reshape(shape)
            out.append(g)

        if hess:
            # This Hessian is for Studholme's normalization only!
            # I need to derive one for Arithmetic/None cases

            # True Hessian: not positive definite
            ones = torch.ones([bins, bins])
            g0 = g0.flatten(start_dim=-2)
            pxy = pxy.flatten(start_dim=-2)
            h = (g0[..., :, None] * (1 + pxy.log()[..., None, :]))
            h = h + h.transpose(-1, -2)
            h = h.abs().sum(-1)
            tmp = linalg.kron2(ones, px[..., 0, :].reciprocal().diag_embed())
            # h -= tmp
            tmp += linalg.kron2(py[..., :, 0].reciprocal().diag_embed(), ones)
            # h -= tmp
            h += tmp.abs().sum(-1)
            # h += (nmi/pxy).flatten(start_dim=-2).diag_embed()
            h += (nmi/pxy).flatten(start_dim=-2).abs()
            h /= hxy.flatten(start_dim=-2)
            # h.neg_()
            # h = h.abs().sum(-1)
            # h.diagonal(0, -1, -2).abs()
            # h = h.reshape([*h.shape[:-1], bins, bins])

            # h = (2 - nmi) * py.reciprocal() / hxy
            # h = h.diag_embed()

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
            h = hist.backward(idx, h/nvox, hess=True, mask=mask)[..., 0]
            h = h.reshape(shape)
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

    order = 2  # Hessian defined

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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False,
                  mask=mask)
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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g = mse(moving, fixed, dim=dim, lam=lam, hess=False, **kwargs)
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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, mask=mask, **kwargs)
        return llx + lll, g, h


class MAD(OptimizationLoss):
    """Median absolute deviation (using IRLS)"""

    order = 2  # Hessian defined

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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint,
                                        dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False,
                  **kwargs)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim, weights=mask)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint,
                                        dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=False)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        recompute_lam = lam is None
        if lam is None:
            lam = weighted_precision(moving, fixed, dim=dim)
        weights = irls_laplace_reweight(moving, fixed, lam=lam, joint=joint,
                                        dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if recompute_lam:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=True)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g, h


class Tukey(OptimizationLoss):
    """Tukey's biweight loss (using IRLS)"""

    order = 2  # Hessian defined

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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        c = kwargs.pop('c', self.c)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint,
                                      dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx = mse(moving, fixed, dim=dim, lam=lam, grad=False, hess=False)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        c = kwargs.pop('c', self.c)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint,
                                      dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=False)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
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
        lam = kwargs.pop('lam', self.lam)
        dim = kwargs.pop('dim', self.dim)
        joint = kwargs.pop('joint', self.joint)
        c = kwargs.pop('c', self.c)
        mask = kwargs.pop('mask', None)
        dim = dim or (fixed.dim() - 1)
        nvox = py.prod(fixed.shape[-dim:])
        weights = irls_tukey_reweight(moving, fixed, lam=lam, c=c, joint=joint,
                                      dim=dim, mask=mask)
        if mask is not None:
            weights *= mask
        lll = 0
        if lam is None:
            lam = weighted_precision(moving, fixed, weights, dim=dim)
            lll = -0.5 * lam.log().sum()  # mse: no need to divide by voxels
        lam = lam * weights
        llx, g, h = mse(moving, fixed, dim=dim, lam=lam, grad=True, hess=True)
        llw = weights[weights > 1e-9].reciprocal_().sum().div_(2*nvox)
        return llx + llw + lll, g, h


class Cat(OptimizationLoss):
    """Categorical cross-entropy"""

    order = 2  # Hessian defined

    def __init__(self, log=False, acceleration=0, dim=None):
        """

        Parameters
        ----------
        log : bool, default=False
            Whether the input are logits (pre-softmax) or probits (post-softmax)
        acceleration : (0..1) float
            Acceleration. Only used if `log is True`.
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.log = log
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
        acceleration = kwargs.pop('acceleration', self.acceleration)
        dim = kwargs.pop('dim', self.dim)
        log = kwargs.pop('log', self.log)
        if log:
            return cat(moving, fixed, acceleration=acceleration, dim=dim,
                       grad=False, hess=False, **kwargs)
        else:
            return cat_nolog(moving, fixed, dim=dim,
                             grad=False, hess=False, **kwargs)

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
        acceleration = kwargs.pop('acceleration', self.acceleration)
        dim = kwargs.pop('dim', self.dim)
        log = kwargs.pop('log', self.log)
        if log:
            return cat(moving, fixed, acceleration=acceleration, dim=dim,
                       hess=False, **kwargs)
        else:
            return cat_nolog(moving, fixed, dim=dim,
                             grad=True, hess=False, **kwargs)

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
        acceleration = kwargs.pop('acceleration', self.acceleration)
        dim = kwargs.pop('dim', self.dim)
        log = kwargs.pop('log', self.log)
        if log:
            return cat(moving, fixed, acceleration=acceleration, dim=dim,
                       **kwargs)
        else:
            return cat_nolog(moving, fixed, dim=dim,
                             grad=True, hess=True, **kwargs)


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


class NMI(HistBasedOptimizationLoss):
    """Normalized cross-correlation"""

    order = 1  # Gradient defined

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
        dim = kwargs.pop('dim', self.dim)
        bins = kwargs.pop('bins', self.bins)
        order = kwargs.pop('order', self.order)
        norm = kwargs.pop('norm', self.norm)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, *minmax = nmi(moving, fixed, grad=False, hess=False,
                              norm=norm, bins=bins, dim=dim, fwhm=fwhm,
                              minmax=True, **kwargs)
            self.minmax = minmax
            return ll
        else:
            return nmi(moving, fixed, grad=False, hess=False, norm=norm,
                       bins=bins, order=order, dim=dim, fwhm=fwhm,
                       minmax=self.minmax, **kwargs)

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
        bins = kwargs.pop('bins', self.bins)
        order = kwargs.pop('order', self.order)
        norm = kwargs.pop('norm', self.norm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, *minmax = nmi(moving, fixed, hess=False, norm=norm,
                                    bins=bins, dim=dim, minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g
        else:
            return nmi(moving, fixed, hess=False, norm=norm,
                       bins=bins, order=order, dim=dim,
                       minmax=self.minmax, **kwargs)

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
        bins = kwargs.pop('bins', self.bins)
        order = kwargs.pop('order', self.order)
        norm = kwargs.pop('norm', self.norm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, h, *minmax = nmi(moving, fixed, norm=norm,
                                    bins=bins, dim=dim, minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g, h
        else:
            return nmi(moving, fixed, norm=norm,
                       bins=bins, order=order, dim=dim,
                       minmax=self.minmax, **kwargs)


class AutoGradLoss(OptimizationLoss):
    """Loss class built on an autodiff function"""

    order = 1

    def __init__(self, function, **kwargs):
        super().__init__()
        self.function = function
        self.options = list(kwargs.keys())
        for key, val in kwargs.items():
            setattr(self, key, val)

    def loss(self, moving, fixed, **overload):
        options = {key: getattr(self, key) for key in self.options}
        for key, value in overload.items():
            options[key] = value
        return self.function(moving, fixed, **options)

    def loss_grad(self, moving, fixed, **overload):
        options = {key: getattr(self, key) for key in self.options}
        for key, value in overload.items():
            options[key] = value
        if moving.requires_grad:
            raise ValueError('`moving` already requires gradients')
        moving.requires_grad_()
        if moving.grad is not None:
            moving.grad.zero_()
        with torch.enable_grad():
            loss = self.function(moving, fixed, **options)
            loss.backward()
            grad = moving.grad
        loss = loss.detach()
        moving.requires_grad_(False)
        moving.grad = None
        return loss, grad


class Dice(AutoGradLoss):

    def __init__(self, log=False, implicit=True, exclude_background=True,
                 weighted=False):
        from nitorch.nn.losses import DiceLoss
        dice = DiceLoss(log=log, implicit=implicit, weighted=weighted,
                        exclude_background=exclude_background)
        super().__init__(dice)


class AutoCat(AutoGradLoss):

    def __init__(self, log=False, implicit=True, weighted=False):
        from nitorch.nn.losses import CategoricalLoss
        cat = CategoricalLoss(log=log, implicit=implicit, weighted=weighted)
        super().__init__(cat)
