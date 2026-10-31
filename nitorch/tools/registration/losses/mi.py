from nitorch.core import py, linalg
from nitorch.core._hist import JointHistCount
import torch
from .base import HistBasedOptimizationLoss
pyutils = py


def entropy(moving, fixed, dim=None, bins=64, order=3, fwhm=2,
            grad=True, hess=True, minmax=False, mask=None):
    """(Negative) Entropy

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
    grad : bool, default=True
        Compute an return gradient
    hess : bool, default=True
        Compute and return approximate Hessian

    Returns
    -------
    ll : () tensor
        Negative Entropy
    grad : (..., K, *spatial) tensor
    hess : (..., K, *spatial) tensor

    """

    hist = JointHistCount(bins=bins, order=order, fwhm=fwhm)

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
        pxy, mn, mx = hist.forward(idx, min=mn, max=mx, w=mask, return_minmax=True)
    else:
        pxy, mn, mx = hist.forward(idx, w=mask, return_minmax=True)
    pxy = pxy.clamp(1e-8)
    pxy /= nvox
    hxy = -(pxy * pxy.log()).sum([-1, -2], keepdim=True)

    out = []
    if grad or hess:
        g = pxy.log().add_(1).neg_()
        g = g.div_(nvox)

    if grad:
        gx = hist.backward(g, idx, min=mn, max=mx, w=mask)[..., 0]
        gx = gx.reshape(shape)
        out.append(gx)

    if hess:
        # The joint entropy is concave in the histogram, so the
        # Gauss-Newton term is negative semi-definite: bound it by zero
        # and use the absolute value of the (voxel-diagonal) second term.
        # See `mi`.
        H = hist.backward_hess(g, idx, min=mn, max=mx, w=mask)[..., 0]
        H = H.abs_().reshape(shape)
        out.append(H)

    out = [hxy, *out]
    if minmax is True:
        out.extend([mn, mx])
    return tuple(out) if len(out) > 1 else out[0]


def _degenerate(idx, mask=None):
    """Batch elements without information: empty mask, or constant
    moving or fixed image, i.e. (numerically) zero intensity range (same
    criterion as `core._hist._guard_range`, which widens such ranges).

    idx : (..., N, 2) tensor, mask : (..., N) tensor -> (...) bool tensor
    """
    if mask is not None:
        inside = (mask > 0).unsqueeze(-1)
        mn = idx.masked_fill(~inside, float('inf')).amin(-2)
        mx = idx.masked_fill(~inside, -float('inf')).amax(-2)
        empty = ~inside.any(-2).squeeze(-1)
        mn, mx = mn.masked_fill(empty[..., None], 0), mx.masked_fill(empty[..., None], 0)
    else:
        mn, mx = idx.amin(-2), idx.amax(-2)
        empty = None
    scale = torch.maximum(mn.abs(), mx.abs()).clamp_min(1)
    bad = ((mx - mn) <= 4 * torch.finfo(mx.dtype).eps * scale).any(-1)
    return bad if empty is None else bad | empty


def mi(moving, fixed, dim=None, bins=64, order=3, fwhm=2, norm='studholme',
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
        The Hessian uses the second derivative of the B-splines: with
        order 1 (where it is zero almost everywhere), its average over a
        bin is used instead.
    norm : {'studholme', 'arithmetic', 'geometric', None}, default='studholme'
        Normalization method:
        None : mi = H[x] + H[y] - H[xy]
        'arithmetic' : nmi = mi / (H[x] + H[y])
        'geometric' : nmi = mi / sqrt(H[x] * H[y])
        'studholme' : nmi = mi / H[xy]
    grad : bool, default=True
        Compute an return gradient
    hess : bool, default=True
        Compute and return approximate Hessian

    Returns
    -------
    ll : () tensor
        1 -  NMI (or -MI if `norm` is None)
    grad : (..., K, *spatial) tensor
    hess : (..., K, *spatial) tensor
        Non-negative diagonal preconditioner (see notes in the code)

    """
    if norm not in ('studholme', 'arithmetic', 'geometric', None, 'none', 'no'):
        raise ValueError(f'Unknown normalization: {norm}')
    hist = JointHistCount(bins=bins, order=order, fwhm=fwhm)

    shape = moving.shape
    dim = dim or fixed.dim() - 1
    moving = moving.reshape([*moving.shape[:-dim], -1])
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])
    idx = torch.stack([moving, fixed], -1)
    if mask is not None:
        mask = mask.to(fixed.device)
        mask = mask.reshape([*mask.shape[:-dim], -1])
        nvox = mask.sum(-1, keepdim=True).unsqueeze(-1)  # [*batch, 1, 1, 1]
        nvox.masked_fill_(nvox == 0, 1)  # empty (e.g., slice) mask -> no NaNs
    else:
        nvox = pyutils.prod(shape[-dim:])
    del moving, fixed

    if minmax not in (True, False, None):
        mn, mx = minmax
        h, mn, mx = hist.forward(idx, min=mn, max=mx, w=mask, return_minmax=True)
    else:
        h, mn, mx = hist.forward(idx, w=mask, return_minmax=True)
    # Batch elements (channels, slices) without information: empty mask,
    # or constant moving or fixed image (whose histogram range is then
    # widened artificially by `JointHistCount`). Their mutual information
    # is zero, i.e. the worst possible value, and so are their gradient
    # and Hessian. Otherwise, the histogram of an empty overlap or of a
    # constant image is spread over a few central bins by the spline
    # window, which gives a spuriously *good* value (e.g., when an
    # optimizer pushes the moving image out of the field of view).
    empty = _degenerate(idx, mask)
    empty = empty[..., None, None]

    h = h.clamp(1e-8)
    h /= nvox

    pxy = h
    px = pxy.sum(-1, keepdim=True)
    py = pxy.sum(-2, keepdim=True)

    hxy = -(pxy * pxy.log()).sum([-1, -2], keepdim=True)
    hx = -(px * px.log()).sum([-1, -2], keepdim=True)
    hy = -(py * py.log()).sum([-1, -2], keepdim=True)

    if norm == 'studholme':
        # studholme nmi = mi / hxy = (hx + hy) / hxy - 1
        nmi = (hx + hy) / hxy
    elif norm == 'arithmetic':
        # arithmetic nmi = mi / (hx + hy) = 1 - hxy / (hx + hy)
        nmi = -hxy / (hx + hy)
    elif norm == 'geometric':
        # geometric nmi = mi / sqrt(hx * hy)
        hxhy = (hx * hy).sqrt()
        nmi = (hx + hy - hxy) / hxhy
        # weights of d(hx) and d(hy) in sqrt(hx * hy) * d(nmi)
        ax = 1 - 0.5 * nmi * (hy / hx).sqrt()
        ay = 1 - 0.5 * nmi * (hx / hy).sqrt()
    else:
        # unnormalized mutual info
        nmi = hx + hy - hxy

    out = []
    if grad or hess:
        # gradient of the loss with respect to the histogram
        if norm == 'studholme':
            g = ((1 + px.log()) + (1 + py.log())) - nmi * (1 + pxy.log())
            g /= hxy
        elif norm == 'arithmetic':
            g = -nmi * ((1 + px.log()) + (1 + py.log())) - (1 + pxy.log())
            g /= (hx + hy)
        elif norm == 'geometric':
            g = ax * (1 + px.log()) + ay * (1 + py.log()) - (1 + pxy.log())
            g /= hxhy
        else:
            # g = ((1 + px.log()) + (1 + py.log())) - (1 + pxy.log())
            g = pxy.log().neg_().add_(px.log()).add_(py.log()).add_(1)
        g = g.div_(nvox)
        g = g.masked_fill_(empty, 0)

    if grad:
        gx = hist.backward(g, idx, min=mn, max=mx, w=mask)[..., 0]
        gx = gx.reshape(shape)
        out.append(gx)

    if hess:
        # The Hessian with respect to the moving voxels is
        #   J^T F J + diag_n(sum_ab g_ab d2 p_ab / dx_n^2)
        # Along moving-image directions, every norm is a function of
        # H(Y|X) and Hx with non-negative coefficients, and both are
        # concave in the histogram, so the first (Gauss-Newton) term is
        # negative semi-definite: we bound it by zero. The rank-2 terms
        # of the normalisation are dropped, which amounts to Newton on
        # the Dinkelbach surrogate of the ratio. The second term is
        # voxel-diagonal; we use its absolute value as preconditioner
        # (Balbastre et al., MedIA 2021, eq. 24).
        H = hist.backward_hess(g, idx, min=mn, max=mx, w=mask)[..., 0]
        H = H.abs_().reshape(shape)
        out.append(H)

    # zero mutual information where there is no information
    nmi_zero = {'studholme': 1., 'arithmetic': -1.}.get(norm, 0.)
    nmi = nmi.masked_fill(empty, nmi_zero)

    nmi = -nmi
    if norm == 'studholme':
        nmi = (2+nmi)
    elif norm == 'geometric':
        nmi = (1+nmi)
    nmi = nmi.sum()

    out = [nmi, *out]
    if minmax is True:
        out.extend([mn, mx])
    return tuple(out) if len(out) > 1 else out[0]


class Entropy(HistBasedOptimizationLoss):
    """entropy"""

    order = 2  # Gradient defined

    def __init__(self, dim=None, bins=None, spline=3, fwhm=2):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__(dim, bins, spline, fwhm=fwhm)
        self.minmax = None

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
        bins = self.bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, *minmax = entropy(moving, fixed, grad=False, hess=False,
                                  bins=bins, order=self.spline,
                                  dim=dim, fwhm=self.fwhm,
                                  minmax=True, **kwargs)
            self.minmax = minmax
            return ll
        else:
            return entropy(moving, fixed, grad=False, hess=False,
                           bins=bins, order=self.spline,
                           dim=dim, fwhm=self.fwhm,
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
        bins = self.bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, *minmax = entropy(moving, fixed, grad=True, hess=False,
                                     bins=bins, order=self.spline,
                                     dim=dim, fwhm=self.fwhm,
                                     minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g
        else:
            return entropy(moving, fixed, grad=True, hess=False,
                           bins=bins, order=self.spline,
                           dim=dim, fwhm=self.fwhm,
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
        bins = self.bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, h, *minmax = entropy(moving, fixed, grad=True, hess=True,
                                        bins=bins, order=self.spline,
                                        dim=dim, fwhm=self.fwhm,
                                        minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g, h
        else:
            return entropy(moving, fixed, grad=True, hess=True,
                           bins=bins, order=self.spline,
                           dim=dim, fwhm=self.fwhm,
                           minmax=self.minmax, **kwargs)


class MI(HistBasedOptimizationLoss):
    """Normalized cross-correlation"""

    order = 2  # Gradient defined

    def __init__(self, dim=None, bins=None, spline=3, fwhm=2, norm='studholme'):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__(dim, bins, spline, fwhm=fwhm)
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
        spline = kwargs.pop('spline', self.spline)
        norm = kwargs.pop('norm', self.norm)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, *minmax = mi(moving, fixed, grad=False, hess=False,
                             norm=norm, bins=bins, order=spline, dim=dim, fwhm=fwhm,
                             minmax=True, **kwargs)
            self.minmax = minmax
            return ll
        else:
            return mi(moving, fixed, grad=False, hess=False, norm=norm,
                      bins=bins, order=spline, dim=dim, fwhm=fwhm,
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
        spline = kwargs.pop('spline', self.spline)
        norm = kwargs.pop('norm', self.norm)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, *minmax = mi(moving, fixed, hess=False, norm=norm,
                                bins=bins, order=spline, dim=dim, minmax=True,
                                fwhm=fwhm, **kwargs)
            self.minmax = minmax
            return ll, g
        else:
            return mi(moving, fixed, hess=False, norm=norm,
                      bins=bins, order=spline, dim=dim, fwhm=fwhm,
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
        spline = kwargs.pop('spline', self.spline)
        norm = kwargs.pop('norm', self.norm)
        fwhm = kwargs.pop('fwhm', self.fwhm)
        bins = bins or self.autobins(fixed, dim)
        if self.minmax is None:
            ll, g, h, *minmax = mi(moving, fixed, norm=norm, fwhm=fwhm,
                                   bins=bins, order=spline, dim=dim,
                                   minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g, h
        else:
            return mi(moving, fixed, norm=norm, fwhm=fwhm,
                      bins=bins, order=spline, dim=dim,
                      minmax=self.minmax, **kwargs)