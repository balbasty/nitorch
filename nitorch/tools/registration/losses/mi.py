from nitorch.core import py, linalg
from nitorch.core._hist import JointHistCount
import torch
from .base import HistBasedOptimizationLoss
pyutils = py


def entropy(moving, fixed, dim=None, bins=64, order=1, fwhm=2,
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
        if grad:
            g = pxy.log().add_(1).neg_()
            g = g.div_(nvox)
            g = hist.backward(g, idx, min=mn, max=mx, w=mask)[..., 0]
            g = g.reshape(shape)
            out.append(g)

        if hess:
            H = pxy.reciprocal()
            H = H.div_(nvox*nvox)
            H = hist.backward2(H, idx, w=mask)[..., 0]
            H = H.reshape(shape)
            out.append(H)

    out = [hxy, *out]
    if minmax is True:
        out.extend([mn, mx])
    return tuple(out) if len(out) > 1 else out[0]


def mi(moving, fixed, dim=None, bins=64, order=1, fwhm=2, norm='studholme',
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
    order : int, default=1
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
    hist = JointHistCount(bins=bins, order=order, fwhm=fwhm)

    shape = moving.shape
    dim = dim or fixed.dim() - 1
    moving = moving.reshape([*moving.shape[:-dim], -1])
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])
    idx = torch.stack([moving, fixed], -1)
    if mask is not None:
        mask = mask.to(fixed.device)
        mask = mask.reshape([*mask.shape[:-dim], -1])
        nvox = mask.sum(-1, keepdim=True)  # shape: [*batch, 1, 1]
    else:
        nvox = pyutils.prod(shape[-dim:])
    del moving, fixed

    if minmax not in (True, False, None):
        mn, mx = minmax
        h, mn, mx = hist.forward(idx, min=mn, max=mx, w=mask, return_minmax=True)
    else:
        h, mn, mx = hist.forward(idx, w=mask, return_minmax=True)
    h = h.clamp(1e-8)
    h /= (nvox.unsqueeze(-1) if torch.is_tensor(nvox) else nvox)

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
    else:
        # unnormalized mutual info
        nmi = hx + hy - hxy

    out = []
    if grad:
        if norm == 'studholme':
            g = ((1 + px.log()) + (1 + py.log())) - nmi * (1 + pxy.log())
            g /= hxy
        elif norm == 'arithmetic':
            g = -nmi * ((1 + px.log()) + (1 + py.log())) + (1 + pxy.log())
            g /= (hx + hy)
        else:
            # g = ((1 + px.log()) + (1 + py.log())) - (1 + pxy.log())
            g = pxy.log().neg_().add_(px.log()).add_(py.log()).add_(1)
        g = g.div_(nvox)
        g = hist.backward(g, idx, min=mn, max=mx, w=mask)[..., 0]
        g = g.reshape(shape)
        out.append(g)

    if hess:
        if norm == 'studholme':
            H = (nmi / pxy - 1 / px + 1 / py) / hxy
        elif norm == 'arithmetic':
            H = (1 / pxy + nmi * (1 / px + 1 / py)) / (hx + hy)
        else:
            H = 1 / pxy - 1 / px - 1 / py
        H = H.div_(nvox**2)
        H = hist.backward2(H, idx, w=mask)[..., 0]
        H = H.reshape(shape)
        out.append(H)

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


class Entropy(HistBasedOptimizationLoss):
    """entropy"""

    order = 2  # Gradient defined

    def __init__(self, dim=None, bins=None, spline=1, fwhm=2):
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

    def __init__(self, dim=None, bins=None, spline=1, fwhm=2, norm='studholme'):
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