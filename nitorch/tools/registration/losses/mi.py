from nitorch.core import py, linalg
import torch
from .utils_hist import JointHist
from .base import HistBasedOptimizationLoss
pyutils = py


def mi(moving, fixed, dim=None, bins=64, order=5, fwhm=2, norm='studholme',
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

class MI(HistBasedOptimizationLoss):
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
            ll, *minmax = mi(moving, fixed, grad=False, hess=False,
                             norm=norm, bins=bins, dim=dim, fwhm=fwhm,
                             minmax=True, **kwargs)
            self.minmax = minmax
            return ll
        else:
            return mi(moving, fixed, grad=False, hess=False, norm=norm,
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
            ll, g, *minmax = mi(moving, fixed, hess=False, norm=norm,
                                bins=bins, dim=dim, minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g
        else:
            return mi(moving, fixed, hess=False, norm=norm,
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
            ll, g, h, *minmax = mi(moving, fixed, norm=norm,
                                   bins=bins, dim=dim, minmax=True, **kwargs)
            self.minmax = minmax
            return ll, g, h
        else:
            return mi(moving, fixed, norm=norm,
                      bins=bins, order=order, dim=dim,
                      minmax=self.minmax, **kwargs)