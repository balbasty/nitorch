from nitorch.core import utils, py, linalg
from .base import OptimizationLoss


def prod(moving, fixed, dim=None, grad=True, hess=True, mask=None):

    dim = dim or (fixed.dim() - 1)
    nvox = py.prod(fixed.shape[-dim:])

    ll = moving * fixed
    if mask is not None:
        ll *= mask
    ll = ll.sum() / nvox
    out = [ll]

    if grad:
        g = fixed / nvox
        if mask is not None:
            g *= mask
        out.append(g)

    if hess:
        if mask is not None:
            h = mask.to(moving.dtype, copy=True).unsqueeze(-dim-1).div_(nvox)
        else:
            h = moving.new_full([1]*(dim+1), 1/nvox)
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def normprod(moving, fixed, dim=None, grad=True, hess=True, mask=None):

    dim = dim or (fixed.dim() - 1)

    m, f = moving, fixed
    del moving, fixed
    if mask is not None:
        f = f * mask
        m = m * mask

    mf = m * f
    # mm = m * m
    sum_m = m.sum().clamp_min_(1e-3)
    # sum_f = f.sum()
    sum_mf = mf.sum()
    # sum_mm = mm.sum()
    # sum_mm = sum_mm.clamp_min_(1e-3)

    # ll = sum_mf / sum_mm
    ll = sum_mf / sum_m
    out = [ll]

    if grad:
        # g = (f - 2 * m * sum_mf / sum_mm) / sum_mm
        g = (f - ll) / sum_m
        if mask is not None:
            g *= mask
        out.append(g)

    if hess:
        # h = (2 * mm * sum_mf / sum_mm + mf) * (4 / sum_mm ** 2)
        # h = (f + sum_mf) * (2 / sum_m ** 2)
        h = (f + ll) / (sum_m * sum_m)
        if mask is not None:
            h *= mask
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def squeezed_prod(moving, fixed, lam=1, dim=None, grad=True, hess=True, mask=None):

    dim = dim or (fixed.dim() - 1)
    nvox = py.prod(fixed.shape[-dim:])

    e = (moving * fixed).mul_(-lam/2).exp_()

    ll = 1 - e
    if mask is not None:
        ll *= mask
    ll = ll.sum() / nvox
    out = [ll]

    if grad:
        g = (e*fixed).mul_(lam / (2*nvox))
        if mask is not None:
            g *= mask
        out.append(g)

    if hess:
        h = (e*fixed).mul_(fixed).mul_((lam / 2)**2 * nvox)
        if mask is not None:
            h *= mask
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


class ProdLoss(OptimizationLoss):

    order = 2  # Hessian defined

    def __init__(self, dim=None):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
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
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        ll = prod(moving, fixed, dim=dim, grad=False, hess=False, mask=mask)
        return ll

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
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        ll, g = prod(moving, fixed, dim=dim, hess=False, mask=mask, **kwargs)
        return ll, g

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
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        ll, g, h = prod(moving, fixed, dim=dim, mask=mask, **kwargs)
        return ll, g, h



class NormProdLoss(OptimizationLoss):

    order = 2  # Hessian defined

    def __init__(self, dim=None):
        """

        Parameters
        ----------
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
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
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        ll = normprod(moving, fixed, dim=dim, grad=False, hess=False, mask=mask)
        return ll

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
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        ll, g = normprod(moving, fixed, dim=dim, hess=False, mask=mask, **kwargs)
        return ll, g

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
        dim = kwargs.pop('dim', self.dim)
        mask = kwargs.pop('mask', None)
        ll, g, h = normprod(moving, fixed, dim=dim, mask=mask, **kwargs)
        return ll, g, h


class SqueezedProdLoss(OptimizationLoss):

    order = 2  # Hessian defined

    def __init__(self, lam=1, dim=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision
        dim : int, default=`fixed.dim() - 1`
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
        ll = squeezed_prod(moving, fixed, dim=dim, lam=lam,
                           grad=False, hess=False, mask=mask)
        return ll

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
        ll, g = squeezed_prod(moving, fixed, dim=dim, lam=lam, mask=mask,
                              hess=False, **kwargs)
        return ll, g

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
        ll, g, h = squeezed_prod(moving, fixed, dim=dim, lam=lam, mask=mask,
                                 **kwargs)
        return ll, g, h