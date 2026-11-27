from nitorch.core import utils, py, linalg
from .base import OptimizationLoss
from .mse import sumweights


def prod(moving, fixed, dim=None, grad=True, hess=True, mask=None):

    dim = dim or (fixed.dim() - 1)
    # mean over the (weighted) voxels of each channel, as in `mse`
    nvox = sumweights(mask, fixed.shape[-dim:])

    ll = moving * fixed
    if mask is not None:
        ll = ll * mask
    ll = (ll / nvox).sum()
    out = [ll]

    if grad:
        g = fixed / nvox
        if mask is not None:
            g *= mask
        out.append(g)

    if hess:
        if mask is not None:
            h = mask.to(moving.dtype) / nvox
        else:
            h = moving.new_full([1]*(dim+1), 1/nvox)
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def normprod(moving, fixed, dim=None, grad=True, hess=True, mask=None):

    dim = dim or (fixed.dim() - 1)
    sumdim = lambda x: x.sum(list(range(-dim, 0)), keepdim=True)

    m, f = moving, fixed
    del moving, fixed
    if mask is not None:
        f = f * mask
        m = m * mask

    mf = m * f
    sum_m = sumdim(m).clamp_min_(1e-3)
    sum_mf = sumdim(mf)
    ll = (sum_mf * sum_mf) / (sum_m * sum_m)
    out = [0.5 * ll.sum()]

    if grad or hess:
        # ll = r^2, with r = sum(m*f) / sum(m) the mean of f over m
        # g = d(r^2/2)/dm = r * (f - r) / sum(m)
        # (written without dividing by sum(m*f), which is zero at a
        # perfect fit)
        r = sum_mf / sum_m
        g = (f - r).mul_(r / sum_m)
        if mask is not None:
            g *= mask
        if grad:
            out.append(g)

        if hess:
            # Upper bound of the Gauss-Newton diagonal (f - r)^2 / sum(m)^2
            h = (f * f + r * r).div_(sum_m * sum_m)
            eps = h.max() * 1e-4
            h.add_(eps)
            if mask is not None:
                h = h * mask
            out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


def squeezed_prod(moving, fixed, lam=1, dim=None, grad=True, hess=True, mask=None):

    dim = dim or (fixed.dim() - 1)
    # mean over the (weighted) voxels of each channel, as in `mse`
    nvox = sumweights(mask, fixed.shape[-dim:])

    e = (moving * fixed).mul_(-lam/2).exp_()

    ll = 1 - e
    if mask is not None:
        ll = ll * mask
    ll = (ll / nvox).sum()
    out = [ll]

    if grad:
        g = (e*fixed).mul_(lam / 2).div_(nvox)
        if mask is not None:
            g *= mask
        out.append(g)

    if hess:
        # Absolute value of the (negative) true Hessian
        h = (e*fixed).mul_(fixed).mul_((lam / 2)**2).div_(nvox)
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