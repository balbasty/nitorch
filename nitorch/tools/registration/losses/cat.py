from nitorch.core import utils, py, math
from .base import OptimizationLoss, AutoGradLoss


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
    moving = moving.clamp_min(1e-5)
    moving = moving / moving.sum(-dim-1, keepdim=True).add_(1e-5).clamp_min_(1-1e-5)
    fixed = fixed.clamp_min(0)
    fixed = fixed / fixed.sum(-dim-1, keepdim=True).clamp_min_(1)
    logmoving = moving.logit()
    last_fixed = fixed.sum(-dim-1, keepdim=True).neg_().add_(1)
    last_moving = moving.sum(-dim-1, keepdim=True).neg_().add_(1)
    ll = (logmoving*fixed).sum(-dim-1, keepdim=True)
    ll += last_fixed * last_moving.logit()
    if mask is not None:
        ll = ll.mul_(mask)
    ll = ll.sum().neg() / nvox
    out = [ll]

    # gradient
    if grad:
        g = last_fixed/last_moving - fixed/moving
        g = g.div_(nvox)
        if mask is not None:
            g = g.mul_(mask)
        out.append(g)

    # hessian
    if hess:
        h = last_fixed/last_moving.square()
        hshape = list(h.shape)
        hshape[-dim-1] = nc*(nc+1)//2
        h = h.expand(hshape).clone()
        diag = utils.slice_tensor(h, range(nc), -dim-1)
        diag += fixed/moving.square()
        h = h.div_(nvox)
        if mask is not None:
            h = h.mul_(mask)
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


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


class AutoCat(AutoGradLoss):

    def __init__(self, log=False, implicit=True, weighted=False):
        from nitorch.nn.losses import CategoricalLoss
        cat = CategoricalLoss(logit=log, implicit=implicit, weighted=weighted)
        super().__init__(cat)
