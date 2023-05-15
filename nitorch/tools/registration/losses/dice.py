from nitorch.core import utils, py
import torch
from typing import Optional
from .base import OptimizationLoss, AutoGradLoss
Tensor = torch.Tensor


class Dice(OptimizationLoss):
    """1 - Dice"""

    order = 2  # Hessian defined

    def __init__(self, add_background=False, weight=False,
                 fisher=False, dim=None):
        """
        Parameters
        ----------
        add_background : bool, default=False
            Include the Dice of the (implicit) background class in the loss.
        weight : bool or tensor, default=False
            Weights for each class. If True, weight by positive rate.
        fisher : bool or {0..1}
            Whether to use Fisher's scoring of the Hessian (True),
            or a more robust approximation (False).
            If a number between 0 and 1, both versions are weighted and used
            (1: full Fisher, 0: full robust).
        dim : int, default=`fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.add_background = add_background
        self.weight = weight
        self.fisher = fisher
        self.dim = dim

    def set_default(self, kwargs):
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('add_background', self.add_background)
        kwargs.setdefault('weighted', self.weight)
        kwargs.setdefault('fisher', self.fisher)
        return kwargs

    def loss(self, moving, fixed, **kwargs):
        """Compute the [negative] Dice

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image. The background class should be omitted.
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        nll : () tensor
            Negative Dice
        """
        kwargs = self.set_default(kwargs)
        return dice(moving, fixed, **kwargs, grad=False, hess=False)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [negative] Dice

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image. The background class should be omitted.
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        nll : () tensor
            Negative Dice
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image
        """
        kwargs = self.set_default(kwargs)
        return dice(moving, fixed, **kwargs, grad=True, hess=False)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [negative] Dice

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image. The background class should be omitted.
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        nll : () tensor
            Negative Dice
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image
        h : (..., K*(K+1)//2, *spatial) tensor, optional
            Hessian with respect to the moving image.
            Its spatial dimensions are singleton when `acceleration == 0`.
        """
        kwargs = self.set_default(kwargs)
        return dice(moving, fixed, **kwargs, grad=True, hess=True)


@torch.jit.script
def dot(x, y, keepdim: bool = False):
    """Dot product along the last dimension"""
    xy = x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1)
    if not keepdim:
        xy = xy.squeeze(-1)
    return xy


def dice(moving, fixed, dim=None, grad=True, hess=True, mask=None,
         add_background=False, weighted=False, fisher=True):
    """Dice loss for optimisation-based registration.

    Dice is implemented as 2 * dot(f, m) / (dot(f, f) + dot(m, m))

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image of probabilities (post-softmax).
        The background class should be omitted.
    fixed : (..., K, *spatial) tensor
        Fixed image of probabilities.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return Hessian
    mask : (..., *spatial) tensor, optional
        Mask of voxels to include in the loss (all by default)
    add_background : bool, default=False
        Include the Dice of the (implicit) background class in the loss.
    weighted : bool or tensor, default=False
        Weights for each class. If True, weight by positive rate.
    fisher, bool or {0..1}, default=True
        Whether to use Fisher's scoring of the Hessian (True),
        or a more robust approximation (False).
        If a number between 0 and 1, both versions are weighted and used
        (1: full Fisher, 0: full robust).

    Returns
    -------
    ll : () tensor
        Negative log-likelihood
    g : (..., K, *spatial) tensor, optional
        Gradient with respect to the moving image.
    h : (..., K, *spatial) tensor, optional
        Hessian with respect to the moving image.

    """
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    nc, *spatial = moving.shape[-dim-1:]                    # nb classes - bck
    fixed = utils.slice_tensor(fixed, slice(nc), -dim-1)    # remove bkg class
    if mask is not None:
        mask = mask.to(moving.device)
        mask = mask.reshape([*mask.shape[:-dim], -1])
        nvox = mask.sum(-1, keepdim=True)
    else:
        nvox = py.prod(fixed.shape[-dim:])

    # flatten spatial dimensions
    moving = moving.reshape([*moving.shape[:-dim], -1])
    fixed = fixed.reshape([*fixed.shape[:-dim], -1])

    if add_background:
        moving = torch.cat([moving, 1 - moving.sum(-2, keepdim=True)], -2)
        fixed = torch.cat([fixed, 1 - fixed.sum(-2, keepdim=True)], -2)

    if mask is not None:
        moving = moving * mask
        fixed = fixed * mask

    if weighted is True:
        weighted = fixed.sum(-1, keepdim=True).div_(nvox)
    elif weighted is not False:
        weighted = torch.as_tensor(weighted, **utils.backend(moving))
        weighted = weighted.unsqueeze(-1)
    else:
        weighted = None

    @torch.jit.script
    def loss_components(moving, fixed, weighted: Optional[Tensor] = None):
        """Compute the (negative) DiceLoss, (positive) Dice and union"""
        overlap = dot(moving, fixed, True)
        union = dot(moving, moving, True) + dot(fixed, fixed, True)
        union += 1e-5
        dice = 2 * overlap / union
        if weighted is not None:
            ll = 1 - weighted * dice
        else:
            ll = 1 - dice
        ll = ll.sum()
        return ll, dice, union

    ll, dice, union = loss_components(moving, fixed, weighted)
    out = [ll]

    fisher, g0 = float(fisher), None

    # gradient
    if grad or (hess and fisher < 1):
        @torch.jit.script
        def do_grad(moving, fixed, dice, union):
            return 2 * (dice * moving - fixed) / union
        g = do_grad(moving, fixed, dice, union)
        if hess and fisher < 1:
            g0 = g.clone()
        if grad:
            if weighted is not None:
                g *= weighted
            if add_background:
                g_last = utils.slice_tensor(g, slice(-1, None), -2)
                g = utils.slice_tensor(g, slice(-1), -2)
                g -= g_last
            if mask is not None:
                g *= mask
            g = g.reshape([*g.shape[:-1], *spatial])
            out.append(g)

    # hessian
    if hess:
        @torch.jit.script
        def do_hess_fisher(dice, union):
            return 2 * dice / union
        @torch.jit.script
        def do_hess_acc(moving, grad, dice, union, acc: float = 1.):
            return 2 * (dice + 2 * (1 - acc) * moving * grad.abs()) / union
        @torch.jit.script
        def do_hess_diag(moving, grad, dice, union):
            return 2 * (dice + 2 * moving * grad.abs()) / union
        if fisher == 1:
            h = do_hess_fisher(dice, union)
        elif fisher == 0:
            h = do_hess_diag(moving, g0, dice, union)
        else:
            h = do_hess_acc(moving, g0, dice, union, fisher)
        if weighted is not None:
            h *= weighted
        if add_background:
            h_foreground = utils.slice_tensor(h, slice(-1), -2)
            h = utils.slice_tensor(h, slice(-1, None), -2)  # h background
            hshape = list(h.shape)
            hshape[-2] = nc*(nc+1)//2
            h = h.expand(hshape).clone()
            diag = utils.slice_tensor(h, range(nc), -2)
            diag += h_foreground
        if mask is not None:
            h *= mask
        h = h.reshape([*h.shape[:-1], *spatial])
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]



class AutoDice(AutoGradLoss):

    def __init__(self, log=False, implicit=True, exclude_background=True,
                 weighted=False):
        from nitorch.nn.losses import DiceLoss
        dice = DiceLoss(logit=log, implicit=implicit, weighted=weighted,
                        exclude_background=exclude_background)
        super().__init__(dice)
