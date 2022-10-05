from nitorch.core import utils, py
import torch
from typing import Optional
from .base import OptimizationLoss, AutoGradLoss
Tensor = torch.Tensor


def dice_nolog(moving, fixed, dim=None, grad=True, hess=True, mask=None,
               add_background=False, weighted=False):
    """Dice loss for optimisation-based registration.

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
    nc = moving.shape[-dim-1]                               # nb classes - bck
    fixed = utils.slice_tensor(fixed, slice(nc), -dim-1)    # remove bkg class
    if mask is not None:
        mask = mask.to(moving.device)
        nvox = mask.sum(list(range(-dim-1)), keepdim=True)
    else:
        nvox = py.prod(fixed.shape[-dim:])

    @torch.jit.script
    def rescale(x, dim_channel: int, add_background: bool = False):
        """Ensure that a tensor is in [0, 1]"""
        x = x.clamp_min(0)
        x = x / x.sum(dim_channel, keepdim=True).clamp_min_(1)
        if add_background:
            x = torch.stack([x, 1 - x.sum(dim_channel, keepdim=True)], dim_channel)
        return x

    moving = rescale(moving, -dim-1, add_background)
    fixed = rescale(fixed, -dim-1, add_background)
    if mask is not None:
        moving *= mask
        fixed *= mask

    if weighted is True:
            weighted = fixed.sum(list(range(-dim, 0)), keepdim=True).div_(nvox)
    elif weighted is not False:
        weighted = torch.as_tensor(weighted, **utils.backend(moving))
        for _ in range(dim):
            weighted = weighted.unsqueeze(-1)
    else:
        weighted = None

    @torch.jit.script
    def loss_components(moving, fixed, dim: int, weighted: Optional[Tensor] = None):
        """Compute the (negative) DiceLoss, (positive) Dice and union"""
        dims = [d for d in range(-dim, 0)]
        overlap = (moving * fixed).sum(dims, keepdim=True)
        union = (moving + fixed).sum(dims, keepdim=True)
        union += 1e-5
        dice = 2 * overlap / union
        if weighted is not None:
            ll = 1 - weighted * dice
        else:
            ll = 1 - dice
        ll = ll.sum()
        return ll, dice, union

    ll, dice, union = loss_components(moving, fixed, dim, weighted)
    out = [ll]

    # gradient
    if grad:
        @torch.jit.script
        def do_grad(dice, fixed, union):
            return (dice - 2 * fixed) / union
        g = do_grad(dice, fixed, union)
        if weighted is not None:
            g *= weighted
        if add_background:
            g_last = utils.slice_tensor(g, slice(-1, None), -dim-1)
            g = utils.slice_tensor(g, slice(-1), -dim-1)
            g -= g_last
        if mask is not None:
            g *= mask
        out.append(g)

    # hessian
    if hess:
        @torch.jit.script
        def do_hess(dice, fixed, union, nvox, dim: int):
            dims = [d for d in range(-dim, 0)]
            positive_rate = fixed.sum(dims, keepdim=True) / nvox
            h = (dice - fixed - positive_rate).abs()
            h = 2 * nvox * h / union.square()
            return h
        nvox = torch.as_tensor(nvox, device=moving.device)
        h = do_hess(dice, fixed, union, nvox, dim)
        if weighted is not None:
            h *= weighted
        if add_background:
            h_foreground = utils.slice_tensor(h, slice(-1), -dim-1)
            h = utils.slice_tensor(h, slice(-1, None), -dim-1)  # h background
            hshape = list(h.shape)
            hshape[-dim-1] = nc*(nc+1)//2
            h = h.expand(hshape).clone()
            diag = utils.slice_tensor(h, range(nc), -dim-1)
            diag += h_foreground
        if mask is not None:
            h *= mask
        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]




class Dice(OptimizationLoss):
    """1 - Dice"""

    order = 2  # Hessian defined

    def __init__(self, add_background=False, weighted=False,
                 log=False, acceleration=0, dim=None):
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
        self.add_background = add_background
        self.weighted = weighted
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
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('add_background', self.add_background)
        kwargs.setdefault('weighted', self.weighted)
        log = kwargs.pop('log', self.log)
        if log:
            kwargs.setdefault('acceleration', self.acceleration)
            raise NotImplementedError
        else:
            return dice_nolog(moving, fixed, **kwargs, grad=False, hess=False,)

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
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('add_background', self.add_background)
        kwargs.setdefault('weighted', self.weighted)
        log = kwargs.pop('log', self.log)
        if log:
            kwargs.setdefault('acceleration', self.acceleration)
            raise NotImplementedError
        else:
            return dice_nolog(moving, fixed, **kwargs, grad=True, hess=False)

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
        kwargs.setdefault('dim', self.dim)
        kwargs.setdefault('add_background', self.add_background)
        kwargs.setdefault('weighted', self.weighted)
        log = kwargs.pop('log', self.log)
        if log:
            kwargs.setdefault('acceleration', self.acceleration)
            raise NotImplementedError
        else:
            return dice_nolog(moving, fixed, **kwargs, grad=True, hess=True)


class AutoDice(AutoGradLoss):

    def __init__(self, log=False, implicit=True, exclude_background=True,
                 weighted=False):
        from nitorch.nn.losses import DiceLoss
        dice = DiceLoss(logit=log, implicit=implicit, weighted=weighted,
                        exclude_background=exclude_background)
        super().__init__(dice)
