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
from nitorch.core import py
import math as pymath
import torch


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

    def clear_state(self):
        """Clear persistant state"""
        pass

    def get_state(self):
        pass

    def set_state(self, state):
        pass


class HistBasedOptimizationLoss(OptimizationLoss):
    """Base class for histogram-bases losses"""

    def __init__(self, dim=None, bins=None, spline=3, fwhm=2):
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.spline = spline
        self.fwhm = fwhm

    def autobins(self, image, dim):
        dim = dim or (image.dim() - 1)
        shape = image.shape[-dim:]
        nvox = py.prod(shape)
        bins = 2 ** int(pymath.ceil(pymath.log2(nvox ** (1/4))))
        return bins


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


