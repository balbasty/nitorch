from nitorch import spatial
from nitorch._C import grid as _spatial
from nitorch.core import py
import torch


class JointHist:
    """
    Joint histogram with a backward pass for optimization-based registration.
    """

    def __init__(self, n=64, order=3, fwhm=2, bound='replicate', extrapolate=False):
        """

        Parameters
        ----------
        n : int, default=64
            Number of bins
        order : int, default=3
            B-spline order
        bound : {'zero', 'replicate'}
            How to deal with out-of-bound values
        extrapolate : bool, default=False

        """
        self.n = py.make_list(n, 2)
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate
        self.fwhm = fwhm

    def _prepare(self, x, min, max):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
        min : (..., 2) tensor
        max : (..., 2) tensor

        Returns
        -------
        x : (batch, N, 2) tensor
            Reshaped and index-converted input

        """
        if min is None:
            min = x.detach().min(dim=-2).values
        min = min.unsqueeze(-2)
        if max is None:
            max = x.detach().max(dim=-2).values
        max = max.unsqueeze(-2)

        # apply affine function to transform intensities into indices
        x = x.clone()
        nn = torch.as_tensor(self.n, dtype=x.dtype, device=x.device)
        x = x.mul_(nn / (max - min)).add_(nn / (1 - max / min)).sub_(0.5)

        # reshape as (B, N, 2)
        x = x.reshape([-1, *x.shape[-2:]])
        return x, min.squeeze(-2), max.squeeze(-2)

    def forward(self, x, min=None, max=None, mask=None):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
            Input multivariate vector
        min : (..., 2) tensor, optional
        max : (..., 2) tensor, optional
        mask : (..., N) tensor, optional

        Returns
        -------
        h : (..., B, B) tensor
            Joint histogram

        """
        shape = x.shape
        x, min, max = self._prepare(x, min, max)

        # push data into the histogram
        #   hidden feature: tell pullpush to use +/- 0.5 tolerance when
        #   deciding if a coordinate is inbounds.
        extrapolate = self.extrapolate or 2
        if mask is None:
            h = spatial.grid_count(x[:, None], self.n, self.order,
                                   self.bound, extrapolate)[:, 0]
        else:
            mask = mask.to(x.device, x.dtype)
            h = spatial.grid_push(mask, x[:, None], self.n, self.order,
                                  self.bound, extrapolate)[:, 0]
        h = h.to(x.dtype)
        h = h.reshape([*shape[:-2], *h.shape[-2:]])

        if self.fwhm:
            h = spatial.smooth(h, fwhm=self.fwhm, bound=self.bound, dim=2)

        return h, min, max

    def backward(self, x, g, min=None, max=None, hess=False, mask=None):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
            Input multidimensional vector
        g : (..., B, B) tensor
            Gradient with respect to the histogram
        min : (..., 2) tensor, optional
        max : (..., 2) tensor, optional

        Returns
        -------
        g : (..., N, 2) tensor
            Gradient with respect to x

        """
        if self.fwhm:
            g = spatial.smooth(g, fwhm=self.fwhm, bound=self.bound, dim=2)

        shape = x.shape
        x, min, max = self._prepare(x, min, max)
        nvox = x.shape[-2]
        min = min.unsqueeze(-2)
        max = max.unsqueeze(-2)
        g = g.reshape([-1, *g.shape[-2:]])

        extrapolate = self.extrapolate or 2
        if not hess:
            g = spatial.grid_grad(g[:, None], x[:, None], self.order,
                                  self.bound, extrapolate)
            g = g[:, 0].reshape(shape)
        else:
            # 1) Absolute value of adjoint of gradient
            # we want shapes
            #   o : [batch=1, channel=1, spatial=[1, vox], dim=2]
            #   g : [batch=1, channel=1, spatial=[B(mov), B(fix)]]
            #   x : [batch=1, spatial=[1, vox], dim=2]
            #    -> [batch=1, channel=1, spatial=[B(mov), B(fix)]]
            order = _spatial.inter_to_nitorch([self.order], True)
            bound = _spatial.bound_to_nitorch([self.bound], True)
            o = torch.ones_like(x)
            g.requires_grad_()  # triggers push
            o, = _spatial.grid_grad_backward(o[:, None, None], g[:, None], x[:, None],
                                            bound, order, extrapolate)
            g.requires_grad_(False)
            g *= o[:, 0]
            # 2) Absolute value of gradient
            #   g : [batch=1, channel=1, spatial=[B(mov), B(fix)]]
            #   x : [batch=1, spatial=[1, vox], dim=2]
            #    -> [batch=1, channel=1, spatial=[1, vox], 2]
            g = _spatial.grid_grad(g[:, None], x[:, None],
                                   bound, order, extrapolate)
            g = g.reshape(shape)

        # adjoint of affine function
        nn = torch.as_tensor(self.n, dtype=x.dtype, device=x.device)
        factor = nn / (max - min)
        if hess:
            factor = factor.square_()
        g = g.mul_(factor)
        if mask is not None:
            g *= mask[..., None]

        return g


class TestJointHist(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return JointHist().forward(x)[0]

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return JointHist().backward(x, g)

