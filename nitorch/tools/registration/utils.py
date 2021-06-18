from nitorch.core import py, utils, linalg
from nitorch import spatial
import torch


def defaults_velocity(prm=None):
    if prm is None:
        prm = dict()
    # values from SPM shoot
    prm.setdefault('absolute', 1e-4)
    prm.setdefault('membrane', 1e-3)
    prm.setdefault('bending', 0.2)
    prm.setdefault('lame', (0.05, 0.2))
    prm.setdefault('voxel_size', 1.)
    return prm


def defaults_template(prm=None):
    if prm is None:
        prm = dict()
    # values from SPM shoot
    prm.setdefault('absolute', 1e-4)
    prm.setdefault('membrane', 0.08)
    prm.setdefault('bending', 0.8)
    prm.setdefault('voxel_size', 1.)
    return prm


def jg(jac, grad, dim=None):
    """Jacobian-gradient product: J*g

    Parameters
    ----------
    jac : (..., K, *spatial, D)
    grad : (..., K, *spatial)

    Returns
    -------
    new_grad : (..., *spatial, D)

    """
    dim = dim or (grad.dim() - 1)
    grad = utils.movedim(grad, -dim-1, -1)
    jac = utils.movedim(jac, -dim-2, -1)
    grad = linalg.matvec(jac, grad)
    return grad


def jhj(jac, hess, dim=0):
    """Jacobian-Hessian product: J*H*J', where H is symmetric and stored sparse

    The Hessian can be symmetric (K*(K+1)//2), diagonal (K) or
    a scaled identity (1).

    Parameters
    ----------
    jac : (..., K, *spatial, D)
    hess : (..., 1|K|K*(K+1)//2, *spatial)

    Returns
    -------
    new_hess : (..., *spatial, D*(D+1)//2)

    """

    dim = dim or (hess.dim() - 1)
    hess = utils.movedim(hess, -dim-1, -1)
    jac = utils.movedim(jac, -dim-2, -1)

    @torch.jit.script
    def _jhj(jac, hess):
        # type: (Tensor, Tensor) -> Tensor

        K = jac.shape[-1]
        D = jac.shape[-2]
        D2 = D*(D+1)//2

        if hess.shape[-1] == 1:
            hess = hess.expand(list(hess.shape[:-1]) + [K])
        is_diag = hess.shape[-1] == K
        out = hess.new_zeros(list(jac.shape[:-2]) + [D2])

        dacc = 0
        for d in range(D):
            doffset = (d+1)*D - dacc  # offdiagonal offset
            dacc += d+1
            # diagonal of output
            hacc = 0
            for k in range(K):
                hoffset = (k+1)*K - hacc
                hacc += k+1
                out[..., d] += hess[..., k] * jac[..., d, k].square()
                if not is_diag:
                    for i, l in enumerate(range(k+1, K)):
                        out[..., d] += 2 * hess[..., i+hoffset] * jac[..., d, k] * jac[..., d, l]
            # off diagonal of output
            for j, e in enumerate(range(d+1, D)):
                hacc = 0
                for k in range(K):
                    hoffset = (k+1)*K - hacc
                    hacc += k+1
                    out[..., j+doffset] += hess[..., k] * jac[..., d, k] * jac[..., e, k]
                    if not is_diag:
                        for i, l in enumerate(range(k+1, K)):
                            out[..., j+doffset] += hess[..., i+hoffset] * (jac[..., d, k] * jac[..., e, l] + jac[..., d, l] * jac[..., e, k])
        return out

    return _jhj(jac, hess)


class JointHist:
    """
    Joint histogram with a backward pass for optimization-based registration.
    """

    def __init__(self, n=64, order=3, bound='replicate', extrapolate=False):
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
        self.n = n
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate

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

    def forward(self, x, min=None, max=None):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
            Input multivariate vector
        min : (..., 2) tensor, optional
        max : (..., 2) tensor, optional

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
        h = spatial.grid_count(x[:, None], [self.n, self.n], self.order,
                               self.bound, extrapolate)[:, 0]
        h = h.to(x.dtype)
        h = h.reshape([*shape[:-2], *h.shape[-2:]])

        return h, min, max

    def backward(self, x, g, min=None, max=None, hess=False):
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
        shape = x.shape
        x, min, max = self._prepare(x, min, max)
        min = min.unsqueeze(-2)
        max = max.unsqueeze(-2)
        g = g.reshape([-1, *g.shape[-2:]])

        extrapolate = self.extrapolate or 2
        g = spatial.grid_grad(g[:, None], x[:, None], self.order,
                              self.bound, extrapolate)
        g = g[:, 0].reshape(shape)

        # adjoint of affine function
        nn = torch.as_tensor(self.n, dtype=x.dtype, device=x.device)
        factor = nn / (max - min)
        if hess:
            factor = factor.square_()
        g = g.mul_(factor)

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

