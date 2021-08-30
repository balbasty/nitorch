"""Utility functions for registration algorithms.
Some (jg, jhj, affine_grid_backward) should maybe be moved to the `spatial`
module (?).
"""

from nitorch.core import py, utils, linalg
from nitorch import spatial
from nitorch._C import grid as _spatial
import torch
from . import optim as optm


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


def loadf(x):
    """Load data from disk if needed"""
    return x.fdata() if hasattr(x, 'fdata') else x


def savef(x, parent):
    """Save data to disk if needed"""
    if hasattr(parent, 'fdata'):
        parent[...] = x
    else:
        parent.copy_(x)


def smart_pull(image, grid, **kwargs):
    """spatial.grid_pull that accepts None grid"""
    if image is None or grid is None:
        return image
    return spatial.grid_pull(image, grid, **kwargs)


def smart_push(image, grid, **kwargs):
    """spatial.grid_push that accepts None grid"""
    if image is None or grid is None:
        return image
    return spatial.grid_push(image, grid, **kwargs)


def smart_exp(vel, **kwargs):
    """spatial.exp that accepts None vel"""
    if vel is not None:
        vel = spatial.exp(vel, **kwargs)
    return vel


def smart_pull_grid(vel, grid, type='disp', *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    if grid is None or vel is None:
        return vel
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = vel.shape[-1]
    if type == 'grid':
        id = spatial.identity_grid(vel.shape[-dim-1:-1],  **utils.backend(vel))
        vel = vel - id
    vel = utils.movedim(vel, -1, -dim-1)
    vel_no_batch = vel.dim() == dim + 1
    grid_no_batch = grid.dim() == dim + 1
    if vel_no_batch:
        vel = vel[None]
    if grid_no_batch:
        grid = grid[None]
    vel = spatial.grid_pull(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch:
        vel = vel[0]
    if type == 'grid':
        id = spatial.identity_grid(vel.shape[-dim-1:-1], **utils.backend(vel))
        vel += id
    return vel


def smart_pull_jac(jac, grid, *args, **kwargs):
    """Interpolate a jacobian field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    jac : ([batch], *spatial_in, ndim, ndim) tensor
        Jacobian field
    grid : ([batch], *spatial_out, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_jac : ([batch], *spatial_out, ndim) tensor
        Jacobian field

    """
    if grid is None or jac is None:
        return jac
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = jac.shape[-1]
    jac = jac.reshape([*jac.shape[:-2], dim*dim])  # collapse matrix
    jac = utils.movedim(jac, -1, -dim - 1)
    jac_no_batch = jac.dim() == dim + 1
    grid_no_batch = grid.dim() == dim + 1
    if jac_no_batch:
        jac = jac[None]
    if grid_no_batch:
        grid = grid[None]
    jac = spatial.grid_pull(jac, grid, *args, **kwargs)
    jac = utils.movedim(jac, -dim - 1, -1)
    jac = jac.reshape([*jac.shape[:-1], dim, dim])
    if jac_no_batch:
        jac = jac[0]
    return jac


def smart_push_grid(vel, grid, *args, **kwargs):
    """Push a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_push:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    if grid is None or vel is None:
        return vel
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = vel.shape[-1]
    vel = utils.movedim(vel, -1, -dim-1)
    vel_no_batch = vel.dim() == dim + 1
    grid_no_batch = grid.dim() == dim + 1
    if vel_no_batch:
        vel = vel[None]
    if grid_no_batch:
        grid = grid[None]
    vel = spatial.grid_push(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch and grid_no_batch:
        vel = vel[0]
    return vel


def make_optim_grid(optim, lr=None, sub_iter=None, kernel=None, **prm):
    """Prepare optimizer for displacement/velocity"""
    correct_keys = ('absolute', 'membrane', 'bending', 'lame',
                    'factor', 'voxel_size')
    prm = {k: prm[k] for k in prm if k in correct_keys}

    optim = (optm.GradientDescent() if optim == 'gd' else
             optm.Momentum() if optim == 'momentum' else
             optm.Nesterov() if optim == 'nesterov' else
             optm.OGM() if optim == 'ogm' else
             optm.GridCG(max_iter=sub_iter, **prm) if optim == 'cg' else
             optm.GridRelax(max_iter=sub_iter, **prm) if optim == 'relax' else
             optm.GridJacobi(max_iter=sub_iter, **prm) if optim == 'jacobi' else
             optm.GridNesterov(max_iter=sub_iter, **prm) if optim.startswith('gnnesterov') else
             optim)
    if lr:
        optim.lr = lr
    if kernel is not None and hasattr(optim, 'preconditioner'):
        optim.preconditioner = lambda x: spatial.greens_apply(x, kernel)
    return optim


def make_optim_field(optim, lr=None, sub_iter=None, kernel=None, **prm):
    """Prepare optimizer for vector field"""
    correct_keys = ('absolute', 'membrane', 'bending', 'factor', 'voxel_size')
    prm = {k: prm[k] for k in prm if k in correct_keys}

    optim = (optm.GradientDescent() if optim == 'gd' else
             optm.Momentum() if optim == 'momentum' else
             optm.Nesterov() if optim == 'nesterov' else
             optm.OGM() if optim == 'ogm' else
             optm.FieldCG(max_iter=sub_iter, **prm) if optim == 'cg' else
             optm.FieldRelax(max_iter=sub_iter, **prm) if optim == 'relax' else
             optim)
    if lr:
        optim.lr = lr
    if kernel is not None and hasattr(optim, 'preconditioner'):
        optim.preconditioner = lambda x: spatial.greens_apply(x, kernel)
    return optim


def make_iteroptim_grid(optim, lr=None, ls=None, max_iter=None, sub_iter=None,
                        kernel=None, **prm):
    """Prepare iterative optimizer for displacement/velocity"""
    if optim == 'lbfgs':
        optim = optm.LBFGS(max_iter=max_iter)
    else:
        optim = make_optim_grid(optim, lr=lr, sub_iter=sub_iter,
                                kernel=kernel, **prm)
    if not hasattr(optim, 'iter'):
        optim = optm.IterateOptim(optim, max_iter=max_iter, ls=ls)
    return optim


def make_optim_affine(optim, lr=None):
    """Prepare optimizer for affine matrices"""

    optim = (optm.GradientDescent() if optim == 'gd' else
             optm.Momentum() if optim == 'momentum' else
             optm.Nesterov() if optim == 'nesterov' else
             optm.OGM() if optim == 'ogm' else
             optm.GaussNewton() if optim == 'gn' else
             optim)
    if lr:
        optim.lr = lr
    return optim


def make_iteroptim_affine(optim, lr=None, ls=None, max_iter=None):
    """Prepare iterative optimizer for displacement/velocity"""
    if optim == 'lbfgs':
        optim = optm.LBFGS(max_iter=max_iter)
    else:
        optim = make_optim_affine(optim, lr=lr)
    if not hasattr(optim, 'iter'):
        optim = optm.IterateOptim(optim, max_iter=max_iter, ls=ls)
    return optim


@torch.jit.script
def _affine_grid_backward_g(grid, grad):
    # type: (Tensor, Tensor) -> Tensor
    dim = grid.shape[-1]
    g = torch.empty([grad.shape[0], dim, dim+1], dtype=grad.dtype, device=grad.device)
    for i in range(dim):
        g[..., i, -1] = grad[..., i].sum(1, dtype=torch.double).to(g.dtype)
        for j in range(dim):
            g[..., i, j] = (grad[..., i] * grid[..., j]).sum(1, dtype=torch.double).to(g.dtype)
    return g


@torch.jit.script
def _affine_grid_backward_gh(grid, grad, hess):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    dim = grid.shape[-1]
    g = torch.zeros([grad.shape[0], dim, dim+1], dtype=grad.dtype, device=grad.device)
    h = torch.zeros([hess.shape[0], dim, dim+1, dim, dim+1], dtype=grad.dtype, device=grad.device)
    basecount = dim - 1
    for i in range(dim):
        basecount = basecount + i * (dim-i)
        for j in range(dim+1):
            if j == dim:
                g[..., i, j] = (grad[..., i]).sum(1)
            else:
                g[..., i, j] = (grad[..., i] * grid[..., j]).sum(1)
            for k in range(dim):
                idx = k
                if k < i:
                    continue
                elif k != i:
                    idx = basecount + (k - i)
                for l in range(dim+1):
                    if l == dim and j == dim:
                        h[..., i, j, k, l] = h[..., k, j, i, l] = hess[..., idx].sum(1)
                    elif l == dim:
                        h[..., i, j, k, l] = h[..., k, j, i, l] = (hess[..., idx] * grid[..., j]).sum(1)
                    elif j == dim:
                        h[..., i, j, k, l] = h[..., k, j, i, l] = (hess[..., idx] * grid[..., l]).sum(1)
                    else:
                        h[..., i, j, k, l] = h[..., k, j, i, l] = (hess[..., idx] * grid[..., j] * grid[..., l]).sum(1)
    return g, h


def affine_grid_backward(*grad_hess, grid=None):
    """Converts ∇ wrt dense displacement into ∇ wrt affine matrix

    g = affine_grid_backward(g, [grid=None])
    g, h = affine_grid_backward(g, h, [grid=None])

    Parameters
    ----------
    grad : (..., *spatial, dim) tensor
        Gradient with respect to a dense displacement.
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Hessian with respect to a dense displacement.
    grid : (*spatial, dim) tensor, optional
        Pre-computed identity grid

    Returns
    -------
    grad : (..., dim, dim+1) tensor
        Gradient with respect to an affine matrix
    hess : (..., dim, dim+1, dim, dim+1) tensor, optional
        Hessian with respect to an affine matrix

    """
    has_hess = len(grad_hess) > 1
    grad, *hess = grad_hess
    hess = hess.pop(0) if hess else None
    del grad_hess

    dim = grad.shape[-1]
    shape = grad.shape[-dim-1:-1]
    batch = grad.shape[:-dim-1]
    nvox = py.prod(shape)
    if grid is None:
        grid = spatial.identity_grid(shape, **utils.backend(grad))
    grid = grid.reshape([1, nvox, dim])
    grad = grad.reshape([-1, nvox, dim])
    if hess is not None:
        hess = hess.reshape([-1, nvox, dim*(dim+1)//2])
        grad, hess = _affine_grid_backward_gh(grid, grad, hess)
        hess = hess.reshape([*batch, dim, dim+1, dim, dim+1])
    else:
        grad = _affine_grid_backward_g(grid, grad)
    grad = grad.reshape([*batch, dim, dim+1])
    return (grad, hess) if has_hess else grad


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
    if grad is None:
        return None
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
    if hess is None:
        return None
    dim = dim or (hess.dim() - 1)
    hess = utils.fast_movedim(hess, -dim-1, -1)
    jac = utils.fast_movedim(jac, -dim-2, -1)

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
        self.n = n
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
            h = spatial.grid_count(x[:, None], [self.n, self.n], self.order,
                                   self.bound, extrapolate, abs=False)[:, 0]
        else:
            mask = mask.to(x.device, x.dtype)
            h = spatial.grid_push(mask, x[:, None], [self.n, self.n], self.order,
                                  self.bound, extrapolate, abs=False)[:, 0]
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
                                            bound, order, extrapolate, True)
            g.requires_grad_(False)
            g *= o[:, 0]
            # 2) Absolute value of gradient
            #   g : [batch=1, channel=1, spatial=[B(mov), B(fix)]]
            #   x : [batch=1, spatial=[1, vox], dim=2]
            #    -> [batch=1, channel=1, spatial=[1, vox], 2]
            g = _spatial.grid_grad(g[:, None], x[:, None],
                                   bound, order, extrapolate, True)
            g = g.reshape(shape)
            g /= nvox*nvox

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

