"""Integrate stationary velocity fields."""

from ._grid import grid_pull, add_identity_grid
from ._finite_differences import diff1d
from nitorch.core import utils, py, linalg
import torch
from nitorch.core.optionals import custom_fwd, custom_bwd


__all__ = ['exp1d', 'exp1d_forward', 'exp1d_backward']


def exp1d(vel, dim=-1, steps=8, interpolation='linear', bound='dft',
          anagrad=False, ndim=None, inplace=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial) tensor
        Stationary velocity field.
    dim : int, default=-1
        Dimension along which the transformation is applied
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    anagrad : bool, default=False
        Use analytical gradients rather than autodiff gradients in
        the backward pass. Should be more memory efficient and (maybe)
        faster.
    ndim : int, default=`vel.dim()`
        Number of spatial dimensions

    Returns
    -------
    grid : ([batch], *spatial) tensor
        Exponentiated tranformation (displacement field)

    """
    exp_fn = _Exp1d.apply if anagrad else exp1d_forward
    ndim = ndim or vel.dim()
    return exp_fn(vel, dim, steps, interpolation, bound, ndim, inplace)


def exp1d_forward(vel, dim=-1, steps=8, interpolation='linear', bound='dft',
                  ndim=None, inplace=False, jacobian=False, _anagrad=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    This function always uses autodiff in the backward pass.
    It can also compute Jacobian fields on the fly.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    dim : int, default=-1
        Dimension along which the transformation is applied
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    ndim : int, optional

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation (displacement field)

    """
    if not inplace:
        vel = vel.clone()

    # Precompute identity + aliases
    ndim = ndim or vel.dim()
    jac = torch.ones_like(vel)
    opt = {'interpolation': interpolation, 'bound': bound,
           'dim': dim, 'ndim': ndim}

    if not _anagrad and vel.requires_grad:
        iadd = lambda x, y: x.add(y)
    else:
        iadd = lambda x, y: x.add_(y)

    vel /= (2**steps)
    for i in range(steps):
        if jacobian:
            jac = _composition_jac(jac, vel, **opt)
        vel = iadd(vel, _pull_vel(vel, **opt))

    return (vel, jac) if jacobian else vel


def exp1d_backward(vel, *grad_and_hess, dim=-1, steps=8,
                   interpolation='linear', bound='dft', ndim=None,
                   rotate_grad=False):
    """Backward pass of SVF exponentiation.

    This should be much more memory-efficient than the autograd pass
    as we don't have to store intermediate grids.

    I am using DARTEL's derivatives (from the code, not the paper).
    From what I get, it corresponds to pushing forward the gradient
    (computed in observation space) recursively while squaring the
    (inverse) transform.
    Remember that the push forward of g by phi is
                    |iphi| iphi' * g(iphi)
    where iphi is the inverse of phi. We could also have implemented
    this operation as: inverse(phi)' * push(g, phi), since
    push(g, phi) \approx |iphi| g(iphi). It has the advantage of using
    push rather than pull, which might preserve better positive-definiteness
    of the Hessian, but requires the inversion of (potentially ill-behaved)
    Jacobian matrices.

    Note that gradients must first be rotated using the Jacobian of
    the exponentiated transform so that the denominator refers to the
    initial velocity (we want dL/dV0, not dL/dPsi).
    THIS IS NOT DONE INSIDE THIS FUNCTION YET (see _dartel).

    Parameters
    ----------
    vel : (..., *spatial) tensor
        Velocity
    grad : (..., *spatial) tensor
        Gradient with respect to the output 1D displacement.
    hess : (..., *spatial) tensor, optional
        Hessian with respect to the output 1D displacement.
    steps : int, default=8
        Number of scaling and squaring steps
    interpolation : str or int, default='linear'
    bound : str, default='dft'
    rotate_grad : bool, default=False
        If True, rotate the gradients using the Jacobian of exp(vel).

    Returns
    -------
    grad : (..., *spatial) tensor
        Gradient with respect to the 1D SVF
    hess : (..., *spatial) tensor, optional
        Approximate (diagonal) Hessian with respect to the 1D SVF

    """
    has_hess = len(grad_and_hess) > 1
    grad, *hess = grad_and_hess
    hess = hess[0] if hess else None
    del grad_and_hess

    ndim = ndim or vel.dim()
    dim = (-ndim + dim) if dim >= 0 else dim
    opt = dict(bound=bound, interpolation=interpolation, dim=dim, ndim=ndim)
    vel = vel.clone()

    if rotate_grad:
        # It forces us to perform a forward exponentiation, which
        # is a bit annoying...
        # Maybe save the Jacobian after the forward pass? But it take space
        _, jac = exp1d_forward(vel, jacobian=True, steps=steps,
                               **opt, _anagrad=True)
        grad = grad.clone().mul_(jac)
        if hess is not None:
            hess = hess.clone().mul_(jac).mul_(jac)
        del jac

    vel /= 2**steps
    jac = diff1d(vel, dim=dim, bound=bound, side='c').add_(1)
    for _ in range(steps):
        grad0 = grad
        grad = _pull_vel(grad, vel, **opt)  # |
        grad.mul_(jac).mul_(jac)            # | push forward
        grad.add_(grad0)                    # add all scales (SVF)
        if hess is not None:
            hess0 = hess
            hess = _pull_vel(hess, vel, **opt)
            hess.mul_(jac).mul_(jac).mul_(jac)
            hess.add_(hess0)
        # squaring
        jac = _composition_jac(jac, vel, **opt)
        vel += _pull_vel(vel, **opt)

    grad /= (2**steps)
    if hess is not None:
        hess /= (2**steps)

    return (grad, hess) if has_hess else grad


class _Exp1d(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # det() only implemented in f32
    def forward(ctx, vel, dim, steps, interpolation, bound, ndim, inplace):
        if vel.requires_grad:
            ctx.save_for_backward(vel.clone() if inplace else vel)
            ctx.args = {'steps': steps, 'dim': dim, 'ndim': ndim,
                        'interpolation': interpolation, 'bound': bound}
        return exp1d_forward(vel, dim, steps, interpolation, bound,
                             ndim, inplace, _anagrad=True)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        vel, = ctx.saved_tensors
        grad = exp1d_backward(vel, grad,
                              dim=ctx.args['dim'],
                              steps=ctx.args['steps'],
                              interpolation=ctx.args['interpolation'],
                              bound=ctx.args['bound'],
                              ndim=ctx.args['ndim'],
                              rotate_grad=True)
        return (grad,) + (None,)*6


def _pull_vel(vel, disp=None, dim=-1, ndim=None, *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial) tensor
        Velocity
    disp : ([batch], *spatial) tensor, default=`vel`
        Displacement field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial) tensor
        Velocity

    """
    if disp is None:
        disp = vel
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    ndim = ndim or disp.dim()
    dim = (dim - ndim) if dim >= 0 else dim
    vel = utils.movedim(vel, dim, -1).unsqueeze(-2)
    disp = utils.movedim(disp, dim, -1).unsqueeze(-1)
    disp = add_identity_grid(disp)
    vel = grid_pull(vel, disp, *args, **kwargs)
    vel = utils.movedim(vel.squeeze(-2), -1, dim)
    return vel


def _composition_jac(jac, rhs, lhs=None, dim=-1, ndim=None, **kwargs):
    """Jacobian of the composition `(lhs)o(rhs)`

    Parameters
    ----------
    jac : ([batch], *spatial) tensor
        Jacobian of input RHS transformation
    rhs : ([batch], *spatial) tensor
        RHS displacement
    lhs : ([batch], *spatial) tensor, default=`rhs`
        LHS small displacement
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    composed_jac : ([batch], *spatial, ndim, ndim) tensor
        Jacobian of composition

    """
    if lhs is None:
        lhs = rhs
    ndim = ndim or jac.dim()
    dim = (dim - ndim) if dim >= 0 else dim
    jac_left = diff1d(lhs, dim=dim, bound=kwargs.pop('bound', 'dft'), side='c')
    jac_left = _pull_vel(jac_left, rhs, dim, ndim).add_(1)
    jac = jac_left.mul_(jac)
    return jac


