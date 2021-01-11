"""Integrate/Shoot velocity fields."""

import torch
from ._grid import grid_pull, identity_grid
from nitorch.core.utils import channel2last, last2channel

__all__ = ['exp']


def exp(vel, inverse=False, steps=None, interpolation='linear', bound='dft',
        displacement=False, energy=None, vs=None, greens=None, inplace=False):
    # Deal with inplace computation
    if inplace is None:
        inplace = not vel.requires_grad
    if not inplace and not vel.requires_grad:
        vel = vel.clone()
    elif inplace and vel.requires_grad:
        Warning('Inplace computation may break the computational graph.')

    if energy is None and greens is None:
        # If no energy or greens function: use scaling and squaring
        return _exp_sq(vel, inverse, steps, interpolation, bound,
                       displacement)
    else:
        # If energy or greens function provided: use shoot
        raise NotImplementedError


def _exp_sq(vel, inverse=False, steps=8,
            interpolation='linear', bound='dft', displacement=False):
    # /!\ This function may process inplace without warning

    if steps is None or steps == float('Inf'):
        steps = 8

    # Precompute identity + aliases
    dtype = vel.dtype
    device = vel.device
    id = identity_grid(vel.shape[1:-1], dtype=dtype, device=device)
    opt = {'interpolation': interpolation, 'bound': bound}

    def scl_outplace(v):
        v = v / (2**steps)
        for i in range(steps):
            g = id + v
            v = v + _pull_vel(v, g, **opt)
        return v

    def scl_inplace(v):
        v /= (2**steps)
        for i in range(steps):
            v += _pull_vel(v, id+v, **opt)
        return v

    if vel.requires_grad:
        scl = scl_outplace
        smalldef = lambda v: id + v
        if inverse:
            vel = -vel
    else:
        scl = scl_inplace
        smalldef = lambda v: v.__iadd__(id)
        if inverse:
            torch.neg(vel, out=vel)

    vel = scl(vel)
    if not displacement:
        vel = smalldef(vel)
    return vel


def _pull_vel(vel, grid, *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Parameters
    ----------
    vel : (batch, ..., ndim) tensor
        Velocity
    grid : (batch, ..., ndim) tensor
        Transformation field
    opt : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : (batch, ..., ndim) tensor
        Velocity

    """
    return channel2last(grid_pull(last2channel(vel), grid, *args, **kwargs))

