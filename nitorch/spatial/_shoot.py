"""Integrate/Shoot velocity fields."""

import torch
from ._grid import grid_pull, channel2grid, grid2channel, identity

__all__ = ['exp']


def exp(vel, inverse=False, steps=None, interpolation='linear', bound='dft',
        displacement=False, energy=None, vs=None, greens=None, inplace=None):
    # Deal with inplace computation
    if inplace is None:
        inplace = not vel.requires_grad
    if not inplace and not vel.requires_grad:
        vel = vel.clone()
    elif inplace and vel.requires_grad:
        Warning('Inplace computation may break the computational graph.')

    if energy is None and greens is None:
        # If no energy or greens function: use scaling and squaring
        return _exp_ss(vel, inverse, steps, interpolation, bound,
                       displacement)
    else:
        # If energy or greens function provided: use shoot
        raise NotImplementedError


def _exp_ss(vel, inverse=False, steps=8, interpolation='linear', bound='dft',
            displacement=False):
    # /!\ This function may process inplace without warning

    if steps is None or steps == float('Inf'):
        steps = 8

    # Precompute identity + aliases
    dtype = vel.dtype
    device = vel.device
    id = identity(vel.shape[1:-1], dtype=dtype, device=device)
    c2g = channel2grid
    g2c = grid2channel
    opt = (interpolation, bound)

    def _ss_outplace(v):
        v = v / (2**steps)
        for i in range(steps):
            v = v + c2g(grid_pull(g2c(v), id+v, *opt))
        if not displacement:
            v = id + v
        return v

    def _ss_inplace(v):
        v /= (2**steps)
        for i in range(steps):
            v += c2g(grid_pull(g2c(v), id+v, *opt))
        if not displacement:
            v += id
        return v

    if vel.requires_grad:
        _ss = _ss_outplace
        if inverse:
            vel = -vel
    else:
        _ss = _ss_inplace
        if inverse:
            torch.neg(vel, out=vel)

    return _ss(vel)
