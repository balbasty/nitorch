"""Integrate stationary velocity fields."""

from ._grid import grid_pull, identity_grid
from nitorch.core import utils

__all__ = ['exp']


def exp(vel, inverse=False, steps=8, interpolation='linear', bound='dft',
        displacement=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    inverse : bool, default=False
        Generate the inverse transformation instead of the forward.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    displacement : bool, default=False
        Return a displacement field rather than a transformation field

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation

    """

    vel = -vel if inverse else vel.clone()

    # Precompute identity + aliases
    dim = vel.shape[-1]
    spatial = vel.shape[-1-dim:-1]
    id = identity_grid(spatial, **utils.backend(vel))
    opt = {'interpolation': interpolation, 'bound': bound}

    if vel.requires_grad:
        iadd = lambda x, y: x.add(y)
    else:
        iadd = lambda x, y: x.add_(y)

    vel /= (2**steps)
    for i in range(steps):
        vel = iadd(vel, _pull_vel(vel, id + vel, **opt))

    if not displacement:
        vel += id
    return vel


def _pull_vel(vel, grid, *args, **kwargs):
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
    vel = grid_pull(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch:
        vel = vel[0]
    return vel
