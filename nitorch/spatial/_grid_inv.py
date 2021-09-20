# grid_inv needs its own file to avoid circular imports
import torch
from nitorch.core import utils
from ._grid import grid_push, grid_count, identity_grid
from ._solvers import solve_grid_fmg


def grid_inv(grid, type='grid', bound='dft',
             extrapolate=True, **prm):
    """Invert a dense deformation (or displacement) grid

    Notes
    -----
    The deformation/displacement grid must be expressed in
    voxels, and map from/to the same lattice.

    Let `f = id + d` be the transformation. The inverse
    is obtained as `id - (f.T @ 1 + L) \ (f.T @ d)`
    where `L` is a regulariser, `f.T @ _` is the adjoint
    operation ("push") of `f @ _` ("pull"). and `1` is an
    image of ones.

    The idea behind this is that `f.T @ _` is approximately
    the inverse transformation weighted by the determinant
    of the Jacobian of the tranformation so, in the (theoretical)
    continuous case, `inv(f) @ _ = f.T @ _ / f.T @ 1`.
    However, in the (real) discrete case, this leads to
    lots of holes in the inverse. The solution we use
    therefore fills these holes using a regularised
    least-squares scheme, where the regulariser penalizes
    the spatial gradients of the inverse field.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    membrane : float, default=0.1
        Regularisation
    bound : str, default='dft'
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate the transformation field when
        it is sampled out-of-bounds.

    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid

    """
    prm = prm or dict(membrane=0.1)

    # get shape components
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    shape = grid.shape[-(dim + 1):-1]
    batch = grid.shape[:-(dim + 1)]
    grid = grid.reshape([-1, *shape, dim])
    backend = dict(dtype=grid.dtype, device=grid.device)

    # get displacement
    identity = identity_grid(shape, **backend)
    if type == 'grid':
        disp = grid - identity
    else:
        disp = grid
        grid = disp + identity

    # push displacement
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = utils.movedim(disp, -1, 1)
    disp = grid_push(disp, grid, **push_opt)
    count = grid_count(grid, **push_opt)
    disp = utils.movedim(disp, 1, -1)
    count = utils.movedim(count, 1, -1)

    # Fill missing values using regularised least squares
    disp = solve_grid_fmg(count, disp, bound=bound, **prm)
    disp = disp.reshape([*batch, *shape, dim])

    if type == 'grid':
        return identity - disp
    else:
        return -disp