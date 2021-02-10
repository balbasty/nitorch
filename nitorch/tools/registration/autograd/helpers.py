import torch
from nitorch.spatial import grid_pull, identity_grid, resize_grid
from nitorch.core.utils import movedim

# ----------------------------------------------------------------------
#                           HELPERS
# ----------------------------------------------------------------------


def samespace(aff, inshape, outshape):
    """Check if two spaces are the same

    Parameters
    ----------
    aff : (dim+1, dim+1)
    inshape : tuple of `dim` int
    outshape : tuple of `dim` int

    Returns
    -------
    bool

    """
    eye = torch.eye(4, dtype=aff.dtype, device=aff.device)
    return inshape == outshape and (aff - eye).allclose()


def ffd_exp(prm, shape, order=3, bound='dft', returns='disp'):
    """Transform FFD parameters into a displacement or transformation grid.

    Parameters
    ----------
    prm : (..., *spatial, dim)
        FFD parameters
    shape : sequence[int]
        Exponentiated shape
    order : int, default=3
        Spline order
    bound : str, default='dft'
        Boundary condition
    returns : {'disp', 'grid', 'disp+grid'}, default='grid'
        What to return:
        - 'disp' -> displacement grid
        - 'grid' -> transformation grid

    Returns
    -------
    disp : (..., *shape, dim), optional
        Displacement grid
    grid : (..., *shape, dim), optional
        Transformation grid

    """
    backend = dict(dtype=prm.dtype, device=prm.device)
    dim = prm.shape[-1]
    batch = prm.shape[:-(dim + 1)]
    prm = prm.reshape([-1, *prm.shape[-(dim + 1):]])
    disp = resize_grid(prm, type='displacement', shape=shape,
                       interpolation=order, bound=bound)
    disp = disp.reshape(batch + disp.shape[1:])
    grid = disp + identity_grid(shape, **backend)
    if 'disp' in returns and 'grid' in returns:
        return disp, grid
    elif 'disp' in returns:
        return disp
    elif 'grid' in returns:
        return grid


def pull_grid(gridin, grid):
    """Sample a displacement field.

    Parameters
    ----------
    gridin : (*inshape, dim) tensor
    grid : (*outshape, dim) tensor

    Returns
    -------
    gridout : (*outshape, dim) tensor

    """
    gridin = movedim(gridin, -1, 0)[None]
    grid = grid[None]
    gridout = grid_pull(gridin, grid, bound='dft', extrapolate=True)
    gridout = movedim(gridout[0], 0, -1)
    return gridout


def pull(image, grid, interpolation=1, bound='dct2', extrapolate=False):
    """Sample a multi-channel image

    Parameters
    ----------
    image : (channel, *inshape) tensor
    grid : (*outshape, dim) tensor

    Returns
    -------
    imageout : (channel, *outshape)

    """
    image = image[None]
    grid = grid[None]
    image = grid_pull(image, grid, interpolation=interpolation,
                      bound=bound, extrapolate=extrapolate)[0]
    return image


def smalldef(disp):
    """Transform a displacement grid into a transformation grid

    Parameters
    ----------
    disp : (*shape, dim)

    Returns
    -------
    grid : (*shape, dim)

    """
    id = identity_grid(disp.shape[:-1], dtype=disp.dtype, device=disp.device)
    return disp + id
