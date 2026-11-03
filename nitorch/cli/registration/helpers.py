import os
import json
import torch
from nitorch.spatial import grid_pull, identity_grid, resize_grid
from nitorch.core.utils import movedim
from nitorch.core.pyutils import file_mod

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
    dim = len(inshape)
    eye = torch.eye(dim+1, dtype=aff.dtype, device=aff.device)
    return inshape == outshape and aff.allclose(eye)


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


def pull_grid(gridin, grid, interpolation=1, bound='dft', extrapolate=True):
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
    gridout = grid_pull(gridin, grid,
                        interpolation=interpolation,
                        bound=bound,
                        extrapolate=extrapolate)
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


class BacktrackingLineSearch(torch.optim.Optimizer):

    def __init__(self, optim, armijo=1, max_iter=6):
        self.optim = optim
        self.armijo = float(armijo)
        self.max_iter = max_iter

    def step(self, closure, loss=None):
        """

        Parameters
        ----------
        closure : callable
        loss : tensor, optional

        Returns
        -------
        tensor

        """

        def get_params():
            params = []
            for group in self.optim.param_groups:
                params.extend(group['params'])
            return params

        # the closure backpropagates, so it must run with autograd enabled
        closure = torch.enable_grad()(closure)

        with torch.no_grad():
            if loss is None:
                loss = closure()
            armijo = self.armijo

            params0 = [param.detach().clone() for param in get_params()]
            self.optim.step()
            deltas = [p - p0 for p, p0 in zip(get_params(), params0)]
            self.last_ok = False
            for n_iter in range(self.max_iter):
                new_loss = closure()
                if new_loss < loss:
                    # NOTE: the increased step is not kept across calls
                    armijo = 2 * armijo
                    self.last_ok = True
                    break
                else:
                    armijo = armijo / 2.
                    for param, param0, delta in zip(get_params(), params0,
                                                    deltas):
                        param.copy_(param0 + armijo * delta)
            if not self.last_ok:
                # no step decreased the loss: restore the initial parameters
                # (otherwise we would keep an untested step, and return
                # the loss of the previous one)
                for param, param0 in zip(get_params(), params0):
                    param.copy_(param0)
                new_loss = loss
        return new_loss

    def add_param_group(self, param_group: dict) -> None:
        return self.optim.add_param_group(param_group)

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optim.load_state_dict(state_dict)

    def state_dict(self):
        return self.optim.state_dict()

    def zero_grad(self, *args, **kwargs) -> None:
        return self.optim.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups


# ----------------------------------------------------------------------
#                       VELOCITY EXPONENTIATION
# ----------------------------------------------------------------------
# Shared by `reslice` and `compose`.

# keys of the JSON sidecar (written by `nitorch register`) that are
# forwarded to `spatial.shoot`
shoot_keys = ('absolute', 'membrane', 'bending', 'lame', 'factor',
              'steps', 'fast', 'forward_method', 'fixedpoint_iter')


def load_shoot_parameters(fname):
    """Read shooting parameters from a JSON sidecar"""
    with open(fname) as f:
        prm = json.load(f)
    return {key: val for key, val in prm.items() if key in shoot_keys}


def find_shoot_json(fname):
    """Return the path of the shooting sidecar `<fname>.json`, if any.

    The sidecar is only returned if it exists and contains shooting
    parameters (i.e., a `'bending'` key), as written by `nitorch register`.
    """
    sidecar = file_mod(fname, ext='.json')
    if os.path.exists(sidecar):
        with open(sidecar, "r") as f:
            prm = json.load(f)
        if 'bending' in prm:
            return sidecar
    return None


def exponentiate_velocity(vel, affine, json=None, square=False, inv=False):
    """Exponentiate a velocity field into a voxel displacement field

    Parameters
    ----------
    vel : (*spatial, dim) tensor
        Velocity field, in voxels. Modified in-place if `square`.
    affine : (dim+1, dim+1) tensor
        Orientation matrix of the velocity field.
    json : str, optional
        Path to a JSON sidecar with shooting parameters. If provided,
        the velocity is geodesically shot (`spatial.shoot`); otherwise
        it is treated as a stationary velocity field (`spatial.exp`).
    square : bool
        The velocity encodes the square of the transform to apply:
        halve it before exponentiation.
    inv : bool
        Return the inverse transform.

    Returns
    -------
    disp : (*spatial, dim) tensor
        Displacement field, in voxels.
    """
    from nitorch import spatial
    if square:
        vel.mul_(0.5)
    if json:
        prm = load_shoot_parameters(json)
        prm['voxel_size'] = spatial.voxel_size(affine)
        if inv:
            # the inverse does not depend on how the forward is computed
            prm.pop('forward_method', None)
            prm.pop('fixedpoint_iter', None)
        disp = spatial.shoot(vel[None], displacement=True,
                             return_inverse=inv, **prm)
        if inv:
            disp = disp[-1]
    else:
        disp = spatial.exp(vel[None], displacement=True, inverse=inv)
    return disp[0]  # drop batch dimension
