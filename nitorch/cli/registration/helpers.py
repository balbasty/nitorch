import os
import json
import math
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
              'steps', 'fast', 'inversion')

# The JSON sidecar `<name>.json` written by `nitorch register` next to
# the nonlinear transform `<name>.nii.gz` holds:
# - shoot: the shooting parameters (`shoot_keys`) and `"model": "shoot"`
#   (older sidecars have no `"model"`: a JSON without it is a shooting
#   JSON);
# - svf, smalldef: `{"model": "svf"|"smalldef", "inversion": ...,
#   "bound": ...}`.
# `"inversion"` (absent in older sidecars: 'default') is the method
# used to compute, by numerical inversion, the side of the warp that is
# not computed directly (see `NonLinModel`).
# `"bound"` (absent in older sidecars, and for shoot: 'dft') is the
# boundary condition of the field, used to exponentiate and invert it.


def load_shoot_parameters(fname):
    """Read shooting parameters from a JSON sidecar"""
    with open(fname) as f:
        prm = json.load(f)
    return {key: val for key, val in prm.items() if key in shoot_keys}


def load_sidecar_model(fname):
    """Model (and inversion method) of a JSON sidecar.

    Returns
    -------
    model : {'shoot', 'svf', 'smalldef'}
        'shoot' if the JSON does not say (older sidecars, or
        user-provided shooting parameters).
    inversion : {'default', 'push', 'newton'}
    """
    from nitorch.spatial import check_inversion
    with open(fname) as f:
        prm = json.load(f)
    if not isinstance(prm, dict):
        return 'shoot', 'default'
    return (prm.get('model', 'shoot'),
            check_inversion(prm.get('inversion', 'default')))


def load_sidecar_bound(fname):
    """Boundary condition of the field described by a JSON sidecar
    ('dft' if it does not say: older sidecars, and shooting)."""
    with open(fname) as f:
        prm = json.load(f)
    if not isinstance(prm, dict):
        return 'dft'
    return prm.get('bound', 'dft')


def _read_sidecar(fname):
    sidecar = file_mod(fname, ext='.json')
    if os.path.exists(sidecar):
        with open(sidecar, "r") as f:
            return sidecar, json.load(f)
    return None, None


def find_shoot_json(fname):
    """Return the path of the velocity sidecar `<fname>.json`, if any.

    The sidecar is only returned if it exists and was written by
    `nitorch register` for a velocity: it contains shooting parameters
    (i.e., a `'bending'` key) or describes a stationary velocity
    (`"model": "svf"`, with its inversion method).
    """
    sidecar, prm = _read_sidecar(fname)
    if prm is not None:
        if 'bending' in prm or (isinstance(prm, dict) and
                                prm.get('model') == 'svf'):
            return sidecar
    return None


def find_displacement_json(fname):
    """Return the path of the displacement sidecar `<fname>.json`, if any
    (written by `nitorch register` for a smalldef model)."""
    sidecar, prm = _read_sidecar(fname)
    if isinstance(prm, dict) and prm.get('model') == 'smalldef':
        return sidecar
    return None


def invert_displacement(disp, fname=None):
    """Invert a voxel displacement field (`-id DISP`).

    `spatial.grid_inv`, with the inversion method stored in the sidecar
    `<fname>.json` (smalldef model): 'newton' refines the push inverse;
    'default' and 'push' are the push inverse (as without sidecar).
    """
    from nitorch import spatial
    method = 'push'
    bound = 'dft'
    sidecar = find_displacement_json(fname) if fname else None
    if sidecar:
        _, inversion = load_sidecar_model(sidecar)
        if inversion == 'newton':
            method = 'newton'
        bound = load_sidecar_bound(sidecar)
    if method == 'push':
        return spatial.grid_inv(disp, type='disp', bound=bound)
    return spatial.grid_inv(disp, type='disp', method=method, bound=bound)


def velocity_time_bounds(t0=0, t1=1, square=False, inv=False):
    """Time bounds of the map applied by `-v VEL [JSON] [T0 T1]`

    The shorthands are particular bounds:
    `-v2` (`square`) halves the bounds (`-v2 vel` = `-v vel 0 0.5`) and
    `-iv` (`inv`) swaps them (`-iv vel` = `-v vel 1 0`).

    Parameters
    ----------
    t0, t1 : float
        Time bounds given on the command line.
    square : bool
        The velocity encodes the square of the transform to apply.
    inv : bool
        Apply the inverse transform.

    Returns
    -------
    t0, t1 : float
        Effective time bounds.
    """
    if square:
        t0, t1 = t0 * 0.5, t1 * 0.5
    if inv:
        t0, t1 = t1, t0
    return t0, t1


def pop_velocity_values(args, tag, rest=None):
    """Pop the optional values `[JSON] [T0 T1]` of a velocity option

    Values that parse as numbers are time bounds; any other value is
    the JSON file of shooting parameters (which must come first).
    There must be zero or two time bounds. Once `args` is exhausted
    (the option is the last one before `--`), values are taken from
    `rest` (the arguments after `--`), even if they are dash-prefixed.

    Parameters
    ----------
    args : list[str]
        Arguments that follow the velocity file. Not modified.
    tag : str
        Name of the option (for error messages).
    rest : list[str], optional
        Arguments after `--`. Modified in-place.

    Returns
    -------
    json : str or None
        JSON file of shooting parameters (None if not provided).
    bounds : (float, float) or None
        Time bounds (None if not provided).
    args : list[str]
        Remaining arguments.
    """
    from nitorch.core import cli
    args = list(args)

    def peek():
        if args:
            return args[0] if cli.isvalue(args[0]) else None
        if rest:
            return rest[0]
        return None

    def pop():
        return args.pop(0) if args else rest.pop(0)

    fjson = None
    val = peek()
    if val is not None and not cli._isnumber(val):
        fjson = pop()

    bounds = []
    while len(bounds) < 2:
        val = peek()
        if val is None or not cli._isnumber(val):
            break
        bounds.append(float(pop()))
    if len(bounds) == 1:
        raise cli.ParseError(f'{tag} expects zero or two time bounds '
                             f'(T0 T1), but got one: {bounds[0]:g}')
    if not all(map(math.isfinite, bounds)):
        raise cli.ParseError(f'{tag}: time bounds must be finite, '
                             f'got {bounds}')
    return fjson, (tuple(bounds) or None), args


def exponentiate_velocity(vel, affine, json=None, square=False, inv=False,
                          t0=0, t1=1):
    """Exponentiate a velocity field into a voxel displacement field

    The returned transform carries the path of the velocity from time
    `t0` to time `t1` (`-v VEL [JSON] [T0 T1]`). Let `phi_t` be the
    transform that `-v` applies when the velocity is scaled by `t`
    (`phi_1` = `-v vel`, `phi_0.5` = `-v2 vel`), such that
    `mov o phi_t` is the moving image `mov` at time `t` along the
    path (`mov o phi_1` matches the fixed image). Then, the transform
    returned is (as a sampling map, used like `phi_t`)

        x -> inv(phi_t0)(phi_t1(x)),

    such that `(mov o phi_t0) o result = mov o phi_t1`: an image at
    time `t0` is resliced to time `t1`. Chaining `t0 -> t1` and then
    `t1 -> t2` (in this order on the reslice command line) is the
    transform `t0 -> t2`.

    - Stationary velocity: the transform is exactly `exp((t1-t0)*vel)`.
    - Geodesic shooting: `phi_t` is the shooting of `t*vel` over unit
      time (geodesics can be rescaled in time). If `t0 == 0`, this is
      `shoot(t1*vel)`; if `t1 == 0`, the inverse of `shoot(t0*vel)`;
      else, both are composed (the accuracy of the composition is
      bounded by that of the inverse returned by `spatial.shoot`).

    Inversion (the JSON's `"inversion"`, as `register --inversion`) is
    applied to each map that is built, exactly as the model computes it
    at `t = 1`, before any composition:

    - Stationary velocity: the forward `exp(t*vel)` (t > 0) is computed
      directly; the backward map `exp(-t*vel)` (t = t1 - t0 < 0) is the
      numerical inverse of `exp(t*vel)` (if not 'default').
    - Geodesic shooting: the inverse of `phi_t` is integrated directly;
      `phi_t` itself is the numerical inverse of it (if not 'default').
      For `0 < t0, t1`, `inv(phi_t0)` is direct and `phi_t1` inverted,
      then they are composed.

    Bounds outside of [0, 1] extrapolate along the path.

    Parameters
    ----------
    vel : (*spatial, dim) tensor
        Velocity field, in voxels. Not modified.
    affine : (dim+1, dim+1) tensor
        Orientation matrix of the velocity field.
    json : str, optional
        Path to a JSON sidecar with shooting parameters. If provided
        (and it does not say `"model": "svf"`), the velocity is
        geodesically shot (`spatial.shoot`); otherwise it is treated as
        a stationary velocity field (`spatial.exp`).
    square : bool
        The velocity encodes the square of the transform to apply:
        halve the time bounds (`-v2`).
    inv : bool
        Return the inverse transform: swap the time bounds (`-iv`).
    t0, t1 : float, default=(0, 1)
        Time bounds.

    Returns
    -------
    disp : (*spatial, dim) tensor
        Displacement field, in voxels.
    """
    from nitorch import spatial
    t0, t1 = velocity_time_bounds(t0, t1, square=square, inv=inv)

    def scaled(t):
        # (no multiplication if t == 1: bitwise identical to `-v`/`-iv`)
        return vel if t == 1 else vel * t

    if t0 == t1:
        return torch.zeros_like(vel)
    model, inversion = load_sidecar_model(json) if json else ('svf', 'default')
    if model not in ('shoot', 'svf'):
        raise ValueError(f'{json} describes a {model} model, '
                         f'not a velocity')
    if model == 'shoot':
        prm = load_shoot_parameters(json)
        prm.pop('inversion', None)
        prm['voxel_size'] = spatial.voxel_size(affine)

        def shoot(t, inverse=False):
            kw = prm
            if not inverse and inversion != 'default':
                # (the inverse is integrated directly, whatever `inversion`)
                kw = dict(prm, inversion=inversion)
            out = spatial.shoot(scaled(t)[None], displacement=True,
                                return_inverse=inverse, **kw)
            return (out[-1] if inverse else out)[0]  # drop batch dimension

        if t0 == 0:
            return shoot(t1)
        if t1 == 0:
            return shoot(t0, inverse=True)
        # x -> inv(phi_t0)(phi_t1(x))
        disp = shoot(t1)
        idisp = shoot(t0, inverse=True)
        return disp + pull_grid(idisp, disp + identity_grid(
            disp.shape[:-1], dtype=disp.dtype, device=disp.device))
    else:
        # exp(-t0 v) o exp(t1 v) = exp((t1 - t0) v)
        bound = load_sidecar_bound(json) if json else 'dft'
        t = t1 - t0
        if t < 0 and inversion != 'default':
            disp = spatial.exp(scaled(-t)[None], displacement=True,
                               bound=bound)
            return spatial.grid_inv(disp[0], type='disp', method=inversion,
                                    bound=bound)
        disp = spatial.exp(scaled(abs(t))[None], displacement=True,
                           inverse=t < 0, bound=bound)
        return disp[0]  # drop batch dimension
