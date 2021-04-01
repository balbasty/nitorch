"""Geodesic shooting of initial velocity fields."""
import torch
from nitorch.core import py, utils, linalg
from nitorch.core.fft import ifftshift
from ._finite_differences import diff
from ._regularisers import regulariser, regulariser_grid
from ._grid import identity_grid, grid_pull, grid_push


def greens(shape, absolute=0, membrane=0, bending=0, lame=0,
           voxel_size=1, dtype=None, device=None):
    """Generate the Greens function of a regulariser in Fourier space.

    Parameters
    ----------
    shape : tuple[int]
        Output shape
    absolute : float, default=0
        Penalty on absolute values
    membrane : float, default=0
        Penalty on membrane energy
    bending : float, default=0
        Penalty on bending energy
    lame : float or (float, float), default=0
        Penalty on linear-elastic energy
    voxel_size : [sequence of[ float, default=1
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    greens : (*shape, [dim, dim]) tensor

    """
    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    backend = dict(dtype=dtype, device=device)
    shape = py.make_tuple(shape)
    dim = len(shape)
    lame1, lame2 = py.make_list(lame, 2)
    # absolute = max(absolute, max(membrane, bending, lame1, lame2)*1e-3)
    prm = dict(
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        voxel_size=voxel_size,
        bound='dft')
    if lame1 or lame2:
        prm['lame'] = lame

    # allocate
    if lame1 or lame2:
        kernel = torch.zeros([*shape, dim, dim], **backend)
    else:
        kernel = torch.zeros([*shape, 1, 1], **backend)

    # only use center to generate kernel
    if bending:
        subkernel = kernel[tuple(slice(s//2-2, s//2+3) for s in shape)]
        subsize = 5
    else:
        subkernel = kernel[tuple(slice(s//2-1, s//2+2) for s in shape)]
        subsize = 3

    # generate kernel
    if lame1 or lame2:
        for d in range(dim):
            center = (subsize//2,)*dim + (d, d)
            subkernel[center] = 1
            subkernel[..., :, d] = regulariser_grid(subkernel[..., :, d], **prm)
    else:
        center = (subsize//2,)*dim
        subkernel[center] = 1
        subkernel[..., 0, 0] = regulariser(subkernel[..., 0, 0], **prm, dim=dim)

    kernel = ifftshift(kernel, dim=range(dim))

    # fourier transform
    #   symmetric kernel -> real coefficients

    dtype = kernel.dtype
    kernel = kernel.double()
    if utils.torch_version('>=', (1, 8)):
        kernel = utils.movedim(kernel, [-2, -1], [0, 1])
        kernel = torch.fft.fftn(kernel, dim=dim).real()
        kernel = utils.movedim(kernel, [0, 1], [-2, -1])
    else:
        kernel = utils.movedim(kernel, [-2, -1], [0, 1])
        if torch.backends.mkl.is_available:
            # use rfft
            kernel = torch.rfft(kernel, dim, onesided=False)
        else:
            zero = kernel.new_zeros([]).expand(kernel.shape)
            kernel = torch.stack([kernel, zero], dim=-1)
            kernel = torch.fft(kernel, dim)
        kernel = kernel[..., 0]  # should be real
        kernel = utils.movedim(kernel, [0, 1], [-2, -1])
    kernel = kernel.to(dtype=dtype)

    if lame1 or lame2:
        kernel = kernel.inverse()
    else:
        kernel = kernel[..., 0, 0].reciprocal_()
    return kernel


_greens = greens  # alias to avoid shadowing


def greens_apply(mom, greens, voxel_size=1):
    """Apply the Greens function to a momentum field.

    Parameters
    ----------
    mom : (..., *spatial, dim) tensor
        Momentum
    greens : (*spatial, [dim, dim]) tensor
        Greens function
    voxel_size : [sequence of] float, default=1
        Voxel size. Only needed when no penalty is put on linear-elasticity.

    Returns
    -------
    vel : (..., *spatial, dim) tensor
        Velocity

    """
    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    mom, greens = utils.to_max_backend(mom, greens)
    dim = mom.shape[-1]

    # fourier transform
    mom = utils.movedim(mom, -1, 0)
    if utils.torch_version('>=', (1, 8)):
        mom = torch.fft.fftn(mom, dim=dim)
    else:
        if torch.backends.mkl.is_available:
            # use rfft
            mom = torch.rfft(mom, dim, onesided=False)
        else:
            zero = mom.new_zeros([]).expand(mom.shape)
            mom = torch.stack([mom, zero], dim=-1)
            mom = torch.fft(mom, dim)
    mom = utils.movedim(mom, 0, -1)

    # voxel-wise matrix multiplication
    if greens.dim() == dim:
        voxel_size = utils.make_vector(voxel_size, dim, **utils.backend(mom))
        voxel_size = voxel_size.square()
        if utils.torch_version('<', (1, 8)):
            greens = greens[..., None, None]
        mom = mom * greens
        mom = mom / voxel_size
    else:
        if utils.torch_version('<', (1, 8)):
            mom[..., 0, :] = linalg.matvec(greens, mom[..., 0, :])
            mom[..., 1, :] = linalg.matvec(greens, mom[..., 1, :])
        else:
            mom = torch.matmul(greens, mom)

    # inverse fourier transform
    mom = utils.movedim(mom, -1, 0)
    if utils.torch_version('>=', (1, 8)):
        mom = torch.fft.ifftn(mom, dim=dim).real()
    else:
        mom = torch.ifft(mom, dim)[..., 0]
    mom = utils.movedim(mom, 0, -1)

    return mom


def shoot(vel, greens=None, absolute=0, membrane=0, bending=0, lame=0,
          voxel_size=1, return_inverse=False, displacement=False, steps=8,
          fast=True):
    """Exponentiate a velocity field by geodesic shooting.

    Notes
    -----
    .. This function generates the *inverse* deformation, if we follow
       LDDMM conventions. This allows the velocity to be defined in the
       space of the moving image.
    .. If the greens function is provided, the penalty parameters are
       not used.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Initial velocity in moving space.
    greens : (*spatial, [dim, dim]) tensor, optional
        Greens function generated by `greens`.

    Other Parameters
    ----------------
    absolute : float, default=0
        Penalty on absolute displacements.
    membrane : float, default=0
        Penalty on the membrane energy.
    bending : float, default=0
        Penalty on the bending energy.
    lame : float or (float, float), default=0
        Linear elastic penalty.
    voxel_size : [sequence of] float, default=1
    return_inverse : bool, default=False
        Return the inverse on top of the forward transform.
    displacement : bool, default=False
        Return a displacement field instead of a transformation field.
    steps : int, default=8
        Number of integration steps.
        If None, use an educated guess based on the magnitude of `vel`.
    fast : bool, default=True
        If True, use a faster integration scheme, which may induce
        some numerical error (the energy is not exactly preserved
        aloong time). Else, use the slower but more precise scheme.

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Transformation from fixed to moving space.
        (It is used to warp a moving image to a fixed one).

    igrid : ([batch], *spatial, dim) tensor, if return_inverse
        Inverse transformation, from fixed to moving space.
        (It is used to warp a fixed image to a moving one).

    """
    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    vel = torch.as_tensor(vel)
    backend = utils.backend(vel)
    dim = vel.shape[-1]
    spatial = vel.shape[-dim-1:-1]

    prm = dict(absolute=absolute, membrane=membrane, bending=bending,
               lame=lame, voxel_size=voxel_size)
    pull_prm = dict(bound='dft', interpolation=1, extrapolate=True)
    if greens is None:
        greens = _greens(spatial, **prm, **backend)
    greens = torch.as_tensor(greens, **backend)

    if not steps:
        # Number of time steps from an educated guess about how far to move
        with torch.no_grad():
            steps = vel.square().sum(dim=-1).max().sqrt().floor().int().item() + 1

    id = identity_grid(spatial, **backend)
    mom = mom0 = regulariser_grid(vel, **prm, bound='dft')
    vel = vel / steps
    disp = -vel
    if return_inverse or not fast:
        idisp = vel.clone()

    for _ in range(1, abs(steps)):
        if fast:
            # JA: the update of u_t is not exactly as described in the paper,
            # but describing this might be a bit tricky. The approach here
            # was the most stable one I could find - although it does lose some
            # energy as < v_t, u_t> decreases over time steps.
            jac = _jacobian(-vel)
            mom = linalg.matvec(jac.transpose(-1, -2), mom)
            mom = _push_grid(mom, id + vel, **pull_prm)
        else:
            jac = _jacobian(idisp).inverse()
            mom = linalg.matvec(jac.transpose(-1, -2), mom0)
            mom = _push_grid(mom, id + idisp, **pull_prm)

        # Convolve with Greens function of L
        # v_t \gets L^g u_t
        vel = greens_apply(mom, greens, voxel_size=voxel_size).div_(steps)
        print(f'{0.5*steps*(vel*mom).sum().item()/py.prod(spatial):6f}',
              end=' ', flush=True)

        # $\psi \gets \psi \circ (id - \tfrac{1}{T} v)$
        # JA: I found that simply using
        # $\psi \gets \psi - \tfrac{1}{T} (D \psi) v$ was not so stable.
        disp = _pull_grid(disp, id - vel, **pull_prm).sub_(vel)
        if return_inverse or not fast:
            idisp += _pull_grid(vel, id + idisp, **pull_prm)

    if not displacement:
        disp += id
        if return_inverse:
            idisp += id
    return (disp, idisp) if return_inverse else disp


def _jacobian(disp, bound='dft', voxel_size=1):
    """Compute the Jacobian of a transformation field

    Notes
    -----
    .. Even though a displacement is provided, we compute the Jacobian
       of the transformation field (identity + displacement) by
       adding ones to the diagonal.
    .. This function uses central finite differences to estimate the
       Jacobian.

    Parameters
    ----------
    disp : ([batch], *spatial, dim) tensor
        Displacement (without identity)
    bound : str, default='dft'
    voxel_size : [sequence of] float, default=1

    Returns
    -------
    jac : ([batch], *spatial, dim, dim) tensor
        Jacobian. In each matrix: jac[i, j] = d psi[i] / d xj

    """
    disp = torch.as_tensor(disp)
    dim = disp.shape[-1]
    dims = list(range(-dim-1, -1))
    jac = diff(disp, dim=dims, bound=bound, voxel_size=voxel_size, side='c')
    torch.diagonal(jac, 0, -1, -2).add_(1)
    return jac


def _pull_grid(vel, grid, *args, **kwargs):
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


def _push_grid(vel, grid, *args, **kwargs):
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
    vel = grid_push(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch:
        vel = vel[0]
    return vel
