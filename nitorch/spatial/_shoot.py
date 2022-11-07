"""Geodesic shooting of initial velocity fields."""
import torch
from nitorch.core.optionals import custom_fwd, custom_bwd
from nitorch.core import py, utils, linalg, fft
from ._finite_differences import diff
from ._regularisers import regulariser, regulariser_grid
from ._grid import identity_grid, grid_pull, grid_push

_default_absolute = 1e-4
_default_membrane = 1e-3
_default_bending = 0.2
_default_lame = (0.05, 0.2)


@torch.jit.script
def _det2(A):
    # type: (Tensor) -> Tensor
    dt = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return dt


@torch.jit.script
def _det3(A):
    # type: (Tensor) -> Tensor
    dt = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) + \
         A[0, 1] * (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) + \
         A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    return dt


@torch.jit.script
def _inv2(A):
    # type: (Tensor) -> Tensor
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1]
    F[1, 1] = A[0, 0]
    F[0, 1] = -A[1, 0]
    F[1, 0] = -A[0, 1]
    dt = _det2(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


@torch.jit.script
def _inv3(A):
    # type: (Tensor) -> Tensor
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    F[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    F[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    F[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    F[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    F[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    F[1, 2] = A[1, 0] * A[0, 2] - A[1, 2] * A[0, 0]
    F[2, 0] = A[2, 1] * A[1, 0] - A[2, 0] * A[1, 1]
    F[2, 1] = A[2, 0] * A[0, 1] - A[2, 1] * A[0, 0]
    dt = _det3(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


def _inv(A):
    A = utils.movedim(A, [-2, -1], [0, 1])
    if len(A) == 3:
        A = _inv3(A)
    elif len(A) == 2:
        A = _inv2(A)
    else:
        raise NotImplementedError
    A = utils.movedim(A, [0, 1], [-2, -1])
    return A


def greens(shape, absolute=_default_absolute, membrane=_default_membrane,
           bending=_default_bending, lame=_default_lame, factor=1,
           voxel_size=1, dtype=None, device=None):
    """Generate the Greens function of a regulariser in Fourier space.

    Parameters
    ----------
    shape : tuple[int]
        Output shape
    absolute : float, default=0.0001
        Penalty on absolute values
    membrane : float, default=0.001
        Penalty on membrane energy
    bending : float, default=0.2
        Penalty on bending energy
    lame : float or (float, float), default=(0.05, 0.2)
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
    if not absolute:
        absolute = max(absolute, max(membrane, bending, lame1, lame2)*1e-3)
    prm = dict(
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        # factor=factor,
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
        subkernel[..., 0, 0] = regulariser(subkernel[None, ..., 0, 0], **prm, dim=dim)[0]

    kernel = fft.ifftshift(kernel, dim=range(dim))

    # fourier transform
    #   symmetric kernel -> real coefficients

    dtype = kernel.dtype
    kernel = kernel.double()
    kernel = fft.real(fft.fftn(kernel, dim=list(range(dim)), real=True))

    # if utils.torch_version('>=', (1, 8)):
    #     kernel = utils.movedim(kernel, [-2, -1], [0, 1])
    #     kernel = torch.fft.fftn(kernel, dim=list(range(-dim, 0))).real
    #     if callable(kernel):
    #         kernel = kernel()
    #     kernel = utils.movedim(kernel, [0, 1], [-2, -1])
    # else:
    #     kernel = utils.movedim(kernel, [-2, -1], [0, 1])
    #     if torch.backends.mkl.is_available:
    #         # use rfft
    #         kernel = torch.rfft(kernel, dim, onesided=False)
    #     else:
    #         zero = kernel.new_zeros([]).expand(kernel.shape)
    #         kernel = torch.stack([kernel, zero], dim=-1)
    #         kernel = torch.fft(kernel, dim)
    #     kernel = kernel[..., 0]  # should be real
    #     kernel = utils.movedim(kernel, [0, 1], [-2, -1])

    kernel = kernel.to(dtype=dtype)

    if lame1 or lame2:
        kernel = _inv(kernel)  #kernel.inverse()
    else:
        kernel = kernel[..., 0, 0].reciprocal_()
    return kernel


_greens = greens  # alias to avoid shadowing


def greens_apply(mom, greens, factor=1, voxel_size=1):
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
    mom = fft.fftn(mom, dim=list(range(-dim-1, -1)), real=True)

    # mom = utils.movedim(mom, -1, 0)
    # if utils.torch_version('>=', (1, 8)):
    #     mom = torch.fft.fftn(mom, dim=list(range(-dim, 0)))
    # else:
    #     if torch.backends.mkl.is_available:
    #         # use rfft
    #         mom = torch.rfft(mom, dim, onesided=False)
    #     else:
    #         zero = mom.new_zeros([]).expand(mom.shape)
    #         mom = torch.stack([mom, zero], dim=-1)
    #         mom = torch.fft(mom, dim)
    # mom = utils.movedim(mom, 0, -1)

    # voxel-wise matrix multiplication
    # if greens.dim() == dim:
    #     voxel_size = utils.make_vector(voxel_size, dim, **utils.backend(mom))
    #     voxel_size = voxel_size.square()
    #     if utils.torch_version('<', (1, 8)):
    #         greens = greens[..., None, None]
    #     mom = mom * greens
    #     mom = mom / voxel_size
    # else:
    #     if utils.torch_version('<', (1, 8)):
    #         mom[..., 0, :] = linalg.matvec(greens, mom[..., 0, :])
    #         mom[..., 1, :] = linalg.matvec(greens, mom[..., 1, :])
    #     else:
    #         mom = torch.complex(linalg.matvec(greens, mom.real),
    #                             linalg.matvec(greens, mom.imag))

    if greens.dim() == dim:
        voxel_size = utils.make_vector(voxel_size, dim, **utils.backend(mom))
        voxel_size = voxel_size.square().reciprocal()
        greens = greens.unsqueeze(-1)
        mom = fft.mul(mom, greens, real=(False, True))
        mom = fft.mul(mom, voxel_size, real=(False, True))
    else:
        mom = fft.matvec(greens, mom, real=(True, False))

    # inverse fourier transform
    # mom = utils.movedim(mom, -1, 0)
    # if utils.torch_version('>=', (1, 8)):
    #     mom = torch.fft.ifftn(mom, dim=list(range(-dim, 0))).real
    #     if callable(mom):
    #         mom = mom()
    # else:
    #     mom = torch.ifft(mom, dim)[..., 0]
    # mom = utils.movedim(mom, 0, -1)

    mom = fft.real(fft.ifftn(mom, dim=list(range(-dim-1, -1))))
    mom /= factor

    return mom


def shoot(vel, greens=None,
          absolute=_default_absolute, membrane=_default_membrane,
          bending=_default_bending, lame=_default_lame, factor=1,
          voxel_size=1, return_inverse=False, displacement=False, steps=8,
          fast=True, verbose=False):
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
    absolute : float, default=0.0001
        Penalty on absolute displacements.
    membrane : float, default=0.001
        Penalty on the membrane energy.
    bending : float, default=0.2
        Penalty on the bending energy.
    lame : float or (float, float), default=(0.05, 0.2)
        Linear elastic penalty.
    voxel_size : [sequence of] float, default=1
        Needed when greens is provided if lame == 0.
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
        along time). Else, use the slower but more precise scheme.

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
               lame=lame, voxel_size=voxel_size, factor=factor)
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

    for i in range(1, abs(steps)):
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
        vel = greens_apply(mom, greens, factor=factor, voxel_size=voxel_size)
        vel = vel.div_(steps)
        if verbose:
            print(f'{0.5*steps*(vel*mom).sum().item()/py.prod(spatial):6g}',
                  end='\n' if not (i % 5) else ' ', flush=True)

        # $\psi \gets \psi \circ (id - \tfrac{1}{T} v)$
        # JA: I found that simply using
        # $\psi \gets \psi - \tfrac{1}{T} (D \psi) v$ was not so stable.
        disp = _pull_grid(disp, id - vel, **pull_prm).sub_(vel)
        if return_inverse or not fast:
            idisp += _pull_grid(vel, id + idisp, **pull_prm)

    if verbose:
        print('')
    if not displacement:
        disp += id
        if return_inverse:
            idisp += id
    return (disp, idisp) if return_inverse else disp


# def shoot_backward(vel, mugrad, grad, hess=None, greens=None,
#                    absolute=_default_absolute, membrane=_default_membrane,
#                    bending=_default_bending, lame=_default_lame, factor=1,
#                    voxel_size=1, inverse=False, steps=8,
#                    interpolation='linear', bound='dft'):
#     # I am mostly following the adjoint procedure from:
#     #       "Diffeomorphic 3D Image Registration via Geodesic Shooting
#     #        Using an Efficient Adjoint Calculation"
#     #       Vialard, Risser, Rueckert, Cotter
#     #       IJCV (2012)
#     # with some tweaks.
#     #
#     # Their `J` corresponds to my `f` (the fixed image)
#     # Their `I` corresponds to my `mu` (the moving image)
#     # Their `P` is the scalar momentum, such that their `P0 ∇I0` is my
#     # inital momentum vector `m0`.
#     # I'll write the similarity/log-likelihood term of the objective
#     # function as S, and drop the regularization term, since the adjoint
#     # calculation is only concerned with propagating gradients of the
#     # similarity term.
#     # In the paper, `\hat{I}` corresponds to the flow of the gradient of
#     # `dS(f, mu)/dmu` backward in time, and their P corresponds to the
#     # flow of the gradient of `S` with respect to the scalar momentum
#     # `P`. The adjoint equations are mostly written in terms of `\hat{I} ∇I`,
#     # so I'll rewrite them in terms of the gradient G, defined such that
#     # `G1 = push[phi1](\hat{I}1) ∇I0`.
#
#     I_tilde = grad
#     P_tilde = 0
#     mom_hat = jac(phi) @ mugrad * I_tilde
#     mom_hat = greens_apply(mom_hat)
#     P_tilde += push(dot(jac @ mugrad, mom_hat), phi) / jacdet(phi)
#     I_tilde =


class _ApproximateShoot(torch.autograd.Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx, *args):
        has_inverse = args[7]
        displacement = args[8]
        outputs = shoot(*args)
        if has_inverse:
            # forward + inverse (only save forward)
            ctx.save_for_backward(outputs[0])
        else:
            # forward only
            ctx.save_for_backward(outputs)
        ctx.args = {'nb_args': len(args),
                    'displacement': displacement,
                    'inverse': has_inverse}
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        if ctx.args['inverse']:
            grad_output, igrad_output = grad_output
        pull_prm = dict(bound='dft', interpolation=1, extrapolate=True)
        grad = _push_grid(grad_output, grid, **pull_prm)
        if ctx.args['inverse']:
            grad -= igrad_output
        output = [None] * ctx.args['nb_args']
        output[0] = grad
        return tuple(output)


def shoot_approx(vel, greens=None,
          absolute=_default_absolute, membrane=_default_membrane,
          bending=_default_bending, lame=_default_lame, factor=1,
          voxel_size=1, return_inverse=False, displacement=False, steps=8,
          fast=True, verbose=False):
    """Exponentiate a velocity field by geodesic shooting.

    Notes
    -----
    .. The backward pass uses an approximate (but much faster) gradient.
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
    absolute : float, default=0.0001
        Penalty on absolute displacements.
    membrane : float, default=0.001
        Penalty on the membrane energy.
    bending : float, default=0.2
        Penalty on the bending energy.
    lame : float or (float, float), default=(0.05, 0.2)
        Linear elastic penalty.
    voxel_size : [sequence of] float, default=1
        Needed when greens is provided if lame == 0.
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
        along time). Else, use the slower but more precise scheme.

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Transformation from fixed to moving space.
        (It is used to warp a moving image to a fixed one).

    igrid : ([batch], *spatial, dim) tensor, if return_inverse
        Inverse transformation, from fixed to moving space.
        (It is used to warp a fixed image to a moving one).

    """
    args = (vel, greens, absolute, membrane, bending, lame, factor, voxel_size,
            return_inverse, displacement, steps, fast, verbose)
    return _ApproximateShoot.apply(*args)
        

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
    if vel_no_batch and grid_no_batch:
        vel = vel[0]
    return vel
