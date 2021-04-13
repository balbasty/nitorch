import torch
import itertools
from nitorch.core import utils, py
from ._finite_differences import diff1d
from ._shoot import greens
from ._grid import identity_grid


def mrfield(ds, zdim=-1, dim=None, b0=1, s0=4e-7, s1=-9.5e-6, vx=1):
    """Generate a MR fieldmap from a MR susceptibility map.

    Parameters
    ----------
    ds : (..., *spatial) tensor
        Susceptibility delta map (delta from air susceptibility).
        If bool, `s1` will be used to set the value inside the mask.
        If float, should contain quantitative delta values in ppm.
    zdim : int, default=-1
        Dimension of the main magnetic field.
    dim : int, default=ds.dim()
        Number of spatial dimensions.
    b0 : float, default=1
        Value of the main magnetic field
    s0 : float, default=4e-7
        Susceptibility of air (ppm)
    s1 : float, default=-9.5e-6
        Susceptibility of tissue minus susceptiblity of air (ppm)
        (only used if `ds` is a boolean mask)
    vx : [sequence of] float
        Voxel size

    Returns
    -------
    field : tensor
        MR field map.

    """
    ds = torch.as_tensor(ds)
    backend = utils.backend(ds)
    if ds.dtype is torch.bool:
        backend['dtype'] = torch.get_default_dtype()

    dim = dim or ds.dim()
    shape = ds.shape[-dim:]
    zdim = (ds.dim() + zdim) if zdim < 0 else zdim
    zdim = zdim - ds.dim()  # number from end so that we can apply to vx
    vx = utils.make_vector(vx, 3, dtype=torch.float).tolist()
    vxz = vx[zdim]

    if ds.dtype is torch.bool:
        ds = ds.to(**backend) * s1

    # compute second order finite differences across z
    f = diff1d(ds, order=2, side='c', bound='dft', dim=zdim, voxel_size=vxz)

    # compute greens function
    prm = dict(absolute=0, membrane=1, bending=0, lame=0)
    g = greens(shape, **prm, voxel_size=vx, **backend)

    # apply greens function to ds
    f = mrfield_greens_apply(f, g)

    # apply rest of the equation
    f = b0 * (f + ds / (3. + s0))
    return f


def mrfield_greens_apply(mom, greens):
    """Apply the Greens function to a momentum field.

    Parameters
    ----------
    mom : (..., *spatial) tensor
        Momentum
    greens : (*spatial) tensor
        Greens function

    Returns
    -------
    field : (..., *spatial) tensor
        Field

    """
    mom, greens = utils.to_max_backend(mom, greens)
    dim = greens.dim()

    # fourier transform
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

    # voxel wise multiplication
    mom = mom * greens[..., None]

    # inverse fourier transform
    if utils.torch_version('>=', (1, 8)):
        mom = torch.fft.ifftn(mom, dim=dim).real()
    else:
        mom = torch.ifft(mom, dim)[..., 0]

    return mom


def susceptibility_phantom(shape, radius=None, dtype=None, device=None):
    """Generate a circle/sphere susceptibility phantom

    Parameters
    ----------
    shape : sequence of int
    radius : default=shape/4
    dtype : optional
    backend : optional

    Returns
    -------
    f : (*shape) tensor[bool]
        susceptibility delta map

    """

    shape = py.make_tuple(shape)
    radius = radius or (min(shape)/4.)
    f = identity_grid(shape, dtype=dtype, device=device)
    for comp, s in zip(f.unbind(-1), shape):
        comp -= s/2
    f = f.square().sum(-1).sqrt() < radius
    return f
