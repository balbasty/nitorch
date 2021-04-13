import torch
import itertools
from nitorch.core import utils, py, linalg
from ._finite_differences import diff1d, diff
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


def shim(fmap, zdim=-1, max_order=2, mask=None, isocenter=None, dim=None,
         returns='corrected'):
    """Subtract a linear combination of spherical harmonics that minimize gradients

    Parameters
    ----------
    fmap : (..., *spatial) tensor
        Field map
    zdim : int, default=-1
        Dimension of the main magnetic field.
    max_order : int, default=2
        Maximum order of the spherical harmonics
    mask : tensor
        Mask of voxels to include (typically brain mask)
    isocenter : [sequence of] float, default=shape/2
        Coordinate of isocenter, in voxels
    dim : int, default=fmap.dim()
        Number of spatial dimensions
    returns : combination of {'corrected', 'correction', 'parameters'}, default='corrected'
        Components to return

    Returns
    -------
    corrected : (..., *spatial) tensor, if 'corrected' in `returns`
        Corrected field map (with spherical harmonics subtracted)
    correction : (..., *spatial) tensor, if 'correction' in `returns`
        Linear combination of spherical harmonics.
    parameters : (..., k) tensor, if 'parameters' in `returns`
        Parameters of the linear combination

    """
    fmap = torch.as_tensor(fmap)
    dim = dim or fmap.dim()
    shape = fmap.shape[-dim:]
    batch = fmap.shape[:-dim]
    backend = utils.backend(fmap)
    dims = list(range(-dim, 0))
    zdim = (fmap.dim() + zdim) if zdim < 0 else zdim
    zdim = zdim - fmap.dim()  # number from the end

    if mask is not None:
        mask = ~mask  # make it a mask of background voxels
    if isocenter is None:
        isocenter = [s/2 for s in shape]
    isocenter = utils.make_vector(isocenter, **backend)

    # compute gradients
    gmap = diff(fmap, dim=dims, side='f', bound='dct2')
    if mask is not None:
        gmap[..., mask, :] = 0
    gmap = gmap.reshape([*batch, -1])

    # compute basis of spherical harmonics
    basis = []
    for i in range(1, max_order+1):
        b = spherical_harmonics(shape, i, **backend)
        b = utils.movedim(b, -1, 0)
        b = diff(b, dim=dims, side='f', bound='dct2')
        if mask is not None:
            b[..., mask, :] = 0
        b = b.reshape([b.shape[0], *batch, -1])
        basis.append(b)
    basis = torch.cat(basis, 0)
    basis = utils.movedim(basis, 0, -1)  # (*batch, vox*dim, k)

    # solve system
    prm = linalg.lmdiv(basis, gmap[..., None], method='pinv')[..., 0]
    print(prm)
    # > (*batch, k)

    # rebuild basis (without taking gradients)
    basis = []
    for i in range(1, max_order+1):
        b = spherical_harmonics(shape, i, **backend)
        b = utils.movedim(b, -1, 0)
        if mask is not None:
            b[..., mask, :] = 0
        b = b.reshape([b.shape[0], *batch, *shape])
        basis.append(b)
    basis = torch.cat(basis, 0)
    basis = utils.movedim(basis, 0, -1)  # (*batch, vox*dim, k)

    comb = linalg.matvec(basis.unsqueeze(-2),
                         utils.unsqueeze(prm, -2, dim))
    comb = comb[..., 0]
    fmap = fmap - comb

    returns = returns.split('+')
    out = []
    for ret in returns:
        if ret == 'corrected':
            out.append(fmap)
        elif ret == 'correction':
            out.append(comb)
        elif ret[0] == 'p':
            out.append(prm)
    return out[0] if len(out) == 1 else tuple(out)


def spherical_harmonics(shape, order=2, **backend):
    """Generate a basis of spherical harmonics on a lattice

    Notes
    -----
    .. This should be checked!
    .. Only orders 1 and 2 implemented
    .. I tried to implement some sort of "circular" harmonics in
       dimension 2 but I don't know what I am doing.
    .. The basis is not orthogonal

    Parameters
    ----------
    shape : sequence of int
    order : {1, 2}, default=2
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    b : (*shape, 2*order + 1) tensor
        Basis

    """
    shape = py.make_list(shape)
    dim = len(shape)
    if dim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3')
    if order not in (1, 2):
        raise ValueError('Order must be 1 or 2')

    ramps = identity_grid(shape, **backend)
    for i, ramp in enumerate(ramps.unbind(-1)):
        ramp -= shape[i] / 2
        ramp /= shape[i] / 2

    if order == 1:
        return ramps
    # order == 2
    if dim == 3:
        basis = [ramps[..., 0] * ramps[..., 1],
                 ramps[..., 0] * ramps[..., 2],
                 ramps[..., 1] * ramps[..., 2],
                 ramps[..., 0].square() - ramps[..., 1].square(),
                 ramps[..., 0].square() - ramps[..., 2].square()]
        return torch.stack(basis, -1)
    else:  # basis == 2
        basis = [ramps[..., 0] * ramps[..., 1],
                 ramps[..., 0].square() - ramps[..., 1].square()]
        return torch.stack(basis, -1)

