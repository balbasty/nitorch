import torch

import nitorch.core.version
from nitorch.core import utils, py, linalg, constants
from nitorch.core.fft import ifftshift
from ._finite_differences import diff1d, diff
from ._regularisers import regulariser
from ._grid import identity_grid


"""
Absolute MR susceptibility values.

/!\ the `mrfield` function takes *delta* susceptibility values, with 
respect to the air susceptibility. The susceptibility of the air should 
thererfore be subtracted from these values before being passed to 
`mrfield`.

All values are expressed in ppm (parts per million). 
They get multiplied by 1e-6 in `mrfield`

References
----------
..[1] "Perturbation Method for Magnetic Field Calculations of
       Nonconductive Objects"
      Mark Jenkinson, James L. Wilson, and Peter Jezzard
      MRM, 2004
      https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.20194
..[2] "Susceptibility mapping of air, bone, and calcium in the head"
      Sagar Buch, Saifeng Liu, Yongquan Ye, Yu‐Chung Norman Cheng, 
      Jaladhar Neelavalli, and E. Mark Haacke
      MRM, 2014
..[3] "Whole-brain susceptibility mapping at high field: A comparison 
       of multiple- and single-orientation methods"
      Sam Wharton, and Richard Bowtell
      NeuroImage, 2010
..[4] "Quantitative susceptibility mapping of human brain reflects 
       spatial variation in tissue composition"
      Wei Li, Bing Wua, and Chunlei Liu
      NeuroImage 2011
..[5] "Human brain atlas for automated region of interest selection in 
       quantitative susceptibility mapping: Application to determine iron 
       content in deep gray matter structures"
      Issel Anne L.Lim, Andreia V. Faria, Xu Li, Johnny T.C.Hsu, 
      Raag D.Airan, Susumu Mori, Peter C.M. van Zijl
      NeuroImage, 2013
"""
mr_chi = {
    'air': 0.4,         # Jenkinson (Buch: 0.35)
    'water': -9.1,      # Jenkinson (Buch: -9.05)
    'bone': -11.3,      # Buch
    'teeth': -12.5,     # Buch
}


def mrfield(ds, zdim=-1, dim=None, b0=1, s0=mr_chi['air'],
            s1=mr_chi['water']-mr_chi['air'], vx=1, analytical=False):
    """Generate a MR fieldmap from a MR susceptibility map.

    Parameters
    ----------
    ds : (..., *spatial) tensor
        Susceptibility delta map (delta from air susceptibility) in ppm.
        If bool, `s1` will be used to set the value inside the mask.
        If float, should contain quantitative delta values in ppm.
    zdim : int, default=-1
        Dimension of the main magnetic field.
    dim : int, default=ds.dim()
        Number of spatial dimensions.
    b0 : float, default=1
        Value of the main magnetic field
    s0 : float, default=0.4
        Susceptibility of air (ppm)
    s1 : float, default=-9.5
        Susceptibility of tissue minus susceptiblity of air (ppm)
        (only used if `ds` is a boolean mask)
    vx : [sequence of] float
        Voxel size
    analytical : bool, default=False
        If True, use Mark Jenkinson's analytical greens function.
        More accurate but slower.

    Returns
    -------
    field : tensor
        MR field map.

    References
    ----------
    ..[1] "Perturbation Method for Magnetic Field Calculations of
           Nonconductive Objects"
          Mark Jenkinson, James L. Wilson, and Peter Jezzard
          MRM, 2004
          https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.20194

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

    analytical = analytical and (dim == 3)  # anal. form only imp. in 3d

    if ds.dtype is torch.bool:
        ds = ds.to(**backend)
    else:
        s1 = ds.abs().max()
        ds = ds / s1
    s1 = s1 * 1e-6
    s0 = s0 * 1e-6

    if analytical:
        # Analytical implementation following Jenkinson et al.
        # Should be slighlty more precise
        g = mrfield_greens2(shape, zdim, voxel_size=vx, **backend)
        f = mrfield_greens_apply(ds, g)
    else:
        # Finite-difference based
        # We get the greens function by inversion in Fourier domain
        # which requires regularizing it slightly.

        # compute second order finite differences across z
        f = diff1d(ds, order=2, side='c', bound='dft', dim=zdim, voxel_size=vxz)
        f.neg_()  # That's really important! Did I make a mistake in diff?

        # compute greens function
        prm = dict(absolute=0, membrane=1, bending=0)
        g = mrfield_greens(shape, **prm, voxel_size=vx, **backend)

        # apply greens function to curvature of ds
        f = mrfield_greens_apply(f, g)

    # apply rest of the equation
    out = ds * ((1. + s0) / (3. + s0))
    out -= f
    out *= b0 * s1 / (1 + s0)
    return out


def mrfield_greens2(shape, zdim=-1, voxel_size=1, dtype=None, device=None):
    """Semi-analytical second derivative of the Greens kernel.

    This function implements exactly the solution from Jenkinson et al.
    (Same as in the FSL source code), with the assumption that
    no gradients are played and the main field is constant and has
    no orthogonal components (Bz = B0, Bx = By = 0).

    The Greens kernel and its second derivatives are derived analytically
    and integrated numerically over a voxel.

    The returned tensor has already been Fourier transformed and could
    be cached if multiple field simulations with the same lattice size
    must be performed in a row.

    Parameters
    ----------
    shape : sequence of int
        Lattice shape
    zdim : int, defualt=-1
        Dimension of the main magnetic field
    voxel_size : [sequence of] int
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*shape) tensor
        Fourier transform of the (second derivatives of the) Greens kernel.

    """
    import itertools

    def atan(num, den):
        return torch.where(den.abs() > 1e-8, torch.atan_(num/den),
                                             torch.atan2(num, den))

    dim = len(py.make_list(shape))
    g0 = identity_grid(shape, dtype=dtype, device=device)
    voxel_size = utils.make_vector(voxel_size, dim, dtype=torch.double).tolist()

    if dim == 3:
        if zdim in (-1, 2):
            odims = [-3, -2]
        elif zdim in (-2, 1):
            odims = [-3, -1]
        elif zdim in (-3, 0):
            odims = [-2, -1]
    else:
        raise NotImplementedError

    def make_shifted(shift):
        g = g0.clone()
        for g1, s, v, t in zip(g.unbind(-1), shape, voxel_size, shift):
            g1 -= s//2      # make center voxel zero
            g1 += t         # apply shift
            g1 *= v         # convert to mm
        return g

    g = 0
    for shift in itertools.product([-0.5, 0.5], repeat=dim):
        g1 = make_shifted(shift)
        if dim == 3:
            r = g1.square().sum(-1).sqrt_()
            g1 = atan(g1[..., odims[0]] * g1[..., odims[1]],
                      g1[..., zdim] * r)
        else:
            raise NotImplementedError
        if py.prod(shift) < 0:
            g -= g1
        else:
            g += g1

    g /= 4. * constants.pi
    g = ifftshift(g, range(dim))  # move center voxel to first voxel

    # fourier transform
    #   symmetric kernel -> real coefficients

    if nitorch.core.version.torch_version('>=', (1, 8)):
        g = torch.fft.fftn(g, dim=dim).real()
    else:
        if torch.backends.mkl.is_available:
            # use rfft
            g = torch.rfft(g, dim, onesided=False)
        else:
            zero = g.new_zeros([]).expand(g.shape)
            g = torch.stack([g, zero], dim=-1)
            g = torch.fft(g, dim)
        g = g[..., 0]  # should be real
    return g


def mrfield_greens(shape, absolute=0, membrane=0, bending=0, factor=1,
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
    voxel_size : [sequence of[ float, default=1
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    greens : (*shape, [dim, dim]) tensor

    """
    # Adapted from the geodesic shooting code

    backend = dict(dtype=dtype, device=device)
    shape = py.make_tuple(shape)
    dim = len(shape)
    if not absolute:
        # we need some regularization to invert
        absolute = max(absolute, max(membrane, bending)*1e-6)
    prm = dict(
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        factor=factor,
        voxel_size=voxel_size,
        bound='dft')

    # allocate
    kernel = torch.zeros(shape, **backend)

    # only use center to generate kernel
    if bending:
        subkernel = kernel[tuple(slice(s//2-2, s//2+3) for s in shape)]
        subsize = 5
    else:
        subkernel = kernel[tuple(slice(s//2-1, s//2+2) for s in shape)]
        subsize = 3

    # generate kernel
    center = (subsize//2,)*dim
    subkernel[center] = 1
    subkernel[...] = regulariser(subkernel, **prm, dim=dim)

    kernel = ifftshift(kernel, dim=range(dim))

    # fourier transform
    #   symmetric kernel -> real coefficients

    if nitorch.core.version.torch_version('>=', (1, 8)):
        kernel = torch.fft.fftn(kernel, dim=dim).real()
    else:
        if torch.backends.mkl.is_available:
            # use rfft
            kernel = torch.rfft(kernel, dim, onesided=False)
        else:
            zero = kernel.new_zeros([]).expand(kernel.shape)
            kernel = torch.stack([kernel, zero], dim=-1)
            kernel = torch.fft(kernel, dim)
        kernel = kernel[..., 0]  # should be real

    kernel = kernel.reciprocal_()
    return kernel


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
    if nitorch.core.version.torch_version('>=', (1, 8)):
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
    if nitorch.core.version.torch_version('>=', (1, 8)):
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
    f = f.square().sum(-1).sqrt() <= radius
    return f


def shim(fmap, max_order=2, mask=None, isocenter=None, dim=None,
         returns='corrected'):
    """Subtract a linear combination of spherical harmonics that minimize gradients

    Parameters
    ----------
    fmap : (..., *spatial) tensor
        Field map
    max_order : int, default=2
        Maximum order of the spherical harmonics
    mask : tensor, optional
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

    if mask is not None:
        mask = ~mask  # make it a mask of background voxels

    # compute gradients
    gmap = diff(fmap, dim=dims, side='f', bound='dct2')
    if mask is not None:
        gmap[..., mask, :] = 0
    gmap = gmap.reshape([*batch, -1])

    # compute basis of spherical harmonics
    basis = []
    for i in range(1, max_order+1):
        b = spherical_harmonics(shape, i, isocenter, **backend)
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
    # > (*batch, k)

    # rebuild basis (without taking gradients)
    basis = []
    for i in range(1, max_order+1):
        b = spherical_harmonics(shape, i, isocenter, **backend)
        b = utils.movedim(b, -1, 0)
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


def spherical_harmonics(shape, order=2, isocenter=None, **backend):
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
    isocenter : [sequence of] int, default=shape/2
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

    if isocenter is None:
        isocenter = [s/2 for s in shape]
    isocenter = utils.make_vector(isocenter, **backend)

    ramps = identity_grid(shape, **backend)
    for i, ramp in enumerate(ramps.unbind(-1)):
        ramp -= isocenter[i]
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


# --- values from the literature

# Wharton and Bowtell, NI, 2010
# These are relative susceptibility with respect to surrounding tissue
# (which I assume we can consider as white matter?)
wharton_chi_delta_water = {
    'sn': 0.17,         # substantia nigra
    'rn': 0.14,         # red nucleus
    'ic': -0.01,        # internal capsule
    'gp': 0.19,         # globus pallidus
    'pu': 0.10,         # putamen
    'cn': 0.09,         # caudate nucleus
    'th': 0.045,        # thalamus
    'gm_ppl': 0.043,    # posterior parietal lobe
    'gm_apl': 0.053,    # anterior parietal lobe
    'gm_fl': 0.04,      # frontal lobe
}

# Li et al, NI, 2011
# These are susceptibility relative to CSF
li_chi_delta_water = {
    'sn': 0.053,        # substantia nigra
    'rn': 0.032,        # red nucleus
    'ic': -0.068,       # internal capsule
    'gp': 0.087,        # globus pallidus
    'pu': 0.043,        # putamen
    'cn': 0.019,        # caudate nucleus
    'dn': 0.064,        # dentate nucleus
    'gcc': -0.033,      # genu of corpus callosum
    'scc': -0.038,      # splenium of corpus collosum
    'ss': -0.075,       # sagittal stratum
}

# Buch et al, MRM, 2014
buch_chi_delta_water = {
    'air': 9.2,
    'bone': -2.1,
    'teeth': -3.3,
}

# Jenkinson et al, MRM, 2004
# Absolute susceptibilities
jenkinson_chi = {
    'air': 0.4,
    'parenchyma': -9.1,
}
