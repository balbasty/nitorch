"""Spatial transformation layers."""

import torch
from nitorch import core, spatial
from nitorch.core import utils
from nitorch.core.py import make_list
from .base import Module


_interpolation_doc = \
    """`interpolation` can be an int, a string or an InterpolationType.
    Possible values are:
        - 0 or 'nearest'
        - 1 or 'linear'
        - 2 or 'quadratic'
        - 3 or 'cubic'
        - 4 or 'fourth'
        - 5 or 'fifth'
        - 6 or 'sixth'
        - 7 or 'seventh'
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders."""

_bound_doc = \
    """`bound` can be a string or a BoundType.
    Possible values are:
        - 'replicate'  or 'nearest'
        - 'dct1'       or 'mirror'
        - 'dct2'       or 'reflect'
        - 'dst1'       or 'antimirror'
        - 'dst2'       or 'antireflect'
        - 'dft'        or 'wrap'
        - 'zero'
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    Note that:
    - `dft` corresponds to circular padding
    - `dct2` corresponds to Neumann boundary conditions (symmetric)
    - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
    See https://en.wikipedia.org/wiki/Discrete_cosine_transform
        https://en.wikipedia.org/wiki/Discrete_sine_transform"""


class GridPull(Module):
    __doc__ = f"""
    Pull/Sample an image according to a deformation.

    This module has no learnable parameters.

    {_interpolation_doc}

    {_bound_doc}
    """

    def __init__(self, interpolation='linear', bound='dct2', extrapolate=True):
        """

        Parameters
        ----------
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, grid, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_out, len(spatial_in)) tensor
            Transformation grid
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        pulled : (batch, channel, *spatial_out) tensor
            Deformed image.

        """
        interpolation = overload.get('interpolation', self.interpolation)
        bound = overload.get('bound', self.bound)
        extrapolate = overload.get('extrapolate', self.extrapolate)
        return spatial.grid_pull(x, grid, interpolation, bound, extrapolate)


class GridPush(Module):
    __doc__ = """
    Push/Splat an image according to a deformation.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, shape=None, interpolation='linear', bound='dct2',
                 extrapolate=True):
        """

        Parameters
        ----------
        shape : list[int], optional
            Output spatial shape. Default is the same as the input shape.
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.shape = shape
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, grid, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_out, len(spatial_in)) tensor
            Transformation grid
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        pushed : (batch, channel, *spatial_out) tensor
            Deformed image.

        """
        shape = overload.get('shape', self.shape)
        interpolation = overload.get('interpolation', self.interpolation)
        bound = overload.get('bound', self.bound)
        extrapolate = overload.get('extrapolate', self.extrapolate)
        return spatial.grid_push(x, grid, shape,
                                 interpolation=interpolation,
                                 bound=bound,
                                 extrapolate=extrapolate)


class GridPushCount(Module):
    __doc__ = """
    Push/Splat an image **and** ones according to a deformation.

    Both an input image and an image of ones of the same shape are pushed.
    The results are concatenated along the channel dimension.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, shape=None, interpolation='linear', bound='dct2',
                 extrapolate=True):
        """

        Parameters
        ----------
        shape : list[int], optional
            Output spatial shape. Default is the same as the input shape.
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.shape = shape
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, grid, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_in, dir) tensor
            Transformation grid
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        pushed : (batch, channel, *shape) tensor
            Pushed image.
        count : (batch, 1, *shape) tensor
            Pushed image.

        """
        shape = overload.get('shape', self.shape)
        interpolation = overload.get('interpolation', self.interpolation)
        bound = overload.get('bound', self.bound)
        extrapolate = overload.get('extrapolate', self.extrapolate)
        push = spatial.grid_push(x, grid, shape,
                                 interpolation=interpolation,
                                 bound=bound,
                                 extrapolate=extrapolate)
        count = spatial.grid_count(grid, shape,
                                   interpolation=interpolation,
                                   bound=bound,
                                   extrapolate=extrapolate)
        return push, count


class GridExp(Module):
    """Exponentiate a stationary velocity field."""

    def __init__(self, fwd=True, inv=False, steps=None,
                 interpolation='linear', bound='dft', displacement=False):
        """

        Parameters
        ----------
        fwd : bool, default=True
            Return the forward deformation.
        inv : bool, default=False
            Return the inverse deformation.
        steps : int, optional
            Number of integration steps.
            Use `1` to use a small displacements model instead of a
            diffeomorphic one. Default is an educated guess based on the
            magnitude of the velocity field.
        interpolation : {0..7}, default=1
            Interpolation order. Can also be names ('nearest', 'linear', etc.).
        bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}, default='dft'
            Boundary conditions.
        displacement : bool, default=False
            Return a displacement field rather than a transformation field.
        """
        super().__init__()

        self.fwd = fwd
        self.inv = inv
        self.steps = steps
        self.interpolation = interpolation
        self.bound = bound
        self.displacement = displacement

    def forward(self, velocity, **kwargs):
        """

        Parameters
        ----------
        velocity :(batch, *spatial, dim) tensor
            Stationary velocity field.
        **overload : dict
            all parameters of the module can be overridden at call time.

        Returns
        -------
        forward : (batch, *spatial, dim) tensor, if `forward is True`
            Forward displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.
        inverse : (batch, *spatial, dim) tensor, if `inverse is True`
            Inverse displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.

        """
        fwd = kwargs.get('fwd', self.forward)
        inv = kwargs.get('inverse', self.inv)
        opt = {
            'steps': kwargs.get('steps', self.steps),
            'interpolation': kwargs.get('interpolation', self.interpolation),
            'bound': kwargs.get('bound', self.bound),
            'displacement': kwargs.get('displacement', self.displacement),
        }

        output = []
        if fwd:
            y = spatial.exp(velocity, inverse=False, **opt)
            output.append(y)
        if inv:
            iy = spatial.exp(velocity, inverse=True, **opt)
            output.append(iy)

        return output if len(output) > 1 else \
               output[0] if len(output) == 1 else \
               None


class GridShoot(Module):
    """Exponentiate an initial velocity field by Geodesic Shooting."""

    def __init__(self, fwd=True, inv=False, steps=8,
                 absolute=0.0001, membrane=0.001, bending=0.2, lame=(0.05, 0.2),
                 factor=1, voxel_size=1, displacement=False, cache_greens=False):
        """

        Parameters
        ----------
        fwd : bool, default=True
            Return the forward deformation.
        inv : bool, default=False
            Return the inverse deformation.
        steps : int, default=8
            Number of integration steps.
        absolute : float, default=0.0001
            Penalty on absolute values
        membrane : float, default=0.001
            Penalty on membrane energy
        bending : float, default=0.2
            Penalty on bending energy
        lame : float or (float, float), default=(0.05, 0.2)
            Penalty on linear-elastic energy
        factor : float, default=1
            Common multiplicative factor for the penalties.
        voxel_size : [sequence of] float, default=1
            Voxel size
        displacement : bool, default=False
            Return a displacement field rather than a transformation field.
        cache_greens: bool, default=False
            Precompute and cache the Greens function.
            The greens function depends on the penalty parameters and
            on the image shape. If any of these change between batches,
            it is better to not cache it.
        """
        super().__init__()

        self.fwd = fwd
        self.inv = inv
        self.steps = steps
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.lame = lame
        self.factor = factor
        self.voxel_size = voxel_size
        self.displacement = displacement
        self.cache_greens = cache_greens

    def forward(self, velocity, **overload):
        """

        Parameters
        ----------
        velocity :(batch, *spatial, dim) tensor
            Initial velocity field.
        **overload : dict
            all parameters of the module can be overridden at call time.

        Returns
        -------
        forward : (batch, *spatial, dim) tensor, if `forward is True`
            Forward displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.
        inverse : (batch, *spatial, dim) tensor, if `inverse is True`
            Inverse displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.

        """
        fwd = overload.get('fwd', self.forward)
        inv = overload.get('inverse', self.inv)
        absolute = overload.get('absolute', self.absolute)
        membrane = overload.get('membrane', self.membrane)
        bending = overload.get('bending', self.bending)
        lame = overload.get('lame', self.lame)
        lame = make_list(lame, 2)
        factor = overload.get('factor', self.factor)
        voxel_size = overload.get('voxel_size', self.voxel_size)

        shoot_opt = {
            'steps': overload.get('steps', self.steps),
            'displacement': overload.get('displacement', self.displacement),
            'voxel_size': voxel_size,
            'absolute': absolute,
            'membrane': membrane,
            'bending': bending,
            'lame': lame,
            'factor': factor,
        }
        greens_prm = {
            'absolute': absolute,
            'membrane': membrane,
            'bending': bending,
            'lame': lame,
            'factor': factor,
            'voxel_size': voxel_size,
            'shape': velocity.shape[1:-1],
        }

        if self.cache_greens:
            if getattr(self, '_greens_prm', None) == greens_prm:
                greens = self._greens.to(**utils.backend(velocity))
            else:
                greens = spatial.greens(**greens_prm, **utils.backend(velocity))
                self._greens = greens
                self._greens_prm = greens_prm
        else:
            greens = spatial.greens(**greens_prm, **utils.backend(velocity))

        output = []
        if inv:
            y, iy = spatial.shoot(velocity, greens, return_inverse=True, **shoot_opt)
            if fwd:
                output.append(y)
            output.append(iy)
        elif fwd:
            y = spatial.shoot(velocity, greens, **shoot_opt)
            output.append(y)

        return output if len(output) > 1 else \
               output[0] if len(output) == 1 else \
               None


class AffineExp(Module):
    """Exponentiate an inifinitesimal affine transformation (Lie algebra)."""

    def __init__(self, dim, basis='CSO', fwd=True, inv=False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Spatial dimension
        basis : basis_like or list[basis_like], default='CSO'
            The simplest way to define an affine basis is to choose from
            a list of Lie groups:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
            More complex (hierarchical) encodings can be achieved as well.
            See `affine_matrix`.
        fwd : bool, default=True
            Return the forward transformation.
        inv : bool, default=False
            Return the inverse transformation.
        """
        super().__init__()

        self.dim = dim
        self.basis = spatial.build_affine_basis(basis, dim)
        self.fwd = fwd
        self.inv = inv

    def forward(self, prm, **overload):
        """

        Parameters
        ----------
        prm : (batch, nb_prm) tensor or list[tensor]
            Affine parameters on the Lie algebra.
        overload : dict
            All parameters of the module can be overridden at call time.

        Returns
        -------
        forward : (batch, dim+1, dim+1) tensor, optional
            Forward matrix
        inverse : (batch, dim+1, dim+1) tensor, optional
            Inverse matrix

        """
        fwd = overload.get('fwd', self.forward)
        inv = overload.get('inverse', self.inv)
        basis = overload.get('basis', self.basis)
        basis = spatial.build_affine_basis(basis, self.dim)

        output = []
        if fwd:
            aff = spatial.affine_matrix(prm, basis)
            output.append(aff)
        if inv:
            if isinstance(prm, (list, tuple)):
                prm = [-p for p in prm]
            else:
                prm = -prm
            iaff = spatial.affine_matrix(prm, basis)
            output.append(iaff)

        return output if len(output) > 1 else \
               output[0] if len(output) == 1 else \
               None


class AffineLog(Module):
    """Take the Riemannian logarithm of an affine (recovers Lie algebra).

    Note
    ----
    The matrix logarithm is currently only implemented on cpu
    and is not parallelized across batches (we just call scipy).
    This is therefore a quite slow layer, which causes data transfer
    between cpu and cuda devices. Hopefully it is not too big a
    bottelneck (affines are quite small in size).

    """

    def __init__(self, basis='CSO'):
        """

        Parameters
        ----------
        basis : basis_like or list[basis_like], default='CSO'
            The simplest way to define an affine basis is to choose from
            a list of Lie groups:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
            More complex (hierarchical) encodings can be achieved as well.
            See `affine_matrix`.
        """
        super().__init__()

        self.basis = basis

    def forward(self, affine, **overload):
        """

        Parameters
        ----------
        prm : (batch, nb_prm) tensor or list[tensor]
            Affine parameters on the Lie algebra.
        overload : dict
            All parameters of the module can be overridden at call time.

        Returns
        -------
        forward : (batch, dim+1, dim+1) tensor, optional
            Forward matrix
        inverse : (batch, dim+1, dim+1) tensor, optional
            Inverse matrix

        """
        # I build the matrix at each call, which is not great.
        # Hard to be efficient *and* generic...
        dim = affine.shape[-1] - 1
        backend = dict(dtype=affine.dtype, device=affine.device)
        basis = overload.get('basis', self.basis)
        basis = spatial.build_affine_basis(basis, dim, **backend)

        # When the affine is well conditioned, its log should be real.
        # Here, I take the real part just in case.
        # Another solution could be to regularise the affine (by loading
        # slightly the diagonal) until it is well conditioned -- but
        # how would that work with autograd?
        affine = core.linalg.logm(affine.double())
        if affine.is_complex():
            affine = affine.real
        affine = affine.to(**backend)
        affine = core.linalg.mdot(affine[:, None, ...], basis[None, ...])
        return affine


class AffineClassic(Module):
    """Build an affine by matrix multiplication of individual affines."""

    def __init__(self, dim, basis='CSO', logzooms=False):
        """

        Parameters
        ----------
        dim : {2, 3}
            Spatial dimension
        basis : str, default='CSO'
            Chosen from a list of Lie groups:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
        logzooms : bool, default=False
            If True, this function will exponentiate the input zoom parameters.
        """
        super().__init__()

        self.dim = dim
        self.basis = basis
        self.logzooms = logzooms

    def forward(self, prm, **overload):
        """

        Parameters
        ----------
        prm : (batch, nb_prm) tensor or list[tensor]
            Affine parameters, ordered as
            (*translations, *rotations, *zooms, *shears).
        overload : dict
            All parameters of the module can be overridden at call time.

        Returns
        -------
        affine : (batch, dim+1, dim+1) tensor
            Affine matrix

        """
        dim = overload.get('dim', self.dim)
        basis = overload.get('basis', self.basis)
        logzooms = overload.get('logzooms', self.logzooms)

        def checkdim(expected, got):
            if got != expected:
                raise ValueError('Expected {} parameters for group {}({}) but '
                                 'got {}.'.format(expected, basis, dim, got))

        nb_prm = prm.shape[-1]
        eps = core.constants.eps(prm.dtype)

        if basis == 'T':
            checkdim(dim, nb_prm)
        elif basis == 'SO':
            checkdim(dim*(dim-1)//2, nb_prm)
        elif basis == 'SE':
            checkdim(dim + dim*(dim-1)//2, nb_prm)
        elif basis == 'D':
            checkdim(dim + 1, nb_prm)
            translations = prm[..., :dim]
            zooms = prm[..., -1]
            zooms = zooms.expand([*zooms.shape, dim])
            zooms = zooms.exp() if logzooms else zooms.clamp_min(eps)
            prm = torch.cat((translations, zooms), dim=-1)
        elif basis == 'CSO':
            checkdim(dim + dim*(dim-1)//2 + 1, nb_prm)
            rigid = prm[..., :-1]
            zooms = prm[..., -1]
            zooms = zooms.expand([*zooms.shape, dim])
            zooms = zooms.exp() if logzooms else zooms.clamp_min(eps)
            prm = torch.cat((rigid, zooms), dim=-1)
        elif basis == 'GL+':
            checkdim((dim-1)*(dim+1), nb_prm)
            rigid = prm[..., :dim*(dim-1)//2]
            zooms = prm[..., dim*(dim-1)//2:(dim + dim*(dim-1)//2)]
            zooms = zooms.exp() if logzooms else zooms.clamp_min(eps)
            strides = prm[..., (dim + dim*(dim-1)//2):]
            prm = torch.cat((rigid, zooms, strides), dim=-1)
        elif basis == 'Aff+':
            checkdim(dim*(dim+1), nb_prm)
            rigid = prm[..., :(dim + dim*(dim-1)//2)]
            zooms = prm[..., (dim + dim*(dim-1)//2):(2*dim + dim*(dim-1)//2)]
            zooms = zooms.exp() if logzooms else zooms.clamp_min(eps)
            strides = prm[..., (2*dim + dim*(dim-1)//2):]
            prm = torch.cat((rigid, zooms, strides), dim=-1)
        else:
            raise ValueError(f'Unknown basis {basis}')

        return spatial.affine_matrix_classic(prm, dim=dim)


class AffineClassicInverse(Module):
    """Recover affine parameters from an affine matrix."""

    def __init__(self, basis='CSO', logzooms=False):
        """

        Parameters
        ----------
        basis : str, default='CSO'
            Chosen from a list of Lie groups:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
        logzooms : bool, default=False
            If True, this function will return the logarithm of the zooms.
        """
        super().__init__()

        self.basis = basis
        self.logzooms = logzooms

    def forward(self, affine, **overload):
        """

        Parameters
        ----------
        affine : (batch, dim+1, dim+1) tensor
            Affine matrix
        overload : dict
            All parameters of the module can be overridden at call time.

        Returns
        -------
        prm : (batch, nb_prm) tensor
            Parameters

        """
        logzooms = overload.get('logzooms', self.logzooms)
        basis = overload.get('basis', self.basis)

        T, R, Z, S = spatial.affine_parameters_classic(affine,
                                                       return_stacked=False)
        if logzooms:
            Z = Z.log()

        if basis == 'T':
            return T
        elif basis == 'SO':
            return R
        elif basis == 'SE':
            return torch.cat((T, R), dim=-1)
        elif basis == 'D':
            Z = torch.mean(Z, dim=-1)[..., None]
            return torch.cat((T, Z), dim=-1)
        elif basis == 'CSO':
            Z = torch.mean(Z, dim=-1)[..., None]
            return torch.cat((T, R, Z), dim=-1)
        elif basis == 'GL+':
            return torch.cat((R, Z, S), dim=-1)
        elif basis == 'Aff+':
            return torch.cat((T, R, Z, S), dim=-1)
        else:
            raise ValueError(f'Unknown basis {basis}')


class AffineGrid(Module):
    """Generate a dense grid from an affine transform."""

    def __init__(self, shape=None, shift=False):
        """

        Parameters
        ----------
        shape : sequence[int], optional
            Output shape of the dense grid.
        shift : bool, default=False
            Compose the affine with a shift so that the origin is in
            the center of the output field of view.

        """
        super().__init__()
        self.shape = shape
        self.shift = shift

    def forward(self, affine, **overload):
        """

        Parameters
        ----------
        affine : (batch, ndim[+1], ndim+1) tensor
            Affine matrix
        overload : dict
            All parameters of the module can be overridden at call time.

        Returns
        -------
        grid : (batch, *shape, ndim) tensor
            Dense transformation grid

        """

        nb_dim = affine.shape[-1] - 1
        info = {'dtype': affine.dtype, 'device': affine.device}
        shape = make_list(overload.get('shape', self.shape), nb_dim)
        shift = overload.get('shift', self.shift)

        if shift:
            affine_shift = torch.cat((
                torch.eye(nb_dim, **info),
                -torch.as_tensor(shape, **info)[:, None]/2),
                dim=1)
            affine = spatial.affine_matmul(affine, affine_shift)
            affine = spatial.affine_lmdiv(affine_shift, affine)

        grid = spatial.affine_grid(affine, shape)
        return grid


class Resize(Module):
    __doc__ = """
    Resize an image by a factor.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, factor=None, shape=None, anchor='c',
                 interpolation='linear', bound='dct2', extrapolate=True,
                 output_padding=None):
        """

        Parameters
        ----------
        factor : float or list[float], optional
            Resizing factor
            * > 1 : larger image <-> smaller voxels
            * < 1 : smaller image <-> larger voxels
        shape : (ndim,) sequence[int], optional
            Output shape
        anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
            * In cases 'c' and 'e', the volume shape is multiplied by the
              zoom factor (and eventually truncated), and two anchor points
              are used to determine the voxel size.
            * In cases 'f' and 'l', a single anchor point is used so that
              the voxel size is exactly divided by the zoom factor.
              This case with an integer factor corresponds to subslicing
              the volume (e.g., `vol[::f, ::f, ::f]`).
            * A list of anchors (one per dimension) can also be provided.
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.factor = factor
        self.outshape = shape
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.output_padding = output_padding

    def forward(self, image, affine=None, **overload):
        """

        Parameters
        ----------
        image : (batch, channel, *spatial_in) tensor
            Input image to deform
        affine : (batch, ndim[+1], ndim+1), optional
            Orientation matrix of the input image.
            If provided, the orientation matrix of the resized image is
            returned as well.
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        resized : (batch, channel, ...) tensor
            Resized image.
        affine : (batch, ndim[+1], ndim+1) tensor, optional
            Orientation matrix

        """
        outshape = self.shape(image, **overload)
        kwargs = {
            'shape': outshape[2:],
            'factor': overload.get('factor', self.factor),
            'anchor': overload.get('anchor', self.anchor),
            'interpolation': overload.get('interpolation', self.interpolation),
            'bound': overload.get('bound', self.bound),
            'extrapolate': overload.get('extrapolate', self.extrapolate),
        }
        return spatial.resize(image, affine=affine, **kwargs)

    def __str__(self):
        if self.factor:
            if isinstance(self.factor, (list, tuple)):
                factor = self.factor[0] if len(self.factor) == 1 else self.factor
            else:
                factor = self.factor
            if isinstance(factor, (int, float)):
                if factor > 1:
                    if int(factor) == factor:
                        factor = int(factor)
                    return f'Upsample({factor}, anchor={self.anchor})'
                elif factor < 1:
                    factor = 1/factor
                    if int(factor) == factor:
                        factor = int(factor)
                    return f'Downsample({factor}, anchor={self.anchor})'
            return f'Resize({self.factor}, anchor={self.anchor})'
        else:
            return f'ResizeTo({self.shape}, anchor={self.anchor})'

    __repr__ = __str__

    def shape(self, image, affine=None, **overload):
        factor = overload.get('factor', self.factor)
        shape = overload.get('shape', self.outshape),
        output_padding = overload.get('output_padding', self.output_padding)

        # read parameters
        if torch.is_tensor(image):
            inshape = tuple(image.shape)
        else:
            inshape = image
        nb_dim = len(inshape) - 2
        batch = inshape[:2]
        inshape = inshape[2:]
        factor = utils.make_vector(factor or 0., nb_dim).tolist()
        outshape = make_list(shape or [None], nb_dim)
        output_padding = make_list(output_padding or [0], nb_dim)
        output_padding = [p or 0 for p in output_padding]

        # compute output shape
        outshape = [int(inshp * f) if outshp is None else outshp
                    for inshp, outshp, f in zip(inshape, outshape, factor)]
        outshape = [s+p for s, p in zip(outshape, output_padding)]
        return (*batch, *outshape)


class GridResize(Module):
    __doc__ = """
    Resize a transformation/displacement grid by a factor.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, factor=None, shape=None, type='grid', anchor='c',
                 interpolation='linear', bound='dct2', extrapolate=True):
        """

        Parameters
        ----------
        factor : float or list[float], optional
            Resizing factor
            * > 1 : larger image <-> smaller voxels
            * < 1 : smaller image <-> larger voxels
        shape : (ndim,) sequence[int], optional
            Output shape
        type : {'grid', 'displacement'}, default='grid'
            Grid type:
            * 'grid' correspond to dense grids of coordinates.
            * 'displacement' correspond to dense grid of relative displacements.
            Both types are not rescaled in the same way.
        anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
            * In cases 'c' and 'e', the volume shape is multiplied by the
              zoom factor (and eventually truncated), and two anchor points
              are used to determine the voxel size.
            * In cases 'f' and 'l', a single anchor point is used so that
              the voxel size is exactly divided by the zoom factor.
              This case with an integer factor corresponds to subslicing
              the volume (e.g., `vol[::f, ::f, ::f]`).
            * A list of anchors (one per dimension) can also be provided.
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.factor = factor
        self.shape = shape
        self.type = type
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, grid, affine=None, **overload):
        """

        Parameters
        ----------
        grid : (batch, *spatial_in, ndim) tensor
            Input grid to deform
        affine : (batch, ndim[+1], ndim+1), optional
            Orientation matrix of the input image.
            If provided, the orientation matrix of the resized image is
            returned as well.
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        resized : (batch, *spatial_out, ndim) tensor
            Resized image.
        affine : (batch, ndim[+1], ndim+1) tensor, optional
            Orientation matrix

        """
        kwargs = {
            'factor': overload.get('factor', self.factor),
            'shape': overload.get('shape', self.shape),
            'type': overload.get('type', self.type),
            'anchor': overload.get('anchor', self.anchor),
            'interpolation': overload.get('interpolation', self.interpolation),
            'bound': overload.get('bound', self.bound),
            'extrapolate': overload.get('extrapolate', self.extrapolate),
        }
        return spatial.resize_grid(grid, affine=affine, **kwargs)

