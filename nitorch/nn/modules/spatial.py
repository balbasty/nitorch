"""Spatial transformation layers.

GridPull : resample according to a dense transformation
GridPush : splat (adjoint of pull) according to a dense transformation
GridPushCount : splat an image of ones
GridExp : exponentiate a stationary velocity field
GridShoot : exponentiate a velocity field by geodesic shooting
AffineExp : exponentiate an affine matrix from its Lie algebra
AffineLog : project an affine matrix on its Lie algebra
AffineClassic : build affine by composing translations/rotations/etc.
AffineClassicInverse : recover translations/rotations parameters
AffineGrid : build a dense transformation from an affine matrix
Resize : resize an image by a factor
GridResize : resize a grid (transformation or displacement) by a factor
"""

import torch
from nitorch import core, spatial
from nitorch.core import utils
from nitorch.core.py import make_list
from nitorch.nn.base import Module


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

    def forward(self, x, grid):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_out, len(spatial_in)) tensor
            Transformation grid

        Returns
        -------
        pulled : (batch, channel, *spatial_out) tensor
            Deformed image.

        """
        return spatial.grid_pull(x, grid, self.interpolation, self.bound,
                                 self.extrapolate)


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

    def forward(self, x, grid, shape=None):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_out, len(spatial_in)) tensor
            Transformation grid
        shape : list[int], default=self.shape
            Output spatial shape.

        Returns
        -------
        pushed : (batch, channel, *spatial_out) tensor
            Deformed image.

        """
        shape = shape or self.shape
        return spatial.grid_push(x, grid, shape,
                                 interpolation=self.interpolation,
                                 bound=self.bound,
                                 extrapolate=self.extrapolate)


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

    def forward(self, x, grid, shape=None):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_in, dir) tensor
            Transformation grid
        shape : list[int], default=self.shape
            Output spatial shape. Default is the same as the input shape.

        Returns
        -------
        pushed : (batch, channel, *shape) tensor
            Pushed image.
        count : (batch, 1, *shape) tensor
            Pushed image.

        """
        shape = shape or self.shape
        push = spatial.grid_push(x, grid, shape,
                                 interpolation=self.interpolation,
                                 bound=self.bound,
                                 extrapolate=self.extrapolate)
        count = spatial.grid_count(grid, shape,
                                   interpolation=self.interpolation,
                                   bound=self.bound,
                                   extrapolate=self.extrapolate)
        return push, count


class GridExp(Module):
    """Exponentiate a stationary velocity field."""

    def __init__(self, fwd=True, inv=False, steps=8, anagrad=False,
                 interpolation='linear', bound='dft', displacement=False):
        """

        Parameters
        ----------
        fwd : bool, default=True
            Return the forward deformation.
        inv : bool, default=False
            Return the inverse deformation.
        steps : int, default=8
            Number of integration steps.
            Use `1` to use a small displacements model instead of a
            diffeomorphic one. Default is an educated guess based on the
            magnitude of the velocity field.
        anagrad : bool, default=False
            Use analytical gradients. Uses less memory but less accurate.
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
        self.anagrad = anagrad

    def forward(self, velocity, fwd=None, inv=None):
        """

        Parameters
        ----------
        velocity :(batch, *spatial, dim) tensor
            Stationary velocity field.
        fwd : bool, default=self.fwd
        inv : bool, default=self.inv

        Returns
        -------
        forward : (batch, *spatial, dim) tensor, if `fwd is True`
            Forward displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.
        inverse : (batch, *spatial, dim) tensor, if `inv is True`
            Inverse displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.

        """
        fwd = fwd if fwd is not None else self.fwd
        inv = inv if inv is not None else self.inv
        opt = dict(
            steps=self.steps,
            interpolation=self.interpolation,
            bound=self.bound,
            displacement=self.displacement,
            anagrad=self.anagrad)

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

    def __init__(self, fwd=True, inv=False, steps=8, approx=False,
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
        self.approx = approx
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.lame = lame
        self.factor = factor
        self.voxel_size = voxel_size
        self.displacement = displacement
        self.cache_greens = cache_greens

    def forward(self, velocity, fwd=None, inv=None, voxel_size=None):
        """

        Parameters
        ----------
        velocity :(batch, *spatial, dim) tensor
            Initial velocity field.
        fwd : bool, default=self.fwd
        inv : bool, default=self.inv
        voxel_size : sequence[float], default=self.voxel_size

        Returns
        -------
        forward : (batch, *spatial, dim) tensor, if `forward is True`
            Forward displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.
        inverse : (batch, *spatial, dim) tensor, if `inverse is True`
            Inverse displacement (if `displacement is True`) or
            transformation (if `displacement is False`) field.

        """
        fwd = fwd if fwd is not None else self.fwd
        inv = inv if inv is not None else self.inv
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        shoot_opt = {
            'steps': self.steps,
            'displacement': self.displacement,
            'voxel_size': voxel_size,
            'absolute': self.absolute,
            'membrane': self.membrane,
            'bending': self.bending,
            'lame': self.lame,
            'factor': self.factor,
        }
        greens_prm = {
            'absolute': self.absolute,
            'membrane': self.membrane,
            'bending': self.bending,
            'lame': self.lame,
            'factor': self.factor,
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

        shoot_fn = spatial.shoot_approx if self.approx else spatial.shoot
            
        output = []
        if inv:
            y, iy = shoot_fn(velocity, greens, return_inverse=True, **shoot_opt)
            if fwd:
                output.append(y)
            output.append(iy)
        elif fwd:
            y = shoot_fn(velocity, greens, **shoot_opt)
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

    def forward(self, prm, fwd=None, inv=None):
        """

        Parameters
        ----------
        prm : (batch, nb_prm) tensor or list[tensor]
            Affine parameters on the Lie algebra.

        Returns
        -------
        forward : (batch, dim+1, dim+1) tensor, optional
            Forward matrix
        inverse : (batch, dim+1, dim+1) tensor, optional
            Inverse matrix

        """
        fwd = fwd if fwd is not None else self.fwd
        inv = inv if inv is not None else self.inv

        output = []
        if fwd:
            aff = spatial.affine_matrix(prm, self.basis)
            output.append(aff)
        if inv:
            if isinstance(prm, (list, tuple)):
                prm = [-p for p in prm]
            else:
                prm = -prm
            iaff = spatial.affine_matrix(prm, self.basis)
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

    def __init__(self, dim, basis='CSO', **backend):
        """

        Parameters
        ----------
        dim : int
            Number of spatial dimensions
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
        self.basis = spatial.build_affine_basis(basis, dim, **backend)

    def forward(self, affine):
        """

        Parameters
        ----------
        affine : (batch, dim+1, dim+1) tensor

        Returns
        -------
        logaff : (batch, nbprm) tensor

        """
        # When the affine is well conditioned, its log should be real.
        # Here, I take the real part just in case.
        # Another solution could be to regularise the affine (by loading
        # slightly the diagonal) until it is well conditioned -- but
        # how would that work with autograd?
        backend = utils.backend(affine)
        affine = core.linalg.logm(affine.double())
        if affine.is_complex():
            affine = affine.real
        affine = affine.to(**backend)
        basis = self.basis.to(**backend)
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

    def forward(self, prm):
        """

        Parameters
        ----------
        prm : (batch, nb_prm) tensor or list[tensor]
            Affine parameters, ordered as
            (*translations, *rotations, *zooms, *shears).

        Returns
        -------
        affine : (batch, dim+1, dim+1) tensor
            Affine matrix

        """
        def checkdim(expected, got):
            if got != expected:
                raise ValueError(f'Expected {expected} parameters for '
                                 f'group {self.basis}({self.dim}) but '
                                 f'got {got}.')

        nb_prm = prm.shape[-1]
        eps = core.constants.eps(prm.dtype)

        if self.basis == 'T':
            checkdim(self.dim, nb_prm)
        elif self.basis == 'SO':
            checkdim(self.dim*(self.dim-1)//2, nb_prm)
        elif self.basis == 'SE':
            checkdim(self.dim + self.dim*(self.dim-1)//2, nb_prm)
        elif self.basis == 'D':
            checkdim(self.dim + 1, nb_prm)
            translations = prm[..., :self.dim]
            zooms = prm[..., -1]
            zooms = zooms.expand([*zooms.shape, self.dim])
            zooms = zooms.exp() if self.logzooms else zooms.clamp_min(eps)
            prm = torch.cat((translations, zooms), dim=-1)
        elif self.basis == 'CSO':
            checkdim(self.dim + self.dim*(self.dim-1)//2 + 1, nb_prm)
            rigid = prm[..., :-1]
            zooms = prm[..., -1]
            zooms = zooms.expand([*zooms.shape, self.dim])
            zooms = zooms.exp() if self.logzooms else zooms.clamp_min(eps)
            prm = torch.cat((rigid, zooms), dim=-1)
        elif self.basis == 'GL+':
            checkdim((self.dim-1)*(self.dim+1), nb_prm)
            rigid = prm[..., :self.dim*(self.dim-1)//2]
            zooms = prm[..., self.dim*(self.dim-1)//2:(self.dim + self.dim*(self.dim-1)//2)]
            zooms = zooms.exp() if self.logzooms else zooms.clamp_min(eps)
            strides = prm[..., (self.dim + self.dim*(self.dim-1)//2):]
            prm = torch.cat((rigid, zooms, strides), dim=-1)
        elif self.basis == 'Aff+':
            checkdim(self.dim*(self.dim+1), nb_prm)
            rigid = prm[..., :(self.dim + self.dim*(self.dim-1)//2)]
            zooms = prm[..., (self.dim + self.dim*(self.dim-1)//2):(2*self.dim + self.dim*(self.dim-1)//2)]
            zooms = zooms.exp() if self.logzooms else zooms.clamp_min(eps)
            strides = prm[..., (2*self.dim + self.dim*(self.dim-1)//2):]
            prm = torch.cat((rigid, zooms, strides), dim=-1)
        else:
            raise ValueError(f'Unknown basis {self.basis}')

        return spatial.affine_matrix_classic(prm, dim=self.dim)


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

    def forward(self, affine):
        """

        Parameters
        ----------
        affine : (batch, dim+1, dim+1) tensor
            Affine matrix

        Returns
        -------
        prm : (batch, nb_prm) tensor
            Parameters

        """
        T, R, Z, S = spatial.affine_parameters_classic(affine,
                                                       return_stacked=False)
        if self.logzooms:
            Z = Z.log()

        if self.basis == 'T':
            return T
        elif self.basis == 'SO':
            return R
        elif self.basis == 'SE':
            return torch.cat((T, R), dim=-1)
        elif self.basis == 'D':
            Z = torch.mean(Z, dim=-1)[..., None]
            return torch.cat((T, Z), dim=-1)
        elif self.basis == 'CSO':
            Z = torch.mean(Z, dim=-1)[..., None]
            return torch.cat((T, R, Z), dim=-1)
        elif self.basis == 'GL+':
            return torch.cat((R, Z, S), dim=-1)
        elif self.basis == 'Aff+':
            return torch.cat((T, R, Z, S), dim=-1)
        else:
            raise ValueError(f'Unknown basis {self.basis}')


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

    def forward(self, affine, shape=None):
        """

        Parameters
        ----------
        affine : (batch, ndim[+1], ndim+1) tensor
            Affine matrix
        shape : sequence[int], default=self.shape

        Returns
        -------
        grid : (batch, *shape, ndim) tensor
            Dense transformation grid

        """

        nb_dim = affine.shape[-1] - 1
        backend = {'dtype': affine.dtype, 'device': affine.device}
        shape = shape or self.shape

        if self.shift:
            affine_shift = torch.eye(nb_dim+1, **backend)
            affine_shift[:nb_dim, -1] = torch.as_tensor(shape, **backend)
            affine_shift[:nb_dim, -1].sub(1).div(2).neg()
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
                 prefilter=True):
        """

        Parameters
        ----------
        factor : float or sequence[float], optional
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
        prefilter : bool, default=True
            Apply spline prefilter.
        """
        super().__init__()
        self.factor = factor
        self.output_shape = shape
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.prefilter = prefilter

    def forward(self, image, affine=None, output_shape=None):
        """

        Parameters
        ----------
        image : (batch, channel, *spatial_in) tensor
            Input image to deform
        affine : (batch, ndim[+1], ndim+1), optional
            Orientation matrix of the input image.
            If provided, the orientation matrix of the resized image is
            returned as well.
        output_shape : sequence[int], optional

        Returns
        -------
        resized : (batch, channel, ...) tensor
            Resized image.
        affine : (batch, ndim[+1], ndim+1) tensor, optional
            Orientation matrix

        """
        outshape = self.shape(image, output_shape=output_shape)
        kwargs = {
            'shape': outshape[2:],
            'factor': self.factor,
            'anchor': self.anchor,
            'interpolation': self.interpolation,
            'bound': self.bound,
            'extrapolate': self.extrapolate,
            'prefilter': self.prefilter,
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

    def shape(self, image, affine=None, output_shape=None):
        output_shape = output_shape or self.shape

        # read parameters
        if torch.is_tensor(image):
            inshape = tuple(image.shape)
        else:
            inshape = image
        nb_dim = len(inshape) - 2
        batch = inshape[:2]
        inshape = inshape[2:]
        factor = utils.make_vector(self.factor or 0., nb_dim).tolist()
        output_shape = make_list(output_shape or [None], nb_dim)

        # compute output shape
        output_shape = [int(inshp * f) if outshp is None else outshp
                        for inshp, outshp, f in zip(inshape, output_shape, factor)]
        return (*batch, *output_shape)


class GridResize(Module):
    __doc__ = f"""
    Resize a transformation/displacement grid by a factor.

    This module has no learnable parameters.

    {_interpolation_doc}

    {_bound_doc}
    
    """

    def __init__(self, factor=None, shape=None, type='grid', anchor='c',
                 interpolation='linear', bound='dct2', extrapolate=True,
                 prefilter=True):
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
        prefilter : bool, default=True
            Apply spline prefilter.
        """
        super().__init__()
        self.factor = factor
        self.shape = shape
        self.type = type
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.prefilter = prefilter

    def forward(self, grid, affine=None, output_shape=None, **overload):
        """

        Parameters
        ----------
        grid : (batch, *spatial_in, ndim) tensor
            Input grid to deform
        affine : (batch, ndim[+1], ndim+1), optional
            Orientation matrix of the input image.
            If provided, the orientation matrix of the resized image is
            returned as well.
        output_shape : bool, optional

        Returns
        -------
        resized : (batch, *spatial_out, ndim) tensor
            Resized image.
        affine : (batch, ndim[+1], ndim+1) tensor, optional
            Orientation matrix

        """
        output_shape = output_shape or self.shape
        kwargs = {
            'factor': overload.get('factor', self.factor),
            'shape': output_shape,
            'type': self.type,
            'anchor': self.anchor,
            'interpolation': self.interpolation,
            'bound': self.bound,
            'extrapolate': self.extrapolate,
            'prefilter': self.prefilter
        }
        return spatial.resize_grid(grid, affine=affine, **kwargs)

