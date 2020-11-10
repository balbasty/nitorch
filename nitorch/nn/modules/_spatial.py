"""Spatial transformation layers."""

import torch
from torch import nn as tnn
from ... import spatial
from ._cnn import UNet, CNN
from ._base import Module
from .. import check
from ...core.pyutils import make_list


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
    __doc__ = """
    Pull/Sample an image according to a deformation.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

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
    """Exponentiate an inifinitesimal deformation field (velocity)."""

    def __init__(self, fwd=True, inv=False, steps=None,
                 interpolation='linear', bound='dft', displacement=False,
                 inplace=True):
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
        inplace : bool, default=True
            Perform the integration inplace if possible.
        """
        super().__init__()

        self.fwd = fwd
        self.inv = inv
        self.steps = steps
        self.interpolation = interpolation
        self.bound = bound
        self.displacement = displacement
        self.inplace = inplace

    def forward(self, velocity, **kwargs):
        """

        Parameters
        ----------
        velocity (tensor) : velocity field with shape (batch, *spatial, dim).
        **kwargs : all parameters of the module can be overridden at call time.

        Returns
        -------
        forward (tensor, if `forward is True`) : forward displacement
            (if `displacement is True`) or transformation (if `displacement
            is False`) field, with shape (batch, *spatial, dim)
        inverse (tensor, if `inverse is True`) : forward displacement
            (if `displacement is True`) or transformation (if `displacement
            is False`) field, with shape (batch, *spatial, dim)

        """
        fwd = kwargs.get('fwd', self.forward)
        inv = kwargs.get('inverse', self.inv)
        opt = {
            'steps': kwargs.get('steps', self.steps),
            'interpolation': kwargs.get('interpolation', self.interpolation),
            'bound': kwargs.get('bound', self.bound),
            'displacement': kwargs.get('displacement', self.displacement),
            'inplace': False,  # kwargs.get('inplace', self.inplace)
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
        self.basis, _ = spatial.build_affine_basis(basis, dim)
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
        basis = overload.get('basis', self.basis)
        fwd = overload.get('fwd', self.forward)
        inv = overload.get('inverse', self.inv)

        output = []
        if fwd:
            aff = spatial.affine_matrix(prm, basis, dim=self.dim)
            output.append(aff)
        if inv:
            if isinstance(prm, (list, tuple)):
                prm = [-p for p in prm]
            else:
                prm = -prm
            iaff = spatial.affine_matrix(prm, basis, dim=self.dim)
            output.append(iaff)

        return output if len(output) > 1 else \
               output[0] if len(output) == 1 else \
               None


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
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

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
        kwargs = {
            'factor': overload.get('factor', self.factor),
            'shape': overload.get('shape', self.shape),
            'anchor': overload.get('anchor', self.anchor),
            'interpolation': overload.get('interpolation', self.interpolation),
            'bound': overload.get('bound', self.bound),
            'extrapolate': overload.get('extrapolate', self.extrapolate),
        }
        return spatial.resize(image, affine=affine, **kwargs)


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


class VoxelMorph(Module):
    """VoxelMorph warps a source/moving image to a fixed/target image.

    A VoxelMorph network is obtained by concatenating a UNet and a
    (diffeomorphic) spatial transformer. The loss is made of two terms:
    an image similarity loss and a velocity regularisation loss.

    The UNet used here is slightly different from the original one (we
    use a fully convolutional network -- based on strided convolutions --
    instead of maxpooling and upsampling).

    References
    ----------
    .. [1] "An Unsupervised Learning Model for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        CVPR 2018. eprint arXiv:1802.02604
    .. [2] "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231
    .. [3] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018. eprint arXiv:1805.04605
    .. [4] "Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
    """

    def __init__(self, dim, encoder=None, decoder=None, kernel_size=3,
                 interpolation='linear', grid_bound='dft', image_bound='dct2',
                 downsample_velocity=2, *, _input_channels=2):
        """

        Parameters
        ----------
        dim : int
            Dimensionalityy of the input (1|2|3)
        encoder : list[int], optional
            Number of channels after each encoding layer of the UNet.
        decoder : list[int], optional
            Number of channels after each decoding layer of the Unet.
        kernel_size : int or list[int], default=3
            Kernel size of the UNet.
        interpolation : int, default=1
            Interpolation order.
        grid_bound : bound_type, default='dft'
            Boundary conditions of the velocity field.
        image_bound : bound_type, default='dct2'
            Boundary conditions of the image.
        downsample_velocity : float, default=2
            Downsample the velocity field by a factor when exponentiating.
        """
        resize_factor = make_list(downsample_velocity, dim)
        resize_factor = [1/f for f in resize_factor]

        super().__init__()
        self.unet = UNet(dim,
                         input_channels=_input_channels,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.resize = GridResize(interpolation=interpolation,
                                 bound=grid_bound,
                                 factor=resize_factor)
        self.exp = GridExp(interpolation=interpolation,
                           bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'velocity']

    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        velocity : tensor (batch,, *spatial, len(spatial))
            Velocity field

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim+2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        velocity = self.unet(source_and_target)
        velocity = spatial.channel2grid(velocity)
        velocity_small = self.resize(velocity, type='displacement')
        grid = self.exp(velocity_small)
        grid = self.resize(grid, shape=target.shape[2:], type='grid')
        deformed_source = self.pull(source, grid)

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     velocity=[velocity])

        return deformed_source, velocity


class VoxelMorphSemiSupervised(VoxelMorph):
    """A VoxelMorph network with a Categorical loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags += ['segmentation']

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):

        # sanity checks
        check.dim(self.dim, source, target, source_seg, target_seg)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim+2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim+2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        velocity = self.unet(source_and_target)
        velocity = spatial.channel2grid(velocity)
        velocity_small = self.resize(velocity, type='displacement')
        grid = self.exp(velocity_small)
        grid = self.resize(grid, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)
        if source_seg is not None:
            if source_seg.shape[2:] != source.shape[2:]:
                grid = spatial.resize_grid(grid, shape=source_seg.shape[2:])
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     velocity=[velocity],
                     segmentation=[deformed_source_seg, target_seg])

        if source_seg is None:
            return deformed_source, velocity
        else:
            return deformed_source, deformed_source_seg, velocity


class AffineMorph(Module):
    """Affine registration network.

    This network builds on VoxelMorph, but replaces the U-Net by an
    encoding CNN, and the dense spatial transformer by an affine spatial
    transformer. Like VoxelMorph, this network encodes deformation on
    their tangent space: here, the Lie algebra of a variety of affine
    Lie groups is used. Affine transformation matrices are recovered
    from their Lie algebra representation using matrix exponentiation.

    * VoxelMorph:
        target |-(unet)-> velocity -(exp)-> grid -(pull)-> warped_source
        source |------------------------------------^
    * AffineMorph:
        target |-(cnn)-> lieparam -(exp)-> affine -(pull)-> warped_source
        source |-------------------------------------^
    """

    def __init__(self, dim, basis='CSO', encoder=None, stack=None,
                 kernel_size=3, interpolation='linear', bound='dct2', *,
                 _additional_input_channels=0, _additional_output_channels=0):
        """

        Parameters
        ----------
        dim : int
            Dimensionalityy of the input (1|2|3)
        basis : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='CSO'
            Basis of a matrix Lie group:
                * 'T'   : Translations
                * 'SO'  : Special Orthogonal (rotations)
                * 'SE'  : Special Euclidean (translations + rotations)
                * 'D'   : Dilations (translations + isotropic scalings)
                * 'CSO' : Conformal Special Orthogonal
                          (translations + rotations + isotropic scalings)
                * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
                * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
                * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
        encoder : list[int], optional
            Number of channels after each encoding layer of the CNN.
        stack : list[int], optional
            Number of channels after each fully-connected layer of the CNN.
        kernel_size : int or list[int], default=3
            Kernel size of the UNet.
        interpolation : int, default=1
            Interpolation order.
        bound : bound_type, default='dct2'
            Boundary conditions of the image.
        """

        super().__init__()
        exp = AffineExp(dim, basis=basis)
        nb_prm = sum(b.shape[0] for b in exp.basis) + _additional_output_channels
        self.cnn = CNN(dim,
                       input_channels=2 + _additional_input_channels,
                       output_channels=nb_prm,
                       encoder=encoder,
                       stack=stack,
                       kernel_size=kernel_size,
                       activation=tnn.LeakyReLU(0.2),
                       final_activation=None)
        self.exp = exp
        self.grid = AffineGrid(shift=True)
        self.pull = GridPull(interpolation=interpolation,
                             bound=bound,
                             extrapolate=False)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'affine']

    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : dict, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        affine_prm : tensor (batch,, *spatial, len(spatial))
            affine Lie parameters

        """
        # sanity checks
        check.dim(self.dim, source, target)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim+2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        affine_prm = self.cnn(source_and_target)
        affine_prm = affine_prm.reshape(affine_prm.shape[:2])
        affine = []
        for prm in affine_prm:
            affine.append(self.exp(prm))
        affine = torch.stack(affine, dim=0)
        grid = self.grid(affine, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     affine=[affine_prm])

        return deformed_source, affine_prm


class AffineMorphSemiSupervised(AffineMorph):
    """An AffineMorph network with a Categorical loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags += ['segmentation']

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):

        # sanity checks
        check.dim(self.dim, source, target, source_seg, target_seg)
        check.shape(target, source, dims=[0], broadcast_ok=True)
        check.shape(target, source, dims=range(2, self.dim+2))
        check.shape(target_seg, source_seg, dims=[0], broadcast_ok=True)
        check.shape(target_seg, source_seg, dims=range(2, self.dim+2))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        affine_prm = self.cnn(source_and_target)
        affine_prm = affine_prm.reshape(affine_prm.shape[:2])
        affine = []
        for prm in affine_prm:
            affine.append(self.exp(prm))
        affine = torch.stack(affine, dim=0)
        grid = self.grid(affine, shape=target.shape[2:])
        deformed_source = self.pull(source, grid)
        if source_seg is not None:
            if source_seg.shape[2:] != source.shape[2:]:
                grid = spatial.resize_grid(grid, shape=source_seg.shape[2:])
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     affine=[affine_prm],
                     segmentation=[deformed_source_seg, target_seg])

        if source_seg is None:
            return deformed_source, affine_prm
        else:
            return deformed_source, deformed_source_seg, affine_prm
