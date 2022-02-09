import math
import torch
from nitorch.core import utils, py, linalg
from nitorch.spatial import affine_matrix_classic, affine_matmul, affine_lmdiv, as_euclidean, identity_grid, smooth, resize
from nitorch.nn.base import Module
from ..modules.spatial import GridExp, GridPull
from .field import HyperRandomFieldSpline
from .distribution import _get_dist

__all__ = ['RandomVelocity', 'RandomDiffeo', 'RandomAffineMatrix', 'RandomGrid',
           'RandomDeform', 'RandomPatch', 'RandomFlip', 'RandomSmooth',
           'RandomLowRes2D', 'RandomLowRes3D', 'RandomRubiks', 'RandomPatchSwap',
           'RandomInpaint', 'HyperRandomRubiks', 'HyperRandomPatchSwap',
           'HyperRandomInpaint']


defaults = dict(
    vel_amplitude_exp=3,
    vel_amplitude_scale=1,
    vel_amplitude='lognormal',
    vel_fwhm_exp=20,
    vel_fwhm_scale=10,
    vel_fwhm='lognormal',
)


class RandomVelocity(Module):
    """Sample a random velocity field with randomized hyper-parameters."""

    def __init__(self, shape=None,
                 amplitude=defaults['vel_amplitude'],
                 amplitude_exp=defaults['vel_amplitude_exp'],
                 amplitude_scale=defaults['vel_amplitude_scale'],
                 fwhm=defaults['vel_fwhm'],
                 fwhm_exp=defaults['vel_fwhm_exp'],
                 fwhm_scale=defaults['vel_fwhm_scale'],
                 device=None, dtype=None):
        """
        The geometry of a random field is controlled by two parameters:
            - `amplitude` controls the voxel-wise variance of the field.
            - `fwhm` controls the smoothness of the field.

        Each of these parameter is sampled according to three hyper-parameters:
            - <param>       : distribution family
                              {'normal', 'lognormal', 'uniform', 'gamma', None}
            - <param>_exp   : expected value of the parameter
            - <param>_scale : standard deviation of the parameter

        Parameters
        ----------
        shape : sequence[int]
        amplitude : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        amplitude_exp : float or (dim,) vector_like, default=3
        amplitude_scale : float or (dim,) vector_like, default=1
        fwhm : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        fwhm_exp : float or (dim,) vector_like, default=20
        fwhm_scale : float or (dim,) vector_like, default=10
        device : torch.device, optional
        dtype : torch.dtype, optional

        """
        super().__init__()
        shape = py.make_list(shape)
        self.field = HyperRandomFieldSpline(
            shape=shape, channel=len(shape), amplitude=amplitude,
            amplitude_exp=amplitude_exp, amplitude_scale=amplitude_scale,
            fwhm=fwhm, fwhm_exp=fwhm_exp, fwhm_scale=fwhm_scale,
            device=device, dtype=dtype)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size

        Other Parameters
        ----------------
        shape : sequence[int], optional
        device : torch.device, optional
        dtype : torch.dtype, optional

        Returns
        -------
        vel : (batch, *shape, dim) tensor
            Velocity field

        """
        overload['channel'] = len(overload.get('shape', self.field.shape))
        return utils.channel2last(self.field(batch, **overload))


class RandomDiffeo(Module):
    """Sample a random diffeomorphic field with randomized hyper-parameters."""

    def __init__(self, shape=None,
                 amplitude=defaults['vel_amplitude'],
                 amplitude_exp=defaults['vel_amplitude_exp'],
                 amplitude_scale=defaults['vel_amplitude_scale'],
                 fwhm=defaults['vel_fwhm'],
                 fwhm_exp=defaults['vel_fwhm_exp'],
                 fwhm_scale=defaults['vel_fwhm_scale'],
                 bound='dft', interpolation=1, displacement=False,
                 device=None, dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int]
        amplitude : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        amplitude_exp : float or (dim,) vector_like, default=3
        amplitude_scale : float or (dim,) vector_like, default=1
        fwhm : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        fwhm_exp : float or (dim,) vector_like, default=20
        fwhm_scale : float or (dim,) vector_like, default=10
        bound : BoundType, default='dft'
            Boundary condition when exponentiating the velocity field.
        interpolation : InterpolationType, default=1
            Interpolation order when exponentiating the velocity field
        displacement : bool, default=False
            Return a displacement rather than a transformation grid.
        device : torch.device, optional
        dtype : torch.dtype, optional

        """
        super().__init__()
        self.velocity = RandomVelocity(
            shape=shape, amplitude=amplitude,
            amplitude_exp=amplitude_exp, amplitude_scale=amplitude_scale,
            fwhm=fwhm, fwhm_exp=fwhm_exp, fwhm_scale=fwhm_scale,
            device=device, dtype=dtype)
        self.exp = GridExp(bound=bound, interpolation=interpolation,
                           displacement=displacement)

    def forward(self, batch=1, return_vel=False, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size
        return_vel : bool, default=False
            Return the velocity on top of the exponentiated grid

        Other Parameters
        ----------------
        shape : sequence[int], optional
        device : torch.device, optional
        dtype : torch.dtype, optional

        Returns
        -------
        grid : (batch, *shape, dim) tensor
            Transformation or displacement field

        """
        vel = self.velocity(batch, **overload)
        grid = self.exp(vel)
        return (grid, vel) if return_vel else grid


class RandomAffineMatrix(Module):
    """Sample an affine transformation matrix"""

    def __init__(self, dim=None,
                 translation='normal',
                 translation_exp=0,
                 translation_scale=5,
                 rotation='normal',
                 rotation_exp=0,
                 rotation_scale=5,
                 zoom='lognormal',
                 zoom_exp=1,
                 zoom_scale=1.2,
                 shear='normal',
                 shear_exp=0,
                 shear_scale=0.1,
                 device=None, dtype=None):
        """
        An affine transformation is parameterized by D*(D+1) parameters:
            - 'translation` : D parameters
            - `rotation`    : D*(D-1)/2 parameters
            - `zoom`        : D parameters
            - `shear`       : D*(D-1)/2 parameters

        Each of these parameter is sampled according to three hyper-parameters:
            - <param>       : distribution family
                              {'normal', 'lognormal', 'uniform', 'gamma', None}
            - <param>_exp   : expected value of the parameter
            - <param>_scale : standard deviation of the parameter

        Parameters
        ----------
        dim : int, optional, Number of spatial dimensions
        translation : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        translation_exp : float or vector_like, default=0
        translation_scale : float or vector_like, default=5
        rotation : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        rotation_exp : float or vector_like, default=0
        rotation_scale : float or vector_like, default=5
        zoom : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        zoom_exp : float or vector_like, default=1
        zoom_scale : float or vector_like, default=1.2
        shear : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        shear_exp : float or  vector_like, default=0
        shear_scale : float or vector_like, default=0.1
        device : torch.device, optional
        dtype : torch.dtype, optional

        """
        super().__init__()
        self.dim = dim
        self.translation = _get_dist(translation)
        self.translation_exp = translation_exp
        self.translation_scale = translation_scale
        self.rotation = _get_dist(rotation)
        self.rotation_exp = rotation_exp
        self.rotation_scale = rotation_scale
        self.zoom = _get_dist(zoom)
        self.zoom_exp = zoom_exp
        self.zoom_scale = zoom_scale
        self.shear = _get_dist(shear)
        self.shear_exp = shear_exp
        self.shear_scale = shear_scale
        self.device = device
        if dtype is None or not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        self.dtype = dtype

    def _make_sampler(self, name, dim, **backend):
        exp = getattr(self, name + '_exp')
        scale = getattr(self, name + '_scale')
        dist = getattr(self, name)
        ndim = (dim if name in ('translation', 'zoom')
                else dim*(dim-1)//2)
        exp = utils.make_vector(exp, ndim, **backend)
        scale = utils.make_vector(scale, ndim, **backend)
        if dist and (scale > 0).all():
            sampler = dist(exp, scale)
        else:
            sampler = _get_dist('dirac')(exp)
        return sampler

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size

        Other Parameters
        ----------------
        dim : int, optional
        device : torch.device, optional
        dtype : torch.dtype, optional

        Returns
        -------
        affine : (batch, dim+1, dim+1) tensor
            Affine matrix

        """
        dim = overload.get('dim', self.dim)
        dtype = overload.get('dtype', self.dtype)
        device = overload.get('device', self.device)
        backend = dict(dtype=dtype, device=device)

        # prepare sampler
        translation = self._make_sampler('translation', dim, **backend)
        rotation = self._make_sampler('rotation', dim, **backend)
        zoom = self._make_sampler('zoom', dim, **backend)
        shear = self._make_sampler('shear', dim, **backend)

        # sample parameters
        prm = torch.cat([
            translation.sample([batch]),
            rotation.sample([batch]).mul_(math.pi/180),
            zoom.sample([batch]),
            shear.sample([batch]),
        ], dim=-1)

        # generate affine matrix
        mat = affine_matrix_classic(prm, dim=dim)
        return mat


RandomAffine = RandomAffineMatrix


class RandomGrid(Module):
    """Random spatial deformation (dense + affine)."""

    def __init__(self, shape=None,
                 amplitude=defaults['vel_amplitude'],
                 amplitude_exp=defaults['vel_amplitude_exp'],
                 amplitude_scale=defaults['vel_amplitude_scale'],
                 fwhm=defaults['vel_fwhm'],
                 fwhm_exp=defaults['vel_fwhm_exp'],
                 fwhm_scale=defaults['vel_fwhm_scale'],
                 translation='normal',
                 translation_exp=0,
                 translation_scale=5,
                 rotation='normal',
                 rotation_exp=0,
                 rotation_scale=5,
                 zoom='lognormal',
                 zoom_exp=1,
                 zoom_scale=1.2,
                 shear='normal',
                 shear_exp=0,
                 shear_scale=0.1,
                 bound='dft', interpolation=1, displacement=False,
                 device=None, dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int]
        amplitude : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        amplitude_exp : float or (dim,) vector_like, default=3
        amplitude_scale : float or (dim,) vector_like, default=1
        fwhm : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        fwhm_exp : float or (dim,) vector_like, default=20
        fwhm_scale : float or (dim,) vector_like, default=10
        translation : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        translation_exp : float or vector_like, default=0
        translation_scale : float or vector_like, default=5
        rotation : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        rotation_exp : float or vector_like, default=0
        rotation_scale : float or vector_like, default=5
        zoom : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        zoom_exp : float or vector_like, default=1
        zoom_scale : float or vector_like, default=1.2
        shear : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        shear_exp : float or  vector_like, default=0
        shear_scale : float or vector_like, default=0.1
        bound : BoundType, default='dft'
        interpolation : InterpolationType, default=1
        displacement : bool, default=False
        device : torch.device, optional
        dtype : torch.dtype, optional

        """
        super().__init__()
        self.affine = RandomAffineMatrix(
            dim=len(shape) if shape else 0,
            translation=translation,
            translation_exp=translation_exp,
            translation_scale=translation_scale,
            rotation=rotation,
            rotation_exp=rotation_exp,
            rotation_scale=rotation_scale,
            zoom=zoom,
            zoom_exp=zoom_exp,
            zoom_scale=zoom_scale,
            shear=shear,
            shear_exp=shear_exp,
            shear_scale=shear_scale,
            device=device,
            dtype=dtype)
        self.grid = RandomDiffeo(
            shape=shape,
            amplitude=amplitude,
            amplitude_exp=amplitude_exp,
            amplitude_scale=amplitude_scale,
            fwhm_exp=fwhm_exp,
            fwhm=fwhm,
            fwhm_scale=fwhm_scale,
            bound=bound,
            interpolation=interpolation,
            device=device,
            dtype=dtype)
        self.displacement = displacement

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch shape.

        Other Parameters
        ----------------
        shape : sequence[int], optional
        device : torch.device, optional
        dtype : torch.dtype, optional

        Returns
        -------
        grid : (batch, *shape, 3) tensor
            Resampling grid

        """
        shape = overload.get('shape', self.grid.velocity.field.shape)
        dtype = overload.get('dtype', self.grid.velocity.field.dtype)
        device = overload.get('device', self.grid.velocity.field.device)
        backend = dict(dtype=dtype, device=device)

        if self.grid.velocity.field.amplitude == 0:
            grid = identity_grid(shape, **backend)
        else:
            grid = self.grid(batch, shape=shape, **backend)
        dtype = grid.dtype
        device = grid.device
        backend = dict(dtype=dtype, device=device)

        shape = grid.shape[1:-1]
        dim = len(shape)
        aff = self.affine(batch, dim=dim, **backend)

        # shift center of rotation
        aff_shift = torch.cat((
            torch.eye(dim, **backend),
            torch.as_tensor(shape, **backend)[:, None].sub_(1).div_(-2)),
            dim=1)
        aff_shift = as_euclidean(aff_shift)

        aff = affine_matmul(aff, aff_shift)
        aff = affine_lmdiv(aff_shift, aff)

        # compose
        aff = utils.unsqueeze(aff, dim=-3, ndim=dim)
        lin = aff[..., :dim, :dim]
        off = aff[..., :dim, -1]
        grid = linalg.matvec(lin, grid) + off

        return grid


class RandomDeform(Module):
    """Random spatial deformation of an image"""

    def __init__(self,
                 amplitude=defaults['vel_amplitude'],
                 amplitude_exp=defaults['vel_amplitude_exp'],
                 amplitude_scale=defaults['vel_amplitude_scale'],
                 fwhm=defaults['vel_fwhm'],
                 fwhm_exp=defaults['vel_fwhm_exp'],
                 fwhm_scale=defaults['vel_fwhm_scale'],
                 translation='normal',
                 translation_exp=0,
                 translation_scale=5,
                 rotation='normal',
                 rotation_exp=0,
                 rotation_scale=5,
                 zoom='lognormal',
                 zoom_exp=1,
                 zoom_scale=1.2,
                 shear='normal',
                 shear_exp=0,
                 shear_scale=0.1,
                 vel_bound='dft', image_bound='dct2', interpolation=1):
        """

        Parameters
        ----------
        amplitude : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        amplitude_exp : float or (dim,) vector_like, default=3
        amplitude_scale : float or (dim,) vector_like, default=1
        fwhm : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        fwhm_exp : float or (dim,) vector_like, default=20
        fwhm_scale : float or (dim,) vector_like, default=10
        translation : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        translation_exp : float or vector_like, default=0
        translation_scale : float or vector_like, default=5
        rotation : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        rotation_exp : float or vector_like, default=0
        rotation_scale : float or vector_like, default=5
        zoom : {'normal', 'lognormal', 'uniform', 'gamma'}, default='lognormal'
        zoom_exp : float or vector_like, default=1
        zoom_scale : float or vector_like, default=1.2
        shear : {'normal', 'lognormal', 'uniform', 'gamma'}, default='normal'
        shear_exp : float or  vector_like, default=0
        shear_scale : float or vector_like, default=0.1
        bound : BoundType, default='dft'
        vel_bound : BoundType, default='dft'
            Boundary condition when exponentiating the velocity field.
        image_bound : BoundType, default='dct2'
            Boundary condition when sampling the image.
        interpolation : InterpolationType, default=1
            Interpolation order

        """
        super().__init__()
        self.grid = RandomGrid(
            amplitude=amplitude,
            amplitude_exp=amplitude_exp,
            amplitude_scale=amplitude_scale,
            fwhm=fwhm,
            fwhm_exp=fwhm_exp,
            fwhm_scale=fwhm_scale,
            translation=translation,
            translation_exp=translation_exp,
            translation_scale=translation_scale,
            rotation=rotation,
            rotation_exp=rotation_exp,
            rotation_scale=rotation_scale,
            zoom=zoom,
            zoom_exp=zoom_exp,
            zoom_scale=zoom_scale,
            shear=shear,
            shear_exp=shear_exp,
            shear_scale=shear_scale,
            bound=vel_bound,
            interpolation=interpolation)
        self.pull = GridPull(
            bound=image_bound,
            interpolation=interpolation,
            extrapolate=True)

    def forward(self, *images, return_grid=False):
        """

        Parameters
        ----------
        *images : (batch, channel, *shape) tensor
            Input images
        return_grid : bool, default=False
            Return deformation grid on top of deformed sample.

        Returns
        -------
        warped : (batch, channel, *shape) tensor
            Deformed image
        grid : (batch, *shape, 3) tensor, if `return_grid`
            Resampling grid

        """
        batch, channel, *shape = images[0].shape

        dtype = None
        for image in images:
            if image.dtype.is_floating_point:
                dtype = image.dtype
                break
        if dtype is None:
            dtype = torch.get_default_dtype()

        # get arguments
        opt_grid = {
            'shape': shape,
            'dtype': dtype,
            'device': images[0].device,
        }

        # pull
        grid = self.grid(batch, **opt_grid)
        warped = []
        for image in images:
            warped.append(self.pull(image, grid))

        return ((*warped, grid) if return_grid
                else warped[0] if len(warped) == 1
                else tuple(warped))


class RandomPatch(Module):
    """Extract a random patch from a tensor.

    The patch location is different in each batch element.
    Multiple images of the same shape can be provided, such that identical
    patches are extracted across all images.
    """

    def __init__(self, shape):
        """

        Parameters
        ----------
        shape : sequence[int]
            Patch shape
        """

        super().__init__()
        self.shape = shape

    def forward(self, *image, **overload):
        """

        Parameters
        ----------
        *image : (batch, channel, *spatial)
        **overload : dict

        Returns
        -------
        *image : (batch, channel, *patch_shape)

        """

        image, *other_images = image
        image = torch.as_tensor(image)
        device = image.device

        dim = image.dim() - 2
        shape = py.make_list(overload.get('shape', self.shape), dim)
        shape = [min(s0, s1) for s0, s1 in zip(image.shape[2:], shape)]

        # sample shift
        max_shift = [d0 - d1 for d0, d1 in zip(image.shape[2:], shape)]
        shift = [[torch.randint(0, s, [], device=device) if s > 0 else 0
                  for s in max_shift] for _ in range(len(image))]

        output = image.new_empty([*image.shape[:2], *shape])
        other_outputs = [im.new_empty([*im.shape[:2], *shape])
                         for im in other_images]

        for b in range(len(image)):
            # subslice
            index = (b, slice(None))  # batch, channel
            index = index + tuple(slice(s, s+d) for s, d in zip(shift[b], shape))
            output[b] = image[index]
            for i in range(len(other_images)):
                other_outputs[i][b] = other_images[i][index]

        if len(other_images) > 0:
            return (output, *other_outputs)
        else:
            return output


class RandomFlip(Module):
    """Apply a random flip to a tensor."""

    def __init__(self, prob=0.5, dim=None):
        """

        Parameters
        ----------
        prob : float or sequence[float]
            Probability yo flip (per spatial dimension)
        dim : int or sequence[int], default=all
            Index of spatial dimension to flip
        """

        super().__init__()
        self.prob = prob
        self.dim = dim

    def forward(self, *image, **overload):
        """

        Parameters
        ----------
        image : (batch, channel, *spatial)
        overload

        Returns
        -------

        """

        image = list(image)
        device = image[0].device

        nb_dim = image[0].dim() - 2
        prob = utils.make_vector(overload.get('prob', self.prob),
                                 dtype=torch.float, device=device)
        dim = overload.get('dim', self.dim)
        dim = py.make_list(dim or range(-nb_dim, 0), nb_dim)

        # sample shift
        flip = torch.rand((nb_dim,), device=device) > (1 - prob)
        dim = [d for d, f in zip(dim, flip) if f]

        if dim:
            for i, img in enumerate(image):
                image[i] = img.flip(dim)
        return image[0] if len(image) == 1 else tuple(image)


class RandomSmooth(Module):

    def __init__(self,
                 fwhm='lognormal',
                 fwhm_exp=1,
                 fwhm_scale=3,
                 iso=False):
        super().__init__()
        self.fwhm = _get_dist(fwhm)
        self.fwhm_exp = fwhm_exp
        self.fwhm_scale = fwhm_scale
        self.iso = iso

    def forward(self, x):
        dim = x.dim() - 2
        backend = dict(dtype=x.dtype, device=x.device)

        fwhm_exp = utils.make_vector(self.fwhm_exp, 1 if self.iso else dim, **backend)
        fwhm_scale = utils.make_vector(self.fwhm_scale, 1 if self.iso else dim, **backend)

        out = torch.as_tensor(x)
        for b in range(len(x)):
            fwhm = self.fwhm(fwhm_exp, fwhm_scale).sample().clamp_min_(0).expand([dim]).clone()
            out[b] = smooth(x[b], fwhm=fwhm, dim=dim, padding='same', bound='dct2')
        return out


class RandomLowRes2D(Module):

    def __init__(self,
                 resolution='lognormal',
                 resolution_exp=3,
                 resolution_scale=3):
        super().__init__()
        self.resolution = _get_dist(resolution)
        self.resolution_exp = resolution_exp
        self.resolution_scale = resolution_scale

    def forward(self, x, noise=None, return_resolution=False):

        if noise is not None:
            noise = noise.expand(x.shape)

        dim = x.dim() - 2
        backend = utils.backend(x)
        resolution_exp = utils.make_vector(self.resolution_exp, x.shape[1],
                                           **backend)
        resolution_scale = utils.make_vector(self.resolution_scale, x.shape[1],
                                             **backend)

        all_resolutions = []
        out = torch.empty_like(x)
        for b in range(len(x)):
            for c in range(x.shape[1]):
                resolution = self.resolution(resolution_exp[c],
                                             resolution_scale[c]).sample()
                resolution = resolution.clamp_min(1)
                axis = torch.randint(dim, [])
                gap = torch.rand([], **backend)
                fwhm = [0] * dim
                fwhm[axis] = resolution * gap
                y = smooth(x[b, c], fwhm=fwhm, dim=dim, padding='same', bound='dct2')
                if noise is not None:
                    y += noise[b, c]
                factor = [1] * dim
                factor[axis] = 1/resolution
                y = y[None, None]  # need batch and channel for resize
                y = resize(y, factor=factor, anchor='f')
                factor = [1] * dim
                factor[axis] = resolution
                all_resolutions.append(factor)
                y = resize(y, factor=factor, shape=x.shape[2:], anchor='f')
                out[b, c] = y[0, 0]

        all_resolutions = utils.as_tensor(all_resolutions, **backend)
        return (out, all_resolutions) if return_resolution else out


class RandomLowRes3D(Module):

    def __init__(self,
                 resolution='lognormal',
                 resolution_exp=3,
                 resolution_scale=3):
        super().__init__()
        self.resolution = _get_dist(resolution)
        self.resolution_exp = resolution_exp
        self.resolution_scale = resolution_scale

    def forward(self, x, noise=None, return_resolution=False):

        if noise is not None:
            noise = noise.expand(x.shape)

        dim = x.dim() - 2
        backend = utils.backend(x)
        resolution_exp = utils.make_vector(self.resolution_exp, x.shape[1],
                                           **backend)
        resolution_scale = utils.make_vector(self.resolution_scale, x.shape[1],
                                             **backend)

        all_resolutions = []
        out = torch.empty_like(x)
        for b in range(len(x)):
            for c in range(x.shape[1]):
                resolution = self.resolution(resolution_exp[c],
                                             resolution_scale[c]).sample()
                resolution = resolution.clamp_min(1)
                fwhm = [resolution] * dim
                y = smooth(x[b, c], fwhm=fwhm, dim=dim, padding='same', bound='dct2')
                if noise is not None:
                    y += noise[b, c]
                factor = [1/resolution] * dim
                y = y[None, None]  # need batch and channel for resize
                y = resize(y, factor=factor, anchor='f')
                factor = [resolution] * dim
                all_resolutions.append(factor)
                y = resize(y, factor=factor, shape=x.shape[2:], anchor='f')
                out[b, c] = y[0, 0]

        all_resolutions = utils.as_tensor(all_resolutions, **backend)
        return (out, all_resolutions) if return_resolution else out


class RandomRubiks(Module):
    """
    Shuffle image by cube/squares to re-organise as a self-supervised pretraining task.
    Also works for anisotropic data with non-regular block shapes.

    References
    ----------
    ..[1] "Revisiting Rubik's Cube: Self-supervised Learning with 
            Volume-wise Transformation for 3D Medical Image Segmentation"
            Xing Tao, Yuexiang Li, Wenhui Zhou, Kai Ma, Yefeng Zheng
            MICCAI 2020
            https://arxiv.org/abs/2007.08826

            
    """
    def __init__(self,
                 dim,
                 kernel=[32,32,32]):
        """
        Arguments:
            kernel (int or sequence[int]): Kernel size for blocks to shuffle. Can be
                single int (isotropic) or specified in all dimensions. Default = [32,32,32]
        """
        
        super().__init__()
        if isinstance(kernel, int):
            kernel = [kernel] * dim
        self.kernel = kernel

    def forward(self, x):
        shape = x.shape[2:]
        dim = len(shape)
        pshape = [x+(k-x%k) for x,k in zip(shape,self.kernel)]
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(pshape))
        x = utils.unfold(x, self.kernel, collapse=True)
        x = x[:, :, torch.randperm(x.shape[2])]
        x = utils.fold(x, dim=dim, stride=self.kernel, collapsed=True, shape=pshape)
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(shape))
        return x


class HyperRandomRubiks(Module):
    """
    Shuffle image by cube/squares to re-organise as a self-supervised pretraining task.
    Also works for anisotropic data with non-regular block shapes.

    References
    ----------
    ..[1] "Revisiting Rubik's Cube: Self-supervised Learning with 
            Volume-wise Transformation for 3D Medical Image Segmentation"
            Xing Tao, Yuexiang Li, Wenhui Zhou, Kai Ma, Yefeng Zheng
            MICCAI 2020
            https://arxiv.org/abs/2007.08826

            
    """
    def __init__(self,
                 kernel='normal',
                 kernel_exp=32,
                 kernel_scale=16):
        """
        Arguments:
            kernel (int or sequence[int]): Kernel size for blocks to shuffle. Can be
                single int (isotropic) or specified in all dimensions. Default = [32,32,32]
        """
        
        super().__init__()
        self.kernel = _get_dist(kernel)
        self.kernel_exp = kernel_exp
        self.kernel_scale = kernel_scale

    def forward(self, x):
        dim = x.dim() - 2
        backend = utils.backend(x)
        kernel_exp = utils.make_vector(self.kernel_exp, dim,
                                           **backend)
        kernel_scale = utils.make_vector(self.kernel_scale, dim,
                                             **backend)

        kernel = [self.kernel(k_e, k_s).sample() for k_e,k_s in zip(kernel_exp, kernel_scale)]
        shape = x.shape[2:]
        kernel = [torch.clamp(k, min=4, max=shape[i]).int().item() for i,k in enumerate(kernel)]
        pshape = [x+(k-x%k) for x,k in zip(shape,kernel)]
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(pshape))
        x = utils.unfold(x, kernel, collapse=True)
        x = x[:, :, torch.randperm(x.shape[2])]
        x = utils.fold(x, dim=dim, stride=kernel, collapsed=True, shape=pshape)
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(shape))
        return x


class RandomPatchSwap(Module):
    """
    Swap random patches in image as self-supervised pretraining task.
    Also works for anisotropic data with non-regular block shapes.

    References
    ----------
    ..[1] "Self-supervised learning for medical image analysis using image context restoration"
            Liang Chen, Paul Bentley, Kensaku Mori, Kazunari Misawa, Michitaka Fujiwara, Daniel Rueckert
            Medical Image Analysis 2019
            https://doi.org/10.1016/j.media.2019.101539

            
    """
    def __init__(self,
                 dim,
                 kernel=[32,32,32],
                 nb_swap=4):
        """
        Arguments:
            kernel (int or sequence[int]): Kernel size for blocks to shuffle. Can be
                single int (isotropic) or specified in all dimensions. Default = [32,32,32]
            nb_swap (int): Number of times to swap two random patches. Default = 4
        """
        
        super().__init__()
        if isinstance(kernel, int):
            kernel = [kernel] * dim
        self.kernel = kernel
        self.nb_swap = nb_swap

    def forward(self, x):
        shape = x.shape[2:]
        dim = len(shape)
        pshape = [x+(k-x%k) for x,k in zip(shape,self.kernel)]
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(pshape))
        x = utils.unfold(x, self.kernel, collapse=True)
        for n in range(self.nb_swap):
            i1, i2 = torch.randint(low=0, high=x.shape[2]-1, size=(2,)).tolist()
            x[:,:,i1], x[:,:,i2] = x[:,:,i2], x[:,:,i1]
        x = utils.fold(x, dim=dim, stride=self.kernel, collapsed=True, shape=pshape)
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(shape))
        return x


class HyperRandomPatchSwap(Module):
    """
    Swap random patches in image as self-supervised pretraining task.
    Also works for anisotropic data with non-regular block shapes.

    References
    ----------
    ..[1] "Self-supervised learning for medical image analysis using image context restoration"
            Liang Chen, Paul Bentley, Kensaku Mori, Kazunari Misawa, Michitaka Fujiwara, Daniel Rueckert
            Medical Image Analysis 2019
            https://doi.org/10.1016/j.media.2019.101539

            
    """
    def __init__(self,
                 kernel='normal',
                 kernel_exp=32,
                 kernel_scale=16,
                 nb_swap=4):
        """
        Arguments:
            kernel (int or sequence[int]): Kernel size for blocks to shuffle. Can be
                single int (isotropic) or specified in all dimensions. Default = [32,32,32]
            nb_swap (int): Number of times to swap two random patches. Default = 4
        """
        
        super().__init__()
        self.kernel = _get_dist(kernel)
        self.kernel_exp = kernel_exp
        self.kernel_scale = kernel_scale
        self.nb_swap = nb_swap

    def forward(self, x):
        dim = x.dim() - 2
        backend = utils.backend(x)
        kernel_exp = utils.make_vector(self.kernel_exp, dim,
                                           **backend)
        kernel_scale = utils.make_vector(self.kernel_scale, dim,
                                             **backend)

        kernel = [self.kernel(k_e, k_s).sample() for k_e,k_s in zip(kernel_exp, kernel_scale)]
        shape = x.shape[2:]
        kernel = [torch.clamp(k, min=4, max=shape[i]).int().item() for i,k in enumerate(kernel)]
        pshape = [x+(k-x%k) for x,k in zip(shape,kernel)]
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(pshape))
        x = utils.unfold(x, kernel, collapse=True)
        for n in range(self.nb_swap):
            i1, i2 = torch.randint(low=0, high=x.shape[2]-1, size=(2,)).tolist()
            x[:,:,i1], x[:,:,i2] = x[:,:,i2], x[:,:,i1]
        x = utils.fold(x, dim=dim, stride=kernel, collapsed=True, shape=pshape)
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(shape))
        return x


class RandomInpaint(Module):
    """
    Set random patches to zero in image as self-supervised pretraining task.
    Also works for anisotropic data with non-regular block shapes.
    TODO: Allow custom mask input to generate random shapes with e.g. spline or topology descriptions.

    References
    ----------
    ..[1] 

            
    """
    def __init__(self,
                 dim,
                 kernel=[32,32,32],
                 nb_drop=4):
        """
        Arguments:
            kernel (int or sequence[int]): Kernel size for blocks to shuffle. Can be
                single int (isotropic) or specified in all dimensions. Default = [32,32,32]
            nb_swap (int): Number of times to swap two random patches. Default = 4
        """
        
        super().__init__()
        if isinstance(kernel, int):
            kernel = [kernel] * dim
        self.kernel = kernel
        self.nb_drop = nb_drop

    def forward(self, x):
        shape = x.shape[2:]
        dim = len(shape)
        pshape = [x+(k-x%k) for x,k in zip(shape,self.kernel)]
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(pshape))
        for n in range(self.nb_drop):
            x = utils.unfold(x, self.kernel, collapse=True)
            i1 = torch.randint(low=0, high=x.shape[2]-1, size=(1,)).item()
            x[:,:,i1] = 0
            x = utils.fold(x, dim=dim, stride=self.kernel, collapsed=True, shape=pshape)
        x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(shape))
        return x


class HyperRandomInpaint(Module):
    """
    Set random patches to zero in image as self-supervised pretraining task.
    Also works for anisotropic data with non-regular block shapes.
    TODO: Allow custom mask input to generate random shapes with e.g. spline or topology descriptions.

    References
    ----------
    ..[1] 

            
    """
    def __init__(self,
                 kernel='normal',
                 kernel_exp=32,
                 kernel_scale=16,
                 nb_drop=4):
        """
        Arguments:
            kernel (int or sequence[int]): Kernel size for blocks to shuffle. Can be
                single int (isotropic) or specified in all dimensions. Default = [32,32,32]
            nb_swap (int): Number of times to swap two random patches. Default = 4
        """
        
        super().__init__()
        self.kernel = _get_dist(kernel)
        self.kernel_exp = kernel_exp
        self.kernel_scale = kernel_scale
        self.nb_drop = nb_drop

    def forward(self, x):
        dim = x.dim() - 2
        backend = utils.backend(x)
        kernel_exp = utils.make_vector(self.kernel_exp, dim, **backend)
        kernel_scale = utils.make_vector(self.kernel_scale, dim, **backend)

        shape = x.shape[2:]
        for n in range(self.nb_drop):
            kernel = [self.kernel(k_e, k_s).sample() for k_e,k_s in zip(kernel_exp, kernel_scale)]
            kernel = [torch.clamp(k, min=4, max=shape[i]).int().item() for i,k in enumerate(kernel)]
            pshape = [x+(k-x%k) for x,k in zip(shape,kernel)]
            x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(pshape))
            x = utils.unfold(x, kernel, collapse=True)
            i1 = torch.randint(low=0, high=x.shape[2]-1, size=(1,)).item()
            x[:,:,i1] = 0
            x = utils.fold(x, dim=dim, stride=kernel, collapsed=True, shape=pshape)
            x = utils.ensure_shape(x, (x.shape[0],x.shape[1],) + tuple(shape))
        return x
