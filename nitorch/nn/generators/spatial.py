import math
import torch
import torch.distributions as td
from nitorch.core import utils
from nitorch.core.utils import channel2last, unsqueeze, make_vector
from nitorch.core.py import make_list
from nitorch.core.linalg import matvec
from nitorch.spatial import affine_matrix_classic, affine_matmul, affine_lmdiv, as_euclidean, identity_grid
from nitorch.nn.base import Module
from ..modules.spatial import GridExp, GridPull
from .field import RandomFieldSpline


__all__ = ['RandomVelocity', 'RandomDiffeo', 'RandomAffine', 'RandomGrid',
           'RandomDeform', 'RandomPatch', 'RandomFlip']


defaults = dict(
    vel_amplitude=5,
    vel_fwhm=15
)


class RandomVelocity(Module):
    """Sample a random velocity field."""

    def __init__(self, shape=None,
                 amplitude=defaults['vel_amplitude'],
                 fwhm=defaults['vel_fwhm'],
                 device='cpu', dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int]
            Spatial shape.
        amplitude : float or callable or list, optional
            Amplitude of the random field (per channel).
        fwhm : float or callable or list, optional
            Full-width at half-maximum of the random field (per direction).
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.

        """
        super().__init__()
        shape = make_list(shape)
        self.field = RandomFieldSpline(shape=shape, channel=len(shape),
                                       amplitude=amplitude, fwhm=fwhm,
                                       device=device, dtype=dtype)

    dim = property(lambda self: self.field.channel)
    shape = property(lambda self: self.field.shape)
    amplitude = property(lambda self: self.field.amplitude)
    fwhm = property(lambda self: self.field.fwhm)
    device = property(lambda self: self.field.device)
    dtype = property(lambda self: self.field.dtype)

    def to(self, *args, **kwargs):
        self.field.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        vel : (batch, *shape, dim) tensor
            Velocity field

        """

        # get arguments
        opt = {
            'shape': overload.get('shape', self.field.shape),
            'amplitude': overload.get('amplitude', self.field.amplitude),
            'fwhm': overload.get('fwhm', self.field.fwhm),
            'dtype': overload.get('dtype', self.field.dtype),
            'device': overload.get('device', self.field.device),
        }
        opt['channel'] = len(opt['shape'])

        # preprocess amplitude
        # > RandomField broadcast amplitude to (channel, *shape), with
        #   padding from the left, which means that a 1d amplitude would
        #   be broadcasted to (1, ..., dim) instead of (dim, ..., 1)
        # > We therefore reshape amplitude to avoid left-side padding
        def preprocess(a):
            a = torch.as_tensor(a)
            a = unsqueeze(a, dim=-1, ndim=opt['channel']+1-a.dim())
            return a
        amplitude = opt['amplitude']
        if callable(amplitude):
            amplitude_fn = amplitude
            amplitude = lambda *args, **kwargs: preprocess(amplitude_fn(*args, **kwargs))
        else:
            amplitude = preprocess(amplitude)
        opt['amplitude'] = amplitude

        return channel2last(self.field(batch, **opt))


class RandomDiffeo(Module):
    """Sample a random diffeomorphic transformation field."""

    def __init__(self, shape=None,
                 amplitude=defaults['vel_amplitude'],
                 fwhm=defaults['vel_fwhm'],
                 bound='dft', interpolation=1, device='cpu',
                 dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int]
            Spatial shape.
        amplitude : float or callable, optional
            Amplitude of the random field.
        fwhm : float or callable or list, optional
            Full-width at half-maximum of the random field.
        bound : BoundType, default='dft'
            Boundary condition when exponentiating the velocity field.
        interpolation : InterpolationType, default=1
            Interpolation order when exponentiating the velocity field
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.

        """
        super().__init__()
        self.velocity = RandomVelocity(shape=shape,
                                       amplitude=amplitude, fwhm=fwhm,
                                       device=device, dtype=dtype)
        self.exp = GridExp(bound=bound, interpolation=interpolation)

    dim = property(lambda self: self.velocity.dim)
    shape = property(lambda self: self.velocity.shape)
    amplitude = property(lambda self: self.velocity.amplitude)
    fwhm = property(lambda self: self.velocity.fwhm)
    device = property(lambda self: self.velocity.device)
    dtype = property(lambda self: self.velocity.dtype)
    bound = property(lambda self: self.exp.bound)
    interpolation = property(lambda self: self.exp.interpolation)

    def to(self, *args, **kwargs):
        self.velocity.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def forward(self, batch=1, return_vel=False, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        grid : (batch, *shape, dim) tensor
            Transformation field

        """

        # get arguments
        opt_vel = {
            'shape': overload.get('shape', self.shape),
            'amplitude': overload.get('amplitude', self.amplitude),
            'fwhm': overload.get('fwhm', self.fwhm),
            'dtype': overload.get('dtype', self.dtype),
            'device': overload.get('device', self.device),
        }

        vel = self.velocity(batch, **opt_vel)
        grid = self.exp(vel)

        return (grid, vel) if return_vel else grid


class RandomAffine(Module):
    """Sample an affine transformation matrix"""

    def __init__(self, dim=None, translation=True, rotation=True,
                 zoom=True, shear=True, device='cpu', dtype=None):
        """

        Parameters
        ----------
        dim : int, optional
            Dimension
        translation : bool or callable or list, default=True
            Translation parameters in voxels
            If True -> Normal(0, 5)
        rotation : bool or callable or list, default=True
            Rotation parameters in degrees
            If True -> Normal(0, 5)
        zoom : bool or callable or list, default=True
            Zoom parameters (1 == no zoom)
            If True -> LogNormal(log(1), log(1.2)/3)
        shear : bool or callable or list, default=True
            Shear parameters in voxels
            If True -> Normal(0, 0.1)
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.

        """
        super().__init__()
        self.dim = dim
        self.translation = translation
        self.rotation = rotation
        self.zoom = zoom
        self.shear = shear
        self.device = device
        if dtype is None or not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        self.dtype = dtype

    def default_translation(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, 5*one).sample(*b)

    def default_rotation(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, 5*one).sample(*b)

    def default_shear(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, 0.1*one).sample(*b)

    def default_zoom(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, math.log(1.2)/3*one).sample(*b).exp()

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format \
            = torch._C._nn._parse_to(*args, **kwargs)

        self.dtype = dtype or self.dtype
        self.device = device or self.device
        super().to(*args, **kwargs)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch size
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        affine : (batch, dim[+1], dim+1) tensor
            Velocity field

        """
        dim = overload.get('dim', self.dim)
        translation = make_list(overload.get('translation', self.translation))
        rotation = make_list(overload.get('rotation', self.rotation))
        zoom = make_list(overload.get('zoom', self.zoom))
        shear = make_list(overload.get('shear', self.shear))
        dtype = overload.get('dtype', self.dtype)
        device = overload.get('device', self.device)
        backend = dict(dtype=dtype, device=device)

        # compute dimension
        dim = dim or max(len(translation), len(rotation), len(zoom), len(shear))
        translation = make_list(translation, dim)
        rotation = make_list(rotation, dim*(dim-1)//2)
        zoom = make_list(zoom, dim)
        shear = make_list(shear, dim*(dim-1)//2)

        # sample values if needed
        translation = [x([batch]) if callable(x)
                       else self.default_translation([batch]) if x is True
                       else 0. if x is None or x is False
                       else x for x in translation]
        rotation = [x([batch]) if callable(x)
                    else self.default_rotation([batch]) if x is True
                    else 0. if x is None or x is False
                    else x for x in rotation]
        zoom = [x([batch]) if callable(x)
                else self.default_zoom([batch]) if x is True
                else 1. if x is None or x is False
                else x for x in zoom]
        shear = [x([batch]) if callable(x)
                 else self.default_shear([batch]) if x is True
                 else 0. if x is None or x is False
                 else x for x in shear]
        rotation = [x * math.pi / 180 for x in rotation]  # degree -> radian
        prm = [*translation, *rotation, *zoom, *shear]
        prm = [p.expand(batch) if torch.is_tensor(p) and p.shape[0] != batch
               else make_list(p, batch) if not torch.is_tensor(p)
               else p for p in prm]

        prm = utils.as_tensor(prm)
        prm = prm.transpose(0, 1)

        # generate affine matrix
        prm = prm.to(**backend)
        mat = affine_matrix_classic(prm, dim=dim)

        return mat


class RandomGrid(Module):
    """Random spatial deformation (dense + affine)."""

    def __init__(self, shape=None,
                 vel_amplitude=defaults['vel_amplitude'],
                 vel_fwhm=defaults['vel_fwhm'],
                 translation=True, rotation=True, zoom=True, shear=True,
                 bound='dft', interpolation=1, device='cpu', dtype=None):
        """

        Parameters
        ----------
        shape : sequence[int]
            Spatial shape
        vel_amplitude : float or callable or list, optional
            Amplitude of the (velocity) random field.
        vel_fwhm : float or callable or list, optional
            Full-width at half-maximum of the (velocity) random field.
        translation : bool or callable or list, default=True
            Translation parameters in voxels
            If True -> Normal(0, 15)
        rotation : bool or callable or list, default=True
            Rotation parameters in degrees
            If True -> Normal(0, 60)
        zoom : bool or callable or list, default=True
            Zoom parameters (1 == no zoom)
            If True -> LogNormal(log(1), log(2)/3)
        shear : bool or callable or list, default=True
            Shear parameters in voxels
            If True -> Normal(0, 10)
        bound : BoundType, default='dft'
            Boundary condition when exponentiating the velocity field.
        interpolation : int, default=1
            Interpolation order when exponentiating the velocity field.
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.

        """
        super().__init__()
        self.affine = RandomAffine(
            translation=translation,
            rotation=rotation,
            zoom=zoom,
            shear=shear,
            device=device,
            dtype=dtype)
        self.grid = RandomDiffeo(
            shape=shape,
            amplitude=vel_amplitude,
            fwhm=vel_fwhm,
            bound=bound,
            interpolation=interpolation,
            device=device,
            dtype=dtype)

    translation = property(lambda self: self.affine.translation)
    rotation = property(lambda self: self.affine.rotation)
    zoom = property(lambda self: self.affine.zoom)
    shear = property(lambda self: self.affine.shear)
    vel_amplitude = property(lambda self: self.grid.amplitude)
    shape = property(lambda self: self.grid.shape)
    vel_fwhm = property(lambda self: self.grid.fwhm)
    bound = property(lambda self: self.grid.bound)
    interpolation = property(lambda self: self.grid.interpolation)
    dtype = property(lambda self: self.grid.dtype)
    device = property(lambda self: self.grid.device)

    def forward(self, batch=1, **overload):
        """

        Parameters
        ----------
        batch : int, default=1
            Batch shape.
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        grid : (batch, *shape, 3) tensor
            Resampling grid

        """
        dtype = overload.get('dtype', self.dtype)
        device = overload.get('device', self.device)

        # get arguments
        opt_grid = {
            'shape': overload.get('shape', self.shape),
            'amplitude': overload.get('vel_amplitude', self.vel_amplitude),
            'fwhm': overload.get('vel_fwhm', self.vel_fwhm),
            'bound': overload.get('bound', self.bound),
            'interpolation': overload.get('interpolation', self.interpolation),
            'dtype': dtype,
            'device': device,
        }
        if opt_grid['amplitude']:
            grid = self.grid(batch, **opt_grid)
        else:
            grid = identity_grid(opt_grid['shape'], dtype=dtype, device=device)

        shape = grid.shape[1:-1]
        dim = len(shape)
        opt_affine = {
            'dim': dim,
            'translation': overload.get('translation', self.translation),
            'rotation': overload.get('rotation', self.rotation),
            'zoom': overload.get('zoom', self.zoom),
            'shear': overload.get('shear', self.shear),
            'dtype': dtype,
            'device': device,
        }
        aff = self.affine(batch, **opt_affine)

        backend = dict(dtype=dtype, device=device)

        # shift center of rotation
        aff_shift = torch.cat((
            torch.eye(dim, **backend),
            -torch.as_tensor(shape, **backend)[:, None]/2),
            dim=1)
        aff_shift = as_euclidean(aff_shift)

        aff = affine_matmul(aff, aff_shift)
        aff = affine_lmdiv(aff_shift, aff)

        # compose
        aff = unsqueeze(aff, dim=-3, ndim=dim)
        lin = aff[..., :dim, :dim]
        off = aff[..., :dim, -1]
        grid = matvec(lin, grid) + off

        return grid


class RandomDeform(Module):
    """Random spatial deformation of an image"""

    def __init__(self,
                 vel_amplitude=defaults['vel_amplitude'],
                 vel_fwhm=defaults['vel_fwhm'],
                 translation=True, rotation=True, zoom=True, shear=True,
                 vel_bound='dft', image_bound='dct2', interpolation=1):
        """

        Parameters
        ----------
        vel_amplitude : float or callable or list, optional
            Amplitude of the (velocity) random field.
        vel_fwhm : float or callable or list, optional
            Full-width at half-maximum of the (velocity) random field.
        translation : bool or callable or list, default=True
            Translation parameters in voxels
            If True -> Normal(0, 15)
        rotation : bool or callable or list, default=True
            Rotation parameters in degrees
            If True -> Normal(0, 60)
        zoom : bool or callable or list, default=True
            Zoom parameters (1 == no zoom)
            If True -> LogNormal(log(1), log(2)/3)
        shear : bool or callable or list, default=True
            Shear parameters in voxels
            If True -> Normal(0, 10)
        vel_bound : BoundType, default='dft'
            Boundary condition when exponentiating the velocity field.
        image_bound : BoundType, default='dct2'
            Boundary condition when sampling the image.
        interpolation : InterpolationType, default=1
            Interpolation order

        """
        super().__init__()
        self.grid = RandomGrid(
            vel_amplitude=vel_amplitude,
            vel_fwhm=vel_fwhm,
            translation=translation,
            rotation=rotation,
            zoom=zoom,
            shear=shear,
            bound=vel_bound,
            interpolation=interpolation)
        self.pull = GridPull(
            bound=image_bound,
            interpolation=interpolation,
            extrapolate=True)

    translation = property(lambda self: self.grid.translation)
    rotation = property(lambda self: self.grid.rotation)
    zoom = property(lambda self: self.grid.zoom)
    shear = property(lambda self: self.grid.shear)
    vel_amplitude = property(lambda self: self.grid.vel_amplitude)
    vel_fwhm = property(lambda self: self.grid.vel_fwhm)
    vel_bound = property(lambda self: self.grid.vel_bound)
    interpolation = property(lambda self: self.grid.interpolation)
    image_bound = property(lambda self: self.pull.bound)

    def forward(self, *images, return_grid=False, **overload):
        """

        Parameters
        ----------
        *images : (batch, channel, *shape) tensor
            Input images
        return_grid : bool, default=False
            Return deformation grid on top of deformed sample.
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        warped : (batch, channel, *shape) tensor
            Deformed image
        grid : (batch, *shape, 3) tensor
            Resampling grid

        """
        batch, channel, *shape = images[0].shape

        # get arguments
        opt_grid = {
            'shape': shape,
            'vel_amplitude': overload.get('vel_amplitude', self.vel_amplitude),
            'vel_fwhm': overload.get('vel_fwhm', self.vel_fwhm),
            'translation': overload.get('translation', self.translation),
            'rotation': overload.get('rotation', self.rotation),
            'zoom': overload.get('zoom', self.zoom),
            'shear': overload.get('shear', self.shear),
            'bound': overload.get('vel_bound', self.grid.bound),
            'interpolation': overload.get('interpolation', self.grid.interpolation),
            'dtype': images[0].dtype,
            'device': images[0].device,
        }

        # pull
        grid = self.grid(batch, **opt_grid)
        warped = []
        for image in images:
            warped.append(self.pull(image, grid))

        return (*warped, grid) if return_grid else tuple(warped)


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
        shape = make_list(overload.get('shape', self.shape), dim)
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
        prob = make_vector(overload.get('prob', self.prob),
                           dtype=torch.float, device=device)
        dim = overload.get('dim', self.dim)
        dim = make_list(dim or range(-nb_dim, 0), nb_dim)

        # sample shift
        flip = torch.rand((nb_dim,), device=device) > (1 - prob)
        dim = [d for d, f in zip(dim, flip) if f]

        if dim:
            for i, img in enumerate(image):
                image[i] = img.flip(dim)
        return image[0] if len(image) == 1 else tuple(image)
