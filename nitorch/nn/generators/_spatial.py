import torch
from ..modules import Module, GridExp, GridPull
from ...core import utils
from ...core.utils import channel2last, unsqueeze
from ...core.pyutils import make_list
from ...core.linalg import matvec
from ._field import RandomFieldSample
from ...spatial import affine_matrix_classic, affine_matmul, affine_lmdiv
import torch.distributions as td
import math


class VelocitySample(Module):
    """Sample a random velocity field."""

    def __init__(self, dim=None, shape=None, amplitude=15, fwhm=10,
                 device='cpu', dtype=torch.get_default_dtype()):
        """

        Parameters
        ----------
        dim : int
            Space dimension.
        shape : sequence[int]
            Lattice shape.
        amplitude : float or callable or list, default=15
            Amplitude of the random field (per channel).
        fwhm : float or callable or list, default=10
            Full-width at half-maximum of the random field (per direction).
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.

        """
        super().__init__()
        self.field = RandomFieldSample(shape=shape, channel=dim,
                                       amplitude=amplitude, fwhm=fwhm,
                                       device=device, dtype=dtype)

    dim = property(lambda self: self.field.channel)
    shape = property(lambda self: self.field.shape)
    amplitude = property(lambda self: self.field.amplitude)
    fwhm = property(lambda self: self.field.fwhm)
    device = property(lambda self: self.field.device)
    dtype = property(lambda self: self.field.dtype)

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
            'channel': overload.get('dim', self.field.channel),
            'shape': overload.get('shape', self.field.shape),
            'amplitude': overload.get('amplitude', self.field.amplitude),
            'fwhm': overload.get('fwhm', self.field.fwhm),
            'dtype': overload.get('dtype', self.field.dtype),
            'device': overload.get('device', self.field.device),
        }

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


class DiffeoSample(Module):
    """Sample a random diffeomorphic transformation field."""

    def __init__(self, dim=None, shape=None, amplitude=15, fwhm=10,
                 bound='dft', interpolation=1, device='cpu',
                 dtype=torch.get_default_dtype()):
        """

        Parameters
        ----------
        dim : int
            Space dimension.
        shape : sequence[int]
            Lattice shape.
        amplitude : float or callable, default=15
            Amplitude of the random field.
        fwhm : float or callable or list, default=10
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
        self.velocity = VelocitySample(shape=shape, dim=dim,
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
        grid : (batch, *shape, dim) tensor
            Transformation field

        """

        # get arguments
        opt_vel = {
            'dim': overload.get('dim', self.velocity.dim),
            'shape': overload.get('shape', self.velocity.shape),
            'amplitude': overload.get('amplitude', self.velocity.amplitude),
            'fwhm': overload.get('fwhm', self.velocity.fwhm),
            'dtype': overload.get('dtype', self.velocity.dtype),
            'device': overload.get('device', self.velocity.device),
        }
        opt_exp = {
            'bound': overload.get('bound', self.exp.bound),
            'interpolation': overload.get('interpolation', self.exp.interpolation),
        }

        vel = self.velocity(batch, **opt_vel)
        grid = self.exp(vel, **opt_exp)

        return grid


class AffineSample(Module):
    """Sample an affine transformation matrix"""

    def __init__(self, dim=None, translation=True, rotation=True,
                 zoom=True, shear=True, device='cpu',
                 dtype=torch.get_default_dtype()):
        """

        Parameters
        ----------
        dim : int, optional
            Dimension
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
        self.dtype = dtype if dtype.is_floating_point \
            else torch.get_default_dtype()

    def default_translation(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, 5.*one).sample(*b)

    def default_rotation(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, 0.1*one).sample(*b)

    def default_shear(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, 0.01*one).sample(*b)

    def default_zoom(self, *b):
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        return td.Normal(zero, math.log(2)/3*one).sample(*b).exp()

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
        dtype = make_list(overload.get('dtype', self.dtype))
        device = make_list(overload.get('device', self.device))

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
        print(prm)
        prm = utils.as_tensor(prm)
        prm = prm.transpose(0, 1)

        # generate affine matrix
        mat = affine_matrix_classic(prm, dim=dim).\
            type(self.dtype).to(self.device)

        return mat


class DeformedSample(Module):
    """Random spatial deformation of an image"""

    def __init__(self, vel_amplitude=15, vel_fwhm=10,
                 translation=True, rotation=True, zoom=True, shear=True,
                 vel_bound='dft', image_bound='dct2', interpolation=1,
                 device='cpu', dtype=torch.get_default_dtype()):
        """

        Parameters
        ----------
        vel_amplitude : float or callable or list, default=15
            Amplitude of the (velocity) random field.
        vel_fwhm : float or callable or list, default=10
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
        device : torch.device: default='cpu'
            Output tensor device.
        dtype : torch.dtype, default=torch.get_default_dtype()
            Output tensor datatype.

        """
        super().__init__()
        self.affine = AffineSample(translation=translation, rotation=rotation,
                                   zoom=zoom, shear=shear, device=device,
                                   dtype=dtype)
        self.grid = DiffeoSample(amplitude=vel_amplitude,
                                 fwhm=vel_fwhm, bound=vel_bound,
                                 interpolation=interpolation, device=device,
                                 dtype=dtype)
        self.pull = GridPull(bound=image_bound,
                             interpolation=interpolation)

    translation = property(lambda self: self.affine.translation)
    rotation = property(lambda self: self.affine.rotation)
    zoom = property(lambda self: self.affine.zoom)
    shear = property(lambda self: self.affine.shear)
    dtype = property(lambda self: self.affine.dtype)
    device = property(lambda self: self.affine.device)
    vel_amplitude = property(lambda self: self.grid.amplitude)
    vel_fwhm = property(lambda self: self.grid.fwhm)
    vel_bound = property(lambda self: self.grid.bound)
    interpolation = property(lambda self: self.grid.interpolation)
    dtype = property(lambda self: self.grid.dtype)
    device = property(lambda self: self.grid.device)
    image_bound = property(lambda self: self.pull.bound)

    def forward(self, image, **overload):
        """

        Parameters
        ----------
        image : (batch, channel, *shape) tensor
            Input image
        overload : dict
            All parameters defined at build time can be overridden at call time

        Returns
        -------
        warped : (batch, channel, *shape) tensor
            Deformed image

        """

        image = torch.as_tensor(image)
        dim = image.dim() - 2
        batch, channel, *shape = image.shape
        info = {'dtype': image.dtype, 'device': image.device}

        # get arguments
        opt_grid = {
            'dim': dim,
            'shape': shape,
            'amplitude': overload.get('vel_amplitude', self.grid.amplitude),
            'fwhm': overload.get('vel_fwhm', self.grid.fwhm),
            'bound': overload.get('vel_bound', self.grid.bound),
            'interpolation': overload.get('interpolation', self.grid.interpolation),
            'dtype': overload.get('dtype', self.grid.dtype),
            'device': overload.get('device', self.grid.device),
        }
        opt_affine = {
            'dim': dim,
            'translation': overload.get('translation', self.affine.translation),
            'rotation': overload.get('rotation', self.affine.rotation),
            'zoom': overload.get('zoom', self.affine.zoom),
            'shear': overload.get('shear', self.affine.shear),
            'dtype': overload.get('dtype', self.affine.dtype),
            'device': overload.get('device', self.affine.device),
        }
        opt_pull = {
            'bound': overload.get('image_bound', self.pull.bound),
            'interpolation': overload.get('interpolation', self.pull.interpolation),
        }

        grid = self.grid(batch, **opt_grid)
        aff = self.affine(batch, **opt_affine)

        # shift center of rotation
        aff_shift = torch.cat((
            torch.eye(dim, **info),
            -torch.as_tensor(opt_grid['shape'], **info)[:, None]/2),
            dim=1)
        aff = affine_matmul(aff, aff_shift)
        aff = affine_lmdiv(aff_shift, aff)

        # compose
        aff = unsqueeze(aff, dim=-3, ndim=dim)
        lin = aff[..., :dim, :dim]
        off = aff[..., :dim, -1]
        grid = matvec(lin, grid) + off

        # pull
        warped = self.pull(image, grid, **opt_pull)

        return warped
