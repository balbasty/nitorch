from nitorch import spatial, io
from nitorch.core import py, utils, linalg, math
import torch
from .utils import jg, jhj, defaults_velocity
from . import plot as plt, optim as optm, losses, phantoms, utils as regutils
import functools
import copy


# TODO:
#  [x] fix backward gradients for smalldef/shoot (svf should be ok)
#  [x] implement forward pass for affine
#  [x] implement a forward pass for affine only (when nonlin is None).
#      It deserves its own implementation as it is so much simpler (the
#      two half matrices collapse).
#  [ ] write utility function to set optimizer options based on
#      model options (e.g., reg param in GridCG/GridRelax)
#  [ ] syntaxic sugar class to transform a forward loss into an OptimizationLoss
#      object using autodiff (set `warped.requires_grad`, wrap under a
#      `with torch.grad()`, call backward on loss, gather grad)
#  [ ] integrate cache mechanism into the Displacement/Image classes
#      (e.g., add a `cache` argument to `exp` to retrieve/cache the
#      exponentiated field).
#  [ ] add the possibility to encode velocity.displacement fields using
#      B-splines. Gradients should be relatively straightforward (compute
#      dense gradient and then propagate to nodes?), but regularizers
#      should probably be stored in matrix form (like what JA did for
#      fields encoded with dft/dct is US). If the number of nodes is not
#      too large, it would allow us to use Gauss-Newton directly.


class MeanSpace:
    """Compute a mean space from a bunch of affine + shape"""
    def __init__(self, images, voxel_size=None, vx_unit='mm', pad=0, pad_unit='%'):
        mat, shape = spatial.mean_space(
            [image.affine for image in images],
            [image.shape for image in images],
            voxel_size=voxel_size, vx_unit=vx_unit,
            pad=pad, pad_unit=pad_unit)
        self.affine = mat
        self.shape = shape


class SpatialTensor:
    """Base class for tensors with an orientation"""
    def __init__(self, dat, affine=None, dim=None, **backend):
        """
        Parameters
        ----------
        dat : ([C], *spatial) tensor
        affine : tensor, optional
        dim : int, default=`dat.dim() - 1`
        **backend : dtype, device
        """
        if isinstance(dat, str):
            dat = io.map(dat)[None]
        if isinstance(dat, io.MappedArray):
            if affine is None:
                affine = dat.affine
            dat = dat.fdata(rand=True, **backend)
        self.dim = dim or dat.dim() - 1
        self.dat = dat
        if affine is None:
            affine = spatial.affine_default(self.shape, **utils.backend(dat))
        self.affine = affine.to(**utils.backend(self.dat))

    def to(self, *args, **kwargs):
        return copy.copy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        self.dat = self.dat.to(*args, **kwargs)
        self.affine = self.affine.to(*args, **kwargs)
        return self

    voxel_size = property(lambda self: spatial.voxel_size(self.affine))
    shape = property(lambda self: self.dat.shape[-self.dim:])
    dtype = property(lambda self: self.dat.dtype)
    device = property(lambda self: self.dat.device)

    def _prm_as_str(self):
        s = [f'shape={list(self.shape)}']
        v = [f'{vx:.2g}' for vx in self.voxel_size.tolist()]
        v = ', '.join(v)
        s += [f'voxel_size=[{v}]']
        if self.dtype != torch.float32:
            s += [f'dtype={self.dtype}']
        if self.device.type != 'cpu':
            s +=[f'device={self.device}']
        return s

    def __repr__(self):
        s = ', '.join(self._prm_as_str())
        s = f'{self.__class__.__name__}({s})'
        return s

    __str__ = __repr__


class Image(SpatialTensor):
    """Data + Metadata (affine, boundary) of an Image"""
    def __init__(self, dat, affine=None, dim=None,
                 bound='dct2', extrapolate=True):
        """
        Parameters
        ----------
        dat : ([C], *spatial) tensor
        affine : tensor, optional
        dim : int, default=`dat.dim() - 1`
        bound : str, default='dct2'
        extrapolate : bool, default=True
        """
        super().__init__(dat, affine, dim)
        self.bound = bound
        self.extrapolate = extrapolate

    def _prm_as_str(self):
        s = []
        if self.bound != 'dct2':
            s += [f'bound={self.bound}']
        if not self.extrapolate:
            s += ['extrapolate=False']
        s = super()._prm_as_str() + s
        return s

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, Image):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('affine', dat.affine)
            kwargs.setdefault('bound', dat.bound)
            kwargs.setdefault('extrapolate', dat.extrapolate)
            dat = dat.dat
        return cls(dat.dat, **kwargs)

    def pull(self, grid):
        """Sample the image at dense coordinates.

        Parameters
        ----------
        grid : (*spatial, dim) tensor or None
            Dense transformation field.

        Returns
        -------
        warped : ([C], *spatial) tensor

        """
        return regutils.smart_pull(self.dat, grid, bound=self.bound,
                                   extrapolate=self.extrapolate)

    def pull_grad(self, grid, rotate=False):
        """Sample the image gradients at dense coordinates.

        Parameters
        ----------
        grid : (*spatial, dim) tensor or None
            Dense transformation field.
        rotate : bool, default=False
            Rotate the gradients using the Jacobian of the transformation.

        Returns
        -------
        grad : ([C], *spatial, dim) tensor

        """
        if grid is None:
            return self.grad()
        grad = spatial.grid_grad(self.dat, grid, bound=self.bound,
                                 extrapolate=self.extrapolate)
        if rotate:
            jac = spatial.grid_jacobian(grid)
            jac = jac.transpose(-1, -2)
            grad = linalg.matvec(jac, grad)
        return grad

    def grad(self):
        """Compute the image gradients in each voxel.
        Almost equivalent to `self.pull_grad(identity)`.

        Returns
        -------
        grad : ([C], *spatial, dim) tensor

        """
        return spatial.diff(self.dat, dim=list(range(-self.dim, 0)),
                            bound=self.bound)


class ImagePyramid(Image):
    """Compute a multiscale image pyramid.
    This object can be used as an Image (in which case the highest
    resolution is returned) or as a list of Images.
    """

    def __init__(self, dat, levels=1, affine=None, dim=None,
                 bound='dct2', extrapolate=True, method='gauss', **backend):
        """

        Parameters
        ----------
        dat : [list of] (..., *shape) tensor or Image
        levels : int or list[int], default=1
            If an int, it is the number of levels.
            If a list, they are the indices of levels to compute.
            `1` is the native resolution, `2` is half of it, etc.
        affine : [list of] tensor, optional
        dim : int, optional
        bound : str, default='dct2'
        extrapolate : bool, default=True
        method : {'gauss', 'average', 'median'}, default='gauss'
        """
        # I don't call super().__init__() on purpose

        if isinstance(dat, str):
            dat = io.map(dat)
        if isinstance(dat, io.MappedArray):
            if affine is None:
                affine = dat.affine
            dat = dat.fdata(rand=True, **backend)[None]

        if not isinstance(dat, (list, tuple)):
            dim = dim or dat.dim() - 1
            if not levels:
                raise ValueError('levels required to compute pyramid')
            if isinstance(levels, int):
                levels = range(1, levels+1)
            if affine is None:
                shape = dat.shape[-dim:]
                affine = spatial.affine_default(shape, **utils.backend(dat[0]))
            dat = self._build_pyramid(dat, levels, method, dim, bound)
        dat = list(dat)

        if all(isinstance(d, Image) for d in dat):
            self._dat = dat
            return

        dim = dim or (dat[0].dim() - 1)
        if affine is None:
            shape = dat[0].shape[-dim:]
            affine = spatial.affine_default(shape, **utils.backend(dat[0]))
        if not isinstance(affine, (list, tuple)):
            if not levels:
                raise ValueError('levels required to compute affine pyramid')
            shape = dat[0].shape[-dim:]
            affine = self._build_affine_pyramid(affine, shape, levels, method)
        self._dat = [Image(d, aff, dim=dim, bound=bound, extrapolate=extrapolate)
                     for d, aff in zip(dat, affine)]
        self.method = method

    def _prm_as_str(self):
        s = []
        if self.method[0] != 'g':
            s += [f'method={self.method}']
        return [f'nb_levels={len(self)}'] + super()._prm_as_str() + s

    def __len__(self):
        return len(self._dat)

    def __getitem__(self, item):
        return self._dat[item]

    def __iter__(self):
        for dat in self._dat:
            yield dat

    dim = property(lambda self: self[0].dim)
    shape = property(lambda self: self[0].shape)
    dat = property(lambda self: self[0].dat)
    affine = property(lambda self: self[0].affine)
    bound = property(lambda self: self[0].bound)
    extrapolate = property(lambda self: self[0].extrapolate)
    voxel_size = property(lambda self: self[0].voxel_size)

    def to(self, *args, **kwargs):
        return copy.deepcopy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        for dat in self:
            dat.to_(*args, **kwargs)
        return self

    def _build_pyramid(self, dat, levels, method, dim, bound):
        levels = list(levels)
        indexed_levels = list(enumerate(levels))
        indexed_levels.sort(key=lambda x: x[1])
        nb_levels = max(levels)
        dats = [dat] * levels.count(1)
        for level in range(2, nb_levels+1):
            if method[0] == 'g':  # gaussian pyramid
                dat = spatial.smooth(dat, fwhm=2, stride=2, dim=dim,
                                     bound=bound)
            elif method[0] == 'a':  # average window
                dat = spatial.pool(dim, dat, kernel_size=2, stride=2,
                                   reduction='mean')
            elif method[0] == 'm':  # median window
                dat = spatial.pool(dim, dat, kernel_size=2, stride=2,
                                   reduction='median')
            dats += [dat] * levels.count(level)
        reordered_dats = [None] * len(levels)
        for (i, level), dat in zip(indexed_levels, dats):
            reordered_dats[i] = dat
        return reordered_dats

    def _build_affine_pyramid(self, affine, shape, levels, method):
        levels = list(levels)
        indexed_levels = list(enumerate(levels))
        indexed_levels.sort(key=lambda x: x[1])
        nb_levels = max(levels)
        affines = [affine] * levels.count(1)
        for level in range(2, nb_levels+1):
            if method[0] == 'g':  # gaussian pyramid
                padding = 'auto'
                kernel = 3
            else:  # moving window
                padding = 0
                kernel = 2
            affine, shape = spatial.affine_conv(affine, shape, kernel, 2,
                                                padding=padding)
            affines += [affine] * levels.count(level)
        reordered_affines = [None] * len(levels)
        for (i, level), affine in zip(indexed_levels, affines):
            reordered_affines[i] = affine
        return reordered_affines

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, ImagePyramid):
            dat = dat._dat
        elif isinstance(dat, Image):
            kwargs.setdefault('affine', dat.affine)
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('bound', dat.bound)
            kwargs.setdefault('extrapolate', dat.extrapolate)
            dat = dat.dat
        return cls(dat, **kwargs)


class Displacement(SpatialTensor):
    """Data + Metadata (affine) of a displacement or velocity field"""
    def __init__(self, dat, affine=None, dim=None, **backend):
        """
        Parameters
        ----------
        dat : (*spatial, dim) tensor or list[int]
            Pre-allocated displacement field, or its shape.
        affine : tensor, optional
            Orientation matrix
        dim : int, default=`dat.dim()-1`
            Number of spatial dimensions
        **backend
        """
        if isinstance(dat, (list, tuple)):
            shape = dat
            dim = dim or len(shape)
            dat = torch.zeros([*shape, dim], **backend)
        super().__init__(dat, affine, dim)

    @property
    def shape(self):
        return self.dat.shape[-self.dim-1:-1]

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, Displacement):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('affine', dat.affine)
            dat = dat.dat
        return cls(dat, **kwargs)


class LogAffine:
    """Data + Metadata (basis) of a "Lie" affine matrix"""
    def __init__(self, dat=None, basis=None, dim=None, **backend):
        """
        Parameters
        ----------
        [dat : tensor, optional]
            Pre-allocated log-affine
        basis : tensor or str
            Pre-computed Lie basis, or its name
        dim : int, default=3
            Number of spatial dimensions
        **backend
        """
        if isinstance(dat, str) or \
                (torch.is_tensor(dat) and dat.dim() == 3):
            if basis is not None:
                raise ValueError('`basis` provided but `dat` looks like '
                                 'a basis.')
            basis, dat = dat, None
        if not isinstance(basis, str):
            dim = dim or basis.shape[-1] - 1
        else:
            dim = dim or 3
            basis = spatial.affine_basis(basis, dim, **backend)
        self.basis = basis
        self.dim = dim
        if dat is None:
            dat = torch.zeros(basis.shape[0], **backend)
        self.dat = dat

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, LogAffine):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('basis', dat.basis)
            dat = dat.dat
        return LogAffine(dat, **kwargs)

    def exp(self, q=None, grad=False):
        if q is None:
            q = self.dat
        return linalg._expm(q, self.basis, grad_X=grad)

    def iexp(self, q=None, grad=False):
        if q is None:
            q = self.dat
        return linalg._expm(-q, self.basis, grad_X=grad)

    def exp2(self, q=None, grad=False):
        if grad:
            a, g = self.exp(q, True)
            ia, ig = self.iexp(q, True)
            return a, ia, g, ig
        else:
            return self.exp(q), self.iexp(q)

    def __repr__(self):
        s = [f'shape={list(self.dat.shape)}', f'dim={self.dim}']
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'

    __str__ = __repr__


class LossComponent:
    """Component of a composite loss"""
    def __init__(self, loss, moving, fixed, weight=1, symmetric=False):
        """

        Parameters
        ----------
        loss : OptimizationLoss (or its name)
            Matching loss to minimize.
        moving : Image or ImagePyramid like
        fixed : Image or ImagePyramid like
            Fixed and moving should have the same number of levels.
        weight : float, default=1
            Weight of the loss
        symmetric : bool, default=False
            Whether the loss should be symmetric
            i.e., loss = (loss(moving, fixed) + loss(fixed, moving)) / 2
        """
        self.moving = ImagePyramid.make(moving)
        self.fixed = ImagePyramid.make(fixed)
        self.loss = losses.make_loss(loss, self.moving.dim)
        self.weight = weight
        self.symmetric = symmetric

    def make(self, loss, **kwargs):
        if isinstance(loss, LossComponent):
            kwargs.setdefault('moving', loss.moving)
            kwargs.setdefault('fixed', loss.fixed)
            kwargs.setdefault('weight', loss.weight)
            kwargs.setdefault('symmetric', loss.symmetric)
            loss = loss.loss
        return LossComponent(loss, **kwargs)


def _rotate_grad(grad, aff=None, dense=None):
    """Rotate grad by the jacobian of `aff o dense`.
    grad : (..., dim) tensor       Spatial gradients
    aff : (dim+1, dim+1) tensor    Affine matrix
    dense : (..., dim) tensor      Dense vox2vox displacement field
    returns : (..., dim) tensor    Rotated gradients.
    """
    if aff is None and dense is None:
        return grad
    dim = grad.shape[-1]
    if dense is not None:
        jac = spatial.grid_jacobian(dense, type='disp')
        if aff is not None:
            jac = torch.matmul(aff[:dim, :dim], jac)
    else:
        jac = aff[:dim, :dim]
    grad = linalg.matvec(jac.transpose(-1, -2), grad)
    return grad


class NonLinModel:
    """Base class for non-linear deformations."""
    # TODO: currently tailored for dense fields but could be adapted
    #   to spline-encoded deformations (or velocities)

    def __new__(cls, model, *args, **kwargs):
        if isinstance(model, NonLinModel):
            return cls.make(model, *args, **kwargs)
        elif model.lower() == 'shoot':
            return ShootModel(*args, **kwargs)
        elif model.lower() == 'svf':
            return SVFModel(*args, **kwargs)
        elif model.lower() == 'smalldef':
            return SmallDefModel(*args, **kwargs)
        else:
            raise ValueError('unknown:', model)

    def __init__(self, dat=None, lam=1, prm=None,
                 steps=8, kernel=None, **backend):
        """

        Parameters
        ----------
        dat : Displacement or tensor or list[int], optional
            Pre-allocated displacement or its shape
        lam : float, default=1
            Regularization factor
        prm : dict(absolute, membrane, bending, lame), optional
            Regularization factor for each component.
            Informed values are used by default.
        steps : int, default=8
            Number of integration steps.
        kernel : tensor, optional
            Pre-computed Greens kernel.
        **backend
        """
        super().__init__()
        self.lam = lam
        self.prm = prm or regutils.defaults_velocity()
        self.prm.pop('voxel_size', None)
        self.steps = steps
        self.kernel = kernel
        if dat is not None:
            self.set_dat(dat, **backend)
        else:
            self.dat = None

    affine = property(lambda self: self.dat.affine if self.dat is not None else None)
    shape = property(lambda self: self.dat.shape if self.dat is not None else None)
    dim = property(lambda self: self.dat.dim if self.dat is not None else None)
    voxel_size = property(lambda self: self.dat.voxel_size if self.dat is not None else None)

    def set_dat(self, dat, affine=None, **backend):
        if isinstance(dat, str):
            dat = io.map(dat)
            if affine is None:
                affine = self.dat.affine
        self.dat = Displacement.make(dat, affine=affine).to(**backend)
        if self.kernel is None:
            self.kernel = spatial.greens(self.shape, **self.prm,
                                         factor=self.lam / py.prod(self.shape),
                                         voxel_size=self.voxel_size,
                                         **utils.backend(self.dat))
        return self

    def make(self, model, **kwargs):
        if isinstance(model, type(self)):
            kwargs.setdefault('lam', model.lam)
            kwargs.setdefault('prm', model.prm)
            kwargs.setdefault('steps', model.steps)
            kwargs.setdefault('kernel', model.kernel)
            kwargs.setdefault('dat', model.dat)
        elif isinstance(model, NonLinModel):
            raise TypeError('Cannot convert between `NonLinModel`s')
        if type(self) is NonLinModel:
            return NonLinModel(model, **kwargs)
        else:
            return type(self)(**kwargs)

    def regulariser(self, v=None):
        if v is None:
            v = self.dat
        return spatial.regulariser_grid(v, **self.prm,
                                        factor=self.lam / py.prod(self.shape),
                                        voxel_size=self.voxel_size)

    def greens_apply(self, m):
        return spatial.greens_apply(m, self.kernel, self.voxel_size)

    def __repr__(self):
        s = []
        if self.dat is not None:
            s += [f'velocity={self.dat}']
        else:
            s += ['<uninitialized>']
        s += [f'lam={self.lam}']
        s += [f'{key}={value}' for key, value in self.prm.items()]
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'

    __str__ = __repr__


class ShootModel(NonLinModel):
    """Initial velocity exponentiated by geodesic shooting"""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(ShootModel)
        obj.__init__(*args, **kwargs)
        return obj

    def exp(self, v=None, jacobian=False, type='grid'):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        grid = spatial.shoot(v, self.kernel, steps=self.steps,
                             factor=self.lam / py.prod(self.shape),
                             voxel_size=self.voxel_size, **self.prm,
                             displacement=type != 'grid')
        if jacobian:
            jac = spatial.grid_jacobian(grid, type=type)
            return grid, jac
        else:
            return grid

    def iexp(self, v=None, jacobian=False, type='grid'):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        _, grid = spatial.shoot(v, self.kernel, steps=self.steps,
                                factor=self.lam / py.prod(self.shape),
                                voxel_size=self.voxel_size, **self.prm,
                                return_inverse=True,
                                displacement=type != 'grid')
        if jacobian:
            jac = spatial.grid_jacobian(grid, type=type)
            return grid, jac
        else:
            return grid

    def exp2(self, v=None, jacobian=False, type='grid'):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        grid, igrid = spatial.shoot(v, self.kernel, steps=self.steps,
                                    factor=self.lam / py.prod(self.shape),
                                    voxel_size=self.voxel_size, **self.prm,
                                    return_inverse=True,
                                    displacement=type != 'grid')
        if jacobian:
            jac = spatial.grid_jacobian(grid, type=type)
            ijac = spatial.grid_jacobian(igrid, type=type)
            return grid, igrid, jac, ijac
        else:
            return grid, igrid

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        """Convert derivatives wrt warped image in loss space to
        to derivatives wrt parameters
        parameters:
            g (tensor) : gradient wrt warped image
            h (tensor) : hessian wrt warped image
            moving (Image) : moving image
            phi (tensor) : dense (exponentiated) displacement field
            left (matrix) : left affine
            right (matrix) : right affine
            inv (bool) : whether we're in a backward symmetric pass
        returns:
            g (tensor) : pushed gradient
            h (tensor) : pushed hessian
            gmu (tensor) : rotated spatial gradients
        """
        # build bits of warp
        dim = phi.shape[-1]
        fixed_shape = g.shape[-dim:]
        moving_shape = moving.shape
        if inv:
            # differentiate wrt δ in: Left o (Id + D) o (Id + δ) o Right
            if right is not None:
                right = spatial.affine_grid(right, fixed_shape)
            g = regutils.smart_push(g, right, shape=self.shape)
            h = regutils.smart_push(h, right, shape=self.shape)
            del right

            phi_left = spatial.identity_grid(self.shape, **utils.backend(phi))
            phi_left += phi
            if left is not None:
                phi_left = spatial.affine_matvec(left, phi_left)
            mugrad = moving.pull_grad(phi_left, rotate=False)
            del phi_left

            mugrad = _rotate_grad(mugrad, left, phi)

        else:
            # differentiate wrt δ in: Left o (Id - δ) o (Id + D) o Right
            if right is not None:
                right = spatial.affine_grid(right, fixed_shape)
                phi_right = regutils.smart_pull_grid(phi, right)
                phi_right += right
            else:
                phi_right = spatial.identity_grid(self.shape, **utils.backend(phi))
                phi_right += phi

            g = g.neg_()
            g = regutils.smart_push(g, phi_right, shape=self.shape)
            h = regutils.smart_push(h, phi_right, shape=self.shape)
            del phi_right

            if left is not None:
                grid_left = spatial.affine_grid(left, self.shape)
            else:
                grid_left = left
            mugrad = moving.pull_grad(grid_left)
            del grid_left
            mugrad = _rotate_grad(mugrad, left)

        return g, h, mugrad


class SVFModel(NonLinModel):
    """Stationary velocity field exponentiated by scaling and squaring."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(SVFModel)
        obj.__init__(*args, **kwargs)
        return obj

    def exp(self, v=None, jacobian=False, type='grid'):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        return spatial.exp_forward(v, steps=self.steps,
                                   jacobian=jacobian,
                                   displacement=type != 'grid')

    def iexp(self, v=None, jacobian=False, type='grid'):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        return spatial.exp_forward(v, steps=self.steps,
                                   jacobian=jacobian, inverse=True,
                                   displacement=type != 'grid')

    def exp2(self, v=None, jacobian=False, type='grid'):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            grid, jac = spatial.exp_forward(v, steps=self.steps,
                                            jacobian=True,
                                            displacement=type != 'grid')
            igrid, ijac = spatial.exp_forward(v, steps=self.steps,
                                              jacobian=True, inverse=True,
                                              displacement=type != 'grid')
            return grid, igrid, jac, ijac
        else:
            grid = spatial.exp_forward(v, steps=self.steps,
                                            jacobian=False,
                                            displacement=type != 'grid')
            igrid = spatial.exp_forward(v, steps=self.steps,
                                              jacobian=False, inverse=True,
                                              displacement=type != 'grid')
            return grid, igrid

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        """Convert derivatives wrt warped image in loss space to
        to derivatives wrt parameters
        parameters:
            g (tensor) : gradient wrt warped image
            h (tensor) : hessian wrt warped image
            moving (Image) : moving image
            phi (tensor) : dense (exponentiated) displacement field
            left (matrix) : left affine
            right (matrix) : right affine
            inv (bool) : whether we're in a backward symmetric pass
        returns:
            g (tensor) : pushed gradient
            h (tensor) : pushed hessian
            gmu (tensor) : rotated spatial gradients
        """
        if inv:
            g = g.neg_()

        # build bits of warp
        dim = phi.shape[-1]
        fixed_shape = g.shape[-dim:]
        moving_shape = moving.shape

        # differentiate wrt δ in: Left o Phi o (Id + δ) o Right
        # we'll then propagate them through Phi by scaling and squaring
        if right is not None:
            right = spatial.affine_grid(right, fixed_shape)
        g = regutils.smart_push(g, right, shape=self.shape)
        h = regutils.smart_push(h, right, shape=self.shape)
        del right

        phi_left = spatial.identity_grid(self.shape, **utils.backend(phi))
        phi_left += phi
        if left is not None:
            phi_left = spatial.affine_matvec(left, phi_left)
        mugrad = moving.pull_grad(phi_left, rotate=False)
        del phi_left

        mugrad = _rotate_grad(mugrad, left, phi)

        return g, h, mugrad


class SmallDefModel(NonLinModel):
    """Dense displacement field."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(SmallDefModel)
        obj.__init__(*args, **kwargs)
        return obj

    def exp(self, v=None, jacobian=False, type='grid'):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        grid = v
        if type == 'grid':
            id = spatial.identity_grid(grid.shape[:-1], **utils.backend(grid))
            grid = grid + id
        if jacobian:
            jac = spatial.grid_jacobian(grid, type=type)
            return grid, jac
        else:
            return grid

    def iexp(self, v=None, jacobian=False, type='grid'):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        grid = spatial.grid_inv(v, type='disp', **self.prm)
        if type == 'grid':
            id = spatial.identity_grid(grid.shape[:-1], **utils.backend(grid))
            grid = grid + id
        if jacobian:
            jac = spatial.grid_jacobian(grid, type=type)
            return grid, jac
        else:
            return grid

    def exp2(self, v=None, jacobian=False, type='grid'):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        grid = v
        igrid = spatial.grid_inv(v, type='disp', **self.prm)
        if type == 'grid':
            id = spatial.identity_grid(grid.shape[:-1], **utils.backend(grid))
            grid = grid + id
            igrid = igrid + id
        if jacobian:
            jac = spatial.grid_jacobian(grid, type=type)
            ijac = spatial.grid_jacobian(igrid, type=type)
            return grid, igrid, jac, ijac
        else:
            return grid, igrid

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        """Convert derivatives wrt warped image in loss space to
        to derivatives wrt parameters
        parameters:
            g (tensor) : gradient wrt warped image
            h (tensor) : hessian wrt warped image
            moving (Image) : moving image
            phi (tensor) : dense (exponentiated) displacement field
            left (matrix) : left affine
            right (matrix) : right affine
            inv (bool) : whether we're in a backward symmetric pass
        returns:
            g (tensor) : pushed gradient
            h (tensor) : pushed hessian
            gmu (tensor) : rotated spatial gradients
        """
        if inv:
            g = g.neg_()

        # build bits of warp
        dim = phi.shape[-1]
        fixed_shape = g.shape[-dim:]

        # differentiate wrt δ in: Left o (Id + D + δ) o Right
        # (If `inv`, we assume that inv(Id + D) = Id - D)
        if right is not None:
            right = spatial.affine_grid(right, fixed_shape)
        g = regutils.smart_push(g, right, shape=self.shape)
        h = regutils.smart_push(h, right, shape=self.shape)
        del right

        phi_left = spatial.identity_grid(self.shape, **utils.backend(phi))
        phi_left += phi
        if left is not None:
            phi_left = spatial.affine_matvec(left, phi_left)
        mugrad = moving.pull_grad(phi_left, rotate=False)
        del phi_left

        mugrad = _rotate_grad(mugrad, left)
        return g, h, mugrad


class AffineModel:
    """Affine transformation model encoded in a Lie algebra"""

    # translate common names to Lie group names
    _name_to_basis = {
        'translation': 'T',
        'rotation': 'SO',
        'rigid': 'SE',
        'similitude': 'CSO',
        'affine': 'Aff+',
    }

    def __init__(self, basis, lam=1, prm=None, dat=None, position='symmetric'):
        """

        Parameters
        ----------
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            or the name of a Lie group.
        lam : float, default=1
            Regularization factor
        prm : dict(), unused
        dat : LogAffine or tensor, optional
            Pre-allocated log-parameters.
        position : {'moving', 'fixed', 'symmetric'}, default='symmetric'
            Which image should be rotated by this transformation.
            If 'symmetric', both images are rotated by the transformation and
            its inverse, towards a mean space; thereby making the model fully
            symmetric.
        """
        self.lam = lam
        self.prm = prm or {} # regutils.defaults_affine()
        self._basis = self._name_to_basis.get(basis, basis)
        self.position = position
        if dat is not None:
            self.set_dat(dat)
        else:
            self.dat = None

    def set_dat(self, dat=None, dim=None):
        self.dat = LogAffine.make(dat, basis=self._basis, dim=dim)
        return self

    basis = property(lambda self: self.dat.basis if self.dat is not None else None)

    def make(self, basis, **kwargs):
        if isinstance(basis, AffineModel):
            kwargs.setdefault('lam', basis.lam)
            kwargs.setdefault('prm', basis.prm)
            kwargs.setdefault('dat', basis.dat)
            kwargs.setdefault('position', basis.position)
            basis = basis.basis
        return AffineModel(basis, **kwargs)

    def exp(self, q=None, grad=False):
        return self.dat.exp(q, grad)

    def iexp(self, q=None, grad=False):
        return self.dat.iexp(q, grad)

    def exp2(self, q=None, grad=False):
        return self.dat.exp2(q, grad)

    def __repr__(self):
        s = []
        if self.dat is not None:
            s += [f'log={self.dat}']
        else:
            s += ['<uninitialized>']
        s += [f'basis={self._basis}']
        s += [f'lam={self.lam}']
        s += [f'{key}={value}' for key, value in self.prm.items()]
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'


def _almost_identity(aff):
    return torch.allclose(aff, torch.eye(*aff.shape, **utils.backend(aff)))


class RegisterStep:
    """Forward pass of Diffeo+Affine registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(
            self,
            losses,                 # list[LossComponent]
            affine=None,            # AffineModel
            nonlin=None,            # NonLinModel
            verbose=True,           # verbosity level
            ):
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        self.losses = losses
        self.affine = affine
        self.nonlin = nonlin
        self.verbose = verbose

        # pretty printing
        self.n_iter = 0             # current iteration
        self.ll_prev = None         # previous loss value
        self.ll_max = 0             # max loss value
        self.llv = 0                # last velocity penalty
        self.lla = 0                # last affine penalty

        self.framerate = 1
        self._last_plot = 0
        if self.verbose > 2:
            import matplotlib.pyplot as plt
            self.figure = plt.figure()

    def mov2fix(self, fixed, moving, warped, vel=None, cat=False, dim=None, title=None):
        """Plot registration live"""

        import time
        tic = self._last_plot
        toc = time.time()
        if toc - tic < 1/self.framerate:
            return
        self._last_plot = toc

        import matplotlib.pyplot as plt

        warped = warped.detach()
        if vel is not None:
            vel = vel.detach()

        dim = dim or (fixed.dim() - 1)
        if fixed.dim() < dim + 2:
            fixed = fixed[None]
        if moving.dim() < dim + 2:
            moving = moving[None]
        if warped.dim() < dim + 2:
            warped = warped[None]
        if vel is not None:
            if vel.dim() < dim + 2:
                vel = vel[None]
        nb_channels = fixed.shape[-dim - 1]
        nb_batch = len(fixed)

        if dim == 3:
            fixed = fixed[..., fixed.shape[-1] // 2]
            moving = moving[..., moving.shape[-1] // 2]
            warped = warped[..., warped.shape[-1] // 2]
            if vel is not None:
                vel = vel[..., vel.shape[-2] // 2, :]
        if vel is not None:
            vel = vel.square().sum(-1).sqrt()

        if cat:
            moving = math.softmax(moving, dim=1, implicit=True)
            warped = math.softmax(warped, dim=1, implicit=True)

        checker = fixed.clone()
        patch = max([s // 8 for s in fixed.shape])
        checker_unfold = utils.unfold(checker, [patch] * 2, [2 * patch] * 2)
        warped_unfold = utils.unfold(warped, [patch] * 2, [2 * patch] * 2)
        checker_unfold.copy_(warped_unfold)

        nb_rows = min(nb_batch, 3)
        nb_cols = 4 + (vel is not None)

        if len(self.figure.axes) != nb_rows*nb_cols:
            self.figure.clf()

            for b in range(nb_rows):
                plt.subplot(nb_rows, nb_cols, b * nb_cols + 1)
                plt.imshow(moving[b, 0].cpu())
                plt.title('moving')
                plt.axis('off')
                plt.subplot(nb_rows, nb_cols, b * nb_cols + 2)
                plt.imshow(warped[b, 0].cpu())
                plt.title('moved')
                plt.axis('off')
                plt.subplot(nb_rows, nb_cols, b * nb_cols + 3)
                plt.imshow(checker[b, 0].cpu())
                plt.title('checker')
                plt.axis('off')
                plt.subplot(nb_rows, nb_cols, b * nb_cols + 4)
                plt.imshow(fixed[b, 0].cpu())
                plt.title('fixed')
                plt.axis('off')
                if vel is not None:
                    plt.subplot(nb_rows, nb_cols, b * nb_cols + 5)
                    plt.imshow(vel[b].cpu())
                    plt.title('velocity')
                    plt.axis('off')
                    plt.colorbar()
            if title:
                plt.suptitle(title)

            self.figure.canvas.draw()
            self.plt_saved = [self.figure.canvas.copy_from_bbox(ax.bbox)
                              for ax in self.figure.axes]
            self.figure.canvas.flush_events()
            plt.show()

        else:
            self.figure.canvas.draw()
            for elem in self.plt_saved:
                self.figure.canvas.restore_region(elem)

            for b in range(nb_rows):
                j = b * nb_cols
                self.figure.axes[j].images[0].set_data(moving[b, 0].cpu())
                self.figure.axes[j+1].images[0].set_data(warped[b, 0].cpu())
                self.figure.axes[j+2].images[0].set_data(checker[b, 0].cpu())
                self.figure.axes[j+3].images[0].set_data(fixed[b, 0].cpu())
                if vel is not None:
                    self.figure.axes[j+4].images[0].set_data(vel[b].cpu())
            if title:
                self.figure._suptitle.set_text(title)

            for ax in self.figure.axes:
                ax.draw_artist(ax.images[0])
                self.figure.canvas.blit(ax.bbox)
            self.figure.canvas.flush_events()

    def do_vel(self, vel, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the nonlinear component"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # build affine and displacement field
        if self.affine:
            aff0 = getattr(self.affine.dat, '_cache_aff', None)
            iaff0 = getattr(self.affine.dat, '_cache_iaff', None)
            if aff0 is None and iaff0 is None:
                aff0, iaff0 = self.affine.exp2()
            elif aff0 is None:
                aff0 = self.affine.exp()
            elif iaff0 is None:
                iaff0 = self.affine.iexp()
            self.affine.dat._cache_aff = aff0
            self.affine.dat._cache_iaff = iaff0
        else:
            aff0 = iaff0 = torch.eye(self.nonlin.dim + 1, **utils.backend(self.nonlin.dat))
        vel0 = vel
        if any(loss.symmetric for loss in self.losses):
            phi0, iphi0 = self.nonlin.exp2(vel0, type='displacement')
            ivel0 = -vel0
        else:
            phi0 = self.nonlin.exp(vel0, type='displacement')
            iphi0 = ivel0 = None
        if self.affine and not in_line_search:
            self.nonlin.dat._cache_phi = phi0
            self.nonlin.dat._cache_iphi = iphi0

        # register temporary "backward" loss for symmetric losses
        losses = []
        for loss in self.losses:
            losses.append(loss)
            if loss.symmetric:
                bwdloss = copy.copy(loss)
                bwdloss.moving, bwdloss.fixed = loss.fixed, loss.moving
                bwdloss.symmetric = 'backward'
                losses.append(bwdloss)

        for loss in losses:

            factor = loss.weight
            if loss.symmetric:
                factor = factor / 2
            if loss.symmetric == 'backward':
                phi00 = iphi0
                aff00 = iaff0
                vel00 = ivel0
            else:
                phi00 = phi0
                aff00 = aff0
                vel00 = vel0

            is_level0 = True
            for moving, fixed in zip(loss.moving, loss.fixed):  # pyramid

                # build left and right affine
                if getattr(self.affine, 'position', None) in ('fixed', 'symmetric'):
                    aff_right = spatial.affine_matmul(aff00, fixed.affine)
                else:
                    aff_right = fixed.affine
                aff_right = spatial.affine_lmdiv(self.nonlin.affine, aff_right)
                if getattr(self.affine, 'position', None) in ('moving', 'symmetric'):
                    tmp = spatial.affine_matmul(aff00, self.nonlin.affine)
                    aff_left = spatial.affine_lmdiv(moving.affine, tmp)
                else:
                    aff_left = spatial.affine_lmdiv(moving.affine, self.nonlin.affine)

                # build full transform
                if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                    aff_right = None
                    phi = spatial.identity_grid(fixed.shape, **utils.backend(phi00))
                    phi += phi00
                else:
                    phi = spatial.affine_grid(aff_right, fixed.shape)
                    phi += regutils.smart_pull_grid(phi00, phi)
                if _almost_identity(aff_left) and moving.shape == self.nonlin.shape:
                    aff_left = None
                else:
                    phi = spatial.affine_matvec(aff_left, phi)

                # forward
                warped = moving.pull(phi)

                if is_level0 and self.verbose > 2 and not in_line_search:
                    is_level0 = False
                    from .plot import mov2fix
                    self.mov2fix(fixed.dat, moving.dat, warped, vel0,
                                 dim=fixed.dim,
                                 title=f'(nonlin) {self.n_iter:03d}')

                # gradient/Hessian of the log-likelihood in observed space
                g = h = None
                if not grad and not hess:
                    llx = loss.loss.loss(warped, fixed.dat, dim=fixed.dim)
                elif not hess:
                    llx, g = loss.loss.loss_grad(warped, fixed.dat, dim=fixed.dim)
                else:
                    llx, g, h = loss.loss.loss_grad_hess(warped, fixed.dat, dim=fixed.dim)

                # compose with spatial gradients
                if grad or hess:

                    g, h, mugrad = self.nonlin.propagate_grad(
                        g, h, moving, phi00, aff_left, aff_right,
                        inv=(loss.symmetric == 'backward'))

                    g = regutils.jg(mugrad, g)
                    h = regutils.jhj(mugrad, h)

                    if isinstance(self.nonlin, SVFModel):
                        # propagate backward by scaling and squaring
                        g, h = spatial.exp_backward(vel00, g, h, steps=self.nonlin.steps)

                    sumgrad = g.mul_(factor) if sumgrad is None else sumgrad.add_(g, alpha=factor)
                    if hess:
                        sumhess = h.mul_(factor) if sumhess is None else sumhess.add_(h, alpha=factor)
                sumloss = llx.mul_(factor) if sumloss is None else sumloss.add_(llx, alpha=factor)

        # add regularization term
        vgrad = self.nonlin.regulariser(vel0)
        llv = 0.5 * vel0.flatten().dot(vgrad.flatten())
        if grad:
            sumgrad += vgrad
        del vgrad

        # print objective
        llx = sumloss.item()
        sumloss += llv
        sumloss += self.lla
        if self.verbose and not in_line_search:
            llv = llv.item()
            self.llv = llv
            ll = sumloss.item()
            lla = self.lla
            self.n_iter += 1
            line = '(nonlin) | '
            line += f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} + {lla:12.6g} = {ll:12.6g}'
            if self.ll_prev is not None:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                line += f' | {gain:12.6g}'
            print(line, end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

    def do_affine(self, logaff, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the affine component (nonlin is not None)"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # build affine and displacement field
        logaff0 = logaff
        if any(loss.symmetric for loss in self.losses):
            aff0, iaff0, gaff0, igaff0 = self.affine.exp2(logaff0, grad=True)
            phi0 = getattr(self.nonlin.dat, '_cache_phi', None)
            iphi0 = getattr(self.nonlin.dat, '_cache_iphi0', None)
            if phi0 is None and iphi0 is None:
                phi0, iphi0 = self.nonlin.exp2(type='displacement')
            elif phi0 is None:
                phi0 = self.nonlin.exp(type='displacement')
            elif iphi0 is None:
                iphi0 = self.nonlin.iexp(type='displacement')
            self.nonlin.dat._cache_phi = phi0
            self.nonlin.dat._cache_iphi = iphi0
        else:
            iaff0 = None
            aff0, gaff0 = self.affine.exp(logaff0, grad=True)
            phi0 = getattr(self.nonlin.dat, '_cache_phi', None)
            if phi0 is None:
                phi0 = self.nonlin.exp(type='displacement')
            self.nonlin.dat._cache_phi = phi0
            iphi0 = None
        if not in_line_search:
            self.affine.dat._cache_aff = aff0
            self.affine.dat._cache_iaff = iaff0

        # register temporary "backward" loss for symmetric losses
        losses = []
        for loss in self.losses:
            losses.append(loss)
            if loss.symmetric:
                bwdloss = copy.copy(loss)
                bwdloss.moving, bwdloss.fixed = loss.fixed, loss.moving
                bwdloss.symmetric = 'backward'
                losses.append(bwdloss)

        for loss in losses:

            factor = loss.weight
            if loss.symmetric:
                factor = factor / 2
            if loss.symmetric == 'backward':
                phi00 = iphi0
                aff00 = iaff0
                gaff00 = igaff0
            else:
                phi00 = phi0
                aff00 = aff0
                gaff00 = gaff0

            is_level0 = True
            for moving, fixed in zip(loss.moving, loss.fixed):  # pyramid

                # build complete warp
                if getattr(self.affine, 'position', None) in ('fixed', 'symmetric'):
                    aff_right = spatial.affine_matmul(aff00, fixed.affine)
                    aff_right = spatial.affine_lmdiv(self.nonlin.affine, aff_right)
                    gaff_right = torch.matmul(gaff00, fixed.affine)
                    gaff_right = linalg.lmdiv(self.nonlin.affine, gaff_right)
                else:
                    aff_right = spatial.affine_lmdiv(self.nonlin.affine, fixed.affine)
                    gaff_right = None
                if getattr(self.affine, 'position', None) in ('moving', 'symmetric'):
                    aff_left = spatial.affine_matmul(aff00, self.nonlin.affine)
                    aff_left = spatial.affine_lmdiv(moving.affine, aff_left)
                    gaff_left = torch.matmul(gaff00, self.nonlin.affine)
                    gaff_left = linalg.lmdiv(moving.affine, gaff_left)
                else:
                    aff_left = spatial.affine_lmdiv(moving.affine, self.nonlin.affine)
                    gaff_left = None
                if _almost_identity(aff_right) and fixed.shape == self.nonlin.shape:
                    right = None
                    phi = spatial.identity_grid(fixed.shape, **utils.backend(phi00))
                    phi += phi00
                else:
                    right = spatial.affine_grid(aff_right, fixed.shape)
                    phi = regutils.smart_pull_grid(phi00, right)
                    phi += right
                phi_right = phi
                if _almost_identity(aff_left) and moving.shape == self.nonlin.shape:
                    left = None
                else:
                    left = spatial.affine_grid(aff_left, self.nonlin.shape)
                    phi = spatial.affine_matvec(aff_left, phi)

                # forward
                warped = moving.pull(phi)

                if is_level0 and self.verbose > 2 and not in_line_search:
                    is_level0 = False
                    from .plot import mov2fix
                    self.mov2fix(fixed.dat, moving.dat, warped, dim=fixed.dim,
                                 title=f'(affine) {self.n_iter:03d}')

                # gradient/Hessian of the log-likelihood in observed space
                g = h = None
                if not grad and not hess:
                    llx = loss.loss.loss(warped, fixed.dat, dim=fixed.dim)
                elif not hess:
                    llx, g = loss.loss.loss_grad(warped, fixed.dat, dim=fixed.dim)
                else:
                    llx, g, h = loss.loss.loss_grad_hess(warped, fixed.dat, dim=fixed.dim)

                def compose_grad(g, h, g_mu, g_aff):
                    """
                    g, h : gradient/Hessian of loss wrt moving image
                    g_mu : spatial gradients of moving image
                    g_aff : gradient of affine matrix wrt Lie parameters
                    returns g, h: gradient/Hessian of loss wrt Lie parameters
                    """
                    # Note that `h` can be `None`, but the functions I
                    # use deal with this case correctly.
                    dim = g_mu.shape[-1]
                    g = jg(g_mu, g)
                    h = jhj(g_mu, h)
                    g, h = regutils.affine_grid_backward(g, h)
                    dim2 = dim * (dim + 1)
                    g = g.reshape([*g.shape[:-2], dim2])
                    g_aff = g_aff[..., :-1, :]
                    g_aff = g_aff.reshape([*g_aff.shape[:-2], dim2])
                    g = linalg.matvec(g_aff, g)
                    if h is not None:
                        h = h.reshape([*h.shape[:-4], dim2, dim2])
                        h = g_aff.matmul(h).matmul(g_aff.transpose(-1, -2))
                        h = h.abs().sum(-1).diag_embed()
                    return g, h

                # compose with spatial gradients
                if grad or hess:
                    g0, g = g, None
                    h0, h = h, None
                    if self.affine.position in ('moving', 'symmetric'):
                        g_left = regutils.smart_push(g0, phi_right, shape=self.nonlin.shape)
                        h_left = regutils.smart_push(h0, phi_right, shape=self.nonlin.shape)
                        mugrad = moving.pull_grad(left, rotate=False)
                        g_left, h_left = compose_grad(g_left, h_left, mugrad, gaff_left)
                        g = g_left
                        h = h_left
                    if self.affine.position in ('fixed', 'symmetric'):
                        g_right = g0
                        h_right = h0
                        mugrad = moving.pull_grad(phi, rotate=False)
                        jac = spatial.grid_jacobian(phi0, right, type='disp', extrapolate=False)
                        jac = torch.matmul(aff_left[:-1, :-1], jac)
                        mugrad = linalg.matvec(jac.transpose(-1, -2), mugrad)
                        g_right, h_right = compose_grad(g_right, h_right, mugrad, gaff_right)
                        g = g_right if g is None else g.add_(g_right)
                        h = h_right if h is None else h.add_(h_right)

                    if loss.symmetric == 'backward':
                        g = g.neg_()
                    sumgrad = g.mul_(factor) if sumgrad is None else sumgrad.add_(g, alpha=factor)
                    if hess:
                        sumhess = h.mul_(factor) if sumhess is None else sumhess.add_(h, alpha=factor)
                sumloss = llx.mul_(factor) if sumloss is None else sumloss.add_(llx, alpha=factor)

        # TODO add regularization term
        lla = 0

        # print objective
        llx = sumloss.item()
        sumloss += lla
        sumloss += self.llv
        if self.verbose and not in_line_search:
            self.n_iter += 1
            ll = sumloss.item()
            llv = self.llv
            line = '(affine) | '
            line += f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} + {lla:12.6g} = {ll:12.6g}'
            if self.ll_prev is not None:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                line += f' | {gain:12.6g}'
            print(line, end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]

    def do_affine_only(self, logaff, grad=False, hess=False, in_line_search=False):
        """Forward pass for updating the affine component (nonlin is None)"""

        sumloss = None
        sumgrad = None
        sumhess = None

        # build affine and displacement field
        logaff0 = logaff
        aff0, iaff0, gaff0, igaff0 = self.affine.exp2(logaff0, grad=True)

        # register temporary "backward" loss for symmetric losses
        losses = []
        for loss in self.losses:
            losses.append(loss)
            if loss.symmetric:
                bwdloss = copy.copy(loss)
                bwdloss.moving, bwdloss.fixed = loss.fixed, loss.moving
                bwdloss.symmetric = 'backward'
                losses.append(bwdloss)

        for loss in losses:

            factor = loss.weight
            if loss.symmetric:
                factor = factor / 2
            if loss.symmetric == 'backward':
                aff00 = iaff0
                gaff00 = igaff0
            else:
                aff00 = aff0
                gaff00 = gaff0

            is_level0 = True
            for moving, fixed in zip(loss.moving, loss.fixed):  # pyramid

                # build complete warp
                aff = spatial.affine_matmul(aff00, fixed.affine)
                aff = spatial.affine_lmdiv(moving.affine, aff)
                gaff = torch.matmul(gaff00, fixed.affine)
                gaff = linalg.lmdiv(moving.affine, gaff)
                phi = spatial.affine_grid(aff, fixed.shape)

                # forward
                warped = moving.pull(phi)

                if is_level0 and self.verbose > 2 and not in_line_search:
                    is_level0 = False
                    from .plot import mov2fix
                    self.mov2fix(fixed.dat, moving.dat, warped, dim=fixed.dim,
                                 title=f'(affine) {self.n_iter:03d}')

                # gradient/Hessian of the log-likelihood in observed space
                g = h = None
                if not grad and not hess:
                    llx = loss.loss.loss(warped, fixed.dat, dim=fixed.dim)
                elif not hess:
                    llx, g = loss.loss.loss_grad(warped, fixed.dat, dim=fixed.dim)
                else:
                    llx, g, h = loss.loss.loss_grad_hess(warped, fixed.dat, dim=fixed.dim)

                def compose_grad(g, h, g_mu, g_aff):
                    """
                    g, h : gradient/Hessian of loss wrt moving image
                    g_mu : spatial gradients of moving image
                    g_aff : gradient of affine matrix wrt Lie parameters
                    returns g, h: gradient/Hessian of loss wrt Lie parameters
                    """
                    # Note that `h` can be `None`, but the functions I
                    # use deal with this case correctly.
                    dim = g_mu.shape[-1]
                    g = jg(g_mu, g)
                    h = jhj(g_mu, h)
                    g, h = regutils.affine_grid_backward(g, h)
                    dim2 = dim * (dim + 1)
                    g = g.reshape([*g.shape[:-2], dim2])
                    g_aff = g_aff[..., :-1, :]
                    g_aff = g_aff.reshape([*g_aff.shape[:-2], dim2])
                    g = linalg.matvec(g_aff, g)
                    if h is not None:
                        h = h.reshape([*h.shape[:-4], dim2, dim2])
                        h = g_aff.matmul(h).matmul(g_aff.transpose(-1, -2))
                        h = h.abs().sum(-1).diag_embed()
                    return g, h

                # compose with spatial gradients
                if grad or hess:
                    mugrad = moving.pull_grad(phi, rotate=False)
                    g, h = compose_grad(g, h, mugrad, gaff)

                    if loss.symmetric == 'backward':
                        g = g.neg_()
                    sumgrad = g.mul_(factor) if sumgrad is None else sumgrad.add_(g, alpha=factor)
                    if hess:
                        sumhess = h.mul_(factor) if sumhess is None else sumhess.add_(h, alpha=factor)
                sumloss = llx.mul_(factor) if sumloss is None else sumloss.add_(llx, alpha=factor)

        # TODO add regularization term
        llv = 0

        # print objective
        llx = sumloss.item()
        sumloss += llv
        llv = llv
        ll = sumloss.item()
        if self.verbose and not in_line_search:
            self.n_iter += 1
            if self.ll_prev is None:
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g}', end='\r')
            else:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [sumloss]
        if grad:
            out.append(sumgrad)
        if hess:
            out.append(sumhess)
        return tuple(out) if len(out) > 1 else out[0]


class Register:

    def __init__(self,
                 losses,                 # list[LossComponent]
                 affine=None,            # AffineModel
                 nonlin=None,            # NonLinModel
                 optim=None,             # Optimizer
                 verbose=True,           # verbosity level
                 ):
        self.losses = losses
        self.verbose = verbose
        self.affine = affine
        self.nonlin = nonlin
        self.optim = optim

    def __call__(self):
        if self.affine is not None and self.affine.dat is None:
            self.affine = self.affine.set_dat(dim=self.losses[0].fixed.dim)
        if self.nonlin is not None and self.nonlin.dat is None:
            space = MeanSpace([loss.fixed for loss in self.losses] +
                              [loss.moving for loss in self.losses])
            self.nonlin.set_dat(space.shape, affine=space.affine)

        step = RegisterStep(self.losses, self.affine, self.nonlin, self.verbose)
        if self.affine is not None and self.nonlin is not None:
            if isinstance(self.optim.optim[1], optm.FirstOrder):
                self.optim.optim[1].preconditioner = lambda x: self.nonlin.greens_apply(x)
            elif isinstance(self.optim.optim[1].optim.optim, optm.FirstOrder):
                self.optim.optim[1].preconditioner = lambda x: self.nonlin.greens_apply(x)
            self.optim.iter([self.affine.dat.dat, self.nonlin.dat.dat],
                            [step.do_affine, step.do_vel])
        elif self.affine is not None:
            self.optim.iter(self.affine.dat.dat, step.do_affine_only)
        elif self.nonlin is not None:
            if isinstance(self.optim, optm.FirstOrder):
                self.optim.preconditioner = lambda x: self.nonlin.greens_apply(x)
            elif isinstance(self.optim.optim.optim, optm.FirstOrder):
                self.optim.preconditioner = lambda x: self.nonlin.greens_apply(x)
            self.optim.iter(self.nonlin.dat.dat, step.do_vel)

        return


