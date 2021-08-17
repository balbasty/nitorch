"""Object-oriented representation of images, losses, transforms, etc."""
from nitorch import spatial, io
from nitorch.core import py, utils, linalg
import torch
from . import losses, utils as regutils
import copy


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
                 bound='dct2', extrapolate=False):
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
        if self.extrapolate:
            s += ['extrapolate=True']
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
                 bound='dct2', extrapolate=False, method='gauss', **backend):
        """

        Parameters
        ----------
        dat : [list of] (..., *shape) tensor or Image
        levels : int or list[int] or range, default=0
            If an int, it is the number of levels.
            If a range or list, they are the indices of levels to compute.
            `0` is the native resolution, `1` is half of it, etc.
        affine : [list of] tensor, optional
        dim : int, optional
        bound : str, default='dct2'
        extrapolate : bool, default=True
        method : {'gauss', 'average', 'median'}, default='gauss'
        """
        # I don't call super().__init__() on purpose
        self.method = method

        if isinstance(dat, str):
            dat = io.map(dat)
        if isinstance(dat, io.MappedArray):
            if affine is None:
                affine = dat.affine
            dat = dat.fdata(rand=True, **backend)[None]

        if isinstance(levels, int):
            levels = range(levels)
        if isinstance(levels, range):
            levels = list(levels)

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
        dats = [dat] * levels.count(0)
        for level in range(1, nb_levels+1):
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
        affines = [affine] * levels.count(0)
        for level in range(1, nb_levels+1):
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
        self._cache = None
        self._icache = None

    def clear_cache(self):
        self._cache = None
        self._icache = None

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, LogAffine):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('basis', dat.basis)
            dat = dat.dat
        return LogAffine(dat, **kwargs)

    def exp(self, q=None, grad=False, cache_result=False, recompute=True):
        if q is None:
            q = self.dat
        if grad:
            recompute = True
        if recompute or getattr(self, '_cache') is None:
            aff = linalg._expm(q, self.basis, grad_X=grad)
        else:
            aff = self._cache
        if cache_result:
            self._cache = aff[0] if grad else aff
        return aff

    def iexp(self, q=None, grad=False, cache_result=False, recompute=True):
        if q is None:
            q = self.dat
        if grad:
            recompute = True
        if recompute or self._cache is None:
            iaff = linalg._expm(-q, self.basis, grad_X=grad)
        else:
            iaff = self._cache
        if cache_result:
            self._cache = iaff[0] if grad else iaff
        return iaff

    def exp2(self, q=None, grad=False, cache_result=False, recompute=True):
        if grad:
            recompute = True
        if grad:
            a, g = self.exp(q, True)
            ia, ig = self.iexp(q, True)
            return a, ia, g, ig
        else:
            return (self.exp(q, cache_result=cache_result, recompute=recompute),
                    self.iexp(q, cache_result=cache_result, recompute=recompute))

    def __repr__(self):
        s = [f'shape={list(self.dat.shape)}', f'dim={self.dim}']
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'

    __str__ = __repr__


class LossComponent:
    """Component of a composite loss"""
    def __init__(self, loss, moving, fixed, factor=1, symmetric=False):
        """

        Parameters
        ----------
        loss : OptimizationLoss (or its name)
            Matching loss to minimize.
        moving : Image or ImagePyramid like
        fixed : Image or ImagePyramid like
            Fixed and moving should have the same number of levels.
        factor : float, default=1
            Weight of the loss
        symmetric : bool, default=False
            Whether the loss should be symmetric
            i.e., loss = (loss(moving, fixed) + loss(fixed, moving)) / 2
        """
        self.moving = ImagePyramid.make(moving)
        self.fixed = ImagePyramid.make(fixed)
        self.loss = losses.make_loss(loss, self.moving.dim)
        self.factor = factor
        self.symmetric = symmetric

    def make(self, loss, **kwargs):
        if isinstance(loss, LossComponent):
            kwargs.setdefault('moving', loss.moving)
            kwargs.setdefault('fixed', loss.fixed)
            kwargs.setdefault('factor', loss.factor)
            kwargs.setdefault('symmetric', loss.symmetric)
            loss = loss.loss
        return LossComponent(loss, **kwargs)

    def __repr__(self):
        s = f'{type(self.loss).__name__}[{self.factor:.1g}](\n' \
            f'    moving={self.moving}, \n' \
            f'    fixed={self.fixed}'
        if self.symmetric:
            s += f',\n    symmetric=True'
        s += ')'
        return s

    __str_ = __repr__


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

    @classmethod
    def subclass(cls, model):
        model = model.lower()
        if model == 'shoot':
            return ShootModel
        elif model == 'svf':
            return SVFModel
        elif model == 'smalldef':
            return SmallDefModel
        else:
            raise ValueError('unknown:', model)

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

    def __init__(self, dat=None, factor=1, prm=None,
                 steps=8, kernel=None, **backend):
        """

        Parameters
        ----------
        dat : Displacement or tensor or list[int], optional
            Pre-allocated displacement or its shape
        factor : float, default=1
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
        self.factor = factor
        self.prm = prm or regutils.defaults_velocity()
        self.prm.pop('voxel_size', None)
        self.steps = steps
        self.kernel = kernel
        if dat is not None:
            self.set_dat(dat, **backend)
        else:
            self.dat = None
        self._cache = None
        self._icache = None

    affine = property(lambda self: self.dat.affine if self.dat is not None else None)
    shape = property(lambda self: self.dat.shape if self.dat is not None else None)
    dim = property(lambda self: self.dat.dim if self.dat is not None else None)
    voxel_size = property(lambda self: self.dat.voxel_size if self.dat is not None else None)

    def set_dat(self, dat, affine=None, **backend):
        if isinstance(dat, str):
            dat = io.map(dat)
            if affine is None:
                affine = self.dat.affine
        if isinstance(dat, SpatialTensor) and affine is None:
            affine = dat.affine
        self.dat = Displacement.make(dat, affine=affine).to(**backend)
        if self.kernel is None:
            self.kernel = spatial.greens(self.shape, **self.prm,
                                         factor=self.factor / py.prod(self.shape),
                                         voxel_size=self.voxel_size,
                                         **utils.backend(self.dat))
        return self

    def make(self, model, **kwargs):
        if isinstance(model, type(self)):
            kwargs.setdefault('factor', model.factor)
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

    @classmethod
    def add_identity(cls, disp):
        dim = disp.shape[-1]
        shape = disp.shape[-dim-1:-1]
        return spatial.identity_grid(shape, **utils.backend(disp)).add_(disp)

    def clear_cache(self):
        self._cache = None
        self._icache = None

    def regulariser(self, v=None):
        if v is None:
            v = self.dat
        return spatial.regulariser_grid(v, **self.prm,
                                        factor=self.factor / py.prod(self.shape),
                                        voxel_size=self.voxel_size)

    def greens_apply(self, m):
        return spatial.greens_apply(m, self.kernel, self.voxel_size)

    def __repr__(self):
        s = []
        if self.dat is not None:
            s += [f'velocity={self.dat}']
        else:
            s += ['<uninitialized>']
        s += [f'factor={self.factor}']
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

    def exp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        if recompute or self._cache is None:
            grid = spatial.shoot(v, self.kernel, steps=self.steps,
                                 factor=self.factor / py.prod(self.shape),
                                 voxel_size=self.voxel_size, **self.prm,
                                 displacement=True)
        if cache_result:
            self._cache = grid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            if add_identity:
                grid = self.add_identity(grid)
            return grid, jac
        else:
            if add_identity:
                grid = self.add_identity(grid)
            return grid

    def iexp(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        if recompute or self._icache is None:
            _, grid = spatial.shoot(v, self.kernel, steps=self.steps,
                                    factor=self.factor / py.prod(self.shape),
                                    voxel_size=self.voxel_size, **self.prm,
                                    return_inverse=True,  displacement=True)
        else:
            grid = self._icache
        if cache_result:
            self._icache = grid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            if add_identity:
                grid = self.add_identity(grid)
            return grid, jac
        else:
            if add_identity:
                grid = self.add_identity(grid)
            return grid

    def exp2(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        if recompute or self._cache is None or self._icache is None:
            grid, igrid = spatial.shoot(v, self.kernel, steps=self.steps,
                                        factor=self.factor / py.prod(self.shape),
                                        voxel_size=self.voxel_size, **self.prm,
                                        return_inverse=True, displacement=True)
        if cache_result:
            self._cache = grid
            self._icache = igrid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            ijac = spatial.grid_jacobian(igrid, type='displacement')
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            return grid, igrid, jac, ijac
        else:
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
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

    def exp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            recompute = True
        if recompute or self._cache is None:
            grid = spatial.exp_forward(v, steps=self.steps,
                                       jacobian=jacobian,
                                       displacement=True)
        else:
            grid = self._cache
        if cache_result:
            self._cache = grid[0] if jacobian else grid
        if add_identity:
            if jacobian:
                grid = (self.add_identity(grid[0]), grid[1])
            else:
                grid = self.add_identity(grid)
        return grid

    def iexp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            recompute = True
        if recompute or self._icache is None:
            igrid = spatial.exp_forward(v, steps=self.steps,
                                        jacobian=jacobian, inverse=True,
                                        displacement=True)
        else:
            igrid = self._icache
        if cache_result:
            self._icache = igrid[0] if jacobian else igrid
        if add_identity:
            if jacobian:
                igrid = (self.add_identity(igrid[0]), igrid[1])
            else:
                igrid = self.add_identity(igrid)
        return igrid

    def exp2(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            grid, jac = spatial.exp_forward(v, steps=self.steps,
                                            jacobian=True,
                                            displacement=True)
            igrid, ijac = spatial.exp_forward(v, steps=self.steps,
                                              jacobian=True, inverse=True,
                                              displacement=True)
            if cache_result:
                self._cache = grid
                self._icache = igrid
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            return grid, igrid, jac, ijac
        else:
            if recompute or self._cache is None:
                grid = spatial.exp_forward(v, steps=self.steps,
                                           jacobian=False,
                                           displacement=True)
            else:
                grid = self._cache
            if recompute or self._icache is None:
                igrid = spatial.exp_forward(v, steps=self.steps,
                                            jacobian=False, inverse=True,
                                            displacement=True)
            else:
                igrid = self._icache
            if cache_result:
                self._cache = grid
                self._icache = igrid
            grid = self.add_identity(grid)
            igrid = self.add_identity(igrid)
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

    def exp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        grid = v
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        if recompute or self._icache is None:
            grid = spatial.grid_inv(v, type='disp', **self.prm)
        else:
            grid = self._icache
        if cache_result:
            self._icache = grid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        grid = v
        if recompute or self._icache is None:
            igrid = spatial.grid_inv(v, type='disp', **self.prm)
        else:
            igrid = self._icache
        if cache_result:
            self._icache = igrid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            ijac = spatial.grid_jacobian(igrid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
            igrid = self.add_identity(igrid)
        return (grid, igrid, jac, ijac) if jacobian else (grid, igrid)

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

    def __init__(self, basis, factor=1, prm=None, dat=None, position='symmetric'):
        """

        Parameters
        ----------
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            or the name of a Lie group.
        factor : float, default=1
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
        self.factor = factor
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
            kwargs.setdefault('factor', basis.factor)
            kwargs.setdefault('prm', basis.prm)
            kwargs.setdefault('dat', basis.dat)
            kwargs.setdefault('position', basis.position)
            basis = basis.basis
        return AffineModel(basis, **kwargs)

    def exp(self, q=None, grad=False, cache_result=False, recompute=True):
        return self.dat.exp(q, grad, cache_result, recompute)

    def iexp(self, q=None, grad=False, cache_result=False, recompute=True):
        return self.dat.iexp(q, grad, cache_result, recompute)

    def exp2(self, q=None, grad=False, cache_result=False, recompute=True):
        return self.dat.exp2(q, grad, cache_result, recompute)

    def __repr__(self):
        s = []
        if self.dat is not None:
            s += [f'log={self.dat}']
        else:
            s += ['<uninitialized>']
        s += [f'basis={self._basis}']
        s += [f'factor={self.factor}']
        s += [f'{key}={value}' for key, value in self.prm.items()]
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'
