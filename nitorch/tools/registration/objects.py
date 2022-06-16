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

    def __repr__(self):
        vx = spatial.voxel_size(self.affine).tolist()
        return f'{type(self).__name__}(shape={self.shape}, vx={vx})'


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
        self.affine = affine.to(utils.backend(self.dat)['device'])

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
    def __init__(self, dat, affine=None, dim=None, mask=None,
                 bound='dct2', extrapolate=False, preview=None):
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
        self.mask = mask
        if self.masked and mask.shape[-self.dim:] != self.shape:
            raise ValueError('Wrong shape for mask')
        self._preview = preview
        if self.previewed and preview.shape[-self.dim:] != self.shape:
            raise ValueError('Wrong shape for preview')

    masked = property(lambda self: self.mask is not None)
    previewed = property(lambda self: self._preview is not None)
    preview = property(lambda self: self._preview if self.previewed else self.dat)

    @preview.setter
    def preview(self, x):
        self._preview = x

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
            kwargs.setdefault('mask', dat.mask)
            dat = dat.mask
        return cls(dat.dat, **kwargs)

    def pull(self, grid, dat=True, mask=False, preview=False):
        """Sample the image at dense coordinates.

        Parameters
        ----------
        grid : (*spatial, dim) tensor or None
            Dense transformation field.

        Returns
        -------
        warped : ([C], *spatial) tensor

        """
        out = []
        if dat or (preview and not self.previewed):
            img = regutils.smart_pull(self.dat, grid, bound=self.bound,
                                      extrapolate=self.extrapolate)
            if dat:
                out += [img]
        if mask:
            msk = None
            if self.masked:
                msk = self.mask.to(self.dat.dtype)
                msk = regutils.smart_pull(msk, grid, bound=self.bound,
                                          extrapolate=self.extrapolate)
            out += [msk]
        if preview:
            if self.previewed:
                prv = self.preview.to(self.dat.dtype)
                prv = regutils.smart_pull(prv, grid, bound=self.bound,
                                          extrapolate=self.extrapolate)
            else:
                prv = img
            out += [prv]
        return tuple(out) if len(out) > 1 else out[0]

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

    def _prm_as_str(self):
        s = super()._prm_as_str()
        if self.masked:
            s += ['masked=True']
        return s


class ImageSequence(Image):
    def __init__(self, dat, affine=None, dim=None, mask=None,
                 bound='dct2', extrapolate=False, **backend):
        # I don't call super().__init__() on purpose
        if torch.is_tensor(affine):
            affine = [affine] * len(dat)
        elif affine is None:
            affine = []
            for dat1 in dat:
                dim1 = dim or dat1.dim
                if callable(dim1):
                    dim1 = dim1()
                if hasattr(dat1, 'affine'):
                    aff1 = dat1.affine
                else:
                    shape1 = dat1.shape[-dim1:]
                    aff1 = spatial.affine_default(shape1, **utils.backend(dat1))
                affine.append(aff1)
        affine = py.make_list(affine, len(dat))
        mask = py.make_list(mask, len(dat))
        self._dat = []
        for dat1, aff1, mask1 in zip(dat, affine, mask):
            if not isinstance(dat1, Image):
                dat1 = Image(dat1, aff1, mask=mask1, dim=dim,
                             bound=bound, extrapolate=extrapolate)
            self._dat.append(dat1)

    def __len__(self):
        return len(self._dat)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._dat[item]
        return ImageSequence(self._dat[item])

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
    mask = property(lambda self: self[0].mask)

    def to(self, *args, **kwargs):
        return copy.deepcopy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        for dat in self:
            dat.to_(*args, **kwargs)
        return self

    def _prm_as_str(self):
        return [f'nb_images={len(self)}'] + super()._prm_as_str()


class ImagePyramid(ImageSequence):
    """Compute a multiscale image pyramid.
    This object can be used as an Image (in which case the highest
    resolution is returned) or as a list of Images.
    """

    def __init__(self, dat, levels=1, affine=None, dim=None,
                 mask=None, preview=None,
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
        method : {'gauss', 'average', 'median', 'stride'}, default='gauss'
        """
        # I don't call super().__init__() on purpose
        self.method = method
        
        if isinstance(dat, Image):
            if affine is None:
                affine = dat.affine
            dim = dat.dim
            mask = dat.mask
            preview = dat._preview
            bound = dat.bound
            extrapolate = dat.extrapolate
            dat = dat.dat
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
            dat, mask, preview = self._build_pyramid(
                dat, levels, method, dim, bound, mask, preview)
        dat = list(dat)
        if not mask:
            mask = [None] * len(dat)
        if not preview:
            preview = [None] * len(dat)

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
        self._dat = [Image(d, aff, dim=dim,  mask=m, preview=p,
                           bound=bound, extrapolate=extrapolate)
                     for d, m, p, aff in zip(dat, mask, preview, affine)]

    def _prm_as_str(self):
        s = []
        if self.method[0] != 'g':
            s += [f'method={self.method}']
        return [f'nb_levels={len(self)}'] + super()._prm_as_str() + s

    def _build_pyramid(self, dat, levels, method, dim, bound,
                       mask=None, preview=None):
        levels = list(levels)
        indexed_levels = list(enumerate(levels))
        indexed_levels.sort(key=lambda x: x[1])
        nb_levels = max(levels)
        if mask is not None:
            mask = mask.to(dat.device)
        dats = [dat] * levels.count(0)
        masks = [mask] * levels.count(0)
        previews = [preview] * levels.count(0)
        if mask is not None:
            mask = mask.to(dat.dtype)
        if preview is not None:
            preview = preview.to(dat.dtype)
        for level in range(1, nb_levels+1):
            shape = dat.shape[-dim:]
            kernel_size = [min(2, s) for s in shape]
            if method[0] == 'g':  # gaussian pyramid
                # We assume the original data has a PSF of 1 input voxel.
                # We smooth by an additional 1-vx FWHM so that the data has a
                # PSF of 2 input voxels == 1 output voxel, then subsample.
                smooth = lambda x: spatial.smooth(x, fwhm=1, stride=2,
                                                  dim=dim, bound=bound)
            elif method[0] == 'a':  # average window
                smooth = lambda x: spatial.pool(dim, x, kernel_size=kernel_size,
                                                stride=2, reduction='mean')
            elif method[0] == 'm':  # median window
                smooth = lambda x: spatial.pool(dim, x, kernel_size=kernel_size,
                                                stride=2, reduction='median')
            elif method[0] == 's':  # strides
                slicer = [slice(None, None, 2)] * dim
                smooth = lambda x: x[(Ellipsis, *slicer)]
            else:
                raise ValueError(method)
            dat = smooth(dat)
            if mask is not None:
                mask = smooth(mask)
            if preview is not None:
                preview = smooth(preview)
            dats += [dat] * levels.count(level)
            masks += [mask] * levels.count(level)
            previews += [preview] * levels.count(level)
        reordered_dats = [None] * len(levels)
        reordered_masks = [None] * len(levels)
        reordered_previews = [None] * len(levels)
        for (i, level), dat, mask, preview \
                in zip(indexed_levels, dats, masks, previews):
            reordered_dats[i] = dat
            reordered_masks[i] = mask
            reordered_previews[i] = preview
        return reordered_dats, reordered_masks, reordered_previews

    def _build_affine_pyramid(self, affine, shape, levels, method):
        levels = list(levels)
        indexed_levels = list(enumerate(levels))
        indexed_levels.sort(key=lambda x: x[1])
        nb_levels = max(levels)
        affines = [affine] * levels.count(0)
        for level in range(1, nb_levels+1):
            if method[0] == 's':  # stride pyramid
                slicer = (slice(None, None, 2),) * len(shape)
                affine, shape = spatial.affine_sub(affine, shape, slicer)
            else:  # conv pyramid
                if method[0] == 'g':  # gaussian pyramid
                    padding = 'auto'
                    kernel = 3
                else:  # moving window
                    padding = 0
                    kernel = [min(2, s) for s in shape]
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


class SpatialTransform:
    pass


class AffineTransform(SpatialTransform):
    pass


class DenseTransform(SpatialTransform):
    pass


class Displacement(SpatialTensor, DenseTransform):
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

    def downsample_(self, factor=2, **kwargs):
        kwargs.setdefault('interpolation', 1)
        kwargs.setdefault('bound', 'dft')
        kwargs.setdefault('anchor', 'c')
        factor = 1 / factor
        self.dat, self.affine = spatial.resize_grid(
            self.dat, factor, type='disp', affine=self.affine, **kwargs)
        return self

    def downsample(self, factor=2, **kwargs):
        kwargs.setdefault('interpolation', 1)
        kwargs.setdefault('bound', 'dft')
        kwargs.setdefault('anchor', 'c')
        factor = 1 / factor
        dat, aff = spatial.resize_grid(self.dat, factor, type='displacement',
                                       affine=self.affine, **kwargs)
        return type(self)(dat, aff, dim=self.dim)

    def upsample_(self, factor=2, **kwargs):
        kwargs.setdefault('interpolation', 1)
        kwargs.setdefault('bound', 'dft')
        kwargs.setdefault('anchor', 'c')
        self.dat, self.affine = spatial.resize_grid(
            self.dat, factor, type='disp', affine=self.affine, **kwargs)
        return self

    def upsample(self, factor=2, **kwargs):
        kwargs.setdefault('interpolation', 1)
        kwargs.setdefault('bound', 'dft')
        kwargs.setdefault('anchor', 'c')
        dat, aff = spatial.resize_grid(self.dat, factor, type='displacement',
                                       affine=self.affine, **kwargs)
        return type(self)(dat, aff, dim=self.dim)


class LogAffine(AffineTransform):
    """Data + Metadata (basis) of a "Lie" affine matrix"""
    def __init__(self, dat=None, basis=None, dim=3, **backend):
        """
        Parameters
        ----------
        [dat : tensor, optional]
            Pre-allocated log-affine
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Name of an Affine basis
        dim : int, default=3
            Number of spatial dimensions
        **backend
        """
        if isinstance(dat, str):
            if basis is not None:
                raise ValueError('`basis` provided but `dat` looks like '
                                 'a basis.')
            basis, dat = dat, None
        elif basis is None:
            raise ValueError('A basis should be provided')
        self._basis = None
        self.basis = basis
        self.dim = dim
        if dat is None:
            dat = torch.zeros(spatial.affine_basis_size(basis, dim), **backend)
        self.dat = dat
        self._cache = None
        self._icache = None

    @property
    def basis(self):
        if self._basis is None:
            self._basis = spatial.affine_basis(self._basis_name, self.dim,
                                               **utils.backend(self.dat))
        return self._basis

    @basis.setter
    def basis(self, x):
        self._basis_name = x

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
    def __init__(self, loss, moving, fixed, factor=1, backward=False):
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
        self.backward = backward

    def make(self, loss, **kwargs):
        if isinstance(loss, LossComponent):
            kwargs.setdefault('moving', loss.moving)
            kwargs.setdefault('fixed', loss.fixed)
            kwargs.setdefault('factor', loss.factor)
            kwargs.setdefault('backward', loss.backward)
            loss = loss.loss
        return LossComponent(loss, **kwargs)

    def __repr__(self):
        s = f'{type(self.loss).__name__}[{self.factor:.1g}](\n' \
            f'    moving={self.moving}, \n' \
            f'    fixed={self.fixed}'
        if self.backward:
            s += f',\n    backward=True'
        p = getattr(self.loss, 'patch', None)
        if p is not None:
            s += f',\n    patch={p}'
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
        self.dat = Displacement.make(dat, affine=affine, **backend)
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
        return spatial.greens_apply(m, self.kernel, self.factor, self.voxel_size)

    def downsample_(self, factor=2, **kwargs):
        self.clear_cache()
        self.dat.downsample_(factor, **kwargs)
        return self

    def downsample(self, factor=2, **kwargs):
        dat = self.dat.downsample(factor, **kwargs)
        return type(self)(dat, self.factor, self.prm, self.steps, self.kernel)

    def upsample_(self, factor=2, **kwargs):
        self.clear_cache()
        self.dat.upsample_(factor, **kwargs)
        return self

    def upsample(self, factor=2, **kwargs):
        dat = self.dat.upsample(factor, **kwargs)
        return type(self)(dat, self.factor, self.prm, self.steps, self.kernel)

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

    def set_kernel(self, kernel=None):
        if kernel is None:
            kernel = spatial.greens(self.shape, **self.prm,
                                    factor=self.factor / py.prod(self.shape),
                                    voxel_size=self.voxel_size,
                                    **utils.backend(self.dat))
        self.kernel = kernel
        return self

    def reset_kernel(self, kernel=None):
        return self.set_kernel(kernel)  # backward compatibility

    def set_dat(self, dat, affine=None, **backend):
        super().set_dat(dat, affine, **backend)
        self.reset_kernel()
        return self

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
        else:
            grid = self._cache
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

    def __init__(self, basis, factor=1, prm=None, dat=None, position='symmetric'):
        """

        Parameters
        ----------
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Affine basis name
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
        self.prm = prm or {}  # regutils.defaults_affine()
        self._basis = basis
        self.position = position
        if dat is not None:
            self.set_dat(dat)
        else:
            self.dat = None

    def set_dat(self, dat=None, dim=None, **backend):
        self.dat = LogAffine.make(dat, basis=self._basis, dim=dim, **backend)
        return self

    basis = property(lambda self: self.dat.basis if self.dat is not None else None)
    basis_name = property(lambda self: self._basis)

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
        s += [f'position={self.position}']
        s += [f'{key}={value}' for key, value in self.prm.items()]
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'
