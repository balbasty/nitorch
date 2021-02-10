from copy import copy
import os
from nitorch import io, spatial
from nitorch.core.dtypes import dtype as nitype
from nitorch.core.utils import make_vector
from nitorch import nn
import torch


class Base:
    """Base class that implements initialization/copy of parameters"""
    def __init__(self, **kwargs):
        # make a copy of all class attributes to avoid cross-talk
        for k in dir(self):
            if k.startswith('_'):
                continue
            v = getattr(self, k)
            if callable(v):
                continue
            setattr(self, k, copy(v))
        # update user-provided attributes
        for k, v in kwargs.items():
            setattr(self, k, copy(v))

    def _ordered_keys(self):
        """Get all class attributes, including inherited ones, in order.
        Private members and methods are skipped.
        """
        unrolled = [type(self)]
        while unrolled[0].__base__ is not object:
            unrolled = [unrolled[0].__base__, *unrolled]
        keys = []
        for klass in unrolled:
            for key in klass.__dict__.keys():
                if key.startswith('_'):
                    continue
                if key in keys:
                    continue
                val = getattr(self, key)
                if callable(val):
                    continue
                keys.append(key)
        return keys

    def _lines(self):
        """Build all lines of the representation of this object.
        Returns a list of str.
        """
        lines = []
        for k in self._ordered_keys():
            v = getattr(self, k)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], Base):
                lines.append(f'{k} = [')
                pad = '  '
                for vv in v:
                    l = [pad + ll for ll in vv._lines()]
                    lines.extend(l)
                    lines[-1] += ','
                lines.append(']')
            elif isinstance(v, Base):
                ll = v._lines()
                lines.append(f'{k} = {ll[0]}')
                lines.extend(ll[1:])
            else:
                lines.append(f'{k} = {v}')

        superpad = '  '
        lines = [superpad + line for line in lines]
        lines = [f'{type(self).__name__}('] + lines + [')']
        return lines

    def __repr__(self):
        return '\n'.join(self._lines())

    def __str__(self):
        return repr(self)


class FileWithInfo(Base):
    fname: str = None      # Full path
    shape: tuple = None    # Spatial shape
    affine = None          # Orientation matrix
    dir: str = None        # Directory
    base: str = None       # Base name (without extension)
    ext: str = None        # Extension
    channels: int = None   # Number of channels
    float: bool = True     # Is raw dtype floating point


class ImageFile(Base):
    """An image to register (fixed or moving)"""
    files: list = []                # full path to inputs files
    updated: list or bool = None    # filename for the "estimated" image
    resliced: list or bool = False  # filename for the "resliced" image
    interpolation: int = None       # interpolation order
    bound: str = None               # boundary conditions for interpolation
    extrapolate: bool = None        # extrapolate out-of-bounds using `bound`


class FixedImageFile(ImageFile):
    """A fixed image -> nothing is written by default"""
    updated: list or bool = False


class MovingImageFile(ImageFile):
    """A moving image -> updated version is written by default"""
    updated: list or bool = True


class MatchingLoss(Base):
    """A loss between two images"""
    name: str = None                # Name of the loss
    factor: float = 1               # Multiplicative factor
    fixed = FixedImageFile()        # Fixed / Target image
    moving = MovingImageFile()      # Moving / Source image
    interpolation: int = None       # Interpolation order (default for f+m)
    bound: str = None               # Bound order         (default for f+m)
    extrapolate: bool = None        # Extrapolate order   (default for f+m)


class NoLoss(MatchingLoss):
    """Additional image to warp -- not involved in the loss"""
    fixed: FixedImageFile = None


class DiceLoss(MatchingLoss):
    name = 'dice'
    labels: list = []
    weights: list = []

    def call(self, x, y):
        fn = nn.DiceLoss(discard_background=True, weighted=self.weights)
        loss = fn(x[None], y[None])
        if self.factor != 1:
            loss = loss * self.factor
        return loss


class MILoss(MatchingLoss):
    name = 'mi'
    patch: list = []
    bins: int = 32
    fwhm: float = 1.

    def call(self, x, y):
        xs = x.unbind(0)
        ys = y.unbind(0)
        loss = 0
        nb_channels = len(xs)
        for x, y in zip(xs, ys):
            x = x[None, None]
            y = y[None, None]
            loss += nn.MutualInfoLoss(patch_size=self.patch)(x, y) / nb_channels
        # I take the average of MI across channels to be consistent
        # with how MSE works.
        if self.factor != 1:
            loss = loss * self.factor
        return loss


class CatLoss(MatchingLoss):
    name = 'cat'

    def call(self,x, y):
        fn = nn.CategoricalLoss(log=False, implicit=True)
        loss = fn(x[None], y[None])
        if self.factor != 1:
            loss = loss * self.factor
        return loss


class JTVLoss(MatchingLoss):
    name = 'jtv'

    def call(self, x, y):
        raise NotImplementedError


class MSELoss(MatchingLoss):
    name = 'mse'

    def call(self, x, y):
        loss = torch.nn.MSELoss()(x[None], y[None])
        if self.factor != 1:
            loss = loss * self.factor
        return loss


matching_losses = [NoLoss, DiceLoss, MILoss, CatLoss, JTVLoss, MSELoss]


class TransformationLoss(Base):
    name = None
    factor: float = 1.


class AbsoluteLoss(TransformationLoss):
    name = 'absolute'

    def call(self, v):
        m = spatial.absolute_grid(v)
        loss = (v*m).sum()
        if self.factor != 1:
            loss = loss * self.factor
        return loss


class MembraneLoss(TransformationLoss):
    name = 'membrane'

    def call(self, v):
        m = spatial.membrane_grid(v)
        loss = (v*m).sum()
        if self.factor != 1:
            loss = loss * self.factor
        return loss


class BendingLoss(TransformationLoss):
    name = 'Bending'

    def call(self, v):
        m = spatial.bending_grid(v)
        loss = (v*m).sum()
        if self.factor != 1:
            loss = loss * self.factor
        return loss


class LinearElasticLoss(TransformationLoss):
    name = 'LinearElastic'

    def call(self, v):
        factor = make_vector(self.factor, 2, dtype=v.dtype, device=v.device)
        loss = 0
        if factor[0]:
            m = spatial.lame_div(v)
            loss += (v*m).sum()
            if self.factor[0] != 1:
                loss = loss * self.factor[0]
        if factor[1]:
            m = spatial.lame_shear(v)
            loss += (v*m).sum()
            if self.factor[1] != 1:
                loss = loss * self.factor[1]
        return loss


transformation_losses = [AbsoluteLoss, MembraneLoss, BendingLoss, LinearElasticLoss]


class Transformation(Base):
    name = None
    factor: float = 1.
    init: list = []
    lr: float = 1.
    stop: float = 1.
    output = True
    losses: list = []


class NonLinear(Transformation):
    ext: str = '.nii.gz'

    def freeable(self):
        return not hasattr(self, 'optdat')

    def free(self):
        if not self.freeable():
            return
        print('Free nonlin')
        self.optdat = torch.nn.Parameter(self.dat, requires_grad=True)
        self.dat = self.optdat


class FFD(NonLinear):
    name = 'ffd'
    grid: list or int = 10


class Diffeo(NonLinear):
    name = 'diffeo'


class Linear(Transformation):
    nb_prm: callable
    basis = None
    ext: str = '.lta'

    def freeable(self):
        return not hasattr(self, 'optdat') or len(self.dat) != len(self.optdat)

    def free(self):
        if not self.freeable():
            return
        nb_prm = len(self.optdat) if hasattr(self, 'optdat') else 0
        if nb_prm == 0:
            print('Free translations')
            self.optdat = torch.nn.Parameter(self.dat[:3], requires_grad=True)
            self.dat = torch.cat([self.optdat, self.dat[3:]])
        elif nb_prm == 3:
            print('Free rotations')
            self.optdat = torch.nn.Parameter(self.dat[:6], requires_grad=True)
            self.dat = torch.cat([self.optdat, self.dat[6:]])
        elif nb_prm == 6:
            print('Free isotropic scaling')
            self.optdat = torch.nn.Parameter(self.dat[:7], requires_grad=True)
            self.dat = torch.cat([self.optdat, self.dat[7:]])
        elif nb_prm == 12:
            print('Free full affine')
            self.optdat = torch.nn.Parameter(self.dat[:12], requires_grad=True)
            self.dat = self.optdat


class Translation(Linear):
    name = 'translation'
    basis = spatial.affine_basis('T', 3)
    nb_prm = staticmethod(lambda dim: dim)


class Rigid(Linear):
    name = 'rigid'
    basis = spatial.affine_basis('SE', 3)
    nb_prm = staticmethod(lambda dim: dim*(dim+1)//2)


class Similitude(Linear):
    name = 'similitude'
    basis = spatial.affine_basis('CSO', 3)
    nb_prm = staticmethod(lambda dim: dim*(dim+1)//2 + 1)


class Affine(Linear):
    name = 'affine'
    basis = spatial.affine_basis('Aff+', 3)
    nb_prm = staticmethod(lambda dim: dim*(dim+1))


class Optimizer(Base):
    name = None
    max_iter: int = None
    lr: float = None
    stop: float = None
    ls: int = None


class Adam(Optimizer):
    name = 'adam'

    def call(self, *args, **kwargs):
        return torch.optim.Adam(*args, **kwargs)


class GradientDescent(Optimizer):
    name = 'gd'

    def call(self, *args, **kwargs):
        return torch.optim.SGD(*args, **kwargs)


class Defaults(Base):
    interpolation: int = 1
    bound: str = 'dct2'
    extrapolate: bool = False
    updated = '{dir}/{base}.registered{ext}'
    resliced = '{dir}/{base}.resliced{ext}'
    output = '{name}{ext}'
    init: float or str = 0.
    max_iter: int = 1000
    lr: int = 0.1
    stop: float = 1e-4
    ls: int = 0


class AutoReg(Base):

    # main blocks of parameters
    losses: list = []
    transformations: list = []
    optimizers: list = []
    # other parameters
    defaults: Defaults = Defaults()
    device: str = 'cpu'
    progressive: bool = False
    pyramid: list = [1]

    def propagate_defaults(self):
        """Propagate defaults from root to leaves"""
        def set_default(s, key, ref=self.defaults):
            if getattr(s, key) is None:
                setattr(s, key, getattr(ref, key))

        def set_fname(s, key, ref=self.defaults):
            if getattr(s, key) is False:
                setattr(s, key, [])
            elif getattr(s, key) is True:
                setattr(s, key, getattr(ref, key))

        for loss in self.losses:
            set_default(loss, 'interpolation')
            set_default(loss, 'bound')
            set_default(loss, 'extrapolate')
            if loss.fixed:
                set_default(loss.fixed, 'interpolation', loss)
                set_default(loss.fixed, 'bound', loss)
                set_default(loss.fixed, 'extrapolate', loss)
                set_fname(loss.fixed, 'updated')
                set_fname(loss.fixed, 'resliced')
            if loss.moving:
                set_default(loss.moving, 'interpolation', loss)
                set_default(loss.moving, 'bound', loss)
                set_default(loss.moving, 'extrapolate', loss)
                set_fname(loss.moving, 'updated')
                set_fname(loss.moving, 'resliced')

        for opt in self.optimizers:
            set_default(opt, 'max_iter')
            set_default(opt, 'lr')
            set_default(opt, 'stop')
            set_default(opt, 'ls')

        for trf in self.transformations:
            set_default(trf, 'init')
            set_default(trf, 'output')
            if isinstance(trf.output, bool) and trf.output:
                trf.output = self.defaults.output

    def read_info(self):
        """Extract info about the input files (affine, shape, etc)"""

        def allsame(x):
            """Check that multiple lists are identical"""
            x = list(x)
            if x:
                x0 = x[0]
                for x1 in x[1:]:
                    if x1 != x0:
                        return False
            return True

        def info(fname):
            """Extract info about one file"""
            file = FileWithInfo()

            file.fname = fname
            file.dir = os.path.dirname(fname)
            file.base = os.path.basename(fname)
            file.base, file.ext = os.path.splitext(file.base)
            if file.ext in ('.gz', '.bz2'):
                file.base, ext = os.path.splitext(file.base)
                file.ext = ext + file.ext
            f = io.volumes.map(fname)
            file.float = nitype(f.dtype).is_floating_point
            file.shape = tuple(f.shape[:3])
            file.affine = f.affine.float()
            while len(f.shape) > 3:
                if f.shape[-1] == 1:
                    f = f[..., 0]
                    continue
                elif f.shape[3] == 1:
                    f = f[:, :, :, 0, ...]
                    continue
                else:
                    break
            if len(f.shape) > 4:
                raise RuntimeError('Input has more than 1 channel dimensions.')
            if len(f.shape) > 3:
                file.channels = f.shape[-1]
            else:
                file.channels = 1
            return file

        def read_info1(image, loss):
            """Read info from one image"""
            image.files = [info(fname) for fname in image.files]
            image.channels = sum(file.channels for file in image.files)
            if not allsame(file.shape for file in image.files):
                raise RuntimeError('All files should have the same (spatial) '
                                   'shape for multi-file inputs.')
            image.shape = image.files[0].shape
            image.affine = image.files[0].affine
            if loss is NoLoss:
                image.type = None
            elif loss is CatLoss:
                image.type = 'proba'
            elif loss is DiceLoss:
                if image.channels > 1:
                    image.type = 'proba'
                elif image.files[0].float:
                    image.type = 'proba'
                else:
                    image.type = 'labels'
            else:
                image.type = 'intensity'
            return image

        for loss in self.losses:
            if loss.fixed:
                loss.fixed = read_info1(loss.fixed, type(loss))
            if loss.moving:
                loss.moving = read_info1(loss.moving, type(loss))

    def propagate_filenames(self):
        """Define output filenames based on patterns and input filenames"""

        def propagate1(image, key):
            pattern = getattr(image, key)
            if isinstance(pattern, str):
                setattr(image, key, [pattern] * len(image.files))
            pattern = getattr(image, key)
            if pattern:
                new = [out.format(dir=inp.dir, base=inp.base, ext=inp.ext)
                                  for out, inp in zip(pattern, image.files)]
                setattr(image, key, new)
            return image

        for loss in self.losses:
            if loss.fixed:
                propagate1(loss.fixed, 'resliced')
                propagate1(loss.fixed, 'updated')
            if loss.moving:
                propagate1(loss.moving, 'resliced')
                propagate1(loss.moving, 'updated')

        for trf in self.transformations:
            if trf.output:
                trf.output = trf.output.format(ext=trf.ext, name=trf.name)

