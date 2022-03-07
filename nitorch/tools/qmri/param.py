from nitorch.tools.qmri import io as qio
from nitorch import spatial
from nitorch.core import utils, py
import torch
import copy


class _ParameterMap(qio.BaseND):
    """
    Wrapper object for Parameter maps
    """

    min = None   # minimum value
    max = None   # maximum value

    def __init__(self, input=None, fill=None, dtype=None, device=None, **kwargs):
        """

        Parameters
        ----------
        input : sequence[int] or tensor_like or flie_like
            If a sequence[int], allocate a tensor of that shape
            Else, wrap the underlying `Volume3D` object.
        fill : number, optional
            A value to fill the tensor with
        dtype : torch.dtype, optional
        device : torch.device, optional
        kwargs : dict
            Attributes for `Volume3D`.
        """
        if isinstance(input, (list, tuple)):
            if fill is not None:
                volume = torch.full(input, fill, dtype=dtype, device=device)
            else:
                volume = torch.zeros(input, dtype=dtype, device=device)
            super().__init__(volume, **kwargs)
        else:
            super().__init__(input, **kwargs)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


class ParameterMap(_ParameterMap, qio.Volume3D):
    """
    Wrapper object for Parameter maps
    """

    min = None   # minimum value
    max = None   # maximum value

    def __init__(self, input=None, fill=None, dtype=None, device=None, **kwargs):
        """

        Parameters
        ----------
        input : sequence[int] or tensor_like or file_like
            If a sequence[int], allocate a tensor of that shape
            Else, wrap the underlying `Volume3D` object.
        fill : number, optional
            A value to fill the tensor with
        dtype : torch.dtype, optional
        device : torch.device, optional
        kwargs : dict
            Attributes for `Volume3D`.
        """
        if isinstance(input, (list, tuple)):
            if fill is not None:
                volume = torch.full(input, fill, dtype=dtype, device=device)
            else:
                volume = torch.zeros(input, dtype=dtype, device=device)
            super().__init__(volume, **kwargs)
        else:
            super().__init__(input, **kwargs)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


class MultiParameterMaps(_ParameterMap):

    min = None   # minimum value
    max = None   # maximum value

    def __init__(self, input=None, fill=None, dtype=None, device=None, **kwargs):
        """

        Parameters
        ----------
        input : sequence[int] or tensor_like or file_like
            If a sequence[int], allocate a tensor of that shape.
            Else, wrap the underlying `BaseND` object.
        fill : [sequence of] number, optional
            A value to fill the tensor(s) with
        dtype : torch.dtype, optional
        device : torch.device, optional
        kwargs : dict
            Attributes for `Volume3D`.
        """
        if isinstance(input, (list, tuple)):
            if fill is not None:
                volume = torch.full(input, fill, dtype=dtype, device=device)
            else:
                volume = torch.zeros(input, dtype=dtype, device=device)
            super().__init__(volume, **kwargs)
        else:
            super().__init__(input, **kwargs)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.volume)

    def __iter__(self):
        for vol in self.volume:
            yield ParameterMap(vol, affine=self.affine)

    def __getitem__(self, index):
        if isinstance(index, slice) or isinstance(index, list):
            return MultiParameterMaps(self.volume[index], affine=self.affine)
        else:
            return ParameterMap(self.volume[index], affine=self.affine)


class DisplacementField(_ParameterMap):
    spatial_dim = 3

    def __init__(self, input=None, fill=None, dtype=None, device=None, **kwargs):
        if isinstance(input, (list, tuple)):
            input = list(input) + [len(input)]
        super().__init__(input, fill, dtype, device, **kwargs)

    @property
    def spatial_shape(self):
        return self.volume.shape[-self.spatial_dim-1:-1]

    @classmethod
    def add_identity(cls, disp):
        dim = disp.shape[-1]
        shape = disp.shape[-dim-1:-1]
        return spatial.identity_grid(shape, **utils.backend(disp)).add_(disp)


class ParameterizedDeformation(DisplacementField):
    model: str = None

    @classmethod
    def make(cls, input=None, model='svf', **kwargs):
        if model == 'svf':
            return SVFDeformation(input, **kwargs)
        elif model == 'shoot':
            return GeodesicDeformation(input, **kwargs)
        elif model == 'smalldef':
            return DenseDeformation(input, **kwargs)
        else:
            raise NotImplementedError


class DenseDeformation(ParameterizedDeformation):
    model = 'smalldef'

    def exp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate forward transform"""
        grid = self.fdata()
        if alpha:
            grid = grid * alpha
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate inverse transform"""
        grid = -self.fdata()
        if alpha:
            grid = grid * alpha
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate both forward and inverse transforms"""
        grid = self.fdata()
        if alpha:
            grid = grid * alpha
        igrid = -grid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            ijac = spatial.grid_jacobian(igrid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
            igrid = self.add_identity(igrid)
        return (grid, igrid, jac, ijac) if jacobian else (grid, igrid)


class SVFDeformation(ParameterizedDeformation):
    model = 'svf'
    steps: int = 8

    def exp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate forward transform"""
        v = self.fdata()
        if alpha:
            v = v * alpha
        grid = spatial.exp_forward(v, steps=self.steps,
                                   jacobian=jacobian,
                                   displacement=True)
        if jacobian:
            grid, jac = grid
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate inverse transform"""
        v = self.fdata()
        if alpha:
            v = v * alpha
        igrid = spatial.exp_forward(v, steps=self.steps,
                                    jacobian=jacobian, inverse=True,
                                    displacement=True)
        if jacobian:
            igrid, ijac = igrid
        if add_identity:
            igrid = self.add_identity(igrid)
        return (igrid, ijac) if jacobian else igrid

    def exp2(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate both forward and inverse transforms"""
        v = self.fdata()
        if alpha:
            v = v * alpha
        if jacobian:
            grid, jac = spatial.exp_forward(v, steps=self.steps,
                                            jacobian=True,
                                            displacement=True)
            igrid, ijac = spatial.exp_forward(v, steps=self.steps,
                                              jacobian=True, inverse=True,
                                              displacement=True)
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            return grid, igrid, jac, ijac
        else:
            grid = spatial.exp_forward(v, steps=self.steps,
                                       jacobian=False,
                                       displacement=True)
            igrid = spatial.exp_forward(v, steps=self.steps,
                                        jacobian=False, inverse=True,
                                        displacement=True)
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            return grid, igrid


class GeodesicDeformation(ParameterizedDeformation):
    model = 'shoot'
    factor: float = 1
    absolute: float = 0.1
    membrane: float = 0.1
    bending: float = 0.2
    lame: float or tuple = 0.
    steps: int = 8

    @property
    def reg_prm(self):
        return dict(absolute=self.absolute, membrane=self.membrane,
                    bending=self.bending, factor=self.factor)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dim1d is None:
            shape = self.spatial_shape
            vx = self.voxel_size
        else:
            shape = [self.spatial_shape[self.dim1d]]
            vx = self.voxel_size[self.dim1d]

        self.kernel = spatial.greens(shape, **self.reg_prm,
                                     factor=self.factor, voxel_size=vx,
                                     **utils.backend(self))

    def exp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate forward transform"""
        v = self.fdata()
        if alpha:
            v = v * alpha
        grid = spatial.shoot(v, self.kernel, steps=self.steps,
                             factor=self.factor,
                             voxel_size=self.voxel_size, **self.reg_prm,
                             displacement=True)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate inverse transform"""
        v = self.fdata()
        if alpha:
            v = v * alpha
        _, grid = spatial.shoot(v, self.kernel, steps=self.steps,
                                factor=self.factor,
                                voxel_size=self.voxel_size, **self.reg_prm,
                                return_inverse=True,  displacement=True)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
        if add_identity:
            grid = self.add_identity(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate both forward and inverse transforms"""
        v = self.fdata()
        if alpha:
            v = v * alpha
        grid, igrid = spatial.shoot(v, self.kernel, steps=self.steps,
                                    factor=self.factor,
                                    voxel_size=self.voxel_size, **self.reg_prm,
                                    return_inverse=True, displacement=True)
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


# ======================================================================
#                           1D deformations (B0)
# ======================================================================


class DistortionField(_ParameterMap):
    spatial_dim = 3

    def __init__(self, input=None, fill=None, dtype=None, device=None,
                dim=-1, bound='dct2', unit='vx', **kwargs):
        if isinstance(input, (list, tuple)):
            input = list(input)
        super().__init__(input, fill, dtype, device, **kwargs)
        self.displacement_dim = dim
        self.bound = bound
        if unit[0].lower() not in 'vh':
            raise ValueError('Unit must be "vx" or "hz" but got', unit)
        self.unit = unit.lower()

    def fdata(self, *args, unit=None, bandwidth=None, **kwargs):
        if unit is not None:
            unit = unit.lower()
        if not unit or not bandwidth or unit == self.unit:
            return super().fdata(*args, **kwargs)
        else:
            kwargs['copy'] = True
            return self._convert_to_(super().fdata(*args, **kwargs),
                                     unit, bandwidth)

    def _convert_from_(self, other, unit=None, bandwidth=None):
        if unit is not None:
            unit = unit.lower()
        if not unit or not bandwidth or unit == self.unit:
            return other
        if unit[0] == 'v':
            # convert from vx to hz
            other *= bandwidth
        elif unit[0] == 'h':
            # convert from hz to vx
            other /= bandwidth
        else:
            raise ValueError('Unknown unit', unit)
        return other

    def _convert_from(self, other, unit=None, bandwidth=None):
        return self._convert_from_(other.clone(), unit, bandwidth)

    def _convert_to_(self, other, unit=None, bandwidth=None):
        if unit is not None:
            unit = unit.lower()
        if not unit or not bandwidth or unit == self.unit:
            return other
        if unit[0] == 'v':
            # convert from hz to vx
            other /= bandwidth
        elif unit[0] == 'h':
            # convert from vx to hz
            other *= bandwidth
        else:
            raise ValueError('Unknown unit', unit)
        return other

    def _convert_to(self, other, unit=None, bandwidth=None):
        return self._convert_to_(other.clone(), unit, bandwidth)

    def add_(self, other, alpha=None, unit=None, bandwidth=None):
        other = self._convert_from(other, unit, bandwidth)
        if torch.is_tensor(self.volume):
            self.volume.add_(other, alpha=alpha)
        else:
            vol = self.fdata(dtype=other.dtype, device=other.device)
            vol.add_(other, alpha)
            self.volume.set_fdata(vol)

    def sub_(self, other, alpha=None, unit=None, bandwidth=None):
        if alpha is None:
            alpha = -1
        else:
            alpha *= -1
        return self.add_(other, alpha, unit, bandwidth)

    @property
    def spatial_shape(self):
        return self.volume.shape[-self.spatial_dim:]

    def add_identity(self, disp):
        disp = utils.movedim(disp, self.displacement_dim, -1)
        disp = spatial.add_identity_grid(disp.unsqueeze(-1)).squeeze(-1)
        disp = utils.movedim(disp, -1, self.displacement_dim)
        return disp

    def add_identity_(self, disp):
        disp = utils.movedim(disp, self.displacement_dim, -1)
        disp = spatial.add_identity_grid_(disp.unsqueeze(-1)).squeeze(-1)
        disp = utils.movedim(disp, -1, self.displacement_dim)
        return disp


class ParameterizedDistortion(DistortionField):
    model: str = None

    @classmethod
    def make(cls, input=None, model='svf', **kwargs):
        if model == 'svf':
            return SVFDistortion(input, **kwargs)
        elif model == 'smalldef':
            return DenseDistortion(input, **kwargs)
        else:
            raise NotImplementedError


class DenseDistortion(ParameterizedDistortion):
    model = 'smalldef'

    def exp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate forward transform"""
        grid = self.fdata().clone()
        if alpha:
            grid *= alpha
        jac = spatial.diff1d(grid, dim=self.displacement_dim,
                             bound=self.bound, side='c').add_(1)
        if add_identity:
            grid = self.add_identity_(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate inverse transform"""
        grid = -self.fdata()
        if alpha:
            grid *= alpha
        jac = spatial.diff1d(grid, dim=self.displacement_dim,
                             bound=self.bound, side='c').add_(1)
        if add_identity:
            grid = self.add_identity_(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate both forward and inverse transforms"""
        grid = self.fdata().clone()
        if alpha:
            grid *= alpha
        jac = spatial.diff1d(grid, dim=self.displacement_dim,
                             bound=self.bound, side='c')
        ijac = 1 - jac
        jac += 1
        igrid = -grid
        if add_identity:
            grid = self.add_identity_(grid)
            igrid = self.add_identity_(igrid)
        return (grid, igrid, jac, ijac) if jacobian else (grid, igrid)


class SVFDistortion(ParameterizedDistortion):
    model = 'svf'
    steps: int = 8

    def exp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate forward transform"""
        v = self.fdata().clone()
        jac = None
        if alpha:
            v *= alpha
        v = spatial.exp1d_forward(v, bound=self.bound, jacobian=jacobian,
                                  inplace=True)
        if jacobian:
            v, jac = v
        if add_identity:
            v = self.add_identity_(v)
        return (v, jac) if jacobian else v

    def iexp(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate inverse transform"""
        v = -self.fdata()
        jac = None
        if alpha:
            v *= alpha
        v = spatial.exp1d_forward(v, bound=self.bound, jacobian=jacobian,
                                  inplace=True)
        if jacobian:
            v, jac = v
        if add_identity:
            v = self.add_identity_(v)
        return (v, jac) if jacobian else v

    def exp2(self, jacobian=False, add_identity=False, alpha=None):
        """Exponentiate both forward and inverse transforms"""
        v = self.fdata()
        jac = ijac = None
        if alpha:
            v *= alpha
        iv = -v
        v = spatial.exp1d_forward(v, bound=self.bound, jacobian=jacobian,
                                  inplace=True)
        iv = spatial.exp1d_forward(iv, bound=self.bound, jacobian=jacobian,
                                   inplace=True)
        if jacobian:
            v, jac = v
            iv, ijac = iv
        if add_identity:
            v = self.add_identity_(v)
            iv = self.add_identity_(iv)
        return (v, iv, jac, ijac) if jacobian else (v, iv)
