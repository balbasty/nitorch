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

    def __new__(cls, input=None, fill=None, dtype=None, device=None, **kwargs):
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
            return super().__new__(cls, volume, **kwargs)
        return super().__new__(cls, input, **kwargs)

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

    def __new__(cls, input=None, fill=None, dtype=None, device=None, **kwargs):
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
            return super().__new__(cls, volume, **kwargs)
        return super().__new__(cls, input, **kwargs)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


class DisplacementField(_ParameterMap):
    spatial_dim = 3

    def __new__(cls, input=None, fill=None, dtype=None, device=None, dim1d=None, **kwargs):
        if isinstance(input, (list, tuple)):
            if dim1d is None:
                input = list(input) + [len(input)]
        obj = super().__new__(cls, input, fill, dtype, device, **kwargs)
        obj.dim1d = dim1d
        return obj

    @property
    def spatial_shape(self):
        if self.dim1d is not None:
            return self.volume.shape[-self.spatial_dim:]
        else:
            return self.volume.shape[-self.spatial_dim-1:-1]

    @classmethod
    def add_identity(cls, disp, dim1d=None):
        if dim1d:
            shape = disp.shape
            shape1d = [shape[dim1d]]
            id = spatial.identity_grid(shape1d, **utils.backend(disp)).squeeze(-1)
            disp = disp.transpose(-1, dim1d).add(id).transpose(-1, dim1d)
            return disp
        else:
            dim = disp.shape[-1]
            shape = disp.shape[-dim-1:-1]
            return spatial.identity_grid(shape, **utils.backend(disp)).add_(disp)

    def fold1d(self, v):
        """Fold velocity so its shape is [prod_other_dim, dim1d, 1]"""
        if self.dim1d is None:
            return v
        shape1d = v.shape[self.dim1d]
        return v.transpose(self.dim1d, -1).reshape([-1, shape1d, 1])

    def unfold1d(self, v, jac=False):
        """Unfold velocity so its shape is [*spatial_dim]"""
        if self.dim1d is None:
            return v
        if jac:
            v = v.squeeze(-1)
        shape_t = list(self.spatial_shape)
        shape_t[self.dim1d], shape_t[-1] = shape_t[-1], shape_t[self.dim1d]
        return v.reshape(shape_t).transpose(self.dim1d, -1)


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

    def exp(self, jacobian=False, add_identity=False):
        """Exponentiate forward transform"""
        grid = self.fdata()
        grid = self.fold1d(grid)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            jac = self.unfold1d(jac, jac=True)
        if add_identity:
            grid = self.add_identity(grid)
        grid = self.unfold1d(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False):
        """Exponentiate inverse transform"""
        grid = -self.fdata()
        grid = self.fold1d(grid)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            jac = self.unfold1d(jac, jac=True)
        if add_identity:
            grid = self.add_identity(grid)
        grid = self.unfold1d(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, jacobian=False, add_identity=False):
        """Exponentiate both forward and inverse transforms"""
        grid = self.fdata()
        grid = self.fold1d(grid)
        igrid = -grid
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            ijac = spatial.grid_jacobian(igrid, type='displacement')
            jac = self.unfold1d(jac, jac=True)
            ijac = self.unfold1d(ijac, jac=True)
        if add_identity:
            grid = self.add_identity(grid)
            igrid = self.add_identity(igrid)
        grid = self.unfold1d(grid)
        igrid = self.unfold1d(igrid)
        return (grid, igrid, jac, ijac) if jacobian else (grid, igrid)


class SVFDeformation(ParameterizedDeformation):
    model = 'svf'
    steps: int = 8

    def exp(self, jacobian=False, add_identity=False):
        """Exponentiate forward transform"""
        v = self.fdata()
        v = self.fold1d(v)
        grid = spatial.exp_forward(v, steps=self.steps,
                                   jacobian=jacobian,
                                   displacement=True)
        if jacobian:
            grid, jac = grid
            jac = self.unfold1d(jac, jac=True)
        if add_identity:
            grid = self.add_identity(grid)
        grid = self.unfold1d(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False):
        """Exponentiate inverse transform"""
        v = self.fdata()
        v = self.fold1d(v)
        igrid = spatial.exp_forward(v, steps=self.steps,
                                    jacobian=jacobian, inverse=True,
                                    displacement=True)
        if jacobian:
            igrid, ijac = igrid
            ijac = self.unfold1d(ijac, jac=True)
        if add_identity:
            igrid = self.add_identity(igrid)
        igrid = self.unfold1d(igrid)
        return (igrid, ijac) if jacobian else igrid

    def exp2(self, jacobian=False, add_identity=False):
        """Exponentiate both forward and inverse transforms"""
        v = self.fdata()
        v = self.fold1d(v)
        if jacobian:
            grid, jac = spatial.exp_forward(v, steps=self.steps,
                                            jacobian=True,
                                            displacement=True)
            igrid, ijac = spatial.exp_forward(v, steps=self.steps,
                                              jacobian=True, inverse=True,
                                              displacement=True)
            jac = self.unfold1d(jac, jac=True)
            ijac = self.unfold1d(ijac, jac=True)
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            grid = self.unfold1d(grid)
            igrid = self.unfold1d(igrid)
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
            grid = self.unfold1d(grid)
            igrid = self.unfold1d(igrid)
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

    def exp(self, jacobian=False, add_identity=False):
        """Exponentiate forward transform"""
        v = self.fdata()
        v = self.fold1d(v)
        grid = spatial.shoot(v, self.kernel, steps=self.steps,
                             factor=self.factor,
                             voxel_size=self.voxel_size, **self.reg_prm,
                             displacement=True)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            jac = self.unfold1d(jac, jac=True)
        if add_identity:
            grid = self.add_identity(grid)
        grid = self.unfold1d(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, jacobian=False, add_identity=False):
        """Exponentiate inverse transform"""
        v = self.fdata()
        v = self.fold1d(v)
        _, grid = spatial.shoot(v, self.kernel, steps=self.steps,
                                factor=self.factor,
                                voxel_size=self.voxel_size, **self.reg_prm,
                                return_inverse=True,  displacement=True)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            jac = self.unfold1d(jac, jac=True)
        if add_identity:
            grid = self.add_identity(grid)
        grid = self.unfold1d(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, jacobian=False, add_identity=False):
        """Exponentiate both forward and inverse transforms"""
        v = self.fdata()
        v = self.fold1d(v)
        grid, igrid = spatial.shoot(v, self.kernel, steps=self.steps,
                                    factor=self.factor,
                                    voxel_size=self.voxel_size, **self.reg_prm,
                                    return_inverse=True, displacement=True)
        if jacobian:
            jac = spatial.grid_jacobian(grid, type='displacement')
            ijac = spatial.grid_jacobian(igrid, type='displacement')
            jac = self.unfold1d(jac, jac=True)
            ijac = self.unfold1d(ijac, jac=True)
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            grid = self.unfold1d(grid)
            igrid = self.unfold1d(igrid)
            return grid, igrid, jac, ijac
        else:
            if add_identity:
                grid = self.add_identity(grid)
                igrid = self.add_identity(igrid)
            grid = self.unfold1d(grid)
            igrid = self.unfold1d(igrid)
            return grid, igrid
