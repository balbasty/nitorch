from .base import SecondOrder
from nitorch.core import linalg, py
from nitorch import spatial
import torch


class SymGaussNewton(SecondOrder):
    """Base class for Gauss-Newton"""

    def __init__(self, lr=1, marquardt=True, preconditioner=None,
                 **kwargs):
        super().__init__(lr, **kwargs)
        self.preconditioner = preconditioner
        self.marquardt = marquardt

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            # maj = hess[..., :dim].abs()
            # if hess.shape[-1] > dim:
            #     maj.add_(hess[..., dim:].abs(), alpha=2)
            maj = hess[..., :dim].abs().max(-1, True).values
            hess[..., :dim].add_(maj, alpha=tiny)
            # hess[..., :dim] += tiny
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        step = linalg.sym_solve(hess, grad)
        step.mul_(-self.lr)
        return step


class GaussNewton(SecondOrder):
    """Base class for Gauss-Newton"""

    def __init__(self, lr=1, marquardt=True, preconditioner=None,
                 **kwargs):
        super().__init__(lr, **kwargs)
        self.preconditioner = preconditioner
        self.marquardt = marquardt

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            # maj = hess[..., :dim].abs()
            # if hess.shape[-1] > dim:
            #     maj.add_(hess[..., dim:].abs(), alpha=2)
            maj = hess.diagonal(0, -1, -2).abs().max(-1, True).values
            hess.diagonal(0, -1, -2).add_(maj, alpha=tiny)
            # hess[..., :dim] += tiny
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        step = linalg.lmdiv(hess, grad[..., None])[..., 0]
        step.mul_(-self.lr)
        return step


class GridGaussNewton(SecondOrder):
    """Base class for Gauss-Newton on displacement grids"""

    def __init__(self, lr=1, fmg=2, max_iter=2, factor=1, voxel_size=1,
                 absolute=0, membrane=0, bending=0, lame=0, reduce='mean',
                 marquardt=True, preconditioner=None, **kwargs):
        super().__init__(lr, **kwargs)
        self.preconditioner = preconditioner
        self.factor = factor
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.lame = lame
        self.reduce = reduce in ('mean', True)
        self.voxel_size = voxel_size
        self.marquardt = marquardt
        self.max_iter = max_iter
        self.fmg = fmg

    @property
    def penalty(self):
        return dict(absolute=self.absolute,
                    membrane=self.membrane,
                    bending=self.bending,
                    lame=self.lame,
                    factor=self.factor)

    @penalty.setter
    def penalty(self, x):
        for key, value in x.items():
            if key in ('absolute', 'membrane', 'bending', 'lame', 'factor'):
                setattr(self, key, value)

    def _get_prm(self):
        prm = dict(absolute=self.absolute,
                   membrane=self.membrane,
                   bending=self.bending,
                   lame=self.lame,
                   factor=self.factor,
                   voxel_size=self.voxel_size)
        if self.fmg:
            prm['nb_iter'] = self.max_iter
            prm['nb_cycles'] = self.fmg
        else:
            prm['max_iter'] = self.max_iter
        return prm

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            # maj = hess[..., :dim].abs()
            # if hess.shape[-1] > dim:
            #     maj.add_(hess[..., dim:].abs(), alpha=2)
            maj = hess[..., :dim].abs().max(-1, True).values
            hess[..., :dim].add_(maj, alpha=tiny)
            # hess[..., :dim] += tiny
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess

    def repr_keys(self):
        return super().repr_keys() + ['fmg']


class GridCG(GridGaussNewton):
    """Gauss-Newton on displacement grids using Conjugate Gradients"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess, 1e-5)
        prm = self._get_prm()
        dim = grad.shape[-1]
        if self.reduce:
            prm['factor'] = prm['factor'] / py.prod(grad.shape[-dim-1:-1])
        solver = spatial.solve_grid_fmg if self.fmg else spatial.solve_grid_sym
        step = solver(hess, grad, optim='cg', **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step


class GridRelax(GridGaussNewton):
    """Gauss-Newton on displacement grids using Relaxation"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        dim = grad.shape[-1]
        if self.reduce:
            prm['factor'] = prm['factor'] / py.prod(grad.shape[-dim-1:-1])
        solver = spatial.solve_grid_fmg if self.fmg else spatial.solve_grid_sym
        step = solver(hess, grad, optim='relax', **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step


class GridJacobi(GridGaussNewton):
    """Gauss-Newton on displacement grids using Relaxation"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        dim = grad.shape[-1]
        if self.reduce:
            prm['factor'] = prm['factor'] / py.prod(grad.shape[-dim-1:-1])
        solver = spatial.solve_grid_fmg if self.fmg else spatial.solve_grid_sym
        step = solver(hess, grad, optim='jacobi', **prm)
        step.mul_(-self.lr)
        return step


class FieldGaussNewton(SecondOrder):
    """Base class for Gauss-Newton on vector fields"""

    def __init__(self, fmg=2, max_iter=2, factor=1, voxel_size=1,
                 absolute=0, membrane=0, bending=0, reduce='mean',
                 marquardt=True, preconditioner=None, **kwargs):
        super().__init__(**kwargs)
        self.preconditioner = preconditioner
        self.factor = factor
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.reduce = reduce in ('mean', True)
        self.voxel_size = voxel_size
        self.marquardt = marquardt
        self.max_iter = max_iter
        self.fmg = fmg

    def _get_prm(self):
        prm = dict(absolute=self.absolute,
                   membrane=self.membrane,
                   bending=self.bending,
                   factor=self.factor,
                   voxel_size=self.voxel_size)
        if self.fmg:
            prm['nb_iter'] = self.max_iter
            prm['nb_cycles'] = self.fmg
        else:
            prm['max_iter'] = self.max_iter
        return prm

    def _add_marquardt(self, grad, hess):
        dim = grad.shape[-1]
        if self.marquardt is True:
            hess[..., :dim] += hess[..., :dim].abs().max(-1, True).values * 1e-5
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess


class FieldCG(FieldGaussNewton):
    """Gauss-Newton on vector fields using Conjugate Gradients"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        if self.reduce:
            prm['factor'] = prm['factor'] / py.prod(grad.shape[-dim-1:-1])
        solver = spatial.solve_field_fmg if self.fmg else spatial.solve_field_sym
        step = solver(hess, grad, optim='cg', **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step


class FieldRelax(FieldGaussNewton):
    """Gauss-Newton on vector fields using Relaxation"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm()
        if self.reduce:
            prm['factor'] = prm['factor'] / py.prod(grad.shape[-dim-1:-1])
        solver = spatial.solve_field_fmg if self.fmg else spatial.solve_field_sym
        step = solver(hess, grad, optim='relax', **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step
