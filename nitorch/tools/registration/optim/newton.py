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
            # `hess` is a full (N, N) matrix: only load its diagonal
            hess.diagonal(0, -1, -2).add_(self.marquardt)
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
    """Base class for Gauss-Newton on vector fields

    Vector fields are channel-first:
    - grad : (..., K, *spatial) tensor
    - hess : (..., K2, *spatial) tensor, with K2 in {1, K, K*(K+1)//2}
    """

    def __init__(self, lr=1, fmg=2, max_iter=2, factor=1, voxel_size=1,
                 absolute=0, membrane=0, bending=0, reduce='mean',
                 marquardt=True, preconditioner=None, bound='dct2',
                 dim=None, **kwargs):
        """

        Parameters
        ----------
        lr : float, default=1
            Learning rate
        fmg : int, default=2
            Number of full multi-grid cycles (0 = single resolution solver)
        max_iter : int, default=2
            Number of iterations of the inner linear solver
        factor : float, default=1
            Global regularization factor
        voxel_size : [sequence of] float, default=1
        absolute, membrane, bending : float, default=0
            Regularization weights
        reduce : {'mean', 'sum'} or bool, default='mean'
            If 'mean', `factor` is divided by the number of voxels.
        marquardt : bool or float, default=True
            Levenberg-Marquardt regularization added to the Hessian diagonal
        bound : str, default='dct2'
            Boundary conditions of the regularizer
        dim : int, optional
            Number of spatial dimensions.
            Default: `grad.dim() - 1` (as in `spatial.solve_field_*`)
        """
        super().__init__(lr, **kwargs)
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
        self.bound = bound
        self.dim = dim

    def _get_dim(self, grad):
        """Number of spatial dimensions of a channel-first field"""
        return self.dim or (grad.dim() - 1)

    def _get_prm(self, grad):
        dim = self._get_dim(grad)
        prm = dict(absolute=self.absolute,
                   membrane=self.membrane,
                   bending=self.bending,
                   factor=self.factor,
                   voxel_size=self.voxel_size,
                   bound=self.bound,
                   dim=dim)
        if self.reduce:
            # fields are (..., K, *spatial): voxels are the last `dim` axes
            nvox = py.prod(grad.shape[-dim:])
            if isinstance(prm['factor'], (list, tuple)):
                prm['factor'] = [f / nvox for f in prm['factor']]
            else:
                prm['factor'] = prm['factor'] / nvox
        if self.fmg:
            prm['nb_iter'] = self.max_iter
            prm['nb_cycles'] = self.fmg
        else:
            prm['max_iter'] = self.max_iter
        return prm

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        if not self.marquardt:
            return grad, hess
        dim = self._get_dim(grad)
        # channels live at dimension -dim-1; the first K (or 1)
        # Hessian channels hold the diagonal
        nb_channels = grad.shape[-dim-1]
        nb_diag = min(nb_channels, hess.shape[-dim-1])
        diag = hess.narrow(-dim-1, 0, nb_diag)
        if self.marquardt is True:
            maj = diag.abs().max(-dim-1, keepdim=True).values
            diag.add_(maj, alpha=tiny)
        else:
            diag.add_(self.marquardt)
        return grad, hess


class FieldCG(FieldGaussNewton):
    """Gauss-Newton on vector fields using Conjugate Gradients"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm(grad)
        solver = spatial.solve_field_fmg if self.fmg else spatial.solve_field_sym
        step = solver(hess, grad, optim='cg', **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step


class FieldRelax(FieldGaussNewton):
    """Gauss-Newton on vector fields using Relaxation"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        prm = self._get_prm(grad)
        solver = spatial.solve_field_fmg if self.fmg else spatial.solve_field_sym
        step = solver(hess, grad, optim='relax', **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step
