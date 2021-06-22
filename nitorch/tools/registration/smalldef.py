import torch
from nitorch import spatial
from nitorch.core import py, utils, linalg, math
from .utils import jg, jhj, defaults_velocity
from . import plot as plt, optim as optm, losses, phantoms
import functools


class RegisterStep:
    """Forward pass of Small Deformation registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(self, moving, fixed, loss, verbose=True, plot=False, max_iter=100, **prm):
        """
        moving : tensor
        fixed : tensor
        loss : OptimizationLoss
        verbose : bool
        plot : bool
        max_iter : bool
        prm : dict
        """
        self.moving = moving
        self.fixed = fixed
        self.loss = loss
        self.verbose = verbose
        self.plot = plot
        self.prm = prm

        self.max_iter = max_iter
        self.n_iter = 0
        self.ll_prev = None
        self.ll_max = 0
        self.id = None

    def __call__(self, vel, grad=False, hess=False):
        # This loop performs the forward pass, and computes
        # derivatives along the way.

        dim = vel.shape[-1]
        nvox = py.prod(self.fixed.shape[-dim:])

        in_line_search = not grad and not hess
        logplot = max(self.max_iter // 20, 1)
        do_plot = (not in_line_search) and self.plot \
                  and (self.n_iter - 1) % logplot == 0

        # forward
        if self.id is None:
            self.id = spatial.identity_grid(vel.shape[-dim-1:-1], **utils.backend(vel))
        grid = self.id + vel
        warped = spatial.grid_pull(self.moving, grid, bound='dct2', extrapolate=True)

        if do_plot:
            iscat = isinstance(self.loss, losses.Cat)
            plt.mov2fix(self.fixed, self.moving, warped, vel, cat=iscat, dim=dim)

        # gradient/Hessian of the log-likelihood in observed space
        if not grad and not hess:
            llx = self.loss.loss(warped, self.fixed)
        elif not hess:
            llx, grad = self.loss.loss_grad(warped, self.fixed)
        else:
            llx, grad, hess = self.loss.loss_grad_hess(warped, self.fixed)
        del warped

        # compose with spatial gradients
        if grad is not False or hess is not False:
            mugrad = spatial.grid_grad(self.moving, grid, bound='dct2', extrapolate=True)
            if grad is not False:
                grad = jg(mugrad, grad)
            if hess is not False:
                hess = jhj(mugrad, hess)

        # add regularization term
        vgrad = spatial.regulariser_grid(vel, **self.prm, kernel=True)
        llv = 0.5 * (vel * vgrad).sum()
        if grad is not False:
            grad += vgrad
        del vgrad

        # print objective
        llx = llx.item()
        llv = llv.item()
        ll = llx + llv
        if self.verbose and not in_line_search:
            self.n_iter += 1
            if self.ll_prev is None:
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g}', end='\r')
            else:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [ll]
        if grad is not False:
            out.append(grad)
        if hess is not False:
            out.append(hess)
        return tuple(out) if len(out) > 1 else out[0]


def register(fixed=None, moving=None, dim=None, lam=1., loss='mse',
             optim='nesterov', hilbert=None, max_iter=500, sub_iter=16,
             lr=None, ls=0, plot=False, klosure=RegisterStep, **prm):
    """Nonlinear registration between two images using smooth displacements.

    Parameters
    ----------
    fixed : (..., K, *spatial) tensor
        Fixed image
    moving : (..., K, *spatial) tensor
        Moving image
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions
    lam : float, default=1
        Modulate regularisation
    loss : {'mse', 'cat'} or OptimizationLoss, default='mse'
        'mse': Mean-squared error
        'cat': Categorical cross-entropy
    optim : {'relax', 'cg', 'gd', 'momentum', 'nesterov'}, default='relax'
        'relax'     : Gauss-Newton (linear system solved by relaxation)
        'cg'        : Gauss-Newton (linear system solved by conjugate gradient)
        'gd'        : Gradient descent
        'momentum'  : Gradient descent with momentum
        'nesterov'  : Nesterov-accelerated gradient descent
        'lbfgs'     : Limited-memory BFGS
    hilbert : bool, default=True
        Use hilbert preconditioning (not used if optim is second order)
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent iterations
    sub_iter : int, default=16
        Number of relax/cg iterations per GN step
    lr : float, default=1
        Learning rate.
    ls : int, default=0
        Number of line search iterations.
    absolute : float, default=1e-4
        Penalty on absolute displacements
    membrane : float, default=1e-3
        Penalty on first derivatives
    bending : float, default=0.2
        Penalty on second derivatives
    lame : (float, float), default=(0.05, 0.2)
        Penalty on zooms and shears

    Returns
    -------
    disp : (..., *spatial, dim) tensor
        Displacement field.

    """
    defaults_velocity(prm)

    # If no inputs provided: demo "circle to square"
    if fixed is None or moving is None:
        fixed, moving = phantoms.demo_register(cat=(loss == 'cat'))

    # init tensors
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]
    lam = lam / py.prod(shape)
    prm['factor'] = lam
    velshape = [*fixed.shape[:-dim-1], *shape, dim]
    vel = torch.zeros(velshape, **utils.backend(fixed))

    # init optimizer
    optim = (optm.GradientDescent() if optim == 'gd' else
             optm.Momentum() if optim == 'momentum' else
             optm.Nesterov() if optim == 'nesterov' else
             optm.OGM() if optim == 'ogm' else
             optm.LBFGS(max_iter=max_iter) if optim == 'lbfgs' else
             optm.GridCG(max_iter=sub_iter, **prm) if optim == 'cg' else
             optm.GridRelax(max_iter=sub_iter, **prm) if optim == 'relax' else
             optim)
    if isinstance(optim, optm.Optim):
        lr = lr or (1 if isinstance(optim, optm.SecondOrder) else 0.01)
        optim.lr = lr
    if hilbert is None:
        hilbert = not isinstance(optim, optm.SecondOrder)
    if hilbert and hasattr(optim, 'preconditioner'):
        # Hilbert gradient
        kernel = spatial.greens(shape, **prm, **utils.backend(fixed))
        optim.preconditioner = lambda x: spatial.greens_apply(x, kernel)
    if not hasattr(optim, 'iter'):
        optim = optm.IterateOptim(optim, max_iter=max_iter, ls=ls)

    # init loss
    loss = (losses.MSE(dim=dim) if loss == 'mse' else
            losses.Cat(dim=dim) if loss == 'cat' else
            losses.NCC(dim=dim) if loss == 'ncc' else
            losses.NMI(dim=dim) if loss == 'nmi' else
            loss)

    print(f'{"it":3s} | {"fit":^12s} + {"reg":^12s} = {"obj":^12s} | {"gain":^12s}')
    print('-' * 63)
    closure = klosure(moving, fixed, loss, plot=plot,
                      max_iter=optim.max_iter, **prm)
    vel = optim.iter(vel, closure)
    print('')
    return vel


class AutoRegStep:
    """Forward pass of Small Deformation registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(self, moving, fixed, loss, verbose=True, plot=False, max_iter=100, **prm):
        """
        moving : tensor
        fixed : tensor
        loss : OptimizationLoss
        verbose : bool
        plot : bool
        max_iter : bool
        prm : dict
        """
        self.moving = moving
        self.fixed = fixed
        self.loss = loss
        self.verbose = verbose
        self.plot = plot
        self.prm = prm

        self.max_iter = max_iter
        self.n_iter = 0
        self.ll_prev = None
        self.ll_max = 0
        self.id = None

    def __call__(self, vel, grad=False):
        # This loop performs the forward pass, and computes
        # derivatives along the way.

        # select correct gradient mode
        if grad:
            vel.requires_grad_()
            if vel.grad is not None:
                vel.grad.zero_()
        if grad and not torch.is_grad_enabled():
            with torch.enable_grad():
                return self(vel, grad)
        elif not grad and torch.is_grad_enabled():
            with torch.no_grad():
                return self(vel, grad)

        dim = vel.shape[-1]
        nvox = py.prod(vel.shape[-dim-1:-1])

        in_line_search = not grad
        do_plot = (not in_line_search) and self.plot \
                  and (self.n_iter - 1) % 20 == 0

        # forward
        if self.id is None:
            self.id = spatial.identity_grid(vel.shape[-dim-1:-1], **utils.backend(vel))
        grid = self.id + vel
        warped = spatial.grid_pull(self.moving, grid, bound='dct2', extrapolate=True)

        if do_plot:
            iscat = isinstance(self.loss, losses.Cat)
            plt.mov2fix(self.fixed, self.moving, warped, vel, cat=iscat, dim=dim)

        # log-likelihood in observed space
        llx = self.loss.loss(warped, self.fixed)
        del warped

        # add regularization term
        vgrad = spatial.regulariser_grid(vel, **self.prm).div_(nvox)
        llv = 0.5 * (vel*vgrad).sum()
        lll = llx + llv
        del vgrad

        # print objective
        llx = llx.item()
        llv = llv.item()
        ll = lll.item()
        if not in_line_search:
            self.n_iter += 1
            if self.ll_prev is None:
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g}', end='\r')
            else:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')

        out = [lll]
        if grad:
            lll.backward()
            out.append(vel.grad)
        vel.requires_grad_(False)
        return tuple(out) if len(out) > 1 else out[0]


@functools.wraps(register)
def autoreg(*args, **kwargs):
    return register(*args, klosure=AutoRegStep, **kwargs)