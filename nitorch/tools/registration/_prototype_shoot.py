"""Diffeomorphic registration using Geodesic shooting.

The function `shoot.register` is a prototype that requires input images
to be defined on the same grid. See `registration.joint` for a more flexible
model.

The classes `RegisterStep` and `AutoRegStep` are implementation
utilities. The main entry points are `register` and `autoreg`.
"""
from nitorch import spatial
from nitorch.core import py, utils
import torch
from .utils import jg, jhj, defaults_velocity
from . import plot as plt, losses, phantoms, utils as regutils
import functools


class RegisterStep:
    """Forward pass of Shooting registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(self, moving, fixed, loss, verbose=True, plot=False,
                 max_iter=100, steps=8, kernel=None,
                 bound='dct2', extrapolate=True, **prm):
        """
        moving : tensor
        fixed : tensor
        loss : OptimizationLoss
        verbose : bool
        plot : bool
        max_iter : bool
        steps : int
        kernel : tensor - Precomputed kernel
        prm : dict
        """
        self.moving = moving
        self.fixed = fixed
        self.loss = loss
        self.verbose = verbose
        self.plot = plot
        self.prm = prm
        self.steps = steps
        self.kernel = kernel
        self.bound = bound
        self.extrapolate = extrapolate

        self.max_iter = max_iter
        self.n_iter = 0
        self.ll_prev = None
        self.ll_max = 0
        self.mugrad = None

    def __call__(self, vel, grad=False, hess=False, gradmov=False, hessmov=False):
        # This loop performs the forward pass, and computes
        # derivatives along the way.

        dim = vel.shape[-1]
        pullopt = dict(bound=self.bound, extrapolate=self.extrapolate)

        in_line_search = not grad and not hess
        logplot = max(self.max_iter // 20, 1)
        do_plot = (not in_line_search) and self.plot \
                  and (self.n_iter - 1) % logplot == 0

        # forward
        if self.kernel is None:
            self.kernel = spatial.greens(vel.shape[-dim-1:-1], **self.prm, **utils.backend(vel))
        grid = spatial.shoot(vel, self.kernel, steps=self.steps, **self.prm)
        warped = spatial.grid_pull(self.moving, grid, bound='dct2', extrapolate=True)

        if do_plot:
            iscat = isinstance(self.loss, losses.Cat)
            plt.mov2fix(self.fixed, self.moving, warped, vel, cat=iscat, dim=dim)

        # gradient/Hessian of the log-likelihood in observed space
        if not grad and not hess:
            llx = self.loss.loss(warped, self.fixed)
        elif not hess:
            llx, grad = self.loss.loss_grad(warped, self.fixed)
            if gradmov:
                gradmov = spatial.grid_push(grad, grid, **pullopt)
        else:
            llx, grad, hess = self.loss.loss_grad_hess(warped, self.fixed)
            if gradmov:
                gradmov = spatial.grid_push(grad, grid, **pullopt)
            if hessmov:
                hessmov = spatial.grid_push(hess, grid, **pullopt)
        del warped

        # compose with spatial gradients
        if grad is not False or hess is not False:
            if self.mugrad is None:
                self.mugrad = spatial.diff(self.moving, dim=list(range(-dim, 0)), bound='dct2')
            if grad is not False:
                grad = grad.neg_()  # "final inverse" to "initial"
                grad = spatial.grid_push(grad, grid)
                grad = jg(self.mugrad, grad)
            if hess is not False:
                hess = spatial.grid_push(hess, grid)
                hess = jhj(self.mugrad, hess)

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
        if gradmov is not False:
            out.append(gradmov)
        if hessmov is not False:
            out.append(hessmov)
        return tuple(out) if len(out) > 1 else out[0]


def register(fixed=None, moving=None, dim=None, lam=1., loss='mse',
             optim='ogm', hilbert=None, max_iter=500, sub_iter=16,
             lr=None, ls=6, steps=8, plot=False, klosure=RegisterStep,
             kernel=None, **prm):
    """Diffeomirphic registration between two images using geodesic shooting.

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
    optim : {'relax', 'cg', 'gd', 'momentum', 'nesterov'}, default='ogm'
        'relax'     : Gauss-Newton (linear system solved by relaxation)
        'cg'        : Gauss-Newton (linear system solved by conjugate gradient)
        'gd'        : Gradient descent
        'momentum'  : Gradient descent with momentum
        'nesterov'  : Nesterov-accelerated gradient descent
        'ogm'       : Optimized gradient descent (Kim & Fessler)
        'lbfgs'     : Limited-memory BFGS
    hilbert : bool, default=True
        Use hilbert preconditioning (not used if optim is second order)
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent iterations
    sub_iter : int, default=16
        Number of relax/cg iterations per GN step
    lr : float, default=1
        Learning rate.
    ls : int, default=6
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
    optim = regutils.make_iteroptim_grid(optim, lr, ls, max_iter, sub_iter, **prm)
    if hilbert is None:
        hilbert = not optim.requires_hess
    if hilbert and kernel is None:
        kernel = spatial.greens(shape, **prm, **utils.backend(fixed))
    if kernel is not None:
        optim.preconditioner = lambda x: spatial.greens_apply(x, kernel)

    # init loss
    loss = losses.make_loss(loss, dim)

    print(f'{"it":3s} | {"fit":^12s} + {"reg":^12s} = {"obj":^12s} | {"gain":^12s}')
    print('-' * 63)
    closure = klosure(moving, fixed, loss, steps=steps,
                      plot=plot, max_iter=optim.max_iter, **prm)
    vel = optim.iter(vel, closure)
    # vel = optm.IterateOptim(optm.GridCG(lr=1, sub_iter=16), max_iter=10).iter(vel, closure)
    print('')
    return vel


class AutoRegStep:
    """Forward pass of Shooting registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(self, moving, fixed, loss, verbose=True, plot=False,
                 max_iter=100, steps=8, kernel=None, **prm):
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
        self.steps = steps
        self.kernel = kernel

        self.max_iter = max_iter
        self.n_iter = 0
        self.ll_prev = None
        self.ll_max = 0

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
        if self.kernel is None:
            self.kernel = spatial.greens(vel.shape[-dim-1:-1], **self.prm, **utils.backend(vel))
        grid = spatial.shoot(vel, self.kernel, steps=self.steps, **self.prm)
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
