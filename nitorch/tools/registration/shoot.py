from nitorch import spatial
from nitorch.core import py, utils, linalg, math
import torch
from .losses import MSE, Cat, NCC, NMI
from .utils import jg, jhj, defaults_velocity, defaults_template
from .phantoms import demo_atlas, demo_register
from .optim import Optim, ZerothOrder, FirstOrder, SecondOrder, \
    GradientDescent, Momentum, Nesterov, GridCG, GridRelax, \
    BacktrackingLineSearch
from .utils import jg, jhj, defaults_velocity
from . import plot as plt, optim as optm, losses, phantoms


class RegisterStep:
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

        self.max_iter = max_iter
        self.n_iter = 0
        self.ll_prev = None
        self.ll_max = 0
        self.id = None
        self.mugrad = None

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
        else:
            llx, grad, hess = self.loss.loss_grad_hess(warped, self.fixed)
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
        vgrad = spatial.regulariser_grid(vel, **self.prm).div_(nvox)
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
             optim='nesterov', hilbert=True, max_iter=500, sub_iter=16,
             lr=1, ls=0, steps=8, plot=False, **prm):
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
    prm['factor'] = lam

    # If no inputs provided: demo "circle to square"
    if fixed is None or moving is None:
        fixed, moving = phantoms.demo_register(cat=(loss == 'cat'))

    # init tensors
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]
    velshape = [*fixed.shape[:-dim-1], *shape, dim]
    vel = torch.zeros(velshape, **utils.backend(fixed))

    # init optimizer
    lr = lr or 1
    optim = (optm.GradientDescent(lr=lr) if optim == 'gd' else
             optm.Momentum(lr=lr) if optim == 'momentum' else
             optm.Nesterov(lr=lr) if optim == 'nesterov' else
             optm.OGM(lr=lr) if optim == 'ogm' else
             optm.LBFGS(lr=lr, max_iter=max_iter) if optim == 'lbfgs' else
             optm.GridCG(lr=lr, max_iter=sub_iter, **prm) if optim == 'cg' else
             optm.GridRelax(lr=lr, max_iter=sub_iter, **prm) if optim == 'relax' else
             optm.GridNesterov(lr=lr, max_iter=sub_iter, **prm) if optim.startswith('gnnesterov') else
             optim)
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
    closure = RegisterStep(moving, fixed, loss, steps=steps,
                           plot=plot, max_iter=optim.max_iter, **prm)
    vel = optim.iter(vel, closure)
    # vel = optm.IterateOptim(optm.GridCG(lr=1, sub_iter=16), max_iter=10).iter(vel, closure)
    print('')
    return vel

# def register(fixed=None, moving=None, dim=None, lam=1., loss='mse',
#              optim='nesterov', hilbert=True, max_iter=500, sub_iter=16,
#              lr=1, ls=0, steps=8, plot=False, **prm):
#     """Diffeomorphic registration between two images using geodesic shooting.
#
#     Parameters
#     ----------
#     fixed : (..., K, *spatial) tensor
#         Fixed image
#     moving : (..., K, *spatial) tensor
#         Moving image
#     dim : int, default=`fixed.dim() - 1`
#         Number of spatial dimensions
#     lam : float, default=1
#         Modulate regularisation
#     loss : {'mse', 'cat'} or OptimizationLoss, default='mse'
#         'mse': Mean-squared error
#         'cat': Categorical cross-entropy
#     optim : {'relax', 'cg', 'gd', 'momentum', 'nesterov'}, default='relax'
#         'relax': Gauss-Newton (linear system solved by relaxation)
#         'cg': Gauss-Newton (linear system solved by conjugate gradient)
#         'gd': Gradient descent
#         'momentum': Gradient descent with momentum
#         'nesterov': Nesterov-accelerated gradient descent
#     hilbert : bool, default=True
#         Use hilbert gradient (not used if optim is second order)
#     max_iter : int, default=100
#         Maximum number of Gauss-Newton or Gradient descent iterations
#     sub_iter : int, default=16
#         Number of relax/cg iterations per GN step
#     lr : float, default=1
#         Learning rate.
#     ls : int, default=0
#         Number of line search iterations.
#     absolute : float, default=1e-4
#         Penalty on absolute displacements
#     membrane : float, default=1e-3
#         Penalty on first derivatives
#     bending : float, default=0.2
#         Penalty on second derivatives
#     lame : (float, float), default=(0.05, 0.2)
#         Penalty on zooms and shears
#
#     Returns
#     -------
#     disp : (..., *spatial, dim) tensor
#         Displacement field.
#
#     """
#     defaults_velocity(prm)
#     prm['factor'] = lam
#
#     # If no inputs provided: demo "circle to square"
#     if fixed is None or moving is None:
#         fixed, moving = demo_register(cat=(loss == 'cat'))
#
#     # init tensors
#     fixed, moving = utils.to_max_backend(fixed, moving)
#     dim = dim or (fixed.dim() - 1)
#     shape = fixed.shape[-dim:]
#     nvox = py.prod(shape)
#     velshape = [*fixed.shape[:-dim-1], *shape, dim]
#     vel0 = torch.zeros(velshape, **utils.backend(fixed))
#
#     # init optimizer
#     lr = lr or 1
#     optim = (GradientDescent(lr=lr) if optim == 'gd' else
#              Nesterov(lr=lr) if optim == 'nesterov' else
#              Momentum(lr=lr) if optim == 'momentum' else
#              GridCG(lr=lr, max_iter=sub_iter, **prm) if optim == 'cg' else
#              GridRelax(lr=lr, max_iter=sub_iter, **prm) if optim == 'relax' else
#              optim)
#     if ls:
#         ls = BacktrackingLineSearch(max_iter=ls)
#     kernel = spatial.greens(shape, **prm, **utils.backend(fixed))
#     mugrad = spatial.diff(moving, dim=list(range(-dim, 0)), bound='dct2')
#
#     # init loss
#     iscat = loss == 'cat'
#     loss = (MSE(dim=dim) if loss == 'mse' else
#             Cat(dim=dim) if loss == 'cat' else
#             NCC(dim=dim) if loss == 'ncc' else
#             NMI(dim=dim) if loss == 'nmi' else
#             loss)
#
#     print(f'{"it":3s} | {"fit":^12s} + {"reg":^12s} = {"obj":^12s} | {"gain":^12s}')
#     print('-' * 63)
#
#     def closure(vel=None):
#
#         in_line_search = True
#         if vel is None:
#             in_line_search = False
#             vel = vel0
#
#         # forward
#         grid = spatial.shoot(vel, kernel, steps=steps, **prm)
#         warped = spatial.grid_pull(moving, grid, bound='dct2', extrapolate=True)
#         if (not in_line_search) and plot and ((n_iter - 1) % (max_iter//20)) == 0:
#             plt.mov2fix(fixed, moving, warped, vel, cat=iscat, dim=dim)
#
#         # gradient/Hessian of the log-likelihood in observed space
#         grad = hess = None
#         if in_line_search or isinstance(optim, ZerothOrder):
#             llx = loss.loss(warped, fixed)
#         elif isinstance(optim, FirstOrder):
#             llx, grad = loss.loss_grad(warped, fixed)
#         else:
#             llx, grad, hess = loss.loss_grad_hess(warped, fixed)
#         del warped
#
#         # compose with spatial gradients
#         if grad is not None or hess is not None:
#
#             # push to template space
#             if grad is not None:
#                 grad = grad.neg_()  # "final inverse" to "initial"
#                 grad = spatial.grid_push(grad, grid)
#                 grad = jg(mugrad, grad)
#             if hess is not None:
#                 hess = spatial.grid_push(hess, grid)
#                 hess = jhj(mugrad, hess)
#
#         # add regularization term
#         vgrad = spatial.regulariser_grid(vel, **prm).div_(nvox)
#         llv = 0.5 * (vel*vgrad).sum()
#         if grad is not None:
#             grad += vgrad
#         del vgrad
#
#         # print objective
#         llx = llx.item()
#         llv = llv.item()
#         ll = llx + llv
#         if not in_line_search:
#             if ll_prev is None:
#                 print(f'{n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g}', end='\r')
#             else:
#                 gain = (ll_prev - ll) / max(abs(ll_max - ll), 1e-8)
#                 print(f'{n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')
#
#         if hilbert and grad is not None and hess is None:
#             grad = spatial.greens_apply(grad, kernel)
#
#         if in_line_search:
#             return ll
#         else:
#             out = [ll]
#             if grad is not None:
#                 out.append(grad)
#             if hess is not None:
#                 out.append(hess)
#             return tuple(out)
#
#     ll_prev = None
#     ll_max = None
#     for n_iter in range(1, max_iter+1):
#
#         ll, *derivatives = closure()
#         ll_prev = ll
#         ll_max = ll_max or ll_prev
#
#         step = optim.step(*derivatives)
#         if ls:
#             vel0, success = ls.step(closure, vel0, ll, step)
#             if not success:
#                 print('\nFailed to improve')
#                 break
#         else:
#             vel0.add_(step)
#
#     print('')
#     return vel0


def autoreg(fixed=None, moving=None, dim=None, lam=1., loss='mse',
             optim='nesterov', hilbert=True, max_iter=500,
             lr=1, ls=0, steps=8, plot=False, **prm):
    """Diffeomorphic registration between two images using geodesic shooting.

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
    optim : {'relax', 'cg', 'gd', 'momentum', 'nesterov'}, default='nesterov'
        'gd': Gradient descent
        'momentum': Gradient descent with momentum
        'nesterov': Nesterov-accelerated gradient descent
    hilbert : bool, default=True
        Use hilbert gradient (not used if optim is second order)
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent iterations
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
    prm['factor'] = lam

    # If no inputs provided: demo "circle to square"
    if fixed is None or moving is None:
        fixed, moving = demo_register(cat=(loss == 'cat'))

    # init tensors
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]
    nvox = py.prod(shape)
    velshape = [*fixed.shape[:-dim-1], *shape, dim]
    vel0 = torch.zeros(velshape, **utils.backend(fixed))
    id = spatial.identity_grid(shape, **utils.backend(vel0))

    # init optimizer
    lr = lr or 1
    optim = (GradientDescent(lr=lr) if optim == 'gd' else
             optm.Momentum(lr=lr) if optim == 'momentum' else
             optm.Nesterov(lr=lr) if optim == 'nesterov' else
             optm.OGM(lr=lr) if optim == 'ogm' else
             optim)
    if ls:
        ls = BacktrackingLineSearch(max_iter=ls)
    kernel = spatial.greens(shape, **prm, **utils.backend(fixed))

    # init loss
    iscat = loss == 'cat'
    loss = (MSE(dim=dim) if loss == 'mse' else
            Cat(dim=dim) if loss == 'cat' else
            NCC(dim=dim) if loss == 'ncc' else
            NMI(dim=dim) if loss == 'nmi' else
            loss)

    print(f'{"it":3s} | {"fit":^12s} + {"reg":^12s} = {"obj":^12s} | {"gain":^12s}')
    print('-' * 63)

    def closure(vel=None, in_line_search=True):

        # we do a bit of magic so that gradients are only tracked if
        # derivatives are being computed.
        # We know that we are not in a line search iteration (and
        # therefore need to compute derivatives) if in_line_search is
        # True or vel is None.
        if vel is None:
            vel = vel0
            vel.requires_grad_()
            if vel.grad is not None:
                vel.grad.zero_()
            with torch.enable_grad():
                return closure(vel, in_line_search=False)

        # forward
        grid = spatial.shoot(vel, kernel, steps=steps, **prm)
        warped = spatial.grid_pull(moving, grid, bound='dct2', extrapolate=True)
        if (not in_line_search) and plot and ((n_iter - 1) % (max_iter//20)) == 0:
            plt.mov2fix(fixed, moving, warped, vel, cat=iscat, dim=dim)

        # log-likelihood in observed space
        llx = loss.loss(warped, fixed)
        del warped

        # add regularization term
        vgrad = spatial.regulariser_grid(vel, **prm).div_(nvox)
        llv = 0.5 * (vel*vgrad).sum()
        lll = llx + llv
        del vgrad

        # print objective
        llx = llx.item()
        llv = llv.item()
        ll = lll.item()
        if not in_line_search:
            if ll_prev is None:
                print(f'{n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g}', end='\r')
            else:
                gain = (ll_prev - ll) / max(abs(ll_max - ll), 1e-8)
                print(f'{n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')

        if in_line_search:
            return lll
        else:
            lll.backward()
            out = [lll, vel.grad]
            vel.requires_grad_(False)
            return tuple(out)

    ll_prev = None
    ll_max = None
    for n_iter in range(1, max_iter+1):

        ll, grad = closure()
        ll_prev = ll
        ll_max = ll_max or ll_prev

        if hilbert:
            grad = spatial.greens_apply(grad, kernel)
        step = optim.step(grad)

        if ls:
            vel0, success = ls.step(closure, vel0, ll, step)
            if not success:
                print('\nFailed to improve')
                break
        else:
            vel0.add_(step)

    print('')
    return vel0

