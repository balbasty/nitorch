from nitorch import spatial
from nitorch.core import py, utils, linalg, math
import torch
from .utils import defaults_template, loadf, savef
from .phantoms import demo_atlas
from .utils import jg, jhj, defaults_velocity
from . import plot as plt, optim as optm, losses, phantoms, utils as regutils
import functools


class RegisterStep:
    """Forward pass of affine registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(self, moving, fixed, loss, basis='CSO', dim=None,
                 affine_moving=None, affine_fixed=None, verbose=True,
                 plot=False, max_iter=100, bound='dct2', extrapolate=True,
                 **prm):
        if dim is None:
            if affine_fixed is not None:
                dim = affine_fixed.shape[-1] - 1
            elif affine_moving is not None:
                dim = affine_moving.shape[-1] - 1
        dim = dim or fixed.dim() - 1
        self.dim = dim
        self.moving = moving        # moving image
        self.fixed = fixed          # fixed image
        self.loss = loss            # similarity loss (`OptimizationLoss`)
        self.verbose = verbose      # print stuff
        self.plot = plot            # plot stuff
        self.prm = prm              # dict of regularization parameters
        self.bound = bound
        self.extrapolate = extrapolate
        self.basis = basis
        if affine_fixed is None:
            affine_fixed = spatial.affine_default(fixed.shape[-dim:], **utils.backend(fixed))
        if affine_moving is None:
            affine_moving = spatial.affine_default(moving.shape[-dim:], **utils.backend(moving))
        self.affine_fixed = affine_fixed
        self.affine_moving = affine_moving

        # pretty printing
        self.max_iter = max_iter    # max number of iterations
        self.n_iter = 0             # current iteration
        self.ll_prev = None         # previous loss value
        self.ll_max = 0             # max loss value
        self.id = None

    def __call__(self, logaff, grad=False, hess=False,
                 gradmov=False, hessmov=False, in_line_search=False):
        """
        logaff : (..., nb) tensor, Lie parameters
        grad : Whether to compute and return the gradient wrt `logaff`
        hess : Whether to compute and return the Hessian wrt `logaff`
        gradmov : Whether to compute and return the gradient wrt `moving`
        hessmov : Whether to compute and return the Hessian wrt `moving`

        Returns
        -------
        ll : () tensor, loss value (objective to minimize)
        g : (..., logaff) tensor, optional, Gradient wrt Lie parameters
        h : (..., logaff) tensor, optional, Hessian wrt Lie parameters
        gm : (..., *spatial, dim) tensor, optional, Gradient wrt moving
        hm : (..., *spatial, ?) tensor, optional, Hessian wrt moving

        """
        # This loop performs the forward pass, and computes
        # derivatives along the way.

        pullopt = dict(bound=self.bound, extrapolate=self.extrapolate)

        logplot = max(self.max_iter // 20, 1)
        do_plot = (not in_line_search) and self.plot \
                  and (self.n_iter - 1) % logplot == 0

        # forward
        if not torch.is_tensor(self.basis):
            self.basis = spatial.affine_basis(self.basis, self.dim,
                                              **utils.backend(logaff))
        aff, gaff, haff = linalg._expm(logaff, self.basis,
                                       grad_X=True, hess_X=True)
        aff = spatial.affine_matmul(aff, self.affine_fixed)
        aff = spatial.affine_lmdiv(self.affine_moving, aff)
        gaff = spatial.affine_matmul(gaff, self.affine_fixed)
        gaff = spatial.affine_lmdiv(self.affine_moving, gaff)
        haff = spatial.affine_matmul(haff, self.affine_fixed)
        haff = spatial.affine_lmdiv(self.affine_moving, haff)
        if self.id is None:
            shape = self.fixed.shape[-self.dim:]
            self.id = spatial.identity_grid(shape, **utils.backend(logaff))
        grid = spatial.affine_matvec(aff, self.id)
        warped = spatial.grid_pull(self.moving, grid, **pullopt)
        if do_plot:
            iscat = isinstance(self.loss, losses.Cat)
            plt.mov2fix(self.fixed, self.moving, warped, cat=iscat, dim=self.dim)

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

        # compose with spatial gradients + dot product with grid
        if grad is not False or hess is not False:
            mugrad = spatial.grid_grad(self.moving, grid, **pullopt)
            grad = jg(mugrad, grad)
            if hess is not False:
                hess = jhj(mugrad, hess)
                grad, hess = regutils.affine_grid_backward(grad, hess, grid=self.id)
            else:
                grad = regutils.affine_grid_backward(grad, grid=self.id)
            dim2 = self.dim*(self.dim+1)
            grad = grad.reshape([*grad.shape[:-2], dim2])
            gaff = gaff[..., :-1, :]
            gaff = gaff.reshape([*gaff.shape[:-2], dim2])
            grad = linalg.matvec(gaff, grad)
            if hess is not False:
                hess = hess.reshape([*hess.shape[:-4], dim2, dim2])
                # haff = haff[..., :-1, :, :-1, :]
                # haff = haff.reshape([*gaff.shape[:-4], dim2, dim2])
                hess = gaff.matmul(hess).matmul(gaff.transpose(-1, -2))
            del mugrad

        # print objective
        llx = llx.item()
        ll = llx
        if self.verbose and not in_line_search:
            self.n_iter += 1
            if self.ll_prev is None:
                print(f'{self.n_iter:03d} | {llx:12.6g} + {0:12.6g} = {ll:12.6g}', end='\r')
            else:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                print(f'{self.n_iter:03d} | {llx:12.6g} + {0:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')
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


class Register:

    def __init__(self, dim=None, loss='mse', basis='CSO',
                 optim='ogm', max_iter=500, sub_iter=16,
                 lr=1, ls=6, plot=False, klosure=RegisterStep,
                 verbose=True, **prm):
        self.dim = dim
        self.loss = loss
        self.optim = optim
        self.max_iter = max_iter
        self.sub_iter = sub_iter
        self.lr = lr
        self.ls = ls
        self.verbose = verbose
        self.plot = plot
        self.prm = prm
        self.klosure = klosure
        self.basis = basis

    def __call__(self, fixed, moving, velocity=None, **overload):
        options = dict(self.__dict__)
        options.update(overload)
        return register(fixed, moving, velocity=velocity, **options)


def register(fixed=None, moving=None, dim=None, loss='mse',
             basis='CSO', optim='ogm', max_iter=500,
             lr=1, ls=6, plot=False, klosure=RegisterStep,
             logaff=None, verbose=True):
    """Affine registration between two images using Lie groups.

    Parameters
    ----------
    fixed : (..., K, *spatial) tensor
        Fixed image
    moving : (..., K, *spatial) tensor
        Moving image
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions
    loss : {'mse', 'cat'} or OptimizationLoss, default='mse'
        'mse': Mean-squared error
        'cat': Categorical cross-entropy
    optim : {'relax', 'cg', 'gd', 'momentum', 'nesterov'}, default='ogm'
        'gn'        : Gauss-Newton
        'gd'        : Gradient descent
        'momentum'  : Gradient descent with momentum
        'nesterov'  : Nesterov-accelerated gradient descent
        'ogm'       : Optimized gradient descent (Kim & Fessler)
        'lbfgs'     : Limited-memory BFGS
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent iterations
    lr : float, default=1
        Learning rate.
    ls : int, default=6
        Number of line search iterations.
    plot : bool, default=False
        Plot progress

    Returns
    -------
    logaff : (...) tensor
        Displacement field.

    """

    # If no inputs provided: demo "circle to square"
    if fixed is None or moving is None:
        fixed, moving = phantoms.demo_register(cat=(loss == 'cat'))

    # init tensors
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    basis = spatial.affine_basis(basis, dim, **utils.backend(fixed))
    if logaff is None:
        logaff = torch.zeros(len(basis), **utils.backend(fixed))

    # init optimizer
    optim = regutils.make_iteroptim_affine(optim, lr, ls, max_iter)

    # init loss
    loss = losses.make_loss(loss, dim)

    # optimize
    if verbose:
        print(f'{"it":3s} | {"fit":^12s} + {"reg":^12s} = {"obj":^12s} | {"gain":^12s}')
        print('-' * 63)
    closure = klosure(moving, fixed, loss, basis=basis, verbose=verbose,
                      plot=plot, max_iter=optim.max_iter)
    logaff = optim.iter(logaff, closure)
    if verbose:
        print('')
    return logaff