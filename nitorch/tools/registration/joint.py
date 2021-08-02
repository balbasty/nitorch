from nitorch import spatial
from nitorch.core import py, utils, linalg, math
import torch
from .utils import defaults_template, loadf, savef
from .phantoms import demo_atlas
from .utils import jg, jhj, defaults_velocity
from . import plot as plt, optim as optm, losses, phantoms, utils as regutils
import functools


class Image:
    """Data + Metadata (affine, boundary) of an Image"""
    def __init__(self, dat, affine=None, dim=None,
                 bound='dct2', extrapolate=True):
        """
        Parameters
        ----------
        dat : (C, *spatial) tensor
        affine : tensor, optional
        dim : int, default=1dat.dim() - 11
        bound : str, default='dct2'
        extrapolate : bool, default=True
        """
        self.dim = dim or dat.dim() - 1
        self.dat = dat
        if affine is None:
            shape = dat.shape[-dim:]
            affine = spatial.affine_default(shape, **utils.backend(dat))
        self.affine = affine
        self.bound = bound
        self.extrapolate = extrapolate

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, Image):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('affine', dat.affine)
            kwargs.setdefault('bound', dat.bound)
            kwargs.setdefault('extrapolate', dat.extrapolate)
            dat = dat.dat
        return cls(dat.dat, **kwargs)


class Displacement:
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
        self.dim = dim or dat.dim() - 1
        self.dat = dat
        if affine is None:
            shape = dat.shape[-dim-1:-1]
            affine = spatial.affine_default(shape, **utils.backend(dat))
        self.affine = affine

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, Displacement):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('affine', dat.affine)
            dat = dat.dat
        return cls(dat.dat, **kwargs)


class LogAffine:
    """Data + Metadata (basis) of a "Lie" affine matrix"""
    def __init__(self, dat, basis=None, dim=None, **backend):
        """
        Parameters
        ----------
        [dat : tensor, optional]
            Pre-allocated log-affine
        basis : tensor or str
            Pre-computed Lie basis, or its name
        dim : int, default=3
            Number of spatial dimensions
        **backend
        """
        if isinstance(dat, str) or \
                (torch.is_tensor(dat) and dat.dim() == 3):
            if basis is not None:
                raise ValueError('`basis` provided but `dat` looks like '
                                 'a basis.')
            basis, dat = dat, None
        if not isinstance(basis, str):
            dim = dim or basis.shape[-1] - 1
        else:
            dim = dim or 3
            basis = spatial.affine_basis(basis, dim, **backend)
        self.basis = basis
        self.dim = dim
        if dat is None:
            dat = torch.zeros(basis.shape[0], **backend)
        self.dat = dat

    @classmethod
    def make(cls, dat, **kwargs):
        if isinstance(dat, LogAffine):
            kwargs.setdefault('dim', dat.dim)
            kwargs.setdefault('basis', dat.basis)
            dat = dat.dat
        return LogAffine(dat.dat, **kwargs)


class LossComponent:
    """Component of a composite loss"""
    def __init__(self, loss, moving, fixed, weight=1, backward=False):
        self.moving = Image.make(moving)
        self.fixed = Image.make(fixed)
        self.loss = losses.make_loss(loss, self.moving.dim)
        self.weight = weight
        self.backward = backward

    def make(self, loss, **kwargs):
        if isinstance(loss, LossComponent):
            kwargs.setdefault('moving', loss.moving)
            kwargs.setdefault('fixed', loss.fixed)
            kwargs.setdefault('weight', loss.weight)
            kwargs.setdefault('backward', loss.backward)
            loss = loss.loss
        return LossComponent(loss, **kwargs)


class NonLinModel:

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

    def __init__(self, lam=1, prm=None, steps=8, kernel=None):
        super().__init__()
        self.lam = lam
        self.prm = prm or regutils.defaults_velocity()
        self.steps = steps
        self.kernel = kernel

    def make(self, model, **kwargs):
        if isinstance(model, self.__class__):
            kwargs.setdefault('lam', model.lam)
            kwargs.setdefault('prm', model.prm)
            kwargs.setdefault('steps', model.steps)
            kwargs.setdefault('kernel', model.kernel)
        elif isinstance(model, NonLinModel):
            raise TypeError('Cannot convert between `NonLinModel`s')
        return self.__class__(**kwargs)


class ShootModel(NonLinModel):
    pass


class SVFModel(NonLinModel):
    pass


class SmallDefModel(NonLinModel):
    pass


class AffineModel:

    def __init__(self, basis, lam=1, prm=None):
        super().__init__()
        self.basis = basis
        self.lam = lam
        self.prm = prm or regutils.defaults_affine()

    def make(self, basis, **kwargs):
        if isinstance(basis, AffineModel):
            kwargs.setdefault('lam', basis.lam)
            kwargs.setdefault('prm', basis.prm)
            self.basis = basis.basis
        return AffineModel(basis, **kwargs)


class RegisterStep:
    """Forward pass of Diffeo+Affine registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(
            self,
            losses,                 # list[LossComponent]
            affine,                 # AffineModel
            nonlin,                 # NonLinModel
            verbose=True,           # verbosity level
            ):
        self.losses = losses
        self.affine = affine
        self.nonlin = nonlin
        self.verbose = verbose

        # pretty printing
        self.n_iter = 0             # current iteration
        self.ll_prev = None         # previous loss value
        self.ll_max = 0             # max loss value

    def do_vel(self):

        # build affine and displacement field
        aff0 = self.affine.get_matrix()
        aff0 = spatial.affine_matmul(aff0, self.nonlin.affine)
        vel0 = self.nonlin.get_displacement()

        for loss in self.losses:

            aff = aff0
            aff = spatial.affine_lmdiv(loss.moving.affine, aff)
            phi = spatial.affine_matmul()


            def do(vel, grad=False, hess=False, gradmov=False, hessmov=False):
                """
                vel : (..., *spatial, dim) tensor, Velocity
                grad : Whether to compute and return the gradient wrt `vel`
                hess : Whether to compute and return the Hessian wrt `vel`
                gradmov : Whether to compute and return the gradient wrt `moving`
                hessmov : Whether to compute and return the Hessian wrt `moving`

                Returns
                -------
                ll : () tensor, loss value (objective to minimize)
                g : (..., *spatial, dim) tensor, optional, Gradient wrt velocity
                h : (..., *spatial, ?) tensor, optional, Hessian wrt velocity
                gm : (..., *spatial, dim) tensor, optional, Gradient wrt moving
                hm : (..., *spatial, ?) tensor, optional, Hessian wrt moving

                """
                # This loop performs the forward pass, and computes
                # derivatives along the way.

                dim = vel.shape[-1]
                pullopt = dict(bound=self.bound, extrapolate=self.extrapolate)

                in_line_search = not grad and not hess
                logplot = max(self.max_iter // 20, 1)
                do_plot = (not in_line_search) and self.plot \
                          and (self.n_iter - 1) % logplot == 0

                # forward
                grid, jac = spatial.exp_forward(vel, steps=self.steps, jacobian=True)
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

                    # sample derivatives and rotate them
                    # (we want `D(mu o phi)`, not `D(mu) o phi`)
                    mugrad = spatial.grid_grad(self.moving, grid, bound='dct2', extrapolate=True)
                    # TODO: add that back? see JA's email
                    # jac = torch.matmul(jac, spatial.grid_jacobian(vel, type='disp').inverse())
                    jac = jac.transpose(-1, -2)
                    mugrad = linalg.matvec(jac, mugrad)

                    derivatives = []
                    if grad is not False:
                        grad = jg(mugrad, grad)
                        derivatives.append(grad)
                    if hess is not False:
                        hess = jhj(mugrad, hess)
                        derivatives.append(hess)

                    # propagate backward
                    derivatives = spatial.exp_backward(vel, *derivatives, steps=self.steps)
                    if hess is not False:
                        grad, hess = derivatives
                    else:
                        grad = derivatives

                # add regularization term
                vgrad = spatial.regulariser_grid(vel, **self.prm)
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
            return do

class Register:

    def __init__(self, dim=None, lam=1., loss='mse',
                 optim='ogm', hilbert=None, max_iter=500, sub_iter=16,
                 lr=1, ls=6, steps=8, plot=False, klosure=RegisterStep,
                 kernel=None, verbose=True, **prm):
        self.dim = dim
        self.lam = lam
        self.loss = loss
        self.optim = optim
        self.hilbert = hilbert
        self.max_iter = max_iter
        self.sub_iter = sub_iter
        self.lr = lr
        self.ls = ls
        self.steps = steps
        self.verbose = verbose
        self.plot = plot
        self.kernel = kernel
        self.prm = prm
        self.klosure = klosure

    def __call__(self, fixed, moving, velocity=None, **overload):
        options = dict(self.__dict__)
        options.update(overload)
        return register(fixed, moving, velocity=velocity, **options)


def register(fixed=None, moving=None, dim=None, lam=1., loss='mse',
             optim='ogm', hilbert=None, max_iter=500, sub_iter=16,
             lr=1, ls=6, steps=8, plot=False, klosure=RegisterStep,
             velocity=None, kernel=None, verbose=True, **prm):
    """Diffeomorphic registration between two images using SVFs.

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
    steps : int, default=8
        Number of scaling and squaring steps
    plot : bool, default=False
        Plot progress
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
    if velocity is None:
        velocity = torch.zeros(velshape, **utils.backend(fixed))

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

    # optimize
    if verbose:
        print(f'{"it":3s} | {"fit":^12s} + {"reg":^12s} = {"obj":^12s} | {"gain":^12s}')
        print('-' * 63)
    closure = klosure(moving, fixed, loss, steps=steps, verbose=verbose,
                      plot=plot, max_iter=optim.max_iter, **prm)
    velocity = optim.iter(velocity, closure)
    if verbose:
        print('')
    return velocity


class AutoRegStep:
    """Forward pass of SVF registration, with derivatives"""
    # We use a class so that we can have a state to keep track of
    # iterations and objectives (mainly for pretty printing)

    def __init__(self, moving, fixed, loss, verbose=True, plot=False,
                 max_iter=100, steps=8, kernel=None, **prm):
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
        nvox = py.prod(self.fixed.shape[-dim:])

        in_line_search = not grad
        logplot = max(self.max_iter // 20, 1)
        do_plot = (not in_line_search) and self.plot \
                  and (self.n_iter - 1) % logplot == 0

        # forward
        grid, jac = spatial.exp_forward(vel, steps=self.steps, jacobian=True)
        warped = spatial.grid_pull(self.moving, grid, bound='dct2', extrapolate=True)

        if do_plot:
            iscat = isinstance(self.loss, losses.Cat)
            plt.mov2fix(self.fixed, self.moving, warped, vel, cat=iscat, dim=dim)

        # gradient/Hessian of the log-likelihood in observed space
        llx = self.loss.loss(warped, self.fixed)

        # add regularization term
        vgrad = spatial.regulariser_grid(vel, **self.prm).div_(nvox)
        llv = 0.5 * (vel * vgrad).sum()
        del vgrad
        lll = llx + llv

        # print objective
        llx = llx.item()
        llv = llv.item()
        ll = lll.item()
        if self.verbose and not in_line_search:
            self.n_iter += 1
            if self.ll_prev is None:
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g}', end='\r')
            else:
                gain = (self.ll_prev - ll) / max(abs(self.ll_max - ll), 1e-8)
                print(f'{self.n_iter:03d} | {llx:12.6g} + {llv:12.6g} = {ll:12.6g} | {gain:12.6g}', end='\r')
            self.ll_prev = ll
            self.ll_max = max(self.ll_max, ll)

        out = [lll]
        if grad:
            lll.backward()
            out.append(vel.grad)
        vel.requires_grad_(False)
        return tuple(out) if len(out) > 1 else out[0]


@functools.wraps(register)
def autoreg(*args, **kwargs):
    return register(*args, klosure=AutoRegStep, **kwargs)
