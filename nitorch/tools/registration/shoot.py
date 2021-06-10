from nitorch import spatial
from nitorch.core import py, utils, linalg, math
import torch
from .losses import MSE, Cat
from .utils import jg, jhj, defaults_velocity, defaults_template
from .phantoms import demo_atlas, demo_register
from . import plot as plt


def register(fixed=None, moving=None, dim=None, lam=1., max_iter=20,
             loss='mse', optim='relax', sub_iter=16, ls=0, plot=False, steps=8,
             **prm):
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
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent optimisation
    loss : {'mse', 'cat'} or OptimizationLoss, default='mse'
        'mse': Mean-squared error
        'cat': Categorical cross-entropy
    optim : {'relax', 'cg', 'gd', 'gdh'}, default='relax'
        'relax': Gauss-Newton (linear system solved by relaxation)
        'cg': Gauss-Newton (linear system solved by conjugate gradient)
        'gd': Gradient descent
        'gdh': Hilbert gradient descent
    sub_iter : int, default=16
        Number of relax/cg iterations per GN step
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
    vel : (..., *spatial, dim) tensor
        Initial velocity field (in template space).

    """
    defaults_velocity(prm)
    prm['factor'] = lam

    # If no inputs provided: demo "circle to square"
    if fixed is None or moving is None:
        fixed, moving = demo_register([128, 128], cat=(loss == 'cat'))

    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]
    velshape = [*fixed.shape[:-dim-1], *shape, dim]
    vel = torch.zeros(velshape, **utils.backend(fixed))

    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)
    kernel = spatial.greens(shape, **prm)

    iscat = loss == 'cat'
    loss = (MSE(dim=dim) if loss == 'mse' else
            Cat(dim=dim) if loss == 'cat' else
            loss)

    ll_prev = None
    for n_iter in range(1, max_iter+1):

        # forward
        grid = spatial.shoot(vel, kernel, steps=steps, **prm)

        # gradient/Hessian of the log-likelihood in observed space
        warped = spatial.grid_pull(moving, grid, bound='dct2', extrapolate=True)
        if plot:
            plt.mov2fix(fixed, moving, warped, vel, cat=iscat, dim=dim)
        ll, grad, hess = loss.loss_grad_hess(warped, fixed)
        del warped

        # push to template space
        grad = spatial.grid_push(grad, grid)
        hess = spatial.grid_push(hess, grid)

        # compose with spatial gradients
        mugrad = spatial.grid_grad(moving, grid, bound='dct2', extrapolate=True)
        grad = jg(mugrad, grad).neg_()  # neg needed, but why?
        hess = jhj(mugrad, hess)
        vgrad = spatial.regulariser_grid(vel, **prm)

        # print objective
        ll += (vel*vgrad).sum()
        ll /= fixed.numel()
        ll = ll.item()
        if ll_prev is None:
            ll_prev = ll
            ll_max = ll
            print(f'{n_iter:03d} | {ll:12.6g}', end='\r')
        else:
            gain = (ll_prev - ll) / max(abs(ll_max - ll), 1e-8)
            ll_prev = ll
            print(f'{n_iter:03d} | {ll:12.6g} | gain = {gain:12.6g}', end='\r')

        # propagate gradients backward
        grad, hess = spatial.exp_backward(vel, grad, hess, steps=steps)
        grad += vgrad

        if optim.startswith('gd'):
            # Gradient descent
            if optim == 'gdh':
                # Hilbert gradient
                # == preconditioning with the regularizer
                grad = spatial.greens_apply(grad, kernel)
        else:
            # Levenberg-Marquardt regularisation
            hess[..., :dim] += hess[..., :dim].abs().max(-1, True).values * 1e-5
            # Gauss-Newton update
            grad = spatial.solve_grid_sym(hess, grad, optim=optim,
                                          max_iter=sub_iter, **prm)
        if lr != 1:
            grad.mul_(lr)

        if not ls:
            vel -= grad
            continue

        # line search
        armijo = 1
        vel0 = vel
        ll0 = ll
        ok = False
        for iter_ls in range(ls):
            vel = vel0 - armijo * grad
            grid = spatial.shoot(vel, kernel, **prm)
            warped = spatial.grid_pull(moving, grid, bound='dct2', extrapolate=True)
            ll = loss.loss(warped, fixed)
            ll += (vel * spatial.regulariser_grid(vel, **prm)).sum()
            ll /= fixed.numel()
            if ll > ll0:
                armijo = armijo / 2
            else:
                ok = True
                break
        if not ok:
            print('\nFailed to improve')
            vel = vel0
            return vel

    print('')
    return vel
