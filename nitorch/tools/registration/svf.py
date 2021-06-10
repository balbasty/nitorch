from nitorch import spatial
from nitorch.core import py, utils, linalg, math
import torch
from .losses import mse, cat
from .utils import jg, jhj, defaults_velocity, defaults_template
from .phantoms import demo_atlas, demo_register
from . import plot as plt


def register(fixed=None, moving=None, dim=None, lam=1., max_iter=20,
             loss='mse', optim='relax', sub_iter=16, plot=False, steps=8,
             **prm):
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
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent optimisation
    loss : {'mse', 'cat'}, default='mse'
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
        Stationary velocity field.

    """
    defaults_velocity(prm)
    prm['factor'] = lam

    # If no inputs provided: demo "circle to square"
    if fixed is None or moving is None:
        fixed, moving = demo_register(cat=(loss == 'cat'))

    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    shape = fixed.shape[-dim:]
    velshape = [*fixed.shape[:-dim-1], *shape, dim]
    vel = torch.zeros(velshape, **utils.backend(fixed))

    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)
    if optim == 'gdh':
        # Greens kernel for Hilbert gradient
        kernel = spatial.greens(shape, **prm)

    iscat = loss == 'cat'
    loss = (mse if loss == 'mse' else
            cat if loss == 'cat' else
            loss)

    ll_prev = None
    for n_iter in range(1, max_iter+1):

        # forward
        grid, jac = spatial.exp_forward(vel, steps=steps, jacobian=True)

        # compute spatial gradients in warped space
        mugrad = spatial.grid_grad(moving, grid, bound='dct2', extrapolate=True)

        # TODO: add that back? see JA's email
        # jac = torch.matmul(jac, spatial.grid_jacobian(vel, type='disp').inverse())

        # rotate gradients (we want `D(mu o phi)`, not `D(mu) o phi`)
        jac = jac.transpose(-1, -2)
        mugrad = linalg.matvec(jac, mugrad)
        del jac

        # gradient/Hessian of the log-likelihood in observed space
        warped = spatial.grid_pull(moving, grid, bound='dct2', extrapolate=True)
        if plot:
            plt.mov2fix(fixed, moving, warped, vel, cat=iscat, dim=dim)
        ll, grad, hess = loss(warped, fixed, dim=dim)
        del warped

        # compose with spatial gradients
        grad = jg(mugrad, grad)
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
        vel -= grad

    print('')
    return vel


def loadf(x):
    return x.fdata() if hasattr(x, 'fdata') else x


def savef(x, parent):
    if hasattr(parent, 'fdata'):
        parent[...] = x
    else:
        parent.copy_(x)


def init_template(images, loss='mse', optim='relax', velocities=None,
                  lam=1, **prm):
    """Initialise the template

    Parameters
    ----------
    images : iterable of (K, *spatial) tensor
    loss : {'mse', 'cat'}, default='mse'
        'mse': Mean-squared error
        'cat': Categorical cross-entropy
    optim : {'relax', 'cg', 'gd', 'gdh'}, default='relax'
        'relax': Gauss-Newton (linear system solved by relaxation)
        'cg': Gauss-Newton (linear system solved by conjugate gradient)
        'gd': Gradient descent
        'gdh': Hilbert gradient descent
    velocities : iterable or (*spatial, D) tensor
        Pre-computed velocities
    lam : float, default=1
        Modulate regularisation
    absolute : float, default=1e-4
        Penalty on absolute displacements
    membrane : float, default=0.08
        Penalty on first derivatives
    bending : float, default=0.8
        Penalty on second derivatives

    Returns
    -------
    template : (K, *shape) tensor

    """
    defaults_template(prm)
    prm['factor'] = lam

    if loss == 'mse':
        loss = mse
    elif loss == 'cat':
        loss = cat
    else:
        raise NotImplementedError(loss)

    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)

    template = torch.zeros([])
    if velocities is None:
        velocities = [None] * len(images)

    grad = 0
    hess = 0
    for image, vel in zip(images, velocities):
        dim = image.dim() - 1
        if vel is not None:
            vel = spatial.exp(vel)
            image = spatial.grid_pull(image, vel, bound='dct2')
        template = template.expand(image.shape).to(**utils.backend(images))
        _, g, h = loss(image, template, dim=dim)
        if vel is not None:
            g = spatial.grid_push(g, vel, bound='dct2')
            h = spatial.grid_push(h, vel, bound='dct2')
        grad += g
        hess += h

    # Levenberg-Marquardt regularisation
    hess[..., :dim] += hess[..., :dim].abs().max(-1, True).values * 1e-5
    # Gauss-Newton update
    template = spatial.solve_field_sym(hess, grad, optim=optim, max_iter=32,
                                       **prm)
    template.mul_(-lr)
    return template


def update_template(template, grad, hess=None, optim='relax', kernel=None,
                    lam=1, **prm):
    """Initialise the template

    Parameters
    ----------
    template : (K, *spatial) tensor
    grad : (K, *spatial) tensor
    hess : (K*(K+1)//2, *spatial) tensor
    optim : {'relax', 'cg', 'gd', 'gdh'}, default='relax'
        'relax': Gauss-Newton (linear system solved by relaxation)
        'cg': Gauss-Newton (linear system solved by conjugate gradient)
        'gd': Gradient descent
        'gdh': Hilbert gradient descent
    kernel : tensor, optional
        Pre-computed Greens kernel for Hilbert gradient descent
    lam : float, default=1
        Modulate regularisation
    absolute : float, default=1e-4
        Penalty on absolute displacements
    membrane : float, default=0.08
        Penalty on first derivatives
    bending : float, default=0.8
        Penalty on second derivatives

    Returns
    -------
    ll : float
        Negative log-likelihood
    template : (K, *shape) tensor
        Updated template

    """
    dim = template.dim() - 1
    shape = template.shape[1:]

    defaults_template(prm)
    prm['factor'] = lam
    if kernel is None and optim == 'gdh':
        # Greens kernel for Hilbert gradient
        kernel = spatial.greens(shape, **prm)

    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)

    # add regularisation
    rgrad = spatial.regulariser(template, **prm)
    ll = (template * rgrad).sum()
    ll = ll.item()
    grad += rgrad

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
        grad = spatial.solve_field_sym(hess, grad, optim=optim,
                                       max_iter=16, **prm)
    if lr != 1:
        grad.mul_(lr)
    template -= grad

    return ll, template


def center_velocities(velocities):
    """Subtract mean across velocities"""
    # compute mean
    meanvel = 0
    for n in range(len(velocities)):
        meanvel += loadf(velocities[n])
    meanvel /= len(velocities)

    # subtract mean
    for n in range(len(velocities)):
        vel = loadf(velocities[n])
        vel -= meanvel
        savef(vel, velocities[n])

    return velocities


def update_velocity(fixed=None, moving=None, lam=1., max_iter=20,
                    loss='mse', optim='relax', sub_iter=16, plot=False,
                    kernel=None, velocity=None, **prm):
    """Diffeomorphic registration between two images using SVFs.

    Parameters
    ----------
    fixed : (K, *spatial) tensor
        Fixed image
    moving : (K, *spatial) tensor
        Moving image
    lam : float, default=1
        Modulate regularisation
    max_iter : int, default=100
        Maximum number of Gauss-Newton or Gradient descent optimisation
    loss : {'mse', 'cat'}, default='mse'
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
    ll : float
        Log-likelihood
    gmoving : (K, *spatial) tensor
        Gradient with respect to moving image
    hmoving : (K*(K+1)//2, *spatial) tensor
        Hessian with respect to moving image
    velocity : (*spatial, dim) tensor
        Stationary velocity field.

    """
    defaults_velocity(prm)
    prm['factor'] = lam

    # initialise velocity
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = fixed.dim() - 1
    shape = fixed.shape[1:]
    if velocity is None:
        velshape = [*shape, dim]
        velocity = torch.zeros(velshape, **utils.backend(fixed))

    # Initialise optimiser
    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)
    if kernel is None and optim == 'gdh':
        # Greens kernel for Hilbert gradient
        kernel = spatial.greens(shape, **prm)

    print('|', end ='', flush=True)
    for n_iter in range(1, max_iter+1):
        print('.', end='', flush=True)

        # forward
        grid = spatial.exp_forward(velocity)

        # compute spatial gradients in warped space
        mugrad = spatial.grid_grad(moving, grid, bound='dct2', extrapolate=True)

        # TODO: add that back? see JA's email
        # jac = spatial.grid_jacobian(vel, type='disp').inverse()
        # jac = torch.matmul(spatial.grid_jacobian(grid), jac)

        # rotate gradients
        jac = spatial.grid_jacobian(grid)
        jac = jac.transpose(-1, -2)
        mugrad = linalg.matvec(jac, mugrad)
        del jac

        # gradient/Hessian of the log=likelihood in observed space
        warped = spatial.grid_pull(moving, grid, bound='dct2', extrapolate=True)
        if plot:
            plt.mov2fix(fixed, moving, warped, velocity, cat=(loss == 'cat'), dim=dim)
        if loss == 'mse':
            ll, lossgrad, losshess = mse(fixed, warped, dim=dim)
        elif loss == 'cat':
            ll, lossgrad, losshess = cat(fixed, warped, dim=dim, acceleration=0)
        else:
            raise NotImplementedError(loss)
        del warped

        # compose with spatial gradients
        grad = jg(mugrad, lossgrad)
        hess = jhj(mugrad, losshess)
        del mugrad
        vgrad = spatial.regulariser_grid(velocity, **prm)

        # print objective
        ll += lam * (velocity*vgrad).sum()
        ll = ll.item()

        # propagate gradients backward
        grad, hess = spatial.exp_backward(velocity, grad, hess)
        grad += lam * vgrad

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
        velocity -= grad

    lossgrad = spatial.grid_push(lossgrad, grid, bound='dct2')
    losshess = spatial.grid_push(losshess, grid, bound='dct2')
    return ll, lossgrad, losshess, velocity


def atlas(images=None, max_outer=10, max_inner=8, sub_iter=16,
          loss='mse', optim='relax', plot=False,
          prm_template=None, prm_velocity=None, dtype=None, device=None):
    """Diffeomorphic registration between two images using SVFs.

    Parameters
    ----------
    images : iterator of (K, *spatial) tensor
        Batch of images
    max_outer : int, default=10
        Maximum number of outer (template update) iterations
    max_inner : int, default=8
        Maximum number of inner (registration) iterations
    sub_iter : int, default=8
        Maximum number of relax/cg iterations per GN step
    loss : {'mse', 'cat'}, default='mse'
        'mse': Mean-squared error
        'cat': Categorical cross-entropy
    optim : {'relax', 'cg', 'gd', 'gdh'}, default='relax'
        'relax': Gauss-Newton (linear system solved by relaxation)
        'cg': Gauss-Newton (linear system solved by conjugate gradient)
        'gd': Gradient descent
        'gdh': Hilbert (preconditioned) gradient descent
    prm_template : dict
        lam : float, default=1
            Modulate regularisation
        absolute : float, default=1e-4
            Penalty on absolute displacements
        membrane : float, default=0.08
            Penalty on first derivatives
        bending : float, default=0.8
            Penalty on second derivatives
    prm_velocity : dict
        lam : float, default=1
            Modulate regularisation
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
    template : (K, *spatial) tensor
        Template
    vel : (N, *spatial, dim) tensor or MappedArray
        Stationary velocity fields.

    """
    prm_velocity = prm_velocity or dict()
    defaults_velocity(prm_velocity)
    prm_template = prm_template or dict()
    defaults_template(prm_template)

    import matplotlib.pyplot as plt

    def get_shape(xx):
        N = len(xx)
        for x in xx:
            shape = x.shape
            break
        return (N, *shape)

    def get_backend(xx):
        for x in xx:
            backend = utils.backend(x)
            break
        return backend

    # If no inputs provided: demo "squircles"
    if images is None:
        images = demo_atlas(batch=8, cat=(loss == 'cat'))

    backend = get_backend(images)
    backend = dict(dtype=dtype or backend['dtype'],
                   device=device or backend['device'])
    N, K, *shape = get_shape(images)
    dim = len(shape)

    # allocate velocities (option to save on disk?)
    velshape = [N, *shape, dim]
    vels = torch.zeros(velshape, **backend)

    lr = None
    if isinstance(optim, (tuple, list)):
        optim, lr = optim
    lr = lr or (1e-3 if optim.startswith('gd') else 1)
    # Greens kernel for Hilbert gradient
    kernelv = spatial.greens(shape, **prm_velocity) if optim == 'gdh' else None
    kernelt = spatial.greens(shape, **prm_template) if optim == 'gdh' else None

    template = init_template(images, loss=loss, optim=(optim, lr), **prm_template)
    plt.imshow(template[0])
    plt.show()

    ll_prev = None
    for n_iter in range(max_outer):
        # register
        grad = 0
        hess = 0
        llv = 0
        for n, image in enumerate(images):
            vel = loadf(vels[n])
            l, g, h, vel = update_velocity(
                image, template, velocity=vel, loss=loss, optim=(optim, lr),
                max_iter=max_inner, sub_iter=sub_iter, kernel=kernelv,
                **prm_velocity)
            savef(vel, vels[n])
            llv += l
            grad += g
            hess += h
        print('')

        # center velocities
        vels = center_velocities(vels)

        # update template
        llt, template = update_template(
            template, grad, hess,
            optim=(optim, lr), kernel=kernelt, **prm_template)

        plt.imshow(template[0])
        plt.show()

        llv /= images.numel()
        llt /= images.numel()
        ll = llt + llv
        if ll_prev is None:
            ll_prev = ll
            ll_max = ll
            print(f'{n_iter:03d} | {llv:12.6g} + {llt:12.6g} = {ll:12.6g}')
        else:
            gain = (ll_prev - ll) / max(abs(ll_max - ll), 1e-8)
            ll_prev = ll
            print(f'{n_iter:03d} | {llv:12.6g} + {llt:12.6g} = {ll:12.6g} | gain = {gain:12.6g}')

    return template, vel

