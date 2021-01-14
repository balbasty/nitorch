import torch
from nitorch import core, spatial
from ._utils import first_to_last, last_to_first, complexmean


def unwrap_phase_gn(obs):
    """

    Parameters
    ----------
    obs : (*shape) tensor[complex] or (2, *shape) tensor[float]

    Returns
    -------

    """

    obs = torch.as_tensor(obs)
    if not obs.is_complex:
        obs = first_to_last(obs)
        obs = torch.view_as_complex(obs)

    logfit = torch.empty_like(obs)
    logfit[...] = complexmean(obs)

    lam_mag = 1e7
    lam_ang = 1e13

    for n_iter in range(16):

        lam_ang /= 10
        lam_mag /= 10
        lam = [lam_mag, lam_ang]

        # data term
        fit = logfit.exp()
        grad = fit - obs
        llx = -0.5 * grad.abs().square_().sum(dtype=torch.double)
        grad *= fit.conj()
        hess = fit.abs().square_()
        grad = last_to_first(torch.view_as_real(grad))
        # hess = torch.stack((hess + grad[0], hess - grad[0], -grad[1]))

        # regularisation term
        logfit = last_to_first(torch.view_as_real(logfit))
        lly = 0
        lly1, g1 = _nonlin_reg(logfit[0], lam=lam[0])
        lly += lly1
        grad[0] += g1
        lly1, g1 = _nonlin_reg(logfit[1], lam=lam[1])
        lly += lly1
        grad[1] += g1
        del g1

        # gauss-newton
        delta = _nonlin_solve(hess, grad, lam=lam)
        logfit -= delta

        logfit = torch.view_as_complex(first_to_last((logfit)))

        print('{:3d} | {:12.6g} + {:12.6g} = {:12.6g}'
              .format(n_iter, llx, lly, llx+lly))



def _nonlin_reg(map, rls=None, lam=1., do_grad=True):
    """Compute the gradient of the regularisation term.

    The regularisation term has the form:
    `-0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
    where `i` indexes a voxel, `lam` is the regularisation factor,
    `w[i]` is the RLS weight, `g+` and `g-` are the forward and
    backward spatial gradients of the parameter map.

    Parameters
    ----------
    map : (*shape) ParameterMap
        Parameter map
    rls : (*shape) tensor, optional
        Weights from the reweighted least squares scheme
    lam : float, default=1
        Regularisation factor
    do_grad : bool, default=True
        Return both the criterion and gradient

    Returns
    -------
    reg : () tensor
        Regularisation term
    grad : (*shape) tensor
        Gradient with respect to the parameter map

    """

    grad_fwd = spatial.diff(map, dim=[0, 1, 2], side='f')
    grad_bwd = spatial.diff(map, dim=[0, 1, 2], side='b')
    if rls is not None:
        grad_fwd *= rls[..., None]
        grad_bwd *= rls[..., None]
    grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], side='f')
    grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], side='b')

    grad = grad_fwd
    grad += grad_bwd
    grad *= lam / (2*3)   # average across directions (3) and side (2)

    if do_grad:
        reg = (map * grad).sum(dtype=torch.double)
        return -0.5 * reg, grad
    else:
        grad *= map
        return -0.5 * grad.sum(dtype=torch.double)


def _nonlin_solve(hess, grad, lam):
    """Solve the regularized linear system

    Parameters
    ----------
    hess : (2*P+1, *shape) tensor
    grad : (P+1, *shape) tensor
    rls : (P+1, *shape) tensor_like
    lam : (P,) sequence[float]
    opt : Options

    Returns
    -------
    delta : (P+1, *shape) tensor

    """

    def hess_fn(x):
        # result_real = hess[0] * x[0] + hess[-1] * x[1]
        # result_imag = hess[1] * x[1] + hess[-1] * x[0]
        result_real = hess * x[0]
        result_imag = hess * x[1]
        result_real += _nonlin_reg(x[0], lam=lam[0], do_grad=True)[1]
        result_imag += _nonlin_reg(x[1], lam=lam[1], do_grad=True)[1]
        return torch.stack((result_real, result_imag), dim=0)

    result = core.optim.cg(hess_fn, grad,
                           verbose=False, stop='norm',
                           max_iter=100, tolerance=1e-5)
    return result
