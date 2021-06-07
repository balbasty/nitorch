from nitorch.core import py, utils, linalg
from nitorch._C.grid import GridGrad
import torch


def jg(jac, grad, dim=None):
    """Jacobian-gradient product: J*g

    Parameters
    ----------
    jac : (..., K, *spatial, D)
    grad : (..., K, *spatial)

    Returns
    -------
    new_grad : (..., *spatial, D)

    """
    dim = dim or (grad.dim() - 1)
    grad = utils.movedim(grad, -dim-1, -1)
    jac = utils.movedim(jac, -dim-2, -1)
    grad = linalg.matvec(jac, grad)
    return grad


def jhj(jac, hess, dim=None):
    """Jacobian-Hessian product: J*H*J', where H is symmetric and stored sparse

    The Hessian can be symmetric (K*(K+1)//2), diagonal (K) or
    a scaled identity (1).

    Parameters
    ----------
    jac : (..., K, *spatial, D)
    hess : (..., 1|K|K*(K+1)//2, *spatial)

    Returns
    -------
    new_hess : (..., *spatial, D*(D+1)//2)

    """

    dim = dim or (hess.dim() - 1)
    hess = utils.movedim(hess, -dim-1, -1)
    jac = utils.movedim(jac, -dim-2, -1)

    K = jac.shape[-1]
    D = jac.shape[-2]
    D2 = D*(D+1)//2

    if hess.shape[-1] == 1:
        hess = hess.expand([*hess.shape[:-1], K])
    is_diag = hess.shape[-1] == K
    out = hess.new_zeros([*jac.shape[:-2], D2])

    for d in range(D):
        # diagonal of output
        for k in range(K):
            out[..., d] += hess[..., k] * jac[..., d, k].square()
            if not is_diag:
                for i, l in enumerate(range(k+1, K)):
                    out[..., d] += 2 * hess[..., i] * jac[..., d, k] * jac[..., d, l]
        # off diagonal of output
        for j, e in enumerate(range(d+1, D)):
            for k in range(K):
                out[..., j] += hess[..., k] * jac[..., d, k] * jac[..., e, k]
                if not is_diag:
                    for i, l in enumerate(range(k+1, K)):
                        out[..., e] += 2 * hess[..., i] * jac[..., d, k] * jac[..., e, l]

    return out
