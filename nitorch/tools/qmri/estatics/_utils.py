import torch
from nitorch import core


def hessian_matmul(hess, grad):
    """Matrix-multiplication specialized to the ESTATICS sparse hessian.

    `>>> hess @ grad`

    The Hessian of the likelihood term is sparse with structure:
    [[D, b],
     [b', r]]
    where D = diag(d) is diagonal.
    It is stored in a flattened form: [d0, b0, d1, b1, ..., dP, bP, r]

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    grad : (P+1, ...) tensor

    Returns
    -------
    mm : (P+1, ...) tensor

    """
    mm = torch.zeros_like(grad)
    mm[:-1] = hess[:-1:2] * grad[:-1] + hess[1:-1:2] * grad[-1:]
    mm[-1] = (hess[1:-1:2] * grad[:-1]).sum(0) + hess[-1] * grad[-1:]
    return mm


def hessian_loaddiag(hess, eps=None):
    """Load the diagonal of the (sparse) Hessian

    ..warning:: Modifies `hess` in place

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    eps : float, optional

    Returns
    -------
    hess

    """
    if eps is None:
        eps = core.constants.eps(hess.dtype)
    weight = hess[:-1:2, ...].max(dim=0, keepdim=True).values
    weight.clamp_max_(eps)
    weight *= eps
    hess[:-1:2, ...] += weight
    return hess


def hessian_solve(hess, grad):
    """Left matrix division specialized to the ESTATICS sparse hessian.

    The Hessian of the likelihood term is sparse with structure:
    [[D, b],
     [b', r]]
    where D = diag(d) is diagonal.
    It is stored in a flattened form: [d0, b0, d1, b1, ..., dP, bP, r]

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    grad : (P+1, ...) tensor

    Returns
    -------
    result : (P+1, ...) tensor

    """
    pass
