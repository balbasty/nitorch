import torch
from nitorch import core, spatial


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


def hessian_loaddiag_(hess, eps=None, eps2=None):
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
    weight = hess[::2].max(dim=0, keepdim=True).values
    weight.clamp_min_(eps)
    weight *= eps
    if eps2:
        weight += eps2
    hess[::2] += weight
    return hess


def hessian_solve(hess, grad, lam=None):
    """Left matrix division specialized to the ESTATICS sparse hessian.

    The Hessian of the likelihood term is sparse with structure:
    `[[D, b], [b.T, r]]` where `D = diag(d)` is diagonal.
    It is stored in a flattened form: `[d0, b0, d1, b1, ..., dP, bP, r]`

    Because of this specific structure, the Hessian is inverted in
    closed-form using the formula for the inverse of a 2x2 block matrix.
    See: https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    grad : (P+1, ...) tensor
    lam : float or (P+1,) sequence[float], optional
        Smoothing term added to the diagonal of H

    Returns
    -------
    result : (P+1, ...) tensor

    """

    backend = dict(dtype=hess.dtype, device=hess.device)
    nb_prm = len(grad)
    
    
    # H = [[diag, vec], [vec.T, scal]]
    diag = hess[:-1:2]
    vec = hess[1:-1:2]
    scal = hess[-1]

    if lam is not None:
        # add smoothing term
        lam = torch.as_tensor(lam, **backend).flatten()
        lam = torch.cat([lam, lam[-1].expand(nb_prm-len(lam))])
        lam = lam.reshape([len(lam)] + [1] * (hess.dim()-1))
        diag = diag + lam[:-1]
        scal = scal + lam[1]
                                              
    # precompute stuff
    vec_norm = vec/diag
    mini_inv = scal - (vec*vec_norm).sum(dim=0)
    result = torch.empty_like(grad)

    # top left corner
    result[:-1] = ((vec_norm * grad[:-1]).sum(dim=0) / mini_inv) * vec_norm
    result[:-1] += grad[:-1]/diag

    # top right corner:
    result[:-1] -= vec_norm * grad[-1] / mini_inv

    # bottom left corner:
    result[-1] = - (vec_norm * grad[:-1]).sum(dim=0) / mini_inv

    # bottom right corner:
    result[-1] += grad[-1] / mini_inv

    return result
