import torch
from nitorch import core, spatial
import math


def hessian_sym_matmul(hess, grad):
    """Matrix-multiplication for small batches of symmetric matrices.

    `>>> hess @ grad`

    `hess` contains only the diagonal and upper part of the matrix, in
    a flattened array. Elements are ordered as:
     `[(i, i) for i in range(P)] +
      [(i, j) for i in range(P) for j in range(i+1, P)]

    Parameters
    ----------
    hess : (P*(P+1)//2, ...) tensor
    grad : (P, ...) tensor

    Returns
    -------
    mm : (P, ...) tensor

    """
    nb_prm = len(grad)
    if nb_prm == 1:
        return hess * grad
    elif nb_prm == 2:
        mm = torch.empty_like(grad)
        mm[0] = hess[0] * grad[0] + hess[2] * grad[1]
        mm[1] = hess[1] * grad[1] + hess[2] * grad[0]
        return mm
    elif nb_prm == 3:
        mm = torch.empty_like(grad)
        mm[0] = hess[0] * grad[0] + hess[3] * grad[1] + hess[4] * grad[2]
        mm[1] = hess[1] * grad[1] + hess[3] * grad[0] + hess[5] * grad[2]
        mm[2] = hess[2] * grad[2] + hess[4] * grad[0] + hess[5] * grad[1]
        return mm
    elif nb_prm == 4:
        mm = torch.empty_like(grad)
        mm[0] = hess[0] * grad[0]
        mm[0] += hess[4] * grad[1]
        mm[0] += hess[5] * grad[2]
        mm[0] += hess[6] * grad[3]
        mm[1] = hess[1] * grad[1]
        mm[1] += hess[4] * grad[0]
        mm[1] += hess[7] * grad[2]
        mm[1] += hess[8] * grad[3]
        mm[2] = hess[2] * grad[2]
        mm[2] += hess[5] * grad[0]
        mm[2] += hess[7] * grad[1]
        mm[2] += hess[9] * grad[3]
        mm[3] = hess[3] * grad[3]
        mm[3] += hess[6] * grad[0]
        mm[3] += hess[8] * grad[1]
        mm[3] += hess[9] * grad[2]
        return mm
    else:
        mm = torch.empty_like(grad)
        for i in range(nb_prm):
            mm[i] = hess[i] * grad[i]
        c = nb_prm
        for i in range(nb_prm):
            for j in range(i+1, nb_prm):
                mm[i] += hess[c] * grad[j]
                mm[j] += hess[c] * grad[i]
                c += 1
        return mm


def hessian_sym_loaddiag(hess, eps=None, eps2=None):
    """Load the diagonal of the (symmetric) Hessian

    `hess` contains only the diagonal and upper part of the matrix, in
    a flattened array. Elements are ordered as:
     `[(i, i) for i in range(P)] +
      [(i, j) for i in range(P) for j in range(i+1, P)]
    ..warning:: Modifies `hess` in place

    Parameters
    ----------
    hess : (P*(P+1)//2, ...) tensor
    eps : float, optional

    Returns
    -------
    hess

    """
    if eps is None:
        eps = core.constants.eps(hess.dtype)
    nb_prm = int((math.sqrt(1 + 8 * len(hess)) - 1)//2)
    weight = hess[:nb_prm].max(dim=0, keepdim=True).values
    weight.clamp_min_(eps)
    weight *= eps
    if eps2:
        weight += eps2
    hess[:nb_prm] += weight
    return hess


def hessian_sym_inv(hess, diag=False):
    """Matrix inversion for sparse symmetric hessians.

    `hess` contains only the diagonal and upper part of the matrix, in
    a flattened array. Elements are ordered as:
     `[(i, i) for i in range(P)] +
      [(i, j) for i in range(P) for j in range(i+1, P)]

    Orders up to 4 are implemented in closed-form.
    Orders > 4 use torch's batched implementation but require
    building the full matrices.

    Parameters
    ----------
    hess : (P*(P+1)//2, ...) tensor
        Sparse symmetric matrix
    diag : bool, default=False
        If True, only return the diagonal of the inverse

    Returns
    -------
    result : (P*(P+1)//2, ...) tensor

    """
    nb_prm = int((math.sqrt(1 + 8 * len(hess)) - 1)//2)
    if diag:
        out = hess.new_empty([nb_prm, *hess.shape[1:]])
    else:
        out = torch.empty_like(hess)

    cnt = nb_prm
    for i in range(nb_prm):
        e = hess.new_zeros(nb_prm)
        e[i] = 1
        vec = hessian_sym_solve(hess, e)
        out[i] = vec[i]
        if not diag:
            for j in range(i+1, nb_prm):
                out[cnt] = vec[j]
                cnt += 1
    return out


def hessian_sym_solve(hess, grad, lam=None):
    """Left matrix division for sparse symmetric hessians.

    `>>> hess \ grad`

    `hess` contains only the diagonal and upper part of the matrix, in
    a flattened array. Elements are ordered as:
     `[(i, i) for i in range(P)] +
      [(i, j) for i in range(P) for j in range(i+1, P)]

    Orders up to 4 are implemented in closed-form.
    Orders > 4 use torch's batched implementation but require
    building the full matrices.

    Parameters
    ----------
    hess : (P*(P+1)//2, ...) tensor
    grad : (P, ...) tensor
    lam : float or (P,) sequence[float], optional
        Smoothing term added to the diagonal of H

    Returns
    -------
    result : (P, ...) tensor

    """

    backend = dict(dtype=hess.dtype, device=hess.device)
    nb_prm = len(grad)

    diag = hess[:nb_prm]  # diagonal
    uppr = hess[nb_prm:]  # upper triangular part

    if lam is not None:
        # add smoothing term
        lam = torch.as_tensor(lam, **backend).flatten()
        lam = torch.cat([lam, lam[-1].expand(nb_prm - len(lam))])
        lam = lam.reshape([len(lam)] + [1] * (hess.dim() - 1))
        diag = diag + lam[:-1]

    if nb_prm == 1:
        return grad / diag
    elif nb_prm == 2:
        det = uppr[0].square().neg_()
        det += diag[0] * diag[1]
        res = torch.empty_like(grad)
        res[0] = diag[1] * grad[0] - uppr[0] * grad[1]
        res[1] = diag[0] * grad[1] - uppr[0] * grad[0]
        res /= det
        return res
    elif nb_prm == 3:
        det = diag.prod(0) + 2 * uppr.prod(0) \
            - (diag[0] * uppr[2].square() +
               diag[2] * uppr[0].square() +
               diag[1] * uppr[1].square())
        res = torch.empty_like(grad)
        res[0] = (diag[1] * diag[2] - uppr[2].square()) * grad[0] \
               + (uppr[1] * uppr[2] - diag[2] * uppr[0]) * grad[1] \
               + (uppr[0] * uppr[2] - diag[1] * uppr[1]) * grad[2]
        res[1] = (uppr[1] * uppr[2] - diag[2] * uppr[0]) * grad[0] \
               + (diag[0] * diag[2] - uppr[1].square()) * grad[1] \
               + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * grad[2]
        res[2] = (uppr[0] * uppr[2] - diag[1] * uppr[1]) * grad[0] \
               + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * grad[1] \
               + (diag[0] * diag[1] - uppr[0].square()) * grad[2]
        res /= det
        return res
    elif nb_prm == 4:
        det = diag.prod(0) \
             + ((uppr[0] * uppr[5]).square() +
                (uppr[1] * uppr[4]).square() +
                (uppr[2] * uppr[3]).square()) + \
             - 2 * (uppr[0] * uppr[1] * uppr[4] * uppr[5] +
                    uppr[0] * uppr[2] * uppr[3] * uppr[5] +
                    uppr[1] * uppr[2] * uppr[3] * uppr[4]) \
             + 2 * (diag[0] * uppr[3] * uppr[4] * uppr[5] +
                    diag[1] * uppr[1] * uppr[2] * uppr[5] +
                    diag[2] * uppr[0] * uppr[2] * uppr[4] +
                    diag[3] * uppr[0] * uppr[1] * uppr[3]) \
             - (diag[0] * diag[1] * uppr[5].square() +
                diag[0] * diag[2] * uppr[4].square() +
                diag[0] * diag[3] * uppr[3].square() +
                diag[1] * diag[2] * uppr[2].square() +
                diag[1] * diag[3] * uppr[1].square() +
                diag[2] * diag[3] * uppr[0].square())
        inv01 = (- diag[2] * diag[3] * uppr[0]
                 + diag[2] * uppr[2] * uppr[4]
                 + diag[3] * uppr[1] * uppr[3]
                 + uppr[0] * uppr[5].square()
                 - uppr[1] * uppr[4] * uppr[5]
                 - uppr[2] * uppr[3] * uppr[5])
        inv02 = (- diag[1] * diag[3] * uppr[1]
                 + diag[1] * uppr[2] * uppr[5]
                 + diag[3] * uppr[0] * uppr[3]
                 + uppr[1] * uppr[4].square()
                 - uppr[0] * uppr[4] * uppr[5]
                 - uppr[2] * uppr[3] * uppr[4])
        inv03 = (- diag[1] * diag[2] * uppr[2]
                 + diag[1] * uppr[1] * uppr[5]
                 + diag[2] * uppr[0] * uppr[4]
                 + uppr[2] * uppr[3].square()
                 - uppr[0] * uppr[3] * uppr[5]
                 - uppr[1] * uppr[3] * uppr[4])
        inv12 = (- diag[0] * diag[3] * uppr[3]
                 + diag[0] * uppr[4] * uppr[5]
                 + diag[3] * uppr[0] * uppr[1]
                 + uppr[3] * uppr[2].square()
                 - uppr[0] * uppr[2] * uppr[5]
                 - uppr[1] * uppr[2] * uppr[4])
        inv13 = (- diag[0] * diag[2] * uppr[4]
                 + diag[0] * uppr[3] * uppr[5]
                 + diag[2] * uppr[0] * uppr[2]
                 + uppr[4] * uppr[1].square()
                 - uppr[0] * uppr[1] * uppr[5]
                 - uppr[1] * uppr[2] * uppr[3])
        inv23 = (- diag[0] * diag[1] * uppr[5]
                 + diag[0] * uppr[4] * uppr[3]
                 + diag[1] * uppr[1] * uppr[2]
                 + uppr[5] * uppr[0].square()
                 - uppr[0] * uppr[1] * uppr[4]
                 - uppr[0] * uppr[2] * uppr[3])
        res = torch.empty_like(grad)
        res[0] = (diag[1] * diag[2] * diag[3]
                  - diag[1] * uppr[5].square()
                  - diag[2] * uppr[4].square()
                  - diag[3] * uppr[3].square()
                  + 2 * uppr[3] * uppr[4] * uppr[5]) * grad[0]
        res[0] += inv01 * grad[1]
        res[0] += inv02 * grad[2]
        res[0] += inv03 * grad[3]
        res[1] = (diag[0] * diag[2] * diag[3]
                  - diag[0] * uppr[5].square()
                  - diag[2] * uppr[2].square()
                  - diag[3] * uppr[1].square()
                  + 2 * uppr[1] * uppr[2] * uppr[5]) * grad[1]
        res[1] += inv01 * grad[0]
        res[1] += inv12 * grad[2]
        res[1] += inv13 * grad[3]
        res[2] = (diag[0] * diag[1] * diag[3]
                  - diag[0] * uppr[4].square()
                  - diag[1] * uppr[2].square()
                  - diag[3] * uppr[0].square()
                  + 2 * uppr[0] * uppr[2] * uppr[4]) * grad[2]
        res[2] += inv02 * grad[0]
        res[2] += inv12 * grad[1]
        res[2] += inv23 * grad[3]
        res[3] = (diag[0] * diag[1] * diag[2]
                  - diag[0] * uppr[3].square()
                  - diag[1] * uppr[1].square()
                  - diag[2] * uppr[0].square()
                  + 2 * uppr[0] * uppr[1] * uppr[3]) * grad[3]
        res[3] += inv03 * grad[0]
        res[3] += inv13 * grad[1]
        res[3] += inv23 * grad[2]
        res /= det
        return res
    else:
        raise NotImplemented


def smart_grid(aff, shape, inshape=None):
    """Generate a sampling grid iff it is not the identity.

    Parameters
    ----------
    aff : (D+1, D+1) tensor
        Affine transformation matrix (voxels to voxels)
    shape : (D,) tuple[int]
        Output shape
    inshape : (D,) tuple[int], optional
        Input shape

    Returns
    -------
    grid : (*shape, D) tensor or None
        Sampling grid

    """
    backend = dict(dtype=aff.dtype, device=aff.device)
    identity = torch.eye(aff.shape[-1], **backend)
    inshape = inshape or shape
    if torch.allclose(aff, identity) and shape == inshape:
        return None
    return spatial.affine_grid(aff, shape)


def smart_pull(tensor, grid):
    """Pull iff grid is defined (+ add/remove batch dim).

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*output_shape, D) tensor or None
        Sampling grid

    Returns
    -------
    pulled : (channels, *output_shape) tensor
        Sampled volume

    """
    if grid is None:
        return tensor
    return spatial.grid_pull(tensor[None, ...], grid[None, ...])[0]


def smart_push(tensor, grid, shape=None):
    """Pull iff grid is defined (+ add/remove batch dim).

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*input_shape, D) tensor or None
        Sampling grid
    shape : (D,) tuple[int], default=input_shape
        Output shape

    Returns
    -------
    pushed : (channels, *output_shape) tensor
        Sampled volume

    """
    if grid is None:
        return tensor
    return spatial.grid_push(tensor[None, ...], grid[None, ...], shape)[0]


def reg(tensor, vx=1., rls=None, lam=1., do_grad=True):
    """Compute the gradient of the regularisation term.

    The regularisation term has the form:
    `0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
    where `i` indexes a voxel, `lam` is the regularisation factor,
    `w[i]` is the RLS weight, `g+` and `g-` are the forward and
    backward spatial gradients of the parameter map.

    Parameters
    ----------
    tensor : (K, *shape) tensor
        Parameter map
    vx : float or sequence[float], default=1
        Voxel size
    rls : (K|1, *shape) tensor, optional
        Weights from the reweighted least squares scheme
    lam : float or sequence[float], default=1
        Regularisation factor
    do_grad : bool, default=True
        Return both the criterion and gradient

    Returns
    -------
    reg : () tensor[double]
        Regularisation term
    grad : (K, *shape) tensor
        Gradient with respect to the parameter map

    """
    nb_prm = tensor.shape[0]
    backend = dict(dtype=tensor.dtype, device=tensor.device)
    vx = core.utils.make_vector(vx, 3, **backend)
    lam = core.utils.make_vector(lam, nb_prm, **backend)
    
    grad_fwd = spatial.diff(tensor, dim=[1, 2, 3], voxel_size=vx, side='f')
    grad_bwd = spatial.diff(tensor, dim=[1, 2, 3], voxel_size=vx, side='b')
    if rls is not None:
        grad_fwd *= rls[..., None]
        grad_bwd *= rls[..., None]
    grad_fwd = spatial.div(grad_fwd, dim=[1, 2, 3], voxel_size=vx, side='f')
    grad_bwd = spatial.div(grad_bwd, dim=[1, 2, 3], voxel_size=vx, side='b')

    grad = grad_fwd
    grad += grad_bwd
    grad *= lam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 2.
    # ^ average across side

    if do_grad:
        reg = (tensor * grad).sum(dtype=torch.double)
        return 0.5 * reg, grad
    else:
        grad *= tensor
        return 0.5 * grad.sum(dtype=torch.double)
    

def reg1(tensor, vx=1., rls=None, lam=1., do_grad=True):
    """Compute the gradient of the regularisation term.

    The regularisation term has the form:
    `0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
    where `i` indexes a voxel, `lam` is the regularisation factor,
    `w[i]` is the RLS weight, `g+` and `g-` are the forward and
    backward spatial gradients of the parameter map.

    Parameters
    ----------
    tensor : (*shape) tensor
        Parameter map
    vx : float or sequence[float], default=1
        Voxel size
    rls : (*shape) tensor, optional
        Weights from the reweighted least squares scheme
    lam : float, default=1
        Regularisation factor
    do_grad : bool, default=True
        Return both the criterion and gradient

    Returns
    -------
    reg : () tensor[double]
        Regularisation term
    grad : (*shape) tensor
        Gradient with respect to the parameter map

    """

    grad_fwd = spatial.diff(tensor, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.diff(tensor, dim=[0, 1, 2], voxel_size=vx, side='b')
    if rls is not None:
        grad_fwd *= rls[..., None]
        grad_bwd *= rls[..., None]
    grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], voxel_size=vx, side='b')

    grad = grad_fwd
    grad += grad_bwd
    grad *= lam / 2.   # average across directions (3) and side (2)

    if do_grad:
        reg = (tensor * grad).sum(dtype=torch.double)
        return 0.5 * reg, grad
    else:
        grad *= tensor
        return 0.5 * grad.sum(dtype=torch.double)

def rls_maj(rls, vx=1., lam=1.):
    """Diagonal majoriser of the RLS regulariser.
    
    Parameters
    ----------
    rls : (..., *shape) tensor
        Weights from the reweighted least squares scheme
    vx : float or sequence[float], default=1
        Voxel size
    lam : float or sequence[float], default=1
        Regularisation factor

    Returns
    -------
    rls : (*shape) tensor
        Convolved weights

    """
    
    nb_prm = rls.shape[0] if rls.dim() > 3 else 1
    backend = dict(dtype=rls.dtype, device=rls.device)
    vx = core.utils.make_vector(vx, 3, **backend)
    lam = core.utils.make_vector(lam, nb_prm, **backend)
    vx = vx.square().reciprocal()
    
    if rls.dim() > 3:
        rls = core.utils.movedim(rls, 0, -1)
    
    out = (2*vx.sum())*rls
    # center
    out[1:-1, 1:-1, 1:-1] += ((rls[ :-2, 1:-1, 1:-1] + rls[2:  , 1:-1, 1:-1])*vx[0] + 
                              (rls[1:-1,  :-2, 1:-1] + rls[1:-1, 2:,   1:-1])*vx[1] + 
                              (rls[1:-1, 1:-1,  :-2] + rls[1:-1, 1:-1, 2:  ])*vx[2])
    # sides
    out[0, 1:-1, 1:-1]  += ((rls[   0, 1:-1, 1:-1] + rls[   1, 1:-1, 1:-1])*vx[0] + 
                            (rls[   0,  :-2, 1:-1] + rls[   0, 2:  , 1:-1])*vx[1] + 
                            (rls[   0, 1:-1,  :-2] + rls[   0, 1:-1, 2:  ])*vx[2])
    out[-1, 1:-1, 1:-1] += ((rls[  -2, 1:-1, 1:-1] + rls[  -1, 1:-1, 1:-1])*vx[0] + 
                            (rls[  -1,  :-2, 1:-1] + rls[  -1, 2:  , 1:-1])*vx[1] + 
                            (rls[  -1, 1:-1,  :-2] + rls[  -1, 1:-1, 2:  ])*vx[2])
    out[1:-1, 0, 1:-1]  += ((rls[ :-2,    0, 1:-1] + rls[2:  ,    0, 1:-1])*vx[0] + 
                            (rls[1:-1,    0, 1:-1] + rls[1:-1,    1, 1:-1])*vx[1] + 
                            (rls[1:-1,    0,  :-2] + rls[1:-1,    0, 2:  ])*vx[2])
    out[1:-1, -1, 1:-1] += ((rls[ :-2,   -1, 1:-1] + rls[2:  ,   -1, 1:-1])*vx[0] + 
                            (rls[1:-1,   -2, 1:-1] + rls[1:-1,   -1, 1:-1])*vx[1] + 
                            (rls[1:-1,   -1,  :-2] + rls[1:-1,   -1, 2:  ])*vx[2])
    out[1:-1, 1:-1, 0]  += ((rls[ :-2, 1:-1,    0] + rls[2:  , 1:-1,    0])*vx[0] + 
                            (rls[1:-1,  :-2,    0] + rls[1:-1, 2:  ,    0])*vx[1] + 
                            (rls[1:-1, 1:-1,    0] + rls[1:-1, 1:-1,    1])*vx[2])
    out[1:-1, 1:-1, -1] += ((rls[ :-2, 1:-1,   -1] + rls[2:  , 1:-1,   -1])*vx[0] + 
                            (rls[1:-1,  :-2,   -1] + rls[1:-1, 2:  ,   -1])*vx[1] + 
                            (rls[1:-1, 1:-1,   -2] + rls[1:-1, 1:-1,   -1])*vx[2])
    # edges
    out[0, 0, 1:-1]   += ((rls[   0,    0, 1:-1] + rls[   1,    0, 1:-1])*vx[0] + 
                          (rls[   0,    0, 1:-1] + rls[   0,    1, 1:-1])*vx[1] + 
                          (rls[   0,    0,  :-2] + rls[   0,    0, 2:  ])*vx[2])
    out[0, -1, 1:-1]  += ((rls[   0,   -1, 1:-1] + rls[   1,   -1, 1:-1])*vx[0] + 
                          (rls[   0,   -2, 1:-1] + rls[   0,   -1, 1:-1])*vx[1] + 
                          (rls[   0,   -1,  :-2] + rls[   0,   -1, 2:  ])*vx[2])
    out[-1, 0, 1:-1]  += ((rls[  -1,    0, 1:-1] + rls[  -1,    0, 1:-1])*vx[0] + 
                          (rls[  -1,    0, 1:-1] + rls[  -1,    1, 1:-1])*vx[1] + 
                          (rls[  -1,    0,  :-2] + rls[  -1,    0, 2:  ])*vx[2])
    out[-1, -1, 1:-1] += ((rls[  -2,   -1, 1:-1] + rls[  -1,   -1, 1:-1])*vx[0] + 
                          (rls[  -1,   -2, 1:-1] + rls[  -1,   -1, 1:-1])*vx[1] + 
                          (rls[  -1,   -1,  :-2] + rls[  -1,   -1, 2:  ])*vx[2])
    out[0, 1:-1, 0]   += ((rls[   0, 1:-1,    0] + rls[   1, 1:-1,    0])*vx[0] + 
                          (rls[   0,  :-2,    0] + rls[   0, 2:  ,    0])*vx[1] + 
                          (rls[   0, 1:-1,    0] + rls[   0, 1:-1,    1])*vx[2])
    out[0, 1:-1, -1]  += ((rls[   0, 1:-1,   -1] + rls[   1, 1:-1,   -1])*vx[0] + 
                          (rls[   0,  :-2,   -1] + rls[   0, 2: ,    -1])*vx[1] + 
                          (rls[   0, 1:-1,   -2] + rls[   0, 1:-1,   -1])*vx[2])
    out[-1, 1:-1, 0]  += ((rls[  -2, 1:-1,    0] + rls[  -1, 1:-1,    0])*vx[0] + 
                          (rls[  -1,  :-2,    0] + rls[  -1, 2:  ,    0])*vx[1] + 
                          (rls[  -1, 1:-1,    0] + rls[  -1, 1:-1,   -1])*vx[2])
    out[-1, 1:-1, -1] += ((rls[  -2, 1:-1,   -1] + rls[  -1, 1:-1,   -1])*vx[0] + 
                          (rls[  -1,  :-2,   -1] + rls[  -1, 2:  ,   -1])*vx[1] + 
                          (rls[  -1, 1:-1,   -2] + rls[  -1, 1:-1,   -1])*vx[2])
    out[1:-1, 0, 0]   += ((rls[ :-2,    0,    0] + rls[2:  ,    0,    0])*vx[0] + 
                          (rls[1:-1,    0,    0] + rls[1:-1,    1,    0])*vx[1] + 
                          (rls[1:-1,    0,    0] + rls[1:-1,    0,    1])*vx[2])
    out[1:-1, 0, -1]  += ((rls[ :-2,    0,   -1] + rls[2:  ,    0,   -1])*vx[0] + 
                          (rls[1:-1,    0,   -1] + rls[1:-1,    1,   -1])*vx[1] + 
                          (rls[1:-1,    0,   -2] + rls[1:-1,    0,   -1])*vx[2])
    out[1:-1, -1, 0]  += ((rls[ :-2,    0,    0] + rls[2:  ,   -1,    0])*vx[0] + 
                          (rls[1:-1,   -2,    0] + rls[1:-1,   -1,    0])*vx[1] + 
                          (rls[1:-1,    0,    0] + rls[1:-1,   -1,    1])*vx[2])
    out[1:-1, -1, -1] += ((rls[ :-2,   -1,   -1] + rls[2:  ,   -1,   -1])*vx[0] + 
                          (rls[1:-1,   -2,   -1] + rls[1:-1,   -1,   -1])*vx[1] + 
                          (rls[1:-1,   -1,   -2] + rls[1:-1,   -1,   -1])*vx[2])
    
    # corners
    out[0, 0, 0]    += ((rls[ 0,  0,  0] + rls[ 1,  0,  0])*vx[0] + 
                        (rls[ 0,  0,  0] + rls[ 0,  1,  0])*vx[1] + 
                        (rls[ 0,  0,  0] + rls[ 0,  0,  1])*vx[2])
    out[0, 0, -1]   += ((rls[ 0,  0, -1] + rls[ 1,  0, -1])*vx[0] + 
                        (rls[ 0,  0, -1] + rls[ 0,  1, -1])*vx[1] + 
                        (rls[ 0,  0, -2] + rls[ 0,  0, -1])*vx[2])
    out[0, -1, 0]   += ((rls[ 0, -1,  0] + rls[ 1, -1,  0])*vx[0] + 
                        (rls[ 0, -2,  0] + rls[ 0, -1,  0])*vx[1] + 
                        (rls[ 0, -1,  0] + rls[ 0, -1,  1])*vx[2])
    out[0, -1, -1]  += ((rls[ 0, -1, -1] + rls[ 1, -1, -1])*vx[0] + 
                        (rls[ 0, -2, -1] + rls[ 0, -1, -1])*vx[1] + 
                        (rls[ 0, -1, -2] + rls[ 0, -1, -1])*vx[2])
    out[-1, 0, 0]   += ((rls[-2,  0,  0] + rls[-1,  0,  0])*vx[0] + 
                        (rls[-1,  0,  0] + rls[-1,  1,  0])*vx[1] + 
                        (rls[-1,  0,  0] + rls[-1,  0,  1])*vx[2])
    out[-1, 0, -1]  += ((rls[-2,  0, -1] + rls[-1,  0, -1])*vx[0] + 
                        (rls[-1,  0, -1] + rls[-1,  1, -1])*vx[1] + 
                        (rls[-1,  0, -2] + rls[-1,  0, -1])*vx[2])
    out[-1, -1, 0]  += ((rls[-2, -1,  0] + rls[-1, -1,  0])*vx[0] + 
                        (rls[-1, -2,  0] + rls[-1, -1,  0])*vx[1] + 
                        (rls[-1, -1,  0] + rls[-1, -1,  1])*vx[2])
    out[-1, -1, -1] += ((rls[-2, -1, -1] + rls[-1, -1, -1])*vx[0] + 
                        (rls[-1, -2, -1] + rls[-1, -1, -1])*vx[1] + 
                        (rls[-1, -1, -2] + rls[-1, -1, -1])*vx[2])
    
    out *= lam
    if out.dim() > 3:
        out = core.utils.movedim(-1, 0)
    return out
