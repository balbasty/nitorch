__all__ = ['sym_to_full', 'sym_det', 'sym_inv', 'sym_outer',
           'sym_solve', 'sym_diag', 'sym_matvec']


import torch
from . import utils
from typing import List
import math


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
    _solve_lu = torch.linalg.solve
else:
    _solve_lu = lambda A, b: torch.solve(b, A)[0]


def sym_to_full(mat):
    """Transform a symmetric matrix into a full matrix

    Notes
    -----
    Backpropagation works at least for torch >= 1.6
    It should be checked on earlier versions.

    Parameters
    ----------
    mat : (..., M * (M+1) // 2) tensor
        A symmetric matrix that is stored in a sparse way.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., [a00, a11, aa22, a01, a02, a12]

    Returns
    -------
    full : (..., M, M) tensor
        Full matrix

    """

    mat = torch.as_tensor(mat)
    mat = utils.movedim(mat, -1, 0)
    nb_prm = int((math.sqrt(1 + 8 * len(mat)) - 1)//2)
    if not mat.requires_grad:
        full = mat.new_empty([nb_prm, nb_prm, *mat.shape[1:]])
        i = 0
        for i in range(nb_prm):
            full[i, i] = mat[i]
        count = i + 1
        for i in range(nb_prm):
            for j in range(i+1, nb_prm):
                full[i, j] = full[j, i] = mat[count]
                count += 1
    else:
        full = [[None] * nb_prm for _ in range(nb_prm)]
        i = 0
        for i in range(nb_prm):
            full[i][i] = mat[i]
        count = i + 1
        for i in range(nb_prm):
            for j in range(i+1, nb_prm):
                full[i][j] = full[j][i] = mat[count]
                count += 1
        full = utils.as_tensor(full)
    return utils.movedim(full, [0, 1], [-2, -1])


@torch.jit.script
def _sym_matvec2(mat, vec):
    mm = mat[:2] * vec
    mm[0].addcmul_(mat[2], vec[1])
    mm[1].addcmul_(mat[2], vec[0])
    return mm


@torch.jit.script
def _sym_matvec3(mat, vec):
    mm = mat[:3] * vec
    mm[0].addcmul_(mat[3], vec[1]).addcmul_(mat[4], vec[2])
    mm[1].addcmul_(mat[3], vec[0]).addcmul_(mat[5], vec[2])
    mm[2].addcmul_(mat[4], vec[0]).addcmul_(mat[5], vec[1])
    return mm


@torch.jit.script
def _sym_matvec4(mat, vec):
    mm = mat[:4] * vec
    mm[0].addcmul_(mat[4], vec[1]) \
        .addcmul_(mat[5], vec[2]) \
        .addcmul_(mat[6], vec[3])
    mm[1].addcmul_(mat[4], vec[0]) \
        .addcmul_(mat[7], vec[2]) \
        .addcmul_(mat[8], vec[3])
    mm[2].addcmul_(mat[5], vec[0]) \
        .addcmul_(mat[7], vec[1]) \
        .addcmul_(mat[9], vec[3])
    mm[3].addcmul_(mat[6], vec[0]) \
        .addcmul_(mat[8], vec[1]) \
        .addcmul_(mat[9], vec[2])
    return mm


@torch.jit.script
def _sym_matvecn(mat, vec, nb_prm: int):
    mm = mat[:nb_prm] * vec
    c = nb_prm
    for i in range(nb_prm):
        for j in range(i+1, nb_prm):
            mm[i].addcmul_(mat[c], vec[j])
            mm[j].addcmul_(mat[c], vec[i])
            c += 1
    return mm


def sym_matvec(mat, vec):
    """Matrix-vector product with a symmetric matrix

    Parameters
    ----------
    mat : (..., M * (M+1) // 2) tensor
        A symmetric matrix that is stored in a sparse way.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., [a00, a11, aa22, a01, a02, a12]
    vec : (..., M) tensor
        A vector

    Returns
    -------
    matvec : (..., M) tensor
        The matrix-vector product

    """

    mat, vec = utils.to_max_backend(mat, vec)

    nb_prm = vec.shape[-1]
    if nb_prm == 1:
        return mat * vec

    # make the vector dimension first so that the code is less ugly
    mat = utils.fast_movedim(mat, -1, 0)
    vec = utils.fast_movedim(vec, -1, 0)

    if nb_prm == 2:
        mm = _sym_matvec2(mat, vec)
    elif nb_prm == 3:
        mm = _sym_matvec3(mat, vec)
    elif nb_prm == 4:
        mm = _sym_matvec4(mat, vec)
    else:
        mm = _sym_matvecn(mat, vec, nb_prm)

    return utils.fast_movedim(mm, 0, -1)


def sym_diag(mat):
    """Diagonal of a symmetric matrix

    Parameters
    ----------
    mat : (..., M * (M+1) // 2) tensor
        A symmetric matrix that is stored in a sparse way.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., [a00, a11, aa22, a01, a02, a12]

    Returns
    -------
    diag : (..., M) tensor
        Main diagonal of the matrix

    """
    mat = torch.as_tensor(mat)
    nb_prm = int((math.sqrt(1 + 8 * mat.shape[-1]) - 1)//2)
    return mat[..., :nb_prm]


@torch.jit.script
def _square(x):
    return x * x


@torch.jit.script
def _square_(x):
    x *= x
    return x


@torch.jit.script
def _sym_det2(diag, uppr):
    det = _square(uppr[0]).neg_()
    det.addcmul_(diag[0], diag[1])
    return det


@torch.jit.script
def _sym_solve2(diag, uppr, vec, shape: List[int]):
    det = _sym_det2(diag, uppr)
    res = vec.new_empty(shape)
    res[0] = diag[1] * vec[0] - uppr[0] * vec[1]
    res[1] = diag[0] * vec[1] - uppr[0] * vec[0]
    res /= det
    return res


@torch.jit.script
def _sym_det3(diag, uppr):
    det = diag.prod(0) + 2 * uppr.prod(0) \
        - (diag[0] * _square(uppr[2]) +
           diag[2] * _square(uppr[0]) +
           diag[1] * _square(uppr[1]))
    return det


@torch.jit.script
def _sym_solve3(diag, uppr, vec, shape: List[int]):
    det = _sym_det3(diag, uppr)
    res = vec.new_empty(shape)
    res[0] = (diag[1] * diag[2] - _square(uppr[2])) * vec[0] \
           + (uppr[1] * uppr[2] - diag[2] * uppr[0]) * vec[1] \
           + (uppr[0] * uppr[2] - diag[1] * uppr[1]) * vec[2]
    res[1] = (uppr[1] * uppr[2] - diag[2] * uppr[0]) * vec[0] \
           + (diag[0] * diag[2] - _square(uppr[1])) * vec[1] \
           + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * vec[2]
    res[2] = (uppr[0] * uppr[2] - diag[1] * uppr[1]) * vec[0] \
           + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * vec[1] \
           + (diag[0] * diag[1] - _square(uppr[0])) * vec[2]
    res /= det
    return res


@torch.jit.script
def _sym_det4(diag, uppr):
    det = diag.prod(0) \
         + (_square(uppr[0] * uppr[5]) +
            _square(uppr[1] * uppr[4]) +
            _square(uppr[2] * uppr[3])) + \
         - 2 * (uppr[0] * uppr[1] * uppr[4] * uppr[5] +
                uppr[0] * uppr[2] * uppr[3] * uppr[5] +
                uppr[1] * uppr[2] * uppr[3] * uppr[4]) \
         + 2 * (diag[0] * uppr[3] * uppr[4] * uppr[5] +
                diag[1] * uppr[1] * uppr[2] * uppr[5] +
                diag[2] * uppr[0] * uppr[2] * uppr[4] +
                diag[3] * uppr[0] * uppr[1] * uppr[3]) \
         - (diag[0] * diag[1] * _square(uppr[5]) +
            diag[0] * diag[2] * _square(uppr[4]) +
            diag[0] * diag[3] * _square(uppr[3]) +
            diag[1] * diag[2] * _square(uppr[2]) +
            diag[1] * diag[3] * _square(uppr[1]) +
            diag[2] * diag[3] * _square(uppr[0]))
    return det


@torch.jit.script
def _sym_solve4(diag, uppr, vec, shape: List[int]):
    det = _sym_det4(diag, uppr)
    inv01 = (- diag[2] * diag[3] * uppr[0]
             + diag[2] * uppr[2] * uppr[4]
             + diag[3] * uppr[1] * uppr[3]
             + uppr[0] * _square(uppr[5])
             - uppr[1] * uppr[4] * uppr[5]
             - uppr[2] * uppr[3] * uppr[5])
    inv02 = (- diag[1] * diag[3] * uppr[1]
             + diag[1] * uppr[2] * uppr[5]
             + diag[3] * uppr[0] * uppr[3]
             + uppr[1] * _square(uppr[4])
             - uppr[0] * uppr[4] * uppr[5]
             - uppr[2] * uppr[3] * uppr[4])
    inv03 = (- diag[1] * diag[2] * uppr[2]
             + diag[1] * uppr[1] * uppr[5]
             + diag[2] * uppr[0] * uppr[4]
             + uppr[2] * _square(uppr[3])
             - uppr[0] * uppr[3] * uppr[5]
             - uppr[1] * uppr[3] * uppr[4])
    inv12 = (- diag[0] * diag[3] * uppr[3]
             + diag[0] * uppr[4] * uppr[5]
             + diag[3] * uppr[0] * uppr[1]
             + uppr[3] * _square(uppr[2])
             - uppr[0] * uppr[2] * uppr[5]
             - uppr[1] * uppr[2] * uppr[4])
    inv13 = (- diag[0] * diag[2] * uppr[4]
             + diag[0] * uppr[3] * uppr[5]
             + diag[2] * uppr[0] * uppr[2]
             + uppr[4] * _square(uppr[1])
             - uppr[0] * uppr[1] * uppr[5]
             - uppr[1] * uppr[2] * uppr[3])
    inv23 = (- diag[0] * diag[1] * uppr[5]
             + diag[0] * uppr[4] * uppr[3]
             + diag[1] * uppr[1] * uppr[2]
             + uppr[5] * _square(uppr[0])
             - uppr[0] * uppr[1] * uppr[4]
             - uppr[0] * uppr[2] * uppr[3])
    res = vec.new_empty(shape)
    res[0] = (diag[1] * diag[2] * diag[3]
              - diag[1] * _square(uppr[5])
              - diag[2] * _square(uppr[4])
              - diag[3] * _square(uppr[3])
              + 2 * uppr[3] * uppr[4] * uppr[5]) * vec[0]
    res[0] += inv01 * vec[1]
    res[0] += inv02 * vec[2]
    res[0] += inv03 * vec[3]
    res[1] = (diag[0] * diag[2] * diag[3]
              - diag[0] * _square(uppr[5])
              - diag[2] * _square(uppr[2])
              - diag[3] * _square(uppr[1])
              + 2 * uppr[1] * uppr[2] * uppr[5]) * vec[1]
    res[1] += inv01 * vec[0]
    res[1] += inv12 * vec[2]
    res[1] += inv13 * vec[3]
    res[2] = (diag[0] * diag[1] * diag[3]
              - diag[0] * _square(uppr[4])
              - diag[1] * _square(uppr[2])
              - diag[3] * _square(uppr[0])
              + 2 * uppr[0] * uppr[2] * uppr[4]) * vec[2]
    res[2] += inv02 * vec[0]
    res[2] += inv12 * vec[1]
    res[2] += inv23 * vec[3]
    res[3] = (diag[0] * diag[1] * diag[2]
              - diag[0] * _square(uppr[3])
              - diag[1] * _square(uppr[1])
              - diag[2] * _square(uppr[0])
              + 2 * uppr[0] * uppr[1] * uppr[3]) * vec[3]
    res[3] += inv03 * vec[0]
    res[3] += inv13 * vec[1]
    res[3] += inv23 * vec[2]
    res /= det
    return res


def sym_solve(mat, vec, eps=None):
    """Left matrix division for sparse symmetric matrices.

    `>>> mat \ vec`

    Warning
    -------
    .. Currently, autograd does not work through this function.
    .. The order of arguments is the inverse of torch.solve

    Notes
    -----
    .. Orders up to 4 are implemented in closed-form.
    .. Orders > 4 use torch's batched implementation but require
       building the full matrices.
    .. Backpropagation works at least for torch >= 1.6
       It should be checked on earlier versions.

    Parameters
    ----------
    mat : (..., M*(M+1)//2) tensor
        A symmetric matrix that is stored in a sparse way.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., [a00, a11, aa22, a01, a02, a12]
    vec : (..., M) tensor
        A vector
    eps : float or (M,) sequence[float], optional
        Smoothing term added to the diagonal of `mat`

    Returns
    -------
    result : (..., M) tensor
    """

    # make the vector dimension first so that the code is less ugly
    mat, vec = utils.to_max_backend(mat, vec)
    backend = dict(dtype=mat.dtype, device=mat.device)
    mat = utils.fast_movedim(mat, -1, 0)
    vec = utils.fast_movedim(vec, -1, 0)
    nb_prm = len(vec)

    shape = utils.expanded_shape(mat.shape[1:], vec.shape[1:])
    shape = [vec.shape[0], *shape]

    diag = mat[:nb_prm]  # diagonal
    uppr = mat[nb_prm:]  # upper triangular part

    if eps is not None:
        # add smoothing term
        eps = torch.as_tensor(eps, **backend).flatten()
        eps = torch.cat([eps, eps[-1].expand(nb_prm - len(eps))])
        eps = eps.reshape([len(eps)] + [1] * (mat.dim() - 1))
        diag = diag + eps[:-1]

    if nb_prm == 1:
        res = vec / diag
    elif nb_prm == 2:
        res = _sym_solve2(diag, uppr, vec, shape)
    elif nb_prm == 3:
        res = _sym_solve3(diag, uppr, vec, shape)
    elif nb_prm == 4:
        res = _sym_solve4(diag, uppr, vec, shape)
    else:
        vec = utils.fast_movedim(vec, 0, -1)
        mat = utils.fast_movedim(mat, 0, -1)
        mat = sym_to_full(mat)
        return _solve_lu(mat, vec.unsqueeze(-1)).squeeze(-1)

    return utils.fast_movedim(res, 0, -1)


def sym_det(mat):
    """Determinant of a sparse symmetric matrix.

    Warning
    -------
    .. Currently, autograd does not work through this function.

    Notes
    -----
    .. Orders up to 4 are implemented in closed-form.
    .. Orders > 4 use torch's batched implementation but require
       building the full matrices.
    .. Backpropagation works at least for torch >= 1.6
       It should be checked on earlier versions.

    Parameters
    ----------
    mat : (..., M*(M+1)//2) tensor
        A symmetric matrix that is stored in a sparse way.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., [a00, a11, aa22, a01, a02, a12]

    Returns
    -------
    result : (..., M) tensor
    """

    # make the vector dimension first so that the code is less ugly
    mat = utils.fast_movedim(mat, -1, 0)
    nb_prm = int((math.sqrt(1 + 8 * mat.shape[-1]) - 1) // 2)

    diag = mat[:nb_prm]  # diagonal
    uppr = mat[nb_prm:]  # upper triangular part

    if nb_prm == 1:
        res = diag
    elif nb_prm == 2:
        res = _sym_det2(diag, uppr)
    elif nb_prm == 3:
        res = _sym_det3(diag, uppr)
    elif nb_prm == 4:
        res = _sym_det4(diag, uppr)
    else:
        mat = utils.fast_movedim(mat, 0, -1)
        mat = sym_to_full(mat)
        return torch.det(mat)

    return utils.fast_movedim(res, 0, -1)


def sym_inv(mat, diag=False):
    """Matrix inversion for sparse symmetric matrices.

    Notes
    -----
    .. Backpropagation works at least for torch >= 1.6
       It should be checked on earlier versions.

    Parameters
    ----------
    mat : (..., M*(M+1)//2) tensor
        A symmetric matrix that is stored in a sparse way.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., [a00, a11, aa22, a01, a02, a12]
    diag : bool, default=False
        If True, only return the diagonal of the inverse

    Returns
    -------
    imat : (..., M or M*(M+1)//2) tensor

    """
    mat = torch.as_tensor(mat)
    nb_prm = int((math.sqrt(1 + 8 * mat.shape[-1]) - 1) // 2)
    if diag:
        imat = mat.new_empty([*mat.shape[:-1], nb_prm])
    else:
        imat = torch.empty_like(mat)

    cnt = nb_prm
    for i in range(nb_prm):
        e = mat.new_zeros(nb_prm)
        e[i] = 1
        vec = sym_solve(mat, e)
        imat[..., i] = vec[..., i]
        if not diag:
            for j in range(i+1, nb_prm):
                imat[..., cnt] = vec[..., j]
                cnt += 1
    return imat


def sym_outer(x):
    """Compute the symmetric outer product of a vector: x @ x.T

    Parameters
    ----------
    x : (..., M)

    Returns
    -------
    xx : (..., M*(M+1)//2)

    """
    M = x.shape[-1]
    MM = M*(M+1)//2
    xx = x.new_empty([*x.shape[:-1], MM])
    if x.requires_grad:
        xx[..., :M] = x.square()
        index = M
        for m in range(M):
            for n in range(m+1, M):
                xx[..., index] = x[..., m] * x[..., n]
    else:
        torch.mul(x, x, out=xx[..., :M])
        index = M
        for m in range(M):
            for n in range(m+1, M):
                torch.mul(x[..., m], x[..., n], out=xx[..., index])
                index += 1
    return xx
