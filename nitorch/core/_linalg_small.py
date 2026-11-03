"""Closed-form linear algebra for fields of small (D <= 3) general matrices.

These are the general (non-symmetric) counterparts of `_linalg_sym`: the
matrices are stored in full, as `(..., D, D)` tensors, and are typically
one per voxel (Jacobians, Greens kernels, ...). For millions of tiny
matrices, closed forms made of elementwise operations are much faster
than batched LU / matmul. Larger matrices fall back to `torch`.
"""
import torch

__all__ = ['small_det', 'small_adj', 'small_inv', 'small_solve',
           'small_matvec']


def _entries(mat):
    dim = mat.shape[-1]
    return [[mat[..., i, j] for j in range(dim)] for i in range(dim)]


def small_det(mat):
    """Determinant of a field of small matrices.

    Parameters
    ----------
    mat : (..., D, D) tensor

    Returns
    -------
    det : (...) tensor
    """
    dim = mat.shape[-1]
    if dim > 3:
        return torch.det(mat)
    a = _entries(mat)
    if dim == 1:
        return a[0][0].clone()
    if dim == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) +
            a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) +
            a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def small_adj(mat):
    """Adjugate (transposed cofactor matrix) of a field of small matrices,
    such that `adj(A) @ A = det(A) * I`.

    Parameters
    ----------
    mat : (..., D, D) tensor

    Returns
    -------
    adj : (..., D, D) tensor
    """
    dim = mat.shape[-1]
    if dim > 3:
        return torch.det(mat)[..., None, None] * torch.inverse(mat)
    a = _entries(mat)
    adj = torch.empty_like(mat)
    if dim == 1:
        adj[..., 0, 0] = 1
    elif dim == 2:
        adj[..., 0, 0] = a[1][1]
        adj[..., 1, 1] = a[0][0]
        adj[..., 0, 1] = -a[0][1]
        adj[..., 1, 0] = -a[1][0]
    else:
        adj[..., 0, 0] = a[1][1] * a[2][2] - a[1][2] * a[2][1]
        adj[..., 1, 1] = a[0][0] * a[2][2] - a[0][2] * a[2][0]
        adj[..., 2, 2] = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        adj[..., 0, 1] = a[0][2] * a[2][1] - a[0][1] * a[2][2]
        adj[..., 0, 2] = a[0][1] * a[1][2] - a[0][2] * a[1][1]
        adj[..., 1, 0] = a[1][2] * a[2][0] - a[1][0] * a[2][2]
        adj[..., 1, 2] = a[1][0] * a[0][2] - a[1][2] * a[0][0]
        adj[..., 2, 0] = a[2][1] * a[1][0] - a[2][0] * a[1][1]
        adj[..., 2, 1] = a[2][0] * a[0][1] - a[2][1] * a[0][0]
    return adj


def small_inv(mat):
    """Inverse of a field of small matrices (adjugate / determinant).

    Parameters
    ----------
    mat : (..., D, D) tensor

    Returns
    -------
    inv : (..., D, D) tensor
    """
    if mat.shape[-1] > 3:
        return torch.inverse(mat)
    return small_adj(mat).div_(small_det(mat)[..., None, None])


def small_solve(mat, vec):
    """Solve `mat @ x = vec` for a field of small matrices.

    Uses Cramer's rule: each component of the solution is accumulated
    from cofactors computed on the fly, so the inverse (a D*D field) is
    never formed.

    Parameters
    ----------
    mat : (..., D, D) tensor
    vec : (..., D) tensor

    Returns
    -------
    x : (..., D) tensor
    """
    dim = vec.shape[-1]
    if dim > 3:
        return torch.linalg.solve(mat, vec) if hasattr(torch, 'linalg') \
            else torch.inverse(mat).matmul(vec[..., None])[..., 0]
    a = _entries(mat)
    if dim == 1:
        return vec / a[0][0][..., None]
    det = small_det(mat)
    if dim == 2:
        x0 = a[1][1] * vec[..., 0] - a[0][1] * vec[..., 1]
        x1 = a[0][0] * vec[..., 1] - a[1][0] * vec[..., 0]
        return torch.stack([x0, x1], -1).div_(det[..., None])

    def cof(i, j):
        # cofactor (i, j)
        return (a[(i+1) % 3][(j+1) % 3] * a[(i+2) % 3][(j+2) % 3] -
                a[(i+1) % 3][(j+2) % 3] * a[(i+2) % 3][(j+1) % 3])

    out = []
    for i in range(3):
        # x_i = sum_j adj[i, j] vec_j = sum_j cof[j, i] vec_j
        xi = cof(0, i).mul_(vec[..., 0])
        xi.addcmul_(cof(1, i), vec[..., 1])
        xi.addcmul_(cof(2, i), vec[..., 2])
        out.append(xi.div_(det))
    return torch.stack(out, -1)


def small_matvec(mat, vec):
    """Matrix-vector product for a field of small matrices.

    A few elementwise multiply-adds: batched matmul is slow for millions
    of tiny matrices, and broadcasting would allocate a D*D field.

    Parameters
    ----------
    mat : (..., D, D) tensor
    vec : (..., D) tensor

    Returns
    -------
    mv : (..., D) tensor
    """
    dim = vec.shape[-1]
    if dim > 3:
        return torch.matmul(mat, vec[..., None])[..., 0]
    rows = []
    for i in range(dim):
        row = mat[..., i, 0] * vec[..., 0]
        for j in range(1, dim):
            row = row.addcmul(mat[..., i, j], vec[..., j])
        rows.append(row)
    return torch.stack(rows, -1)
