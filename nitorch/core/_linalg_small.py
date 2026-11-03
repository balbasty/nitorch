"""Closed-form linear algebra for fields of small (D <= 3) general matrices.

These are the general (non-symmetric) counterparts of `_linalg_sym`: the
matrices are stored in full, as `(..., D, D)` tensors, and are typically
one per voxel (Jacobians, Greens kernels, ...). For millions of tiny
matrices, closed forms made of elementwise operations are much faster
than batched LU / matmul. Larger matrices fall back to `torch`.

The D = 2 and D = 3 kernels are TorchScript compatible and fused when
possible (`jit_policy.maybe_jit`). The decorated kernels only call the
undecorated `_*_impl` functions (no nesting); the public functions are
plain Python dispatchers.
"""
import torch
from torch import Tensor
from .jit_policy import maybe_jit

__all__ = ['small_det', 'small_adj', 'small_inv', 'small_solve',
           'small_matvec']


# ----------------------------------------------------------------------
#   kernels (TorchScript compatible; not decorated)
# ----------------------------------------------------------------------

def _det2_impl(a: Tensor) -> Tensor:
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]


def _det3_impl(a: Tensor) -> Tensor:
    return (a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1]) +
            a[..., 0, 1] * (a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2]) +
            a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]))


def _adj2_impl(a: Tensor) -> Tensor:
    adj = torch.empty_like(a)
    adj[..., 0, 0] = a[..., 1, 1]
    adj[..., 1, 1] = a[..., 0, 0]
    adj[..., 0, 1] = -a[..., 0, 1]
    adj[..., 1, 0] = -a[..., 1, 0]
    return adj


def _adj3_impl(a: Tensor) -> Tensor:
    adj = torch.empty_like(a)
    adj[..., 0, 0] = a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1]
    adj[..., 1, 1] = a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0]
    adj[..., 2, 2] = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    adj[..., 0, 1] = a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2]
    adj[..., 0, 2] = a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1]
    adj[..., 1, 0] = a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2]
    adj[..., 1, 2] = a[..., 1, 0] * a[..., 0, 2] - a[..., 1, 2] * a[..., 0, 0]
    adj[..., 2, 0] = a[..., 2, 1] * a[..., 1, 0] - a[..., 2, 0] * a[..., 1, 1]
    adj[..., 2, 1] = a[..., 2, 0] * a[..., 0, 1] - a[..., 2, 1] * a[..., 0, 0]
    return adj


def _inv2_impl(a: Tensor) -> Tensor:
    return _adj2_impl(a) / _det2_impl(a)[..., None, None]


def _inv3_impl(a: Tensor) -> Tensor:
    return _adj3_impl(a) / _det3_impl(a)[..., None, None]


def _solve2_impl(a: Tensor, v: Tensor) -> Tensor:
    det = _det2_impl(a)
    x0 = a[..., 1, 1] * v[..., 0] - a[..., 0, 1] * v[..., 1]
    x1 = a[..., 0, 0] * v[..., 1] - a[..., 1, 0] * v[..., 0]
    return torch.stack([x0, x1], -1) / det[..., None]


def _solve3_impl(a: Tensor, v: Tensor) -> Tensor:
    # Cramer's rule: x_i = sum_j cof[j, i] v_j / det, with the cofactors
    # computed on the fly (the inverse is never formed)
    det = _det3_impl(a)
    out = []
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        x = (a[..., 1, i1] * a[..., 2, i2] - a[..., 1, i2] * a[..., 2, i1]) * v[..., 0]
        x = x + (a[..., 2, i1] * a[..., 0, i2] - a[..., 2, i2] * a[..., 0, i1]) * v[..., 1]
        x = x + (a[..., 0, i1] * a[..., 1, i2] - a[..., 0, i2] * a[..., 1, i1]) * v[..., 2]
        out.append(x / det)
    return torch.stack(out, -1)


def _matvec2_impl(m: Tensor, v: Tensor) -> Tensor:
    return torch.stack([m[..., 0, 0] * v[..., 0] + m[..., 0, 1] * v[..., 1],
                        m[..., 1, 0] * v[..., 0] + m[..., 1, 1] * v[..., 1]], -1)


def _matvec3_impl(m: Tensor, v: Tensor) -> Tensor:
    out = []
    for i in range(3):
        out.append(m[..., i, 0] * v[..., 0] + m[..., i, 1] * v[..., 1] +
                   m[..., i, 2] * v[..., 2])
    return torch.stack(out, -1)


_det = {2: maybe_jit(_det2_impl), 3: maybe_jit(_det3_impl)}
_adj = {2: maybe_jit(_adj2_impl), 3: maybe_jit(_adj3_impl)}
_inv = {2: maybe_jit(_inv2_impl), 3: maybe_jit(_inv3_impl)}
_solve = {2: maybe_jit(_solve2_impl), 3: maybe_jit(_solve3_impl)}
_matvec = {2: maybe_jit(_matvec2_impl), 3: maybe_jit(_matvec3_impl)}


# ----------------------------------------------------------------------
#   public API
# ----------------------------------------------------------------------

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
    if dim == 1:
        return mat[..., 0, 0].clone()
    if dim in _det:
        return _det[dim](mat)
    return torch.det(mat)


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
    if dim == 1:
        return torch.ones_like(mat)
    if dim in _adj:
        return _adj[dim](mat)
    return torch.det(mat)[..., None, None] * torch.inverse(mat)


def small_inv(mat):
    """Inverse of a field of small matrices (adjugate / determinant).

    Parameters
    ----------
    mat : (..., D, D) tensor

    Returns
    -------
    inv : (..., D, D) tensor
    """
    dim = mat.shape[-1]
    if dim == 1:
        return mat.reciprocal()
    if dim in _inv:
        return _inv[dim](mat)
    return torch.inverse(mat)


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
    if dim == 1:
        return vec / mat[..., 0, 0][..., None]
    if dim in _solve:
        return _solve[dim](mat, vec)
    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
        return torch.linalg.solve(mat, vec)
    return torch.inverse(mat).matmul(vec[..., None])[..., 0]


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
    if dim == 1:
        return mat[..., 0, 0][..., None] * vec
    if dim in _matvec:
        return _matvec[dim](mat, vec)
    return torch.matmul(mat, vec[..., None])[..., 0]
