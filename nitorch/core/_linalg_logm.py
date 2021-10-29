# TODO:
#   we want a differentiable matrix logarithm.
#   I have started working on this, but it cannot be implemented in
#   pure python if we want it to work on batched matrices.
#   In the meantime, we should use scipy's implementation (which
#   does not accept batched matrices either) with a (parallel?) loop
import torch
from .optionals import custom_fwd, custom_bwd
from .optionals import numpy as np


def matrix_chain_rule(A, G, f):
    """Analytical chain rule for functions of square matrices.

    Parameters
    ----------
    A : (..., N, N) tensor or array
        Input matrix
    G : (..., N, N) tensor or array
        Gradient with respect to the output of the function
    f : callable
        Forward function

    Returns
    -------
    G : (..., M, N) tensor or array
        Gradient with respect to the input of the function

    References
    ----------
    .. [1] Roy Mathias, "A Chain Rule for Matrix Functions and Applications".
           SIAM Journal on Matrix Analysis and Applications, July 1996.
           https://dl.acm.org/doi/10.1137/S0895479895283409
    .. [2] github.com/Lezcano/expm

    """
    def transpose(mat):
        if torch.is_tensor(mat):
            return mat.transpose(-1, -2)
        else:
            perm = list(range(len(mat.shape)))
            perm[-1] = -2
            perm[-2] = -1
            return mat.transpose(*perm)

    def new_zeros(mat, shape):
        if torch.is_tensor(mat):
            return mat.new_zeros(shape)
        else:
            return np.zeros(shape, dtype=mat.dtype)

    A = transpose(A)
    n = A.shape[-1]
    shape_M = A.shape[:-2] + (2*n, 2*n)
    M = new_zeros(A, shape_M)
    M[..., :n, :n] = A
    M[..., n:, n:] = A
    M[..., :n, n:] = G
    return f(M)[..., :n, n:]


class _LogM(torch.autograd.Function):
    """Autograd implementation of th ematrix logarithm.

    This function does not work on batched matrices.

    """

    @staticmethod
    @custom_fwd
    def forward(ctx, mat):
        from scipy.linalg import logm
        logm_nowarn = lambda x: logm(x, disp=False)[0]
        if mat.requires_grad:
            ctx.save_for_backward(mat)
        device = mat.device
        input_complex = mat.is_complex()
        mat = mat.cpu().numpy()
        mat = logm_nowarn(mat)
        mat = torch.as_tensor(mat, device=device)
        if not input_complex and mat.is_complex():
            mat = mat.real
        return mat

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        from scipy.linalg import logm
        logm_nowarn = lambda x: logm(x, disp=False)[0]
        mat, = ctx.saved_tensors
        device = output_grad.device
        input_complex = output_grad.is_complex()
        mat = mat.cpu().numpy()
        output_grad = output_grad.cpu().numpy()
        grad = matrix_chain_rule(mat, output_grad, logm_nowarn)
        grad = torch.as_tensor(grad, device=device)
        if not input_complex and grad.is_complex():
            grad = grad.real
        return grad


def logm(mat):
    """Batched matrix logarithm.

    This implementation actually use scipy, so the data will be
    transferred to cpu and transferred back to device afterwards.

    Parameters
    ----------
    mat : (..., N, N) tensor
        Input matrix or batch of matrices

    Returns
    -------
    logmat : (..., N, N) tensor
        Input log-matrix or batch of log-matrices

    """
    mat = torch.as_tensor(mat)
    shape = mat.shape
    mat = mat.reshape([-1, mat.shape[-2], mat.shape[-1]])

    mats = []
    for M in mat:
        mats.append(_LogM.apply(M))
    mat = torch.stack(mats, dim=0)

    mat = mat.reshape(shape)
    return mat
