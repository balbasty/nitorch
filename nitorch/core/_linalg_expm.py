"""Implementation of the matrix exponential and its derivative.

This implementation is based on John Ashburner's in SPM, which relies
on a Taylor approximation for both the exponential and its derivative.
Faster implementations that rely on scaling and squearing or Pade
approximations (as in scipy) could be used instead. This may be the
object of future work.

"""
import torch
import torch.nn.functional as F
from .optionals import custom_fwd, custom_bwd
from . import utils


def _expm(X, basis=None, grad_X=False, grad_basis=False, hess_X=False,
          max_order=10000, tol=1e-32):
    """Matrix exponential (and its derivatives).

    Notes
    -----
    .. This function evaluates the matrix exponential and its
       derivatives using a Taylor approximation. A faster integration
       technique, based  e.g. on scaling and squaring, could have been
       used instead.
    .. PyTorch/NumPy broadcasting rules apply.
       See: https://pytorch.org/docs/stable/notes/broadcasting.html
    .. The output shapes of `dX` and `dB` can be different from the
       shapes of `X` and `basis`, because of broadcasting.
       When computing the backward pass, you should take their dot
       product with the output gradient and then reduce across axes
       that have been expanded by broadcasting. E.g.:
       >>> def backward(grad, X, B):
       >>>     # X.shape     == (*batch_X, F)
       >>>     # B.shape     == (*batch_B, F, D, D)
       >>>     # grad.shape  == (*batch_XB, D, D)
       >>>     _, dX, dB = _dexpm(X, B, grad_X=True, grad_basis=True)
       >>>     # dX.shape    == (*batch_XB, F, D, D)
       >>>     # dB.shape    == (*batch_XB, F, D, D, D, D)
       >>>     dX = torch.sum(dX * grad[..., None, :, :], dim=[-1, -2])
       >>>     dX = utils.broadcast_backward(dX, X.shape)
       >>>     dB = torch.sum(dB * grad[..., None, None, None, :, :], dim=[-1, -2])
       >>>     dB = utils.broadcast_backward(dB, B.shape)
       >>>     return dX, dB

    Parameters
    ----------
    X : {(..., F), (..., D, D)} tensor_like
        If `basis` is None: log-matrix.
        Else:               parameters of the log-matrix in the basis set.
    basis : (..., F, D, D) tensor_like, default=None
        Basis set. If None, basis of all DxD matrices and F = D**2.
    grad_X : bool, default=False
        Compute derivatives with respect to `X`.
    grad_basis : bool, default=False
        Compute derivatives with respect to `basis`.
    max_order : int, default=10000
        Order of the Taylor expansion
    tol : float, default=1e-32
        Tolerance for early stopping
        The criterion is based on the Frobenius norm of the last term of
        the Taylor series.

    Returns
    -------
    eX : (..., D, D) tensor
        Matrix exponential
    dX : (..., F, D, D) tensor, if `grad_X is True`
        Derivative of the matrix exponential with respect to the
        parameters in the basis set
    dB : (..., F, D, D, D, D) tensor, if `grad_basis is True`
        Derivative of the matrix exponential with respect to the basis.
    hX : (..., F, F, D, D) tensor, if `hess_X is True`
        Second derivative of the matrix exponential with respect to the
        parameters in the basis set

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    def smart_incr(a, b):
        """Use inplace or outplace increment based on `a` and `b`s shapes."""
        if a.shape == b.shape:
            a += b
        else:
            a = a + b
        return a

    backend = utils.max_backend(X, basis)
    X = utils.as_tensor(X, **backend)

    if basis is not None:
        # X contains parameters in the Lie algebra -> reconstruct the matrix
        # X.shape = [.., F], basis.shape = [..., F, D, D]
        basis = utils.as_tensor(basis, **backend)
        param = X
        X = torch.sum(basis * X[..., None, None], dim=-3, keepdim=True)
        dim = basis.shape[-1]
    else:
        # X contains matrices in log-space -> build basis set
        # X.shape = [..., D, D]
        dim = X.shape[-1]
        param = X.reshape(X.shape[:-2] + (-1,))
        basis = torch.arange(dim ** 2, dtype=torch.long,
                             device=backend['device'])
        basis = F.one_hot(basis).to(**backend)
        basis = basis.reshape((dim**2, dim, dim))
        X = X[..., None, :, :]

    XB_batch_shape = X.shape[:-3]
    nb_feat = param.shape[-1]

    if grad_basis:
        # Build a basis for the basis
        basis_basis = torch.arange(dim ** 2, dtype=torch.long,
                                   device=backend['device'])
        basis_basis = F.one_hot(basis_basis).to(**backend)
        basis_basis = basis_basis.reshape((1, dim, dim, dim, dim))
        basis_basis = basis_basis * param[..., None, None, None, None]
        basis_basis = basis_basis.reshape(XB_batch_shape + (-1, dim, dim))

    # At this point:
    #   X.shape           = [*XB_batch_shape, 1, D, D]
    #   basis.shape       = [*B_batch_shape,  F, D, D]
    #   param.shape       = [*X_batch_shape,  F]
    #   basis_basis.shape = [*XB_batch_shape, F*D*D, D, D]

    # Aliases
    I = torch.eye(dim, **backend)
    E = I + X                            # expm(X)
    En = X.clone()                       # n-th Taylor coefficient of expm
    if grad_X or hess_X:
        dE = basis.clone()               # dexpm(X)/dx
        dEn = basis.clone()              # n-th Taylor coefficient of dexpm/dx
    if grad_basis:
        dB = basis_basis.clone()         # dexpm(X)/dB
        dBn = basis_basis.clone()        # n-th Taylor coefficient of dexpm/dB
    if hess_X:
        hE = E.new_zeros([*XB_batch_shape, nb_feat, nb_feat, dim, dim])
        hEn = hE.clone()

    for n_order in range(2, max_order+1):
        # Compute coefficients at order `n_order`, and accumulate
        if hess_X:
            dEB = torch.matmul(dEn[..., None, :, :, :], basis[..., None, :, :])
            hEn = torch.matmul(hEn, X[..., None, :, :]) + dEB + dEB.transpose(-3, -4)
            hEn /= n_order
            hE = smart_incr(hE, hEn)
            del dEB
        if grad_X:
            dEn = torch.matmul(dEn, X) + torch.matmul(En, basis)
            dEn /= n_order
            dE = smart_incr(dE, dEn)
        if grad_basis:
            dBn = torch.matmul(dBn, X) + torch.matmul(En, basis_basis)
            dBn /= n_order
            dB = smart_incr(dB, dBn)
        En = torch.matmul(En, X)
        En /= n_order
        E = smart_incr(E, En)
        # Compute sum-of-squares
        sos = torch.sum(En ** 2)
        if sos <= torch.numel(En) * tol:
            break

    E = E[..., 0, :, :]
    if grad_basis:
        dB = dB.reshape(XB_batch_shape + (nb_feat, dim, dim, dim, dim))

    out = [E]
    if grad_X:
        out.append(dE)
    if grad_basis:
        out.append(dB)
    if hess_X:
        out.append(hE)
    return out[0] if len(out) == 1 else out


class _ExpM(torch.autograd.Function):
    """Matrix exponential with automatic differentiation."""

    @staticmethod
    @custom_fwd
    def forward(ctx, X, basis, max_order, tol):
        # Save precomputed components of the backward pass
        needs_grad_X = torch.is_tensor(X) and X.requires_grad
        needs_grad_basis = torch.is_tensor(basis) and basis.requires_grad
        ctx.names = []
        if needs_grad_X or needs_grad_basis:
            if torch.is_tensor(X):
                ctx.save_for_backward(X)
                ctx.names += ['X']
            if torch.is_tensor(basis):
                ctx.save_for_backward(basis)
                ctx.names += ['basis']
            ctx.args = {'X': X, 'basis': basis,
                        'max_order': max_order, 'tol': tol}

        # Compute matrix exponential
        E = _expm(X, basis, max_order=max_order, tol=tol)
        return E

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        # DEBUG
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        grad_X = grad_basis = None
        dim = output_grad.shape[-1]

        # Unpack arguments
        X = ctx.args['X']
        basis = ctx.args['basis']
        max_order = ctx.args['max_order']
        tol = ctx.args['tol']
        needs_grad_X = utils.requires_grad(ctx, 'X')
        needs_grad_basis = utils.requires_grad(ctx, 'basis')

        # Compute derivative of output w.r.t. input
        _, *input_grad = _expm(X, basis, max_order=max_order, tol=tol,
                               grad_X=needs_grad_X,
                               grad_basis=needs_grad_basis)

        # Chain rule: dot product with derivative of loss w.r.t. output
        if needs_grad_X:
            grad_X, *input_grad = input_grad
            grad_for_X = output_grad[..., None, :, :]
            grad_X = torch.sum(grad_X * grad_for_X, dim=[-1, -2])
            if basis is None:
                grad_X = grad_X.reshape(grad_X.shape[:-1] + (dim, dim))
            grad_X = utils.broadcast_backward(grad_X, X.shape)

        if needs_grad_basis:
            grad_basis, *input_grad = input_grad
            grad_for_basis = output_grad[..., None, None, None, :, :]
            grad_basis = torch.sum(grad_basis * grad_for_basis, dim=[-1, -2])
            grad_basis = utils.broadcast_backward(grad_basis, basis.shape)

        return grad_X, grad_basis, None, None


def _expm_torch(X, basis=None):
    backend = utils.max_backend(X, basis)
    X = utils.as_tensor(X, **backend)

    if basis is not None:
        # X contains parameters in the Lie algebra -> reconstruct the matrix
        # X.shape = [.., F], basis.shape = [..., F, D, D]
        basis = utils.as_tensor(basis, **backend)
        X = torch.sum(basis * X[..., None, None], dim=-3)

    return torch.matrix_exp(X)


def expm(X, basis=None, max_order=10000, tol=1e-32):
    """Matrix exponential.

    Notes
    -----
    .. This function evaluates the matrix exponential and its
       derivatives using a Taylor approximation. A faster integration
       technique, based  e.g. on scaling and squaring, could have been
       used instead.
    .. PyTorch/NumPy broadcasting rules apply.
       See: https://pytorch.org/docs/stable/notes/broadcasting.html
    .. This function is automtically differentiable.

    Parameters
    ----------
    X : {(..., F), (..., D, D)} tensor_like
        If `basis` is None: log-matrix.
        Else:               parameters of the log-matrix in the basis set.
    basis : (..., F, D, D) tensor_like, optional.
        Basis set.
    max_order : int, default=10000
        Order of the Taylor expansion
    tol : float, default=1e-32
        Tolerance for early stopping
        The criterion is based on the Frobenius norm of the last term of
        the Taylor series.

    Returns
    -------
    eX : (..., D, D) tensor
        Matrix exponential

    """
    return _ExpM.apply(X, basis, max_order, tol)
    # if hasattr(torch, 'matrix_exp'):
    #     return _expm_torch(X, basis)
    # else:
    #     return _ExpM.apply(X, basis, max_order, tol)
