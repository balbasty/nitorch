"""Schur decomposition.

This implementation uses torch.qr under the hood. See algorithm 4.1 from:
    "Lecture Notes on Solving Large Scale Eigenvalue Problems"
    Peter Arbenz, 2016.
    https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf

It might be better to use an implicit/Hessenberg QR algorithm instead:
    https://en.wikipedia.org/wiki/QR_algorithm#The_implicit_QR_algorithm
or section 4.2 from the previous lecture notes.

Options and pre/post processing are inspired by the implementation in
scipy (which uses lapack under the hood):
    https://github.com/scipy/scipy/blob/master/scipy/linalg/decomp_schur.py

Here's Lapack's documentation on Schur decomposition and Hessenberg reduction.
"""
import torch
from . import utils
from . import constants


def householder(x, basis=0, inplace=False, check_finite=True, return_alpha=False):
    """Compute the Householder reflector of a (batch of) vector(s).

    Householder reflectors are matrices of the form :math:`P = I - 2uu^*`,
    where `u` is a Householder vector. Reflectors are typically used
    to project a (complex) vector onto a Euclidean basis:
    :math:`Px = r ||x|| e_i`, with `r = -exp(1j*angle(x_i))`. This function
    returns a Householder vector tailored to a specific (complex) vector.

    ..note:: Complex tensors only exist in PyTorch >= 1.6

    Parameters
    ----------
    x : (..., n) tensor_like
        Input vector. Can be batched.
    basis : int, default=1
        Index of the Euclidean basis.
    inplace : bool, default=False
        If True, overwrite `x`.
    check_finite : bool, default=True
        If True, checks that the input matrix does not contain any
        non finite value. Disabling this may speed up the algorithm.
    return_alpha : bool, default=False
        Return alpha, the 'projection' of x to the Euclidean basis:
        :math:`-||x|| * exp(1j*angle(x_i))`

    Returns
    -------
    u : (..., n) tensor
        Householder vector.
    a : (...) tensor, optional
        Projection on the Euclidean basis.

    """
    x = utils.as_tensor(x)
    if check_finite and not torch.isfinite(x).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        x = x.clone()

    x, alpha = householder_(x, basis)

    if return_alpha:
        return x, alpha
    else:
        return x


def householder_(x, basis=0):
    """Inplace version of ``householder``, without any checks."""

    # Compute unitary parameter
    rho = x[..., basis:basis+1].clone()
    rho.div_(rho.abs()).neg_()
    rho[rho == 0] = 1
    rho *= x.norm(dim=-1, keepdim=True)

    # Compute Householder reflector
    x[..., basis:basis+1] -= rho
    x /= x.norm(dim=-1, keepdim=True)

    return x, rho[..., 0]


def householder_apply(a, u, k=None, side='both', inverse=False,
                      inplace=False, check_finite=True):
    """Apply a series of Householder reflectors to a matrix.

    Parameters
    ----------
    a : (..., n, n) tensor_like
    u : tensor_like or list[tensor_like]
        A list of Householder reflectors :math:`u_k`.
        Each reflector forms Householder matrix :math:`P_k = I - 2 u_k u_k^H`.
    k : int or list[int], optional
        The index corresponding to each reflector.
    side : {'left', 'right', 'both'}, default='both'
        Side to apply to.
    inverse : bool, default=False
        Apply the inverse transform
    inplace : bool, default=False
        Apply transformation inplace.
    check_finite : bool, default=True
        Check that the input does not have any nonfinite values

    Returns
    -------
    h : (..., n, n) tensor
        The transformed matrix :math:`U \times A \times U^H`
        with :math:`U = P_{k-2} \times ... \times P_1`

    """
    a = utils.as_tensor(a)
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()

    return householder_apply_(a, u, k, side, inverse)


def householder_apply_(a, u, k=None, side='both', inverse=False):
    """Inplace version of ``householder_apply``, without any checks."""

    if inverse:
        u = reversed(u)
        # Reversed order (= transpose), but we haven't taken the conjugate yet
    do_left = side.lower() in ('left', 'both')
    do_right = side.lower() in ('right', 'both')

    n = a.shape[-1]
    if not isinstance(u, (list, tuple)):
        u = [u]
    k_range = k if k is not None else range(len(u))
    if not isinstance(k_range, (list, tuple)):
        k_range = [k_range]

    for k, uk in zip(k_range, u):
        # Householder reflector
        uk = torch.as_tensor(uk)[..., None]
        uk_h = uk.conj()
        if inverse:
            uk, uk_h = (uk_h, uk)
        uk_h = uk_h.transpose(-1, -2)

        # Apply P from the left
        if do_left:
            rhs = uk.matmul(uk_h.matmul(a[..., k:, :]))
            a[..., k:, :] -= 2*rhs
        # Apply P from the right
        if do_right:
            rhs = a[..., :, k:].matmul(uk).matmul(uk_h)
            a[..., :, k:] -= 2*rhs

    return a


def _householder_apply_(a, u, side):
    if side == 'left':
        a -= 2 * u.matmul(u.conj().transpose(-1, -2).matmul(a))
    elif side == 'right':
        a -= 2 * a.matmul(u).matmul(u.conj().transpose(-1, -2))
    else:
        raise ValueError()
    return a


def hessenberg(a, inplace=False, check_finite=True, compute_u=False):
    """Return an hessenberg form of the matrix (or matrices) ``a``.

    Parameters
    ----------
    a : (..., n, n) tensor_like
        Input matrix. Can be complex.
    inplace : bool, default=False
        Overwrite ``a``.
    check_finite : bool, default=True
        Check that all values in ``a`` ar finite.
    compute_u : bool, default=False
        Compute and return the transformation matrix ``u``.

    Returns
    -------
    h : (..., n, n) tensor
        Hessenberg form of ``a``.
    u : list[tensor], optional
        Set of Householder reflectors.


    """
    a = utils.as_tensor(a)
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))

    return hessenberg_(a, compute_u)


def hessenberg_(a, compute_u=False):
    """Inplace version of ``hessenberg``, withouth any checks."""
    n = a.shape[-1]
    u = []
    for k in range(n-2):
        # Householder reflector: P_k = I_k (+) (I_{n-k} - 2 u_k u_k*)
        uk, alpha = householder_(a[..., k+1:, k])
        if compute_u:
            u.append(uk.clone())
        uk = uk[..., None]
        uk_h = uk.conj().transpose(-1, -2)
        # Apply P from the left
        rhs = uk.matmul(uk_h.matmul(a[..., k+1:, k+1:]))
        a[..., k+1:, k+1:] -= 2*rhs
        # Apply P from the right
        rhs = a[..., :, k+1:].matmul(uk).matmul(uk_h)
        a[..., :, k+1:] -= 2*rhs
        # Set k-th column to [alpha; zeros]
        a[..., k+1, k] = alpha
        a[..., k+2:, k] = 0

    if compute_u:
        return a, u
    else:
        return a


def givens(x, y):
    r"""Compute a Givens rotation matrix.

    A Givens rotation is a rotation in a plane that aligns a specific
    vector (in this plane) with the first axis of the plane:
    :math:`G \times x = [ ||x||, 0 ]`.

    Parameters
    ----------
    x : tensor_like
        First element
    y : tensor_like
        Second element

    Returns
    -------
    c : tensor
        ``c = x / norm([x, y])``
    s : tensor
     ``s = -y / norm([x, y])``

    """
    x = utils.as_tensor(x)
    y = utils.as_tensor(y)
    nrm = (x.square() + y.square()).sqrt()
    x = x / nrm
    y = (y / nrm).neg_()
    return x, y


def givens_apply(a, c, s, i=0, j=None, side='both',
                 inplace=False, check_finite=True):
    a = utils.as_tensor(a)
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))
    return givens_apply_(a, c, s, i, j, side)


def givens_apply_(a, c, s, i=0, j=None, side='both'):
    j = i+1 if j is None else j

    if side in ('both', 'left'):
        a0 = a[..., i, :]
        a1 = a[..., j, :]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)

    if side in ('both', 'right'):
        a0 = a[..., :, i]
        a1 = a[..., :, j]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)

    return a


def qr_hessenberg(h, inplace=False, check_finite=True):
    """QR decomposition for Hessenberg matrices.

    ..note:: This is slower than torch.qr when the batch size is small,
             even though torch.qr does not know that ``h`` has a
             Hessenberg form. It's just hard to beat lapack. With
             larger batch size, both algorithms are on par.

    Parameters
    ----------
    h : (..., n, n) tensor_like
        Hessenberg matrix (all zeros below the first lower diagonal).
    inplace : bool, default=False
        Process inplace.
    check_finite : bool, default=True
        Check that all input values are finite.

    Returns
    -------
    q : tensor
    r : tensor

    """
    h = utils.as_tensor(h)
    if check_finite and not torch.isfinite(h).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        h = h.clone()
    if h.shape[-1] != h.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(h.shape[-2], h.shape[-1]))
    return qr_hessenberg_(h)


def qr_hessenberg_(a):
    """Inplace version of `qr_hessenberg`, without any checks."""
    n = a.shape[-1]
    q = torch.empty_like(a)
    q[..., :, :] = torch.eye(n, dtype=a.dtype, device=a.device)
    for k in range(n-1):
        c, s = givens(a[..., k, k], a[..., k+1, k])
        c = c[..., None]
        s = s[..., None]
        # Compute R
        a0 = a[..., k, k:]
        a1 = a[..., k+1, k:]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        # Compute Q
        q0 = q[..., :k+2, k]
        q1 = q[..., :k+2, k+1]
        tmp = s * q0
        q0.mul_(c).sub_(s * q1)
        q1.mul_(c).add_(tmp)

    return q, a


def rq_hessenberg(h, inplace=False, check_finite=True):
    """Compute the QR decomposition of a Hessenberg matrix and ``R` @ Q``.

    ..note:: This is slower than torch.qr when the batch size is small,
             even though torch.qr does not know that ``h`` has a
             Hessenberg form. It's just hard to beat lapack. With
             larger batch size, this algorithm is faster.

    Parameters
    ----------
    h : (..., n, n) tensor_like
        Hessenberg matrix (all zeros below the first lower diagonal).
    inplace : bool, default=False
        Process inplace.
    check_finite : bool, default=True
        Check that all input values are finite

    Returns
    -------
    rq : tensor
        Reverse product of the QR decomposition: ``rq = r @ q``.

    """
    h = utils.as_tensor(h)
    if check_finite and not torch.isfinite(h).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        h = h.clone()
    if h.shape[-1] != h.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(h.shape[-2], h.shape[-1]))
    return rq_hessenberg_(h)


def rq_hessenberg_(a, u=None):
    """Inplace version of `rq_hessenberg`, without any checks."""
    n = a.shape[-1]

    g = []
    for k in range(n-1):
        c, s = givens(a[..., k, k], a[..., k+1, k])
        c = c[..., None]
        s = s[..., None]
        g.append([c, s])
        a0 = a[..., k, k:]
        a1 = a[..., k+1, k:]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
    for k in range(n-1):
        c, s = g[k]
        a0 = a[..., :k+2, k]
        a1 = a[..., :k+2, k+1]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        if u is not None:
            u0 = u[..., :, k]
            u1 = u[..., :, k+1]
            tmp = s * u0
            u0.mul_(c).sub_(s * u1)
            u1.mul_(c).add_(tmp)
    if u is not None:
        return a, u
    else:
        return a


def schur(a, output='real', compute_u=True, inplace=False, check_finite=True,
          max_iter=1024, tol=1e-32):
    """Compute the Schur decomposition of a square matrix.

    Under the Schur decomposition, :math:`A = U T U^H`, with `U` unitary
    (:math:`QQ^H = I`) and `T` upper triangular.

    Parameters
    ----------
    a : (..., m, m) tensor_like
        Input matrif or field of matrices
    output : {'real', 'complex'}, default='real'
        When ``a`` is real whether to return the real or complex form
        of the Schur decomposition. If ``a`` is complex, this is not used.
        The complex form is upper-triangular while the real form is
        quasi upper-triangular, with elements on the lower diagonal
        when a eigenvalue (precisely: a pair of conjugate eigenvalues)
        is complex.
    compute_u : bool, default=True
        Compute the unitary matrix. If False, only return ``t``.
    inplace : bool, default=False
        If True, overwrite ``a``.
    check_finite : bool, default=True
        If True, checks that the input matrix does not contain any
        non finite value. Disabling this may speed up the algorithm.
    max_iter : int, default=1024
        Maximum number of iterations.
    tol : float, optional
        Tolerance for early stopping.
        Default: machine precision for ``a.dtype``.

    Returns
    -------
    t : (..., m, m) tensor
        Upper triangular matrix.
    u : (..., m, m) tensor, optional
        Unitary matrix.

    """
    # Check arguments
    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("output must be 'real', or 'complex'. Got {}."
                         .format(output))
    a = utils.as_tensor(a)
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))

    # Initialization: reduction to Hessenberg form
    if compute_u:
        a, q = hessenberg_(a, compute_u=True)
    else:
        a = hessenberg_(a, compute_u=False)
        q = None

    # Main part
    schur_ = schur_complex_ if a.is_complex() else schur_real_
    if compute_u:
        a, u = schur_(a, max_iter=max_iter, tol=tol, compute_u=True)
    else:
        a = schur_(a, max_iter=max_iter, tol=tol, compute_u=False)
        u = None

    # TODO: u <- q@u

    # Convert to complex form if required
    # if not a.is_complex() and output.lower() in ('c', 'complex'):
    #     if compute_u:
    #         a, u = schur_real_to_complex(a, u, check_finite=False)
    #     else:
    #         a = schur_real_to_complex(a, check_finite=False)

    if compute_u:
        return a, u
    else:
        return a


def schur_complex_(h, max_iter=1024, tol=None, compute_u=False):
    """Complex Schur decomposition.

    This function is applied inplace and does not perform checks.
    It uses the Hessenberg QR algorithm with Rayleigh quotient shift.
    The input matrix ``h`` should be an upper Hessenberg matrix.

    References
    ----------
    ..[1] Alg 4.4, Ch. 4 in "Lecture Notes on Solving Large Scale
          Eigenvalue Problems", Arbenz P.

    """

    tol = constants.eps(h.dtype) if tol is None else tol

    dtype = h.dtype
    device = h.device

    n = h.shape[-1]
    eye = torch.eye(n, dtype=dtype, device=device)
    if compute_u:
        u = torch.empty_like(h)
        u[..., :, :] = utils.unsqueeze(eye, ndim=h.dim() - 2)

    # Deflation loop
    h0 = h
    for k in reversed(range(1, n)):
        # QR Algorithm
        for _ in range(max_iter):

            # Estimate eigenvalue
            sigma = h[..., k, k] * eye

            # Hessenberg QR decomposition
            if compute_u:
                h, u = rq_hessenberg_(h - sigma, u)
            else:
                h = rq_hessenberg_(h - sigma)
                h += sigma

            # Extract lower-triangular point and compute its norm
            sos_lower = h[..., k, k-1].abs().square().sum()
            sos_diag = h[..., k, k].abs().square().sum() + \
                       h[..., k-1, k-1].abs().square().sum()
            if sos_lower < tol * sos_diag:
                h[..., k, k-1:] = 0
                break
        h = h[:k, :k]

    h = h0

    if compute_u:
        return h, u
    else:
        return h


def schur_real_(h, max_iter=1024, tol=None, compute_u=False):
    """Real Schur decomposition (Francis algorithm).

    This function is applied inplace and does not perform checks.
    It uses the  Francis double step QR algorithm.
    The input matrix ``h`` should be an upper Hessenberg matrix.

    References
    ----------
    ..[1] Alg 4.5, Ch. 4 in "Lecture Notes on Solving Large Scale
          Eigenvalue Problems", Arbenz P.


    """

    tol = constants.eps(h.dtype) if tol is None else tol

    # Initialize
    n = h.shape[-1]
    if compute_u:
        u = torch.empty_like(h)
        eye = torch.eye(n, dtype=h.dtype, device=h.device)
        u[..., :, :] = eye
    xyz = h.new_empty(h.shape[:-2] + (3,))

    # Deflation loop: we start with h[..., :, :] and reduce the matrix
    # each time the bottom-right eigenvalue (or pair of conjugate
    # eigenvalues) has converged.
    h0 = h
    p = n-1
    n_iter = 0
    while p > 1:
        n_iter += 1
        # ---------------------
        # Double spectral shift
        # ---------------------
        # Compute trace (s) and determinant (t) of the bottom-right 2x2 matrix
        # If this 2x2 matrix has two complex (conjugate) eigenvalues, then:
        # > s = 2 * real(lam)
        # > t = abs(lam) ** 2
        s = h[..., p, p] + h[..., p-1, p-1]
        t = h[..., p, p] * h[..., p-1, p-1] - h[..., p-1, p] * h[..., p, p-1]
        # Let M = (H-lam)(H-lam*) = H^2 - 2*real(lam)*H + I*abs(lam)**2
        #                         = H^2 - s*H + t*I
        # All lower diagonal > 2 of M are zero.
        # Compute first 3 elements of first column of M:
        # > xyz = M[0:3, 0]
        xyz[..., 0] = h[..., 0, 0].square() + h[..., 0, 1]*h[..., 1, 0] \
                      - s*h[..., 0, 0] + t
        xyz[..., 1] = h[..., 1, 0] * (h[..., 0, 0] + h[..., 1, 1] - s)
        xyz[..., 2] = h[..., 1, 0] * h[..., 2, 1]
        # ------------------------------
        # Implicit QR decomposition of M
        # ------------------------------
        for k in range(p-1):
            # Here > xyz = M[k:k+3, k-1]
            # Determine the Householder reflector of xyz
            w, _ = householder_(xyz)
            w = w[..., None]
            r = max(0, k-1)
            _householder_apply_(h[..., k:k+3, r:], w, side='left')
            r = min(k+3, p)
            _householder_apply_(h[..., :r+1, k:k+3], w, side='right')
            if compute_u:
                _householder_apply_(u[..., :, k:k+3], w, side='right')
            # Update xyz
            xyz[..., 0] = h[..., k+1, k]
            xyz[..., 1] = h[..., k+2, k]
            if k < p-2:
                xyz[..., 2] = h[..., k+3, k]
        # Here > xy = M[p-1:p+1, p-2]
        # Determine the Givens rotation of [x; y]
        c, s = givens(xyz[..., 0], xyz[..., 1])
        c = c[..., None]
        s = s[..., None]
        givens_apply_(h[..., p-1:p+1, p-2:], c, s, side='left')
        givens_apply_(h[..., :p+1, p-1:p+1], c, s, side='right')
        if compute_u:
            givens_apply_(u[..., :, p-1:p+1], c, s, side='right')
        print(n_iter, p, h[..., p, p-1].item(), h[..., p-1, p-2].item())
        if n_iter in (6, 10, 11):
            import numpy as np
            np.set_printoptions(4)
            print(h.numpy())
        # Convergence
        sos_lower = h[..., p, p-1].abs().sum()
        sos_diag = h[..., p, p].abs().sum() + \
                   h[..., p-1, p-1].abs().sum()
        if sos_lower <= tol * sos_diag:
            h[..., p, p-1] = 0
            h = h[..., :-1, :]
            p -= 1
        else:
            sos_lower = h[..., p-1, p-2].abs().sum()
            sos_diag = h[..., p-1, p-1].abs().sum() + \
                       h[..., p-2, p-2].abs().sum()
            if sos_lower <= tol * sos_diag:
                h[..., p-1, p-2] = 0
                h = h[..., :-2, :]
                p -= 2

    h = h0

    if compute_u:
        return h, u
    else:
        return h

