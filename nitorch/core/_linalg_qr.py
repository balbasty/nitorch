"""QR decomposition and QR algorithm

Only the QR algorithm for symmetric matrices is implemented because
it is the only one that scales easily to batched matrices (the
real schur decomposition that must be used in the general case has
a deflation step that depends on each matrix).

It is *much* slower than torch.symeig on the CPU, but I hope that it will
be better than torch on GPU for large numbers of small matrices (where
the GPU implementation is just unusable).

References
----------
..[1] "Lecture Notes on Solving Large Scale Eigenvalue Problems"
      Peter Arbenz, 2016.
      https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
..[2] "Implicit QR algorithm"
      https://en.wikipedia.org/wiki/QR_algorithm#The_implicit_QR_algorithm
"""
import torch
from .optionals import custom_fwd, custom_bwd
from . import utils, py, constants


def smart_conj(x):
    """Take conjugate if complex (saves a copy when real)."""
    return x.conj() if x.is_complex() else x


def smart_real(x):
    """Take real part if complex (saves a copy when real)."""
    return torch.real(x) if x.is_complex() else x


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
    rho.sign_().neg_()
    rho[rho == 0] = 1
    rho *= x.norm(dim=-1, keepdim=True)

    # Compute Householder reflector
    x[..., basis:basis+1] -= rho
    x /= x.norm(dim=-1, keepdim=True)
    x[torch.isfinite(x).bitwise_not_()] = 0

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

    u = py.make_list(u)
    if inverse:
        u = u[::-1]
        # Transpose of the product = Reversed order
        # Each factor in the product is Hermitian so no need to transpose them
    do_left = side.lower() in ('left', 'both')
    do_right = side.lower() in ('right', 'both')

    n = a.shape[-1]
    u = py.make_list(u)
    k_range = k if k is not None else range(len(u))
    k_range = py.make_list(k_range)

    for k, uk in zip(k_range, u):
        # Householder reflector
        uk = torch.as_tensor(uk)[..., None]
        uk_h = uk.conj()
        uk_h = uk_h.transpose(-1, -2)

        # Apply P from the left
        k0 = n - uk.shape[-2]
        if do_left:
            rhs = uk.matmul(uk_h.matmul(a[..., k0:, :]))
            a[..., k0:, :] -= 2*rhs
        # Apply P from the right
        if do_right:
            rhs = a[..., :, k0:].matmul(uk).matmul(uk_h)
            a[..., :, k0:] -= 2*rhs

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
    """Inplace version of ``hessenberg``, without any checks."""
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
        rhs = uk.matmul(uk_h.matmul(a[..., k+1:, k+1:])).mul_(2)
        a[..., k+1:, k+1:] -= rhs
        # Apply P from the right
        rhs = a[..., :, k+1:].matmul(uk).matmul(uk_h).mul_(2)
        a[..., :, k+1:] -= rhs
        # Set k-th column to [alpha; zeros]
        a[..., k+1, k] = alpha
        a[..., k+2:, k] = 0

    if compute_u:
        return a, u
    else:
        return a


def mask_tri(n, diagonal=0, upper=True, **backend):
    """Return a mask of the upper or lower triangular elements."""

    backend['dtype'] = torch.int
    def mask_upper():
        """Return a mask of upper triangular elements"""
        i = torch.arange(n, **backend)
        i, j = utils.meshgrid_ij(i, i)
        i = i - j
        return i <= -diagonal

    return mask_upper() if upper else mask_upper().T


def fill_sym(a, upper=True):
    """Fill the other half of a triangular matrix by symmetry

    Parameters
    ----------
    a : (..., n, n) tensor
        Triangular matrix.
    upper : bool, default=False
        Whether the input tensor is upper or lower triangulat.
        If `upper` is True, the *lower* half is filled by copying the
        upper half.

    Returns
    -------
    a

    """
    return fill_sym_(a.clone(), upper)


def fill_sym_(a, upper=True):
    """Fill the other half of a triangular matrix by symmetry, inplace

    Parameters
    ----------
    a : (..., n, n) tensor
        Triangular matrix.
    upper : bool, default=False
        Whether the input tensor is upper or lower triangulat.
        If `upper` is True, the *lower* half is filled by copying the
        upper half.

    Returns
    -------
    a

    """
    mask = mask_tri(a.shape[-1], diagonal=1, upper=upper, **utils.backend(a))
    a.transpose(-1, -2)[..., mask] = a[..., mask]
    return a


def matvec_sym(a, x, upper=True):
    """Hermitian matrix-vector product.

    Parameters
    ----------
    a : (..., n, n) tensor
        Symmetric matrix
    x : (..., n) tensor
        Vector
    upper : bool, default=False
        Whether to use the upper or lower half of the matrix.

    Returns
    -------
    y : (..., n) tensor
        Matrix-vector product

    """

    def matvec_sym_upper(a, x):
        n = a.shape[-1]
        if n <= 4:
            shape = utils.expanded_shape(a.shape[:-2], x.shape[:-1])
            shape = [*shape, n]
            y = x.new_zeros(shape)
            for i in range(n):
                for j in range(n):
                    aij = a[..., i, j] if i <= j else a[..., j, i]
                    y[..., i] += aij * x[..., j]
        else:
            a = fill_sym(a, upper=True)
            y = torch.matmul(a, x)
        return y

    if not upper:
        a = a.transpose(-1, -2)
    return matvec_sym_upper(a, x)


def outer_sym(u, v):
    """Symmetric part of the outer product of two vectors.

    ..warning: I do not divide by two

    Parameters
    ----------
    u : (..., n) tensor
        Left vector
    v : (..., n) tensor
        Right vector

    Returns
    -------
    uv : (..., n, n) tensor
        Symmetric part of the outer product of u and v
        (i.e., u @ v.conj().T + v @ u.conj().T)

    """
    n = u.shape[-1]
    if n > 4:
        out = u[..., :, None] * smart_conj(v[..., None, :])
        out += out.transpose(-1, -2)
        return out
    shape = utils.expanded_shape(u.shape[:-1], v.shape[:-1])
    shape = [*shape, n, n]
    out = u.new_empty(shape)
    for i in range(n):
        out[..., i, i] = smart_real(u[..., i] * smart_conj(v[..., i]))
        out[..., i, i] *= 2
        for j in range(i+1, n):
            vjc = smart_conj(v[..., j])
            ujc = smart_conj(u[..., j])
            out[..., i, j] = u[..., i] * vjc + v[..., i] * ujc
            out[..., j, i] = out[..., i, j].conj()
    return out


def hessenberg_sym(a, upper=True, fill=True, inplace=False,
                   check_finite=True, compute_u=False):
    """Return a tridiagonal form of the hermitian matrix (or matrices) ``a``.

    Parameters
    ----------
    a : (..., n, n) tensor_like
        Input hermitian matrix. Can be complex.
    upper : bool, default=True
        Whether to use the upper or lower triangular part of the matrix.
    fill : bool, default=False
        Fill the other half of the output matrix by conjugate symmetry.
    inplace : bool, default=False
        Overwrite ``a``.
    check_finite : bool, default=True
        Check that all values in ``a`` ar finite.
    compute_u : bool, default=False
        Compute and return the transformation matrix ``u``.

    Returns
    -------
    h : (..., n, n) tensor
        Hessenberg (tridiagonal) form of ``a``.
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
    return (hessenberg_sym_upper_(a, compute_u, fill) if upper else
            hessenberg_sym_lower_(a, compute_u, fill))


def hessenberg_sym_upper_(a, compute_u=False, fill=False):
    """Inplace version of ``hessenberg_sym``, without any checks.
    This function does not use the lower triangular part. """
    # I take the transpose of a and use A = USU' => A.' = conj(U)S.'U.'
    a = a.transpose(-1, -2)
    a = hessenberg_sym_lower_(a, compute_u, fill)
    if compute_u:
        a, u = a
        a.transpose_(-1, -2)
        u = [smart_conj(uk) for uk in u]
        a = (a, u)
    else:
        a.transpose_(-1, -2)
    return a


def hessenberg_sym_lower_(a, compute_u=False, fill=False):
    """Inplace version of ``hessenberg_sym``, without any checks.
    This function does not use the upper triangular part. """
    n = a.shape[-1]
    u = []
    for k in range(n-2):
        # Householder reflector: P_k = I_k (+) (I_{n-k} - 2 u_k u_k*)
        uk, alpha = householder_(a[..., k+1:, k])
        if compute_u:
            u.append(uk.clone())
        uk_h = smart_conj(uk)
        ak = a[..., k+1:, k+1:]
        vk = matvec_sym(ak, uk, upper=False)
        vk -= uk * (uk_h * vk).sum(-1, keepdim=True)
        vk *= 2
        uvk = outer_sym(uk, vk)
        msk = mask_tri(n-k-1, upper=False, **utils.backend(a))
        ak[..., msk] -= uvk[..., msk]
        # Set k-th column to [alpha; zeros]
        a[..., k+1, k] = alpha
        a[..., k+2:, k] = 0

    if fill:
        a = fill_sym(a, upper=False)
    if compute_u:
        return a, u
    else:
        return a


@torch.jit.script
def _givens_jit(x, y):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
    nrm = (x*x + y*y).sqrt_()
    x = x / nrm
    y = (y / nrm).neg_()
    msk = nrm == 0
    x.masked_fill_(msk, 1.)
    y.masked_fill_(msk, 0.)
    return x, y


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
    return _givens_jit(x, y)


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

    @torch.jit.script
    def _givens_apply_left(a, c, s, i, j):
        # type : (Tensor, Tensor, Tensor, Tensor, int, int) -> Tensor
        a0 = a[..., i, :]
        a1 = a[..., j, :]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        return a

    @torch.jit.script
    def _givens_apply_right(a, c, s, i, j):
        # type : (Tensor, Tensor, Tensor, Tensor, int, int) -> Tensor
        a0 = a[..., :, i]
        a1 = a[..., :, j]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        return a

    @torch.jit.script
    def _givens_apply_both(a, c, s, i, j):
        # type : (Tensor, Tensor, Tensor, Tensor, int, int) -> Tensor
        a0 = a[..., i, :]
        a1 = a[..., j, :]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        a0 = a[..., :, i]
        a1 = a[..., :, j]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        return a

    if side == 'left':
        return _givens_apply_left(a, c, s, i, j)
    if side == 'right':
        return _givens_apply_right(a, c, s, i, j)
    if side == 'both':
        return _givens_apply_both(a, c, s, i, j)

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


def rq_hessenberg(h, u=None, inplace=False, check_finite=True, sym=True):
    """Compute the QR decomposition of a Hessenberg matrix and ``R` @ Q``.

    ..note:: This is slower than torch.qr when the batch size is small,
             even though torch.qr does not know that ``h`` has a
             Hessenberg form. It's just hard to beat lapack. With
             larger batch size, this algorithm is faster.

    Parameters
    ----------
    h : (..., n, n) tensor_like
        Hessenberg matrix (all zeros below the first lower diagonal).
    u : (..., n) tensor_like, optional
        Vectors to transform with
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
    return rq_hessenberg_(h, u)


@torch.jit.script
def _rq_hessenberg_jit_(a, sym):
    # type: (Tensor, bool) -> Tensor
    n = a.shape[-1]
    list_c = []
    list_s = []
    for k in range(n - 1):
        c, s = _givens_jit(a[..., k, k], a[..., k + 1, k])
        c = c[..., None]
        s = s[..., None]
        list_c.append(c)
        list_s.append(s)
        if sym:
            a0 = a[..., k, k:k+3]
            a1 = a[..., k+1, k:k+3]
        else:
            a0 = a[..., k, k:]
            a1 = a[..., k+1, k:]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
    for k in range(n - 1):
        c = list_c[k]
        s = list_s[k]
        a0 = a[..., max(0, k-1):k+2, k]
        a1 = a[..., max(0, k-1):k+2, k+1]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
    return a


@torch.jit.script
def _rq_hessenberg_vectors_jit_(a, u, sym):
    # type: (Tensor, Tensor, bool) -> Tuple[Tensor, Tensor]
    n = a.shape[-1]
    list_c = []
    list_s = []
    for k in range(n - 1):
        c, s = _givens_jit(a[..., k, k], a[..., k + 1, k])
        c = c[..., None]
        s = s[..., None]
        list_c.append(c)
        list_s.append(s)
        if sym:
            a0 = a[..., k, k:k+3]
            a1 = a[..., k+1, k:k+3]
        else:
            a0 = a[..., k, k:]
            a1 = a[..., k+1, k:]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
    for k in range(n - 1):
        c = list_c[k]
        s = list_s[k]
        a0 = a[..., max(0, k-1):k+2, k]
        a1 = a[..., max(0, k-1):k+2, k+1]
        tmp = s * a0
        a0.mul_(c).sub_(s * a1)
        a1.mul_(c).add_(tmp)
        # vectors
        u0 = u[..., :, k]
        u1 = u[..., :, k+1]
        tmp = s * u0
        u0.mul_(c).sub_(s * u1)
        u1.mul_(c).add_(tmp)
    return a, u


def rq_hessenberg_(a, u=None, sym=False):
    """Inplace version of `rq_hessenberg`, without any checks."""
    if u is None:
        return _rq_hessenberg_jit_(a, sym)
    else:
        return _rq_hessenberg_vectors_jit_(a, u, sym)


def qr_explicit_(h, max_iter=1024, tol=None, compute_u=False, sym=False):
    """Explicit QR decomposition.

    This function is applied inplace and does not perform checks.
    It uses the Hessenberg QR algorithm with Wilkinson shift.
    The input matrix ``h`` should be an upper Hessenberg matrix.

    If h is not tridiagonal symmetric, complex eigenvalues can exist
    and complex arithmetic will be used.

    This is not the most stable implementation of the QR algorithm,
    but it has the benefit of working well with batched matrices.

    References
    ----------
    ..[1] Alg 4.4, Ch. 4 in "Lecture Notes on Solving Large Scale
          Eigenvalue Problems", Arbenz P.
    """
    tol = constants.eps(h.dtype) if tol is None else tol
    if not compute_u:
        return _qr_explicit_jit_(h, max_iter, tol, sym)
    else:
        return _qr_explicit_vectors_jit_(h, max_iter, tol, sym)


@torch.jit.script
def wilkinson(h):
    # type: (Tensor) -> Tensor
    h0 = h[..., 0, 0]
    h1 = h[..., 1, 1]
    b2 = h[..., 1, 0]
    b2 = b2*b2
    d = (h0 - h1) / 2
    s = d.sign()
    s.masked_fill_(s == 0, 1)
    d = d.abs() + (d*d + b2).sqrt()
    d.masked_fill_(d == 0, 1)
    return h1 - s * b2 / d


@torch.jit.script
def _qr_explicit_vectors_jit_(h, max_iter, tol, sym):
    # type: (Tensor, int, float, bool) -> Tuple[Tensor, Tensor]

    n = h.shape[-1]
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    u = torch.empty_like(h)
    u[..., :, :] = eye

    u0 = u
    h0 = h
    sigma = h.new_empty(h.shape[:-2])
    buf = h.new_empty(h.shape[:-2])
    for k in range(n-1, 0, -1):
        # QR Algorithm
        for it in range(max_iter):

            # Estimate eigenvalue
            if sym:
                sigma = wilkinson(h[..., -2:, -2:])
            else:
                sigma.copy_(h[..., -1, -1])  # Rayleigh

            # Hessenberg QR decomposition
            h.diagonal(0, -1, -2).sub_(sigma[..., None])
            _rq_hessenberg_vectors_jit_(h, u, sym)
            h.diagonal(0, -1, -2).add_(sigma[..., None])

            # Extract lower-triangular point and compute its norm
            b = torch.abs(h[..., -1, -2], out=buf).pow_(2).sum()
            a0 = torch.abs(h[..., -1, -1], out=buf).pow_(2).sum()
            a1 = torch.abs(h[..., -2, -2], out=buf).pow_(2).sum()
            sos_lower = b
            sos_diag = a0 + a1
            if sos_lower < tol * sos_diag:
                h[..., -1, :-1] = 0
                break
        h = h[..., :-1, :-1]
        u = u[..., :-1]

    h = h0
    u = u0
    return h, u


@torch.jit.script
def _qr_explicit_jit_(h, max_iter, tol, sym):
    # type: (Tensor, int, float, bool) -> Tensor

    n = h.shape[-1]
    h0 = h
    sigma = h.new_empty(h.shape[:-2])
    buf = h.new_empty(h.shape[:-2])
    for k in range(n-1, 0, -1):
        # QR Algorithm
        sos_prev = 0.
        for it in range(max_iter):

            # Estimate eigenvalue
            if sym:
                sigma = wilkinson(h[..., -2:, -2:])
            else:
                sigma.copy_(h[..., -1, -1])  # Rayleigh

            # Hessenberg QR decomposition
            h.diagonal(0, -1, -2).sub_(sigma[..., None])
            _rq_hessenberg_jit_(h, sym)
            h.diagonal(0, -1, -2).add_(sigma[..., None])

            # Extract lower-triangular point and compute its norm
            b = torch.abs(h[..., -1, -2], out=buf).pow_(2).sum()
            a0 = torch.abs(h[..., -1, -1], out=buf).pow_(2).sum()
            a1 = torch.abs(h[..., -2, -2], out=buf).pow_(2).sum()
            sos_lower = b
            sos_diag = a0 + a1
            if sos_lower < tol * sos_diag:
                h[..., -1, :-1] = 0
                break
            sos_new = sos_lower/sos_diag
            if sos_prev and abs((sos_prev - sos_new)/sos_prev) < tol * 1e-3:
                # We're stuck -> better to return a shitty value now than
                # be stuck forever
                break
            sos_prev = sos_new
        h = h[..., :-1, :-1]

    h = h0
    return h


def eig_sym(a, compute_u=False, upper=True, inplace=False,
            check_finite=True, max_iter=1024, tol=1e-32):
    """Compute the eigendecomposition of a Hermitian square matrix.

    Notes
    -----
    .. Eigenvalues are *not* sorted
    .. We use the explicit QR algorithm, which is probably less
       stable than the implicit QR algorithm used in Lapack.

    To sort eigenvalues and eigenvectors according to some criterion:
    ```python
    >> s, u = eig_sym(x, compute_u=True)
    >> _, i = crit(s).sort()
    >> s = s.gather(-1, i)    # permute eigenvalues
    >> i = i.unsqueeze(-2).expand(y.shape)
    >> u = u.gather(-1, i)    # permute eigenvectors
    ```
    If the criterion is the value of the eigenvalues, it simplifies:
    ```python
    >> s, u = eig_sym(x, compute_u=True)
    >> s, i = s.sort()
    >> i = i.unsqueeze(-2).expand(y.shape)
    >> u = u.gather(-1, i)    # permute eigenvectors
    ```

    Parameters
    ----------
    a : (..., m, m) tensor_like
        Input hermitian matrix or field of matrices
    compute_u : bool, default=False
        Compute the eigenvectors. If False, only return ``s``.
    upper : bool, default=True
        Whether to use the upper or lower triangular component.
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
    s : (..., m) tensor
        Eigenvalues.
    u : (..., m, m) tensor, optional
        Corresponding eigenvectors.

    """
    # Check arguments
    a = utils.as_tensor(a)
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))
    return eig_sym_(a, compute_u, upper, max_iter, tol)


def eig_sym_(a, compute_u=False, upper=True, max_iter=1024, tol=1e-32):
    """Inplace version of eig_sym, without checks (autodiff)."""
    return EigSym.apply(a, compute_u, upper, max_iter, tol)


def eig_sym_forward_(a, compute_u=False, upper=True, max_iter=1024, tol=1e-32):
    """Inplace version of eig_sym, without checks (forward)."""

    # Initialization: reduction to symmetric tridiagonal form
    hessenberg_ = hessenberg_sym_upper_ if upper else hessenberg_sym_lower_
    a = hessenberg_(a, compute_u=compute_u, fill=True)
    if compute_u:
        a, q = a

    # Main part
    a = qr_explicit_(a, max_iter=max_iter, tol=tol, compute_u=compute_u, sym=True)
    if compute_u:
        a, u = a
        householder_apply_(u, q, side='left', inverse=True)
    a = a.diagonal(0, -1, -2)

    return (a, u) if compute_u else a


class EigSym(torch.autograd.Function):
    """Autodiff implementation of `eig_sym_`.

    References
    ----------
    ..[1] "An extended collection of matrix derivative results for
           forward and reverse mode algorithmic differentiation"
          Mike Giles
          Report 08/01 (2008), Oxford University Computing Laboratory
          https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, a, compute_u, upper, max_iter, tol):

        if hasattr(ctx, 'set_materialize_grads'):
            # pytorch >= 1.7
            ctx.set_materialize_grads(False)

        compute_u_ = compute_u or a.requires_grad
        val = eig_sym_forward_(a, compute_u_, upper, max_iter, tol)

        if compute_u_:
            val, vec = val
            if a.requires_grad:
                ctx.save_for_backward(val, vec)

        return (val, vec) if compute_u else val

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outputs):
        # notations form ref [1] are used

        gD, *gU = grad_outputs
        D, U = ctx.saved_tensors

        if gD is None and (not gU or gU[0] is None):
            return (None,)*5

        if gU and gU[0] is not None:
            gU = gU.pop()
            F = D[..., :, None] - D[..., None, :]
            F = F.reciprocal_()
            F.diagonal(0, -1, -2).fill_(0)
            F *= U.transpose(-1, -2).matmul(gU)
            gD = F if gD is None else gD + F

        gD = smart_conj(gD.unsqueeze(-1) * U.transpose(-1, -2))
        gD = U.matmul(gD)
        return (gD,) + (None,) * 4
