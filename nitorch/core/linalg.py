"""Linear algebra."""
import torch
from . import utils
from warnings import warn
import math


# Expose from private implementation
from ._linalg_expm import expm, _expm
from ._linalg_logm import logm
from ._linalg_qr import eig_sym, eig_sym_


def meanm(mats, max_iter=1024, tol=1e-20):
    """Compute the exponential barycentre of a set of matrices.

    Parameters
    ----------
    mats : (N, M, M) tensor_like[float]
        Set of square invertible matrices
    max_iter : int, default=1024
        Maximum number of iterations
    tol : float, default=1E-20
        Tolerance for early stopping.
        The tolerance criterion is the sum-of-squares of the residuals
        in log-space, _i.e._, :math:`||\sum_n \log_M(A_n) / N||^2`

    Returns
    -------
    mean_mat : (M, M) tensor
        Mean matrix.

    References
    ----------
    .. [1]  Xavier Pennec, Vincent Arsigny.
        "Exponential Barycenters of the Canonical Cartan Connection and
        Invariant Means on Lie Groups."
        Matrix Information Geometry, Springer, pp.123-168, 2012.
        (10.1007/978-3-642-30232-9_7). (hal-00699361)
        https://hal.inria.fr/hal-00699361

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

    # NOTE: all computations are performed in double, else logm is not
    # precise enough

    mats = utils.as_tensor(mats)
    dim = mats.shape[-1] - 1
    dtype = mats.dtype
    device = mats.device
    mats = mats.double()

    mean_mat = torch.eye(dim+1, dtype=torch.double, device=device)
    for n_iter in range(max_iter):
        # Project all matrices to the tangent space about the current mean_mat
        log_mats = lmdiv(mean_mat, mats)
        log_mats = logm(log_mats)
        if log_mats.is_complex():
            warn('`meanm` failed to converge (`logm` -> complex)',
                 RuntimeWarning)
            break
        # Compute the new mean in the tangent space
        mean_log_mat = torch.mean(log_mats, dim=0)
        # Compute sum-of-squares in tangent space (should be zero at optimum)
        sos = mean_log_mat.square().sum()
        # Exponentiate to original space
        mean_mat = torch.matmul(mean_mat, expm(mean_log_mat))
        if sos <= tol:
            break

    return mean_mat.to(dtype)


def lmdiv(a, b, method='lu', rcond=1e-15, out=None):
    r"""Left matrix division ``inv(a) @ b``.

    Parameters
    ----------
    a : (..., m, n) tensor_like
        Left input ("the system")
    b : (..., m, k) tensor_like
        Right input ("the point")
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition.
        * 'pinv' : Moore-Penrose pseudoinverse (by means of svd).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., n, k) tensor
        Solution of the linear system.

    """
    backend = utils.max_backend(a, b)
    a = utils.as_tensor(a, **backend)
    b = utils.as_tensor(b, **backend)
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        # TODO: out keyword
        return torch.solve(b, a)[0]
    elif method.lower().startswith('chol'):
        u = torch.cholesky(a, upper=False)
        return torch.cholesky_solve(b, u, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return v.matmul(u.transpose(-1, -2).matmul(b) / s)
    elif method.lower().startswith('pinv'):
        return torch.pinverse(a, rcond=rcond).matmul(b)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def rmdiv(a, b, method='lu', rcond=1e-15, out=None):
    r"""Right matrix division ``a @ inv(b)``.

    Parameters
    ----------
    a : (..., k, m) tensor_like
        Left input ("the point")
    b : (..., n, m) tensor_like
        Right input ("the system")
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition. ``a`` must be invertible.
        * 'pinv' : Moore-Penrose pseudoinverse
                   (by means of svd with thresholded singular values).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., k, m) tensor
        Solution of the linear system.

    """
    backend = utils.max_backend(a, b)
    a = utils.as_tensor(a, **backend).transpose(-1, -2)
    b = utils.as_tensor(b, **backend).transpose(-1, -2)
    x = lmdiv(b, a, method=method, rcond=rcond).transpose(-1, -2)
    return x


def inv(a, method='lu', rcond=1e-15, out=None):
    r"""Matrix inversion.

    Parameters
    ----------
    a : (..., m, n) tensor_like
        Input matrix.
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition.
        * 'pinv' : Moore-Penrose pseudoinverse (by means of svd).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., n, m) tensor
        Inverse matrix.

    """
    a = utils.as_tensor(a)
    backend = dict(dtype=a.dtype, device=a.device)
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        return torch.inverse(a, out=out)
    elif method.lower().startswith('chol'):
        if a.dim() == 2:
            return torch.cholesky_inverse(a, upper=False, out=out)
        else:
            chol = torch.cholesky(a, upper=False)
            eye = torch.eye(a.shape[-2], **backend)
            return torch.cholesky_solve(eye, chol, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return v.matmul(u.transpose(-1, -2) / s)
    elif method.lower().startswith('pinv'):
        return torch.pinverse(a, rcond=rcond)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def matvec(mat, vec, out=None):
    """Matrix-vector product (supports broadcasting)

    Parameters
    ----------
    mat : (..., M, N) tensor
        Input matrix.
    vec : (..., N) tensor
        Input vector.
    out : (..., M) tensor, optional
        Placeholder for the output tensor.

    Returns
    -------
    mv : (..., M) tensor
        Matrix vector product of the inputs

    """
    mat = torch.as_tensor(mat)
    vec = torch.as_tensor(vec)[..., None]
    if out is not None:
        out = out[..., None]

    mv = torch.matmul(mat, vec, out=out)
    mv = mv[..., 0]
    if out is not None:
        out = out[..., 0]

    return mv


def mdot(a, b):
    """Compute the Frobenius inner product of two matrices

    Parameters
    ----------
    a : (..., N, M) tensor
        Left matrix
    b : (..., N, M) tensor
        Rightmatrix

    Returns
    -------
    dot : (...) tensor
        Matrix inner product

    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/Frobenius_inner_product

    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    mm = torch.matmul(a.conj().transpose(-1, -2), b)
    if a.dim() == b.dim() == 2:
        return mm.trace()
    else:
        return mm.diagonal(0, -1, -2).sum(dim=-1)


def is_orthonormal(basis, return_matrix=False):
    """Check that a basis is an orthonormal basis.

    Parameters
    ----------
    basis : (F, N, [M])
        A basis of a vector or matrix space.
        `F` is the number of elements in the basis.
    return_matrix : bool, default=False
        If True, return the matrix of all pairs of inner products
        between elements if the basis

    Returns
    -------
    check : bool
        True if the basis is orthonormal
    matrix : (F, F) tensor, if `return_matrix is True`
        Matrix of all pairs of inner products

    """
    basis = torch.as_tensor(basis)
    info = dict(dtype=basis.dtype, device=basis.device)
    F = basis.shape[0]
    dot = torch.dot if basis.dim() == 2 else mdot
    mat = basis.new_zeros(F, F)
    for i in range(F):
        mat[i, i] = dot(basis[i], basis[i])
        for j in range(i+1, F):
            mat[i, j] = dot(basis[i], basis[j])
            mat[j, i] = mat[i, j].conj()
    check = torch.allclose(mat, torch.eye(F, **info))
    return (check, mat) if return_matrix else check


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
    mat = utils.movedim(mat, -1, 0)
    vec = utils.movedim(vec, -1, 0)

    if nb_prm == 2:
        mm = mat[:nb_prm] * vec
        mm[0].add_(mat[2] * vec[1])
        mm[1].add_(mat[2] * vec[0])
    elif nb_prm == 3:
        mm = mat[:nb_prm] * vec
        mm[0].add_(mat[3] * vec[1]).add_(mat[4] * vec[2])
        mm[1].add_(mat[3] * vec[0]).add_(mat[5] * vec[2])
        mm[2].add_(mat[4] * vec[0]).add_(mat[5] * vec[1])
    elif nb_prm == 4:
        mm = mat[:nb_prm] * vec
        mm[0] += (mat[4] * vec[1])
        mm[0] += (mat[5] * vec[2])
        mm[0] += (mat[6] * vec[3])
        mm[1] += (mat[4] * vec[0])
        mm[1] += (mat[7] * vec[2])
        mm[1] += (mat[8] * vec[3])
        mm[2] += (mat[5] * vec[0])
        mm[2] += (mat[7] * vec[1])
        mm[2] += (mat[9] * vec[3])
        mm[3] += (mat[6] * vec[0])
        mm[3] += (mat[8] * vec[1])
        mm[3] += (mat[9] * vec[2])
    else:
        mm = mat[:nb_prm] * vec
        c = nb_prm
        for i in range(nb_prm):
            for j in range(i+1, nb_prm):
                mm[i] += mat[c] * vec[j]
                mm[j] += mat[c] * vec[i]
                c += 1

    return utils.movedim(mm, 0, -1)


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
    mat = utils.movedim(mat, -1, 0)
    vec = utils.movedim(vec, -1, 0)
    nb_prm = len(vec)

    shape = utils.expanded_shape(mat.shape[1:], vec.shape[1:])
    shape = (vec.shape[0], *shape)

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
        return utils.movedim(res, 0, -1)
    elif nb_prm == 2:
        det = uppr[0].square().neg_()
        det += diag[0] * diag[1]
        res = vec.new_empty(shape)
        res[0] = diag[1] * vec[0] - uppr[0] * vec[1]
        res[1] = diag[0] * vec[1] - uppr[0] * vec[0]
        res /= det
        return utils.movedim(res, 0, -1)
    elif nb_prm == 3:
        det = diag.prod(0) + 2 * uppr.prod(0) \
            - (diag[0] * uppr[2].square() +
               diag[2] * uppr[0].square() +
               diag[1] * uppr[1].square())
        res = vec.new_empty(shape)
        res[0] = (diag[1] * diag[2] - uppr[2].square()) * vec[0] \
               + (uppr[1] * uppr[2] - diag[2] * uppr[0]) * vec[1] \
               + (uppr[0] * uppr[2] - diag[1] * uppr[1]) * vec[2]
        res[1] = (uppr[1] * uppr[2] - diag[2] * uppr[0]) * vec[0] \
               + (diag[0] * diag[2] - uppr[1].square()) * vec[1] \
               + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * vec[2]
        res[2] = (uppr[0] * uppr[2] - diag[1] * uppr[1]) * vec[0] \
               + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * vec[1] \
               + (diag[0] * diag[1] - uppr[0].square()) * vec[2]
        res /= det
        return utils.movedim(res, 0, -1)
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
        res = vec.new_empty(shape)
        res[0] = (diag[1] * diag[2] * diag[3]
                  - diag[1] * uppr[5].square()
                  - diag[2] * uppr[4].square()
                  - diag[3] * uppr[3].square()
                  + 2 * uppr[3] * uppr[4] * uppr[5]) * vec[0]
        res[0] += inv01 * vec[1]
        res[0] += inv02 * vec[2]
        res[0] += inv03 * vec[3]
        res[1] = (diag[0] * diag[2] * diag[3]
                  - diag[0] * uppr[5].square()
                  - diag[2] * uppr[2].square()
                  - diag[3] * uppr[1].square()
                  + 2 * uppr[1] * uppr[2] * uppr[5]) * vec[1]
        res[1] += inv01 * vec[0]
        res[1] += inv12 * vec[2]
        res[1] += inv13 * vec[3]
        res[2] = (diag[0] * diag[1] * diag[3]
                  - diag[0] * uppr[4].square()
                  - diag[1] * uppr[2].square()
                  - diag[3] * uppr[0].square()
                  + 2 * uppr[0] * uppr[2] * uppr[4]) * vec[2]
        res[2] += inv02 * vec[0]
        res[2] += inv12 * vec[1]
        res[2] += inv23 * vec[3]
        res[3] = (diag[0] * diag[1] * diag[2]
                  - diag[0] * uppr[3].square()
                  - diag[1] * uppr[1].square()
                  - diag[2] * uppr[0].square()
                  + 2 * uppr[0] * uppr[1] * uppr[3]) * vec[3]
        res[3] += inv03 * vec[0]
        res[3] += inv13 * vec[1]
        res[3] += inv23 * vec[2]
        res /= det
        return utils.movedim(res, 0, -1)
    else:
        vec = utils.movedim(vec, 0, -1)
        mat = utils.movedim(mat, 0, -1)
        mat = sym_to_full(mat)
        return torch.solve(vec, mat)


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
