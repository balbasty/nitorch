"""Linear algebra."""
import torch
from . import utils, py
from warnings import warn
import math


# Expose from private implementation
from ._linalg_expm import expm, _expm
from ._linalg_logm import logm
from ._linalg_qr import eig_sym, eig_sym_
from ._linalg_sym import *


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
    _solve_lu = torch.linalg.solve
else:
    _solve_lu = lambda A, b: torch.solve(b, A)[0]


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


def kron2(x, y):
    """Kronecker product of two matrices

    Parameters
    ----------
    x : (..., m, n) tensor
        Left matrix
    y : (..., p, q) tensor
        Right matrix

    Returns
    -------
    xy : (..., p*m, q*n) tensor
        Kronecker product
        `xy.reshape([P, M, Q, N])[p, m, q, n] == x[m, n] * y[n, q]`

    """
    x, y = utils.to_max_backend(x, y)
    *_, m, n = x.shape
    *_, p, q = y.shape
    x = x[..., None, :, None, :]
    y = y[..., :, None, :, None]
    xy = x*y
    xy = xy.reshape([*xy.shape[:-4], m*p, n*q])
    return xy


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
        return _solve_lu(a, b)
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


def outer(a, b, out=None):
    """Outer product of two (batched) tensors

    Parameters
    ----------
    a : (..., N) tensor
    b : (..., M) tensor
    out : (..., N, M) tensor, optional

    Returns
    -------
    out : (..., N, M) tensor

    """
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-2)
    return torch.matmul(a, b)


def dot(a, b, keepdim=False, out=None):
    """(Batched) dot product

    Parameters
    ----------
    a : (..., N) tensor
    b : (..., N) tensor
    keepdim : bool, default=False
    out : tensor, optional

    Returns
    -------
    ab : (..., [1]) tensor

    """
    a = a[..., None, :]
    b = b[..., :, None]
    ab = torch.matmul(a, b, out=out)
    if keepdim:
        ab = ab[..., 0]
    else:
        ab = ab[..., 0, 0]
    return ab


def cholesky(a):
    """Compute the Choleksy decomposition of a positive-definite matrix

    Parameters
    ----------
    a : (..., M, M) tensor
        Positive-definite matrix

    Returns
    -------
    l : (..., M, M) tensor
        Lower-triangular cholesky factor

    """
    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cholesky'):
        return torch.linalg.cholesky(a)
    return torch.cholesky(a)


def trace(a, keepdim=False):
    """Compute the trace of a matrix (or batch)

    Parameters
    ----------
    a : (..., M, M) tensor

    Returns
    -------
    t : (...) tensor

    """
    t = a.diagonal(0, -1, -2).sum(-1)
    if keepdim:
        t = t[..., None, None]
    return t


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


def trapprox(matvec, shape=None, moments=None, samples=10,
             method='rademacher', hutchpp=False, **backend):
    """Stochastic trace approximation (Hutchinson's estimator)

    Parameters
    ----------
    matvec : sparse tensor or callable(tensor) -> tensor
        Function that computes the matrix-vector product
    shape : sequence[int]
        "vector" shape
    moments : int, default=1
        Number of moments
    samples : int, default=10
        Number of samples
    method : {'rademacher', 'gaussian'}, default='rademacher'
        Sampling method
    hutchpp : bool, default=False
        Use Hutch++ instead of Hutchinson.
        /!\ Be aware that it uses more memory.

    Returns
    -------
    trace : ([moments],) tensor

    Reference
    ---------
    ..[1]   "A stochastic estimator ofthe trace of the influence matrix
            for Laplacian smooth-ing  splines"
            Hutchinson
            Communications in  Statistics - Simulation and Computation (1990)
    ..[2]   "Hutch++: Optimal Stochastic Trace Estimation"
            Meyer, Musco, Musco, Woodruff
            Proc SIAM Symp Simplicity Algorithms (2021)

    """
    if torch.is_tensor(matvec):
        mat = matvec
        matvec = lambda x: mat.matmul(x)
        backend.setdefault('dtype', mat.dtype)
        backend.setdefault('device', mat.device)
        shape = [*mat.shape[:-2], mat.shape[-1]]
    else:
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', torch.device('cpu'))
    no_moments = moments is None
    moments = moments or 1

    def rademacher(m=0):
        shape1 = [m, *shape] if m else shape
        x = torch.bernoulli(torch.full([], 0.5, **backend).expand(shape1))
        x.sub_(0.5).mul_(2)
        return x

    def gaussian(m=0):
        shape1 = [m, *shape] if m else shape
        return torch.randn(shape1, **backend)

    samp = rademacher if method[0].lower() == 'r' else gaussian

    if hutchpp:
        samples = int(math.ceil(samples/3))

        def matvecpp(x):
            y = torch.empty_like(x)
            for j in range(samples):
                y[j] = matvec(x[j])
            return y

        def dotpp(x, y):
            d = 0
            for j in range(samples):
                d += x[j].flatten().dot(y[j].flatten())
            return d

        def outerpp(x, y):
            z = x.new_empty([samples, samples])
            for j in range(samples):
                for k in range(samples):
                    z[j, k] = x[j].flatten().dot(y[k].flatten())
            return z

        def mmpp(x, y):
            z = torch.zeros_like(x)
            for j in range(samples):
                for k in range(samples):
                    z[j].addcmul_(x[k], y[k, j])
            return z

        t = torch.zeros([moments], **backend)
        q, g = samp(samples), samp(samples)
        q = torch.qr(matvecpp(q).T, some=True)[0].T
        g -= mmpp(q, outerpp(q, g))
        mq, mg = q, g
        for j in range(moments):
            mq = matvecpp(mq)
            mg = matvecpp(mg)
            t[j] = dotpp(q, mq) + dotpp(g, mg) / samples

    else:
        t = torch.zeros([moments], **backend)
        for i in range(samples):
            m = v = samp()
            for j in range(moments):
                m = matvec(m)
                t[j] += m.flatten().dot(v.flatten())
        t /= samples

    if no_moments:
        t = t[0]
    return t


def vbald(matvec, shape=None, upper=None, moments=5, samples=5, mc_samples=64,
          method='rademacher', **backend):
    """Variational Bayesian Approximation of Log Determinants

    Parameters
    ----------
    matvec : sparse tensor or callable(tensor) -> tensor
        Function that computes the matrix-vector product
    shape : sequence[int]
        "vector" shape
    upper : float
        Upper bound on eigenvalues
    moments : int, default=1
        Number of moments
    samples : int, default=5
        Number of samples for moment estimation
    mc_samples : int, default=64
        Number of samples for Monte Carlo integration
    method : {'rademacher', 'gaussian'}, default='rademacher'
        Sampling method

    Returns
    -------
    logdet : scalar tensor

    Reference
    ---------
    ..[1]   "VBALD - Variational Bayesian Approximation of Log Determinants"
            Granziol, Roberts & Osborne
            https://arxiv.org/abs/1802.08054
    """
    if torch.is_tensor(matvec):
        mat = matvec
        matvec = lambda x: mat.matmul(x)
        backend.setdefault('dtype', mat.dtype)
        backend.setdefault('device', mat.device)
        shape = [*mat.shape[:-2], mat.shape[-1]]
    else:
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', torch.device('cpu'))
    numel = py.prod(shape)

    if not upper:
        upper = maxeig_power(matvec, shape)
    matvec2 = lambda x: matvec(x).div_(upper)
    mom = trapprox(matvec2, shape, moments=moments, samples=samples,
                   method=method, **backend).cpu()
    mom /= numel

    # Compute beta parameters (Maximum Likelihood)
    alpha = mom[0] * (mom[0] - mom[1]) / (mom[1] - mom[0]**2)
    beta = alpha * (1/mom[0] - 1)
    if alpha > 0 and beta > 0:
        prior = torch.distributions.Beta(alpha.item(), beta.item())
    else:
        prior = torch.distributions.Uniform(1e-8, 1)

    # Compute coefficients
    coeff = _vbald_gn(mom, mc_samples, prior)

    # logdet(A) = N * (E[log(lam)] + log(upper))
    logdet = _vbald_mc_log(coeff, mc_samples, prior)
    logdet = numel * (logdet + math.log(upper))
    return logdet.to(backend['device'])


def _vbald_gn(mom, samples, prior, tol=1e-6, max_iter=512):
    dot = lambda u, v: u.flatten().dot(v.flatten())
    coeff = torch.zeros_like(mom)
    for n_iter in range(max_iter):
        loss, grad, hess = _vbald_mc(coeff, samples, prior,
                                     gradient=True, hessian=True)
        loss += dot(coeff, mom)
        grad = mom - grad
        diag = hess.diagonal(0, -1, -2)
        diag += 1e-3 * diag.abs().max() * torch.rand_like(diag)
        delta = lmdiv(hess, grad)

        success = False
        armijo = 1
        loss0 = loss
        coeff0 = coeff
        for n_iter in range(12):
            coeff = coeff0 - armijo * delta
            loss = _vbald_mc(coeff, samples, prior)
            loss += dot(coeff, mom)
            if loss < loss0:
                success = True
                break
            armijo /= 2
        if not success:
            return coeff0

        gain = abs(loss - loss0)
        if gain < tol:
            break
    return coeff


def _vbald_mc(coeff, samples, prior, gradient=False, hessian=False):
    nprm = 1
    if gradient:
        nprm += len(coeff)
    if hessian:
        nprm += len(coeff)

    # compute \int q(lam) * lam**j * exp(-1 - \sum coeff[i] lam**i) dlam
    # for multiple k, using monte carlo integration.
    s = coeff.new_zeros([nprm])
    for i in range(samples):
        lam = prior.sample([])
        q = _vbald_factexp(lam, coeff)
        s[0] += q
        if len(s) > 1:
            for j in range(1, len(s)):
                q = q * lam
                s[j] += q
    s /= samples

    # compute gradient and Hessian from the above integrals
    if gradient:
        g = s[1:len(coeff)+1]
        if hessian:
            h = g.new_zeros(len(coeff), len(coeff))
            for j in range(len(coeff)):
                for k in range(j+1, len(coeff)):
                    h[j, k] = h[k, j] = s[1+j+k]
                h[j, j] = s[1+j+j]
            return s[0], g, h
        return s[0], g
    return s[0]


def _vbald_factexp(lam, coeff):
    lam = lam ** torch.arange(1, len(coeff)+1, dtype=lam.dtype, device=lam.device)
    dot = lambda u, v: u.flatten().dot(v.flatten())
    return (-1 - dot(lam, coeff)).exp()


def _vbald_mc_log(coeff, samples, prior):

    # compute \int q(lam) * log(lam) * exp(-1 - \sum coeff[i] lam**i) dlam
    # for multiple k, using monte carlo integration.
    s = 0
    for i in range(samples):
        lam = prior.sample([])
        s += lam.log() * _vbald_factexp(lam, coeff)
    s /= samples
    return s


def maxeig_power(matvec, shape=None, max_iter=512, tol=1e-6, **backend):
    """Estimate the maximum eigenvalue of a matrix by power iteration

    Parameters
    ----------
    matvec : sparse tensor or callable(tensor) -> tensor
        Function that computes the matrix-vector product
    shape : sequence[int]
        "vector" shape
    max_iter : int, default=512
    tol : float, default=1e-6

    Returns
    -------
    maxeig : scalar tensor
        Largest eigenvalue

    """

    if torch.is_tensor(matvec):
        mat = matvec
        matvec = lambda x: mat.matmul(x)
        backend.setdefault('dtype', mat.dtype)
        backend.setdefault('device', mat.device)
        shape = [*mat.shape[:-2], mat.shape[-1]]
    else:
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', torch.device('cpu'))

    dot = lambda u, v: u.flatten().dot(v.flatten())

    v = torch.bernoulli(torch.full([], 0.5, **backend).expand(shape)).sub_(0.5).mul_(2)
    mu = float('inf')

    for n_iter in range(max_iter):
        w, v = v, matvec(v)
        mu0, mu = mu, dot(w, v)
        v /= dot(v, v).sqrt_()
        if abs(mu - mu0) < tol:
            break
    return mu