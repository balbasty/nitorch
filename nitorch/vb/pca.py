"""Various flavors of component/factor analysis"""
import torch
from nitorch.core import utils, linalg, math

# TODO: the convention is not the same between pca and ppca in terms
#   of which factor is scaled and which is unitary.
#   Currently, pca returns a unitary basis whereas ppca returns
#   unitary latent coordinates.
#   We could have an option to specify which side should be normalized?
#   Or return the (diagonal) covariance matrix as well?


def pca(x, nb_components=None, mean=None,
        returns='latent+basis+scale', norm='latent+basis'):
    """Principal Component Analysis

    Factorize a NxM matrix X into the product ZSU where Z is NxK and
    unitary, U is KxM and unitary, and S is KxK and diagonal.

    By convention, N encodes independent replicates (individuals or samples)
    and M encodes correlated features, although in practice the problem
    is symmetric. Following probabilistic conventions, we say that
    each sample (X[n]) is encoded by "latent" coordinates (Z[n]) in an
    orthogonal "basis" (U).

    This function merely applies a singular value decomposition (SVD)
    under the hood.

    Parameters
    ----------
    x : (..., n, m) tensor_like or sequence[callable]
        Observed variables.
        `n` is the number of independent variables and `m` their dimension.
        If a sequence of callable, a memory efficient (but slower)
        implementation is used, where tensors are loaded by calling the
        corresponding callable when needed.
    nb_components : int, default=`min(n, m)`
        Number of principal components (k) to return.
    mean : float or (..., m) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, subtract the mean of all observations.
        If 0, nothing is done.
    returns : combination of {'latent', 'basis', 'scale'}, default='latent+basis+scale'
        Which variables to return.
    norm : {'latent', 'basis', 'latent+basis', None}, default='latent+basis'
        Which variable to normalize. If normalized, the corresponding
        matrix is unitary (Z @ Z.T == I).
        If 'latent+basis', a tensor of scales is returned.

    Returns
    -------
    latent : (..., n, k), if 'latent' in `returns`
    basis : (..., k, m), if 'basis' in `returns`
    scale : (..., k), if 'scale' in `returns`

    """
    if isinstance(x, (list, tuple)) and callable(x[0]):
        return _pca_callable(x, nb_components, mean, returns, norm)
    else:
        return _pca_tensor(x, nb_components, mean, returns, norm)


def _pca_tensor(x, k, mu, returns, norm):
    """Classic implementation: subtract mean and call SVD."""

    x = torch.as_tensor(x)
    if mu is None:
        mu = torch.mean(x, dim=-2)
    nomu = isinstance(mu, (int, float)) and mu == 0
    mu = torch.as_tensor(mu, **utils.backend(x))
    if not nomu:
        x = x - mu[..., None, :]

    z, s, u = torch.svd(x, some=True)

    if k:
        if k > min(x.shape[-1], x.shape[-2]):
            raise ValueError('Number of components cannot be larger '
                             'than min(N,M)')
        z = z[..., k]
        u = u[..., k]
        s = s[..., k]

    if 'latent' not in norm:
        z.mul_(s[..., None, :])
    if 'basis' not in norm:
        u.mul_(s[..., None, :])
    u = u.transpose(-1, -2)

    out = []
    returns = returns or ''
    for var in returns.split('+'):
        if var == 'latent':
            out.append(z)
        elif var == 'basis':
            out.append(u)
        elif var == 'scale':
            out.append(s)
    return out[0] if len(out) == 1 else tuple(out)


def _pca_callable(x, k, mu, returns, norm):
    """Implementation that loads tensors one at a time.
    1) Compute the NxN covariance matrix
    2) Use SVD to compute the NxK latent vectors
    3) Compute the KxM basis by projection (= matmul by pseudoinversed latent)
    """

    x = list(x)
    n = len(x)

    if callable(mu):
        mu = mu()
    nomu = isinstance(mu, (int, float)) and mu == 0
    if mu is not None:
        mu = torch.as_tensor(mu)

    # infer output shape/dtype/device
    shape = [mu.shape] if mu is not None else []
    dtype = [mu.dtype] if mu is not None else []
    device = [mu.device] if mu is not None else []
    if mu is None:
        mu = 0
    for x1 in x:
        x1 = torch.as_tensor(x1())
        mu += x1
        shape.append(tuple(x1.shape))
        dtype.append(x1.dtype)
        device.append(x1.device)
    mu /= n
    shape = list(utils.expanded_shape(*shape))
    m = shape.pop(-1)
    dtype = utils.max_dtype(dtype)
    device = utils.max_device(device)
    backend = dict(dtype=dtype, device=device)
    mu = mu.to(**backend)

    if k and k > min(n, m):
        raise ValueError('Number of components cannot be larger '
                         'than min(N,M)')
    k = k or min(n, m)

    # build NxN covariance matrix
    cov = torch.empty([*shape, n, n])
    for n1 in range(n):
        x1 = torch.as_tensor(x[n1](), **backend)
        if not nomu:
            x1 = x1 - mu
        cov[..., n1, n1] = x1.square().sum(-1)
        for n2 in range(n1+1, n):
            x2 = torch.as_tensor(x[n2](), **backend)
            if not nomu:
                x2 = x2 - mu
            x2 = x2.mul(x1).sum(-1)
            cov[..., n1, n2] = x2
            cov[..., n2, n1] = x2

    # compute svd
    z, s, _ = torch.svd(cov, some=True)  # [..., n, k]
    s = s.sqrt_()
    z = z[..., :k]
    s = s[..., :k]

    if 'basis' in returns:
        # build basis by projection
        iz = torch.pinverse(z * s[..., None, :])
        u = iz.new_zeros([*shape, k, m])
        for n1 in range(n):
            x1 = torch.as_tensor(x[n1](), **backend)
            if not nomu:
                x1 -= mu
            u += iz[..., :, n1, None] * x1[..., None, :]

        if 'basis' not in norm:
            u *= s[..., None]

    if 'latent' not in norm:
        z *= s[..., None, :]

    out = []
    returns = returns or ''
    for var in returns.split('+'):
        if var == 'latent':
            out.append(z)
        elif var == 'basis':
            out.append(u)
        elif var == 'scale':
            out.append(s)
    return out[0] if len(out) == 1 else tuple(out)


def ppca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False, rca=None):
    """Probabilistic Principal Component Analysis

    Notes
    -----
    .. We manually orthogonalize the subspace within the optimization
       loop so that the output subspace is orthogonal (`z.T @ z` and
       `u @ u.T` are diagonal).
    .. The output basis is not unitary. Each basis is scaled by
       the square root of the corresponding eigenvalue of the sample
       covariance minus the residual variance:
            basis = unitary_basis * sqrt(lambda - sigma ** 2)
       See reference [1].
    .. Probabilistic residual component analysis (RCA) can be performed
       instead of PCA by providing a function that applies the residual
       precision matrix. See reference [2].

    Parameters
    ----------
    x : (..., n, m) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `M` their dimension.
        If a sequence of callable, a memory efficient implementation is
        used, where tensors are loaded by calling the corresponding callable
        when needed.
    nb_components : int, default=min(n, m)-1
        Number of principal components (k) to return.
    mean : float or (..., m) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, use the mean of all observations (maximum-likelihood).
        If 0, nothing is done.
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-5
        Tolerance on log model evidence for early stopping.
    returns : {'latent', 'basis', 'var'}, default='latent+basis+var'
        Which variables to return.
    verbose : {0, 1, 2}, default=0
    rca : callable, optional
        A function (..., m) -> (..., m) that applies a residual precision
        matrix for residual component analysis.

    Returns
    -------
    latent : (..., n, k), if 'latent' in `returns`
        Latent coordinates
    basis : (..., k, m), if 'basis' in `returns`
        Orthogonal basis, scaled by sqrt(eigenvalue - residual variance)
    var : (...), if 'var' in `returns`
        Residual variance

    References
    ----------
    ..[1] "Probabilistic principal component analysis."
          Tipping, Michael E. and Bishop, Christopher M.
          J. R. Stat. Soc., Ser. B (1999)
    ..[2] "Residual Component Analysis: Generalising PCA for more
          flexible inference in linear-Gaussian models."
          Kalaitzis, Alfredo A. and Lawrence, Neil D.
          ICML (2012)

    """

    if isinstance(x, (list, tuple)) and callable(x[0]):
        return _ppca_callable(x, nb_components, mean, max_iter, tol,
                              returns, verbose, rca)
    else:
        return _ppca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose, rca)


def _ppca_tensor(x, k, mu, max_iter, tol, returns, verbose=False, rca=None):
    """Implementation that assumes that all the data is in memory."""

    # --- preproc ---

    x = torch.as_tensor(x)
    n, m = x.shape[-2:]
    backend = utils.backend(x)
    k = k or (min(x.shape[-1], x.shape[-2]) - 1)
    eps = 1e-6
    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    # subtract mean
    nomu = isinstance(mu, (float, int)) and mu == 0
    if mu is None:
        mu = x.mean(-2)
    mu = torch.as_tensor(mu, **backend)
    if not nomu:
        x = x - mu.unsqueeze(-2)

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_diag(x):
        """Quick create diagonal matrix"""
        return torch.diagonal(x, dim1=-2, dim2=-1)

    def trace(x, **kwargs):
        """Batched trace"""
        return get_diag(x).sum(-1, **kwargs)

    def make_sym(x):
        """Make a matrux symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def reg(x, s):
        """Regularize matrix by adding number on the diagonal"""
        return reg_(x.clone(), s)

    def reg_(x, s):
        """Regularize matrix by adding number on the diagonal (inplace)"""
        get_diag(x).add_(s[..., None])
        return x

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        return linalg.inv(x.double()).to(dtype=dtype)

    def rinv(z, s, side='l'):
        """Regularized pseudo-inverse
        z : (..., N, K) matrix to invert
        s : (...) weight
        side : {'l', 'r'}
        returns (..., K, N) -> (z.T @ z + s * I).inv() @ z.T, if 'l'
                (..., N, K) -> (z @ z.T + s * I).inv() @ z,   if 'r'
        """
        if side[0] == 'l':
            zz = make_sym(t(z).matmul(z))
            zz = inv(reg_(zz, s))
            z = zz.matmul(t(z))
        else:
            zz = make_sym(z.matmul(t(z)))
            zz = inv(reg_(zz, s))
            z = zz.matmul(z)
        return z

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    def rescale(uu, s):
        """Rescale after orthonormalization to optimize log-evidence.
        uu : (*batch, K, K) - basis product (u @ u.T) !!must be diagonal!!
        s  : (*batch) - Residual variance
        """
        # The objective function that I optimize here is the one computed
        # in `logev`, so it takes into account an "immediate" update
        # of the posterior covariance. This means that the scaling is
        # only applied to the basis u (and to the mean of z), and the latent
        # covariance is immediately updated according to:
        #                   Sz = inv(inv_scale(uu) + s*I)
        # In the shape (& appearance) papers, we kept the posterior
        # covariance fixed (under VB) and scaled it along with the mean:
        #                   z = scale(z)
        #                   Sz = scale(Sz)
        a = s[..., None] / get_diag(uu)
        scl = (1 + (1 + 4 * a * n).sqrt()) / (2*n)
        return scl.reciprocal_().sqrt_()

    def logev(r, z, uu, s):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        z  : (*batch, N, K) - latent variables
        uu : (*batch, K, K) - basis product (u @ u.T)
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z inside the function (with
        # the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = z.square().sum([-1, -2])
        z = z.sum()
        # uncertainty
        unc = reg(uu, s).logdet() - s.log() * k
        unc = unc.sum() * n
        # log sigma
        s = s.log().sum() * (n * m)
        tot = (r + z + s + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{s.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    # --- initialization ---

    # init residual var with 10% of full var
    if has_rca:
        s = (x * rca(x)).mean([-1, -2]).mul_(0.1)
    else:
        s = x.square().mean([-1, -2]).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*x.shape[:-1], k], **backend)
    z, _, _ = torch.svd(z, some=True)

    # init basis
    iz = rinv(z, s, 'l')
    u = iz.matmul(x)
    uu = make_sym(u.matmul(t(rca(u))))

    # init log-evidence
    if has_rca:
        r = (x - z.matmul(u))
        r = (r * rca(r)).sum([-1, -2])
    else:
        r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, z, uu, s)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = x.matmul(t(rca(u))).matmul(im)              # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        tiny = eps * get_diag(zz).abs().max(-1).values
        sz = im * s[..., None, None].clamp_min(tiny)    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        u = inv(zz).matmul(t(z)).matmul(x)
        uu = make_sym(u.matmul(t(rca(u))))

        # update sigma
        sz = s * inv(reg(uu, s))
        if has_rca:
            r = (x - z.matmul(u))
            r = (r * rca(r)).sum([-1, -2])
        else:
            r = (x - z.matmul(u)).square_().sum([-1, -2])
        s = r / (n*m) + trace(sz.matmul(uu)) / m     # residuals + uncertainty

        # orthogonalize
        zz = make_sym(t(z).matmul(z))
        q, iq = joint_ortho(zz, uu)
        scl = rescale(iq.matmul(uu).matmul(t(iq)), s)
        q *= scl[..., None, :]
        iq /= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update log-evidence
        l = logev(r, z, uu, s)
        gain = (l1-l)/(l0 - l)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end=end)
        if abs(gain) < tol:
            break
        l1 = l
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'var':
            out.append(s)
    return out[0] if len(out) == 0 else tuple(out)


def _ppca_callable(x, k, mu, max_iter, tol, returns, verbose=False, rca=None):
    """Inline implementation that loads data only when needed."""

    # --- preproc ---

    x = list(x)
    n = len(x)

    if callable(mu):
        mu = mu()
    nomu = isinstance(mu, (int, float)) and mu == 0
    if mu is not None:
        mu = torch.as_tensor(mu)

    # infer output shape/dtype/device
    shape = [mu.shape] if mu is not None else []
    dtype = [mu.dtype] if mu is not None else []
    device = [mu.device] if mu is not None else []
    if mu is None:
        mu = 0
    for x1 in x:
        x1 = torch.as_tensor(x1())
        mu += x1
        shape.append(tuple(x1.shape))
        dtype.append(x1.dtype)
        device.append(x1.device)
    mu /= n
    shape = list(utils.expanded_shape(*shape))
    m = shape.pop(-1)
    dtype = utils.max_dtype(dtype)
    device = utils.max_device(device)
    backend = dict(dtype=dtype, device=device)
    mu = mu.to(**backend)

    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    k = k or (min(n, m) - 1)
    eps = 1e-6

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_diag(x):
        """Quick create diagonal matrix"""
        return torch.diagonal(x, dim1=-2, dim2=-1)

    def trace(x, **kwargs):
        """Batched trace"""
        return get_diag(x).sum(-1, **kwargs)

    def make_sym(x):
        """Make a matrux symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def reg(x, s):
        """Regularize matrix by adding number on the diagonal"""
        return reg_(x.clone(), s)

    def reg_(x, s):
        """Regularize matrix by adding number on the diagonal (inplace)"""
        get_diag(x).add_(s[..., None])
        return x

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        return linalg.inv(x.double()).to(dtype=dtype)

    def rinv(z, s, side='l'):
        """Regularized pseudo-inverse
        z : (..., N, K) matrix to invert
        s : (...) weight
        side : {'l', 'r'}
        returns (..., K, N) -> (z.T @ z + s * I).inv() @ z.T, if 'l'
                (..., N, K) -> (z @ z.T + s * I).inv() @ z,   if 'r'
        """
        if side[0] == 'l':
            zz = make_sym(t(z).matmul(z))
            zz = inv(reg_(zz, s))
            z = zz.matmul(t(z))
        else:
            zz = make_sym(z.matmul(t(z)))
            zz = inv(reg_(zz, s))
            z = zz.matmul(z)
        return z

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    def rescale(uu, s):
        """Rescale after orthonormalization to optimize log-evidence.
        uu : (*batch, K, K) - basis product (u @ u.T) !!must be diagonal!!
        s  : (*batch) - Residual variance
        """
        # The objective function that I optimize here is the one computed
        # in `logev`, so it takes into account an "immediate" update
        # of the posterior covariance. This means that the scaling is
        # only applied to the basis u (and to the mean of z), and the latent
        # covariance is immediately updated according to:
        #                   Sz = inv(inv_scale(uu) + s*I)
        # In the shape (& appearance) papers, we kept the posterior
        # covariance fixed (under VB) and scaled it along with the mean:
        #                   z = scale(z)
        #                   Sz = scale(Sz)
        a = s[..., None] / get_diag(uu)
        scl = (1 + (1 + 4 * a * n).sqrt()) / (2*n)
        return scl.reciprocal_().sqrt_()

    def logev(r, z, uu, s):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        z  : (*batch, N, K) - latent variables
        uu : (*batch, K, K) - basis product (u @ u.T)
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z inside the function (with
        # the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = z.square().sum([-1, -2])
        z = z.sum()
        # uncertainty
        unc = reg(uu, s).logdet() - s.log() * k
        unc = unc.sum() * n
        # log sigma
        s = s.log().sum() * (n * m)
        tot = (r + z + s + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{s.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    def matmul(x, y, out=None):
        """Matmul where one of the inputs is a list of callable"""
        if isinstance(x, list):
            if out is None:
                out = y.new_empty(n, y.shape[-1])
            for i, x1 in enumerate(x):
                x1 = x1()
                if not nomu:
                    x1 -= mu
                out[..., i, :] = x1[..., None, :].matmul(y)[..., 0, :]
        elif isinstance(y, list):
            if out is None:
                out = x.new_empty(x.shape[-2], m)
            out.zero_()
            for i, y1 in enumerate(y):
                y1 = y1()
                if not nomu:
                    y1 -= mu
                out += x[..., i, None] * y1[..., None]
        return out

    def get_sqres(x, z, u, out=None):
        """Compute sum of squared residuals"""
        if out is None:
            out = z.new_empty(shape)
        out.zero_()
        for i, x1 in enumerate(x):
            recon = (z[..., n, :, None] * u).sum(-2)
            x1 = x1()
            if not nomu:
                x1 -= mu
            x1 -= recon
            if has_rca:
                out += (x1 * rca(x1)).sum(-1)
            else:

                out += x1.square_().sum(-1)
        return out

    def var(x):
        out = torch.zeros(shape, **backend)
        for x1 in x:
            x1 = x1()
            if not nomu:
                x1 -= mu
            if has_rca:
                out += (x1 * rca(x1)).sum(-1)
            else:
                out += x1.square_().sum(-1)
        out /= (n*m)
        return out

    # --- initialization ---

    # init residual var with 10% of full var
    s = var(x).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*shape, n, k], **backend)
    z, _, _ = torch.svd(z, some=True)

    # init basis
    iz = rinv(z, s, 'l')
    u = matmul(iz, x)
    uu = make_sym(u.matmul(t(rca(u))))

    # init log-evidence
    r = get_sqres(x, z, u)
    l0 = l1 = logev(r, z, uu, s)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = matmul(x, t(rca(u)), out=z).matmul(im)      # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        tiny = eps * get_diag(zz).abs().max(-1).values
        sz = im * s[..., None, None].clamp_min(tiny)    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        u = matmul(inv(zz).matmul(t(z)), x, out=u)
        uu = make_sym(u.matmul(t(rca(u))))

        # update sigma
        sz = s * inv(reg(uu, s))
        r = get_sqres(x, z, u, out=r)
        s = r / (n*m) + trace(sz.matmul(uu)) / m     # residuals + uncertainty

        # orthogonalize
        zz = make_sym(t(z).matmul(z))
        q, iq = joint_ortho(zz, uu)
        scl = rescale(iq.matmul(uu).matmul(t(iq)), s)
        q *= scl[..., None, :]
        iq /= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update log-evidence
        l = logev(r, z, uu, s)
        gain = (l1-l)/(l0 - l)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end=end)
        if abs(gain) < tol:
            break
        l1 = l
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'var':
            out.append(s)
    return out[0] if len(out) == 0 else tuple(out)


def vpca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False, rca=None):
    """Variational Principal Component Analysis

    Notes
    -----
    .. We manually orthogonalize the subspace within the optimization
       loop so that the output subspace is orthogonal (`E[z.T @ z]` and
       `E[u @ u.T]` are diagonal).
    .. The output basis is not unitary.
    .. Variational residual component analysis (RCA) can be performed
       instead of PCA by providing a function that applies the residual
       precision matrix. See reference [2].

    Parameters
    ----------
    x : (..., n, m) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `M` their dimension.
        If a sequence of callable, a memory efficient implementation is
        used, where tensors are loaded by calling the corresponding callable
        when needed.
    nb_components : int, default=min(n, m)-1
        Number of principal components (k) to return.
    mean : float or (..., m) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, use the mean of all observations (maximum-likelihood).
        If 0, nothing is done.
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-5
        Tolerance on log model evidence for early stopping.
    returns : {'latent', 'basis', 'var'}, default='latent+basis+var'
        Which variables to return.
    verbose : {0, 1, 2}, default=0
    rca : callable, optional
        A function (..., m) -> (..., m) that applies a residual precision
        matrix for residual component analysis.

    Returns
    -------
    latent : (..., n, k), if 'latent' in `returns`
        Latent coordinates
    basis : (..., k, m), if 'basis' in `returns`
        Orthogonal basis, scaled by sqrt(eigenvalue - residual variance)
    var : (...), if 'var' in `returns`
        Residual variance

    References
    ----------
    ..[1] "Variational principal components."
          Bishop, Christopher M.
          ICANN (1999)
    ..[2] "Residual Component Analysis: Generalising PCA for more
          flexible inference in linear-Gaussian models."
          Kalaitzis, Alfredo A. and Lawrence, Neil D.
          ICML (2012)

    """
    if isinstance(x, (list, tuple)) and callable(x[0]):
        raise NotImplementedError
        # TODO
        # return _vpca_callable(x, nb_components, mean, max_iter, tol,
        #                       returns, verbose, rca)
    else:
        return _vpca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose, rca)


def _vpca_tensor(x, k, mu, max_iter, tol, returns, verbose=False, rca=None):
    """Implementation that assumes that all the data is in memory."""

    # --- preproc ---

    x = torch.as_tensor(x)
    n, m = x.shape[-2:]
    backend = utils.backend(x)
    k = k or (min(x.shape[-1], x.shape[-2]) - 1)
    eps = 1e-6
    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    # subtract mean
    nomu = isinstance(mu, (float, int)) and mu == 0
    if mu is None:
        mu = x.mean(-2)
    mu = torch.as_tensor(mu, **backend)
    mu = mu.unsqueeze(-2)
    if not nomu:
        x = x - mu

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_diag(x):
        """Quick create diagonal matrix"""
        return torch.diagonal(x, dim1=-2, dim2=-1)

    def trace(x, **kwargs):
        """Batched trace"""
        return get_diag(x).sum(-1, **kwargs)

    def make_sym(x):
        """Make a matrux symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def reg(x, s):
        """Regularize matrix by adding number on the diagonal"""
        return reg_(x.clone(), s)

    def reg_(x, s):
        """Regularize matrix by adding number on the diagonal (inplace)"""
        get_diag(x).add_(s[..., None])
        return x

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        x = x.double()
        scl = get_diag(x).abs().max(-1)[0].mul_(1e-5)
        get_diag(x).add(scl[..., None])
        return linalg.inv(x).to(dtype=dtype)

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    def logev(r, zz, uu, s, a, uuzz):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        zz : (*batch, K, K) - latent product (E[z @ z.T])
        uu : (*batch, K, K) - basis product (E[u @ u.T])
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z and u inside the function
        # (with the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = trace(zz).sum()                         # this should be n*k if optimal
        u = trace(uu.matmul(a)).sum()               # this should be m*k if optimal
        # uncertainty
        unc = ((trace(uu.matmul(zz)) - uuzz)/s).sum()   # uncertainty in likelihood
        az = uu/s[..., None, None]
        get_diag(az).add_(1)
        unc += az.logdet().sum() * n                # -E[log q(z)] in KL
        au = zz/s[..., None, None]
        au += a
        unc += au.logdet().sum() * m                # -E[log q(u)] in KL
        # log sigma
        s = s.log().sum() * (n * m)
        # log prior
        a = -a.logdet().sum() * m
        tot = (r + z + u + s + a + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{u.item() / (n*m):6f} | {s.item() / (n*m):6f} | '
        #       f'{a.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    # --- initialization ---

    # init residual var with 10% of full var
    if has_rca:
        s = (x * rca(x)).mean([-1, -2]).mul_(0.1)
    else:
        s = x.square().mean([-1, -2]).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*x.shape[:-1], k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))
    zz += torch.eye(k, **backend) * n

    # init basis
    im = inv(reg(zz, s))
    u = im.matmul(t(z)).matmul(x)
    uu = make_sym(u.matmul(t(rca(u))))
    uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
    su = im * s[..., None, None]
    uu += m * su
    a = inv(uu/m)

    # init log-evidence
    if has_rca:
        r = (x - z.matmul(u))
        r = (r * rca(r)).sum([-1, -2])
    else:
        r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, zz, uu, s, a, uuzz)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = x.matmul(t(rca(u))).matmul(im)              # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        sz = im * s[..., None, None]                    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        im = inv(zz + a*s[..., None, None])
        u = im.matmul(t(z)).matmul(x)
        uu = make_sym(u.matmul(t(rca(u))))
        uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
        su = im * s[..., None, None]
        uu += m * su

        # update sigma
        if has_rca:
            r = (x - z.matmul(u))
            r = (r * rca(r)).sum([-1, -2])
        else:
            r = (x - z.matmul(u)).square_().sum([-1, -2])
        s = r + trace(uu.matmul(zz)) - uuzz
        s /= (n*m)

        # orthogonalize (jointly)
        # For rescaling, writing out the terms that depend on it and
        # assuming E[zz], E[uu], Sz, Su diagonal (which they are after
        # joint diagonalization) and that A is immediately ML-updated shows
        # that the optimal scaling makes E[zz] an identity matrix.
        q, iq = joint_ortho(zz, uu)
        zz = t(q).matmul(zz).matmul(q)
        scl = get_diag(zz).div(n).sqrt_()
        zz = torch.eye(k, **backend).mul_(n)
        q /= scl[..., None, :]
        iq *= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update A
        a = inv(uu / m)

        # update log-evidence
        l = logev(r, zz, uu, s, a, uuzz)
        gain = (l1-l)/(l0 - l)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end=end)
        if abs(gain) < tol:
            break
        l1 = l
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'var':
            out.append(s)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)


def _vmpca_tensor(x, k, l, mu=None, max_iter=100, tol=1e-5,
                  returns='latent+basis', verbose=False, rca=None):
    """ Variational mixture of PCA
    Implementation that assumes that all the data is in memory.
    """

    # --- preproc ---

    x = torch.as_tensor(x)
    n, m = x.shape[-2:]
    backend = utils.backend(x)
    k = k or (min(x.shape[-1], x.shape[-2]) - 1)
    has_rca = bool(rca)
    rca = rca or (lambda x: x)

    # subtract mean
    nomu = isinstance(mu, (float, int)) and mu == 0
    if mu is None:
        mu = x.mean(-2)
    mu = torch.as_tensor(mu, **backend)
    mu = mu.unsqueeze(-2)
    if not nomu:
        x = x - mu

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_diag(x):
        """Quick create diagonal matrix"""
        return torch.diagonal(x, dim1=-2, dim2=-1)

    def trace(x, **kwargs):
        """Batched trace"""
        return get_diag(x).sum(-1, **kwargs)

    def make_sym(x):
        """Make a matrux symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def reg(x, s):
        """Regularize matrix by adding number on the diagonal"""
        return reg_(x.clone(), s)

    def reg_(x, s):
        """Regularize matrix by adding number on the diagonal (inplace)"""
        get_diag(x).add_(s[..., None])
        return x

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        x = x.double()
        scl = get_diag(x).abs().max(-1)[0].mul_(1e-5)
        get_diag(x).add(scl[..., None])
        return linalg.inv(x).to(dtype=dtype)

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    def logev(r, zz, uu, s, a, uuzz):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        zz : (*batch, K, K) - latent product (E[z @ z.T])
        uu : (*batch, K, K) - basis product (E[u @ u.T])
        s  : (*batch) - Residual variance
        """
        # It is not exactly computed in a EM fashion because we
        # compute the posterior covariance of z and u inside the function
        # (with the most recent sigma) even though sigma was updated while
        # assuming the posterior covariance fixed.
        r = (r/s).sum()
        z = trace(zz).sum()                         # this should be n*k if optimal
        u = trace(uu.matmul(a)).sum()               # this should be m*k if optimal
        # uncertainty
        unc = ((trace(uu.matmul(zz)) - uuzz)/s).sum()   # uncertainty in likelihood
        az = uu/s[..., None, None]
        get_diag(az).add_(1)
        unc += az.logdet().sum() * n                # -E[log q(z)] in KL
        au = zz/s[..., None, None]
        au += a
        unc += au.logdet().sum() * m                # -E[log q(u)] in KL
        # log sigma
        s = s.log().sum() * (n * m)
        # log prior
        a = -a.logdet().sum() * m
        tot = (r + z + u + s + a + unc)
        # print(f'{r.item() / (n*m):6f} | {z.item() / (n*m):6f} | '
        #       f'{u.item() / (n*m):6f} | {s.item() / (n*m):6f} | '
        #       f'{a.item() / (n*m):6f} | {unc.item() / (n*m):6f} | '
        #       f'{tot.item() / (n*m):6f}')
        return 0.5 * tot / (n*m)

    # --- initialization ---

    # init residual var with 10% of full var
    if has_rca:
        s = (x * rca(x)).mean([-1, -2]).mul_(0.1)
    else:
        s = x.square().mean([-1, -2]).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*x.shape[:-1], k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))
    zz += torch.eye(k, **backend) * n

    # init basis
    im = inv(reg(zz, s))
    u = im.matmul(t(z)).matmul(x)
    uu = make_sym(u.matmul(t(rca(u))))
    uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
    su = im * s[..., None, None]
    uu += m * su
    a = inv(uu/m)

    # init log-evidence
    if has_rca:
        r = (x - z.matmul(u))
        r = (r * rca(r)).sum([-1, -2])
    else:
        r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, zz, uu, s, a, uuzz)
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {l0.item():6f}', end=end)

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg(uu, s))
        z = x.matmul(t(rca(u))).matmul(im)              # < E[Z]
        zz = make_sym(t(z).matmul(z))                   # < E[Z].T @ E[Z]
        sz = im * s[..., None, None]                    # < Cov[Z[n]]
        zz += n * sz                                    # < E[Z.T @ Z]

        # update basis
        im = inv(zz + a*s[..., None, None])
        u = im.matmul(t(z)).matmul(x)
        uu = make_sym(u.matmul(t(rca(u))))
        uuzz = trace(uu.matmul(make_sym(t(z).matmul(z))))
        su = im * s[..., None, None]
        uu += m * su

        # update sigma
        if has_rca:
            r = (x - z.matmul(u))
            r = (r * rca(r)).sum([-1, -2])
        else:
            r = (x - z.matmul(u)).square_().sum([-1, -2])
        s = r + trace(uu.matmul(zz)) - uuzz
        s /= (n*m)

        # orthogonalize (jointly)
        # For rescaling, writing out the terms that depend on it and
        # assuming E[zz], E[uu], Sz, Su diagonal (which they are after
        # joint diagonalization) and that A is immediately ML-updated shows
        # that the optimal scaling makes E[zz] an identity matrix.
        q, iq = joint_ortho(zz, uu)
        zz = t(q).matmul(zz).matmul(q)
        scl = get_diag(zz).div(n).sqrt_()
        zz = torch.eye(k, **backend).mul_(n)
        q /= scl[..., None, :]
        iq *= scl[..., None]
        uu = iq.matmul(uu).matmul(t(iq))
        z = z.matmul(q)
        u = iq.matmul(u)

        # update A
        a = inv(uu / m)

        # update log-evidence
        l = logev(r, zz, uu, s, a, uuzz)
        gain = (l1-l)/(l0 - l)
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end=end)
        if abs(gain) < tol:
            break
        l1 = l
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'var':
            out.append(s)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)


def _softmax_lse(x, dim=-1):
    """Implicit softmax that also returns the LSE"""
    x = x.clone()
    lse, _ = torch.max(x, dim=dim, keepdim=True)
    lse.clamp_min_(0)  # don't forget the class full of zeros

    x = x.sub_(lse).exp_()
    sumval = x.sum(dim=dim, keepdim=True)
    sumval += lse.neg().exp_()  # don't forget the class full of zeros
    x /= sumval

    sumval = sumval.log_()
    lse += sumval
    lse = lse.sum(dtype=torch.float64)
    return x, lse


def cpca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False):
    """(Probabilistic) Categorical Principal Component Analysis

    Notes
    -----
    .. We find the basis U that maximizes E_z[ Cat(x | SoftMax(U@z + mu)) ],
       where z stems form a standard Gaussian distribution.
    .. We manually orthogonalize the subspace within the optimization
       loop so that the output subspace is orthogonal (`z.T @ z` and
       `U @ U.T` are diagonal).

    Parameters
    ----------
    x : (..., n, m, i) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `MxI` their dimension,
        where `M` is the number of classes minus one, and `I` is the number
        of voxels.
    nb_components : int, default=min(n, m*i)-1
        Number of principal components (k) to return.
    mean : float or (..., m, i) tensor_like, optional
        Mean tensor. If not provided, it is estimated along with the bases.
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-5
        Tolerance on log model evidence for early stopping.
    returns : {'latent', 'basis', 'mean'}, default='latent+basis+mean'
        Which variables to return.
    verbose : {0, 1, 2}, default=0

    Returns
    -------
    latent : (..., n, k) tensor, if 'latent' in `returns`
        Latent coordinates
    basis : (..., k, m, i) tensor, if 'basis' in `returns`
        Orthogonal basis.
    mean : (..., m, i) tensor, if 'mean' in `returns`
        Mean

    References
    ----------
    ..[1] "Variational bounds for mixed-data factor analysis."
          Khan, Bouchard, Murphy, Marlin
          NeurIPS (2010)
    ..[2] "Factorisation-based Image Labelling."
          Yan, Balbastre, Brudfors, Ashburner.
          Preprint (2021)

    """

    if isinstance(x, (list, tuple)) and callable(x[0]):
        raise NotImplementedError
        # return _cpca_callable(x, nb_components, mean, max_iter, tol,
        #                       returns, verbose)
    else:
        return _cpca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose)


def _cpca_tensor(x, k, mu=None, max_iter=20, tol=1e-5,
                 returns='latent+basis+mean', verbose=False):

    # --- preproc ---
    x = torch.as_tensor(x)
    batch = x.shape[:-3]
    n, m, i = x.shape[-3:]
    backend = utils.backend(x)
    k = k or (min(n, m*i) - 1)

    # --- helpers ---

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_sym(x):
        """Make a matrix symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        return linalg.inv(x.double()).to(dtype=dtype)

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    # --- initialization ---
    sumxn = x.sum(-3)

    # init mean
    optim_mu = mu is None
    if mu is None:
        mu = x.mean(-3)
        mu = math.logit(mu, -2, implicit=True)
    mu = torch.as_tensor(mu, **backend)

    # init latent with random orthogonal tensor
    z = torch.randn([*batch, n, k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))

    # init basis
    u = torch.zeros([*batch, k, m, i], **backend)

    # init approximate Hessian
    a = 0.5*(torch.eye(m, **backend) - 1/(m+1))
    b = torch.eye(m, **backend) + 1/(m+1)
    b = linalg.kron2(torch.eye(k, **backend), b)

    # init log-fit
    eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
    rho, nll = _softmax_lse(eta, -2)
    nll = nll.sum() - eta.flatten().dot(x.flatten())

    # init log-evidence
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {nll.item():6f}', end=end)

    for n_iter in range(max_iter):
        nll0 = nll

        # update mean (FIL, eq [30] simplified)
        if optim_mu and n_iter > 0:
            mu -= torch.einsum('...lm,mi->li', inv(a)/n, rho.sum(-3) - sumxn)

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update basis (FIL, eq [33, 34])
        h = linalg.kron2(zz, a) + b
        u = torch.einsum('...nk,kmi->nmi', z, u)
        u = torch.einsum('...ml,nli->nmi', a, u)
        u += x
        u -= rho
        u = torch.einsum('...nk,nmi->kmi', z, u)
        u = torch.einsum('...kl,li->ki', inv(h), u.reshape([*batch, k*m, i]))
        u = u.reshape([*batch, k, m, i])

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update latent (FIL, eq [25, 26])
        uu = torch.einsum('...kmi,ml,jli->kj', u, a, u)
        get_diag(uu).add_(1)
        uu = inv(uu)
        z = torch.einsum('...nk,kmi->nmi', z, u)
        z = torch.einsum('...ml,nli->nmi', a, z)
        z += x
        z -= rho
        z = torch.einsum('...kmi,nmi->nk', u, z)
        z = torch.einsum('...kj,nj->nk', uu, z)
        zz = make_sym(t(z).matmul(z))
        zz += uu

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # orthogonalize
        q, iq = joint_ortho(zz, uu)
        z = z.matmul(q)
        zz = t(q).matmul(zz).matmul(q)
        u = torch.einsum('...kl,lmi->kmi', iq, u)

        # update log-evidence
        nll = nll.sum() - eta.flatten().dot(x.flatten())
        gain = (nll0 - nll) / x.numel()
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {nll.item():6f} | '
                  f'{gain.item():.3e} ({"-" if nll < nll0 else "+"})', end=end)
        if abs(gain) < tol:
            break
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)


def _mcpca_tensor(x, k, q, mu=None, max_iter=20, tol=1e-5,
                  returns='latent+basis+mean+mixing', verbose=False):
    # x : (..., n, m, i)
    # n : number of independent observations
    # m : number of classes minus one
    # i : number of voxels per patch
    # k : number of latent dimensions
    # q : number of mixture components

    # --- preproc ---
    x = torch.as_tensor(x)
    batch = x.shape[:-3]
    n, m, i = x.shape[-3:]
    backend = utils.backend(x)
    k = k or (min(n, m*i) - 1)
    q = q or 1

    # --- helpers ---

    def flatdot(x, y):
        return x.flatten().dot(y.flatten())

    def t(x):
        """Quick transpose"""
        return x.transpose(-1, -2)

    def get_diag(x):
        """Quick extract diagonal as a view"""
        return x.diagonal(dim1=-2, dim2=-1)

    def make_sym(x):
        """Make a matrix symmetric by averaging with its transpose"""
        return (x + t(x)).div_(2.)

    def inv(x):
        """Robust inverse in double"""
        dtype = x.dtype
        return linalg.inv(x.double()).to(dtype=dtype)

    def joint_ortho(zz, uu):
        """Joint orthogonalization of two matrices:
            Find T such that T' @ A @ T and inv(T) @ B @ inv(T') are diagonal.
            Since the scaling is arbitrary, we make A unitary and B diagonal.
        """
        vz, sz, _ = torch.svd(zz)
        vu, su, _ = torch.svd(uu)
        su = su.sqrt_()
        sz = sz.sqrt_()
        vsz = vz * sz[..., None, :]
        vsu = vu * su[..., None, :]
        v, s, w = torch.svd(torch.matmul(t(vsz), vsu))
        w *= s[..., None, :]

        eu = get_diag(vu).abs().max(-1).values[..., None]
        su = torch.max(su, eu * 1e-3)
        vu /= su[..., None, :]
        ez = get_diag(vz).abs().max(-1).values[..., None]
        sz = torch.max(sz, ez * 1e-3)
        vz /= sz[..., None, :]

        q = vz.matmul(v)
        iq = t(w).matmul(t(vu))
        return q, iq

    # --- initialization ---
    sumxn = x.sum(-3)

    # init responsibilities
    pi = torch.rand([*batch, q], **backend)
    pi /= pi.sum(-1, keepdim=True)
    r = torch.distributions.Categorical(probs=pi).sample(n)
    r = r.transpose(-1, -2)  # (*batch, n, q)
    pi = r.sum(-2)
    pi /= pi.sum(-1, keepdim=True)
    klr = n * flatdot(pi, pi.log()) - flatdot(r, r.log())

    # init mean
    optim_mu = mu is None
    if mu is None:
        mu = torch.einsum('...nmi,nq->qmi', x, r)
        mu /= pi[..., None, None]
        mu = math.logit(mu, -2, implicit=True)
    mu = torch.as_tensor(mu, **backend)

    # init latent with random orthogonal tensor
    z = torch.randn([*batch, n, k], **backend)
    z, _, _ = torch.svd(z, some=True)
    zz = make_sym(t(z).matmul(z))

    # init basis
    u = torch.zeros([*batch, q, k, m, i], **backend)

    # init approximate Hessian
    a = 0.5*(torch.eye(m, **backend) - 1/(m+1))
    b = torch.eye(m, **backend) + 1/(m+1)
    b = linalg.kron2(torch.eye(k, **backend), b)

    # init log-fit
    eta = torch.einsum('...nk,qkmi->nqmi', z, u)
    eta = torch.einsum('...nqmi,qmi->nqmi', eta, mu)
    rho, nll = _softmax_lse(eta, -2)
    nll = flatdot(r, nll.sum(-1))
    nll -= torch.einsum('...nq,nqmi,nmi->', r, eta, x).sum()
    nll += klr

    # init log-evidence
    if verbose:
        end = '\n' if verbose > 1 else '\r'
        print(f'{0:3d} | {nll.item():6f}', end=end)

    for n_iter in range(max_iter):
        nll0 = nll

        # update mean (FIL, eq [30])
        if optim_mu and n_iter > 0:
            mu = torch.einsum('...ml,qli->qmi', a, mu)
            mu.add_(sumxn.unsqueeze(-3), alpha=1/n)
            mu -= torch.einsum('...nqmi,nq,q->qmi', rho, r, 1/(n*pi))
            mu = torch.einsum('...lm,mi->li', inv(a), mu)

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update basis (FIL, eq [33, 34])
        h = linalg.kron2(zz, a) + b
        u = torch.einsum('...nk,kmi->nmi', z, u)
        u = torch.einsum('...ml,nli->nmi', a, u)
        u += x
        u -= rho
        u = torch.einsum('...nk,nmi->kmi', z, u)
        u = torch.einsum('...kl,li->ki', inv(h), u.reshape([*batch, k*m, i]))
        u = u.reshape([*batch, k, m, i])

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # update latent (FIL, eq [25, 26])
        uu = torch.einsum('...kmi,ml,jli->kj', u, a, u)
        get_diag(uu).add_(1)
        uu = inv(uu)
        z = torch.einsum('...nk,kmi->nmi', z, u)
        z = torch.einsum('...ml,nli->nmi', a, z)
        z += x
        z -= rho
        z = torch.einsum('...kmi,nmi->nk', u, z)
        z = torch.einsum('...kj,nj->nk', uu, z)
        zz = make_sym(t(z).matmul(z))
        zz += uu

        eta = torch.einsum('...nk,kmi->nmi', z, u).add_(mu)
        rho, nll = _softmax_lse(eta, -2)

        # orthogonalize
        q, iq = joint_ortho(zz, uu)
        z = z.matmul(q)
        zz = t(q).matmul(zz).matmul(q)
        u = torch.einsum('...kl,lmi->kmi', iq, u)

        # update log-evidence
        nll = nll.sum() - eta.flatten().dot(x.flatten())
        gain = (nll0 - nll) / x.numel()
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{n_iter+1:3d} | {nll.item():6f} | '
                  f'{gain.item():.3e} ({"-" if nll < nll0 else "+"})', end=end)
        if abs(gain) < tol:
            break
    if verbose < 2:
        print('')

    out = []
    returns = returns.split('+')
    for ret in returns:
        if ret == 'latent':
            out.append(z)
        elif ret == 'basis':
            out.append(u)
        elif ret == 'mean':
            out.append(mu)
    return out[0] if len(out) == 0 else tuple(out)
