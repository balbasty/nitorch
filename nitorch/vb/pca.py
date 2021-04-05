"""Various flavors of component/factor analysis"""
import torch
from nitorch.core import utils, linalg

# TODO: the convention is not the same between pca and ppca in terms
#   of which factor is scaled and which is unitary.
#   Currently, pca returns a unitary basis whereas ppca returns
#   unitary latent coordinates.
#   We could have an option to specify which side should be normalized?


def pca(x, nb_components=None, mean=None, returns='latent+basis'):
    """Principal Component Analysis

    Parameters
    ----------
    x : (..., N, M) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `M` their dimension.
        If a sequence of callable, a memory efficient implementation is
        used, where tensors are loaded by calling the corresponding callable
        when needed.
    nb_components : int, default=min(N, M)
        Number of principal components to return.
    mean : float or (..., M) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, subtract the mean of all observations.
        If 0, nothing is done.
    returns : {'latent', 'basis', 'latent+basis'}, default='latent+basis'
        Which variables to return.

    Returns
    -------
    latent : (..., N, nb_components), if 'latent' in `returns`
    basis : (..., nb_components, M), if 'basis' in `returns`

    """
    if isinstance(x, (list, tuple)) and callable(x[0]):
        return _pca_callable(x, nb_components, mean, returns)
    else:
        return _pca_tensor(x, nb_components, mean, returns)


def _pca_tensor(x, k, m, returns):
    """Classic implementation: subtract mean and call SVD."""

    x = torch.as_tensor(x)
    if m is None:
        m = torch.mean(x, dim=-2)
    m = torch.as_tensor(m, **utils.backend(x))
    if not isinstance(m, (int, float)) or m != 0:
        x = x - m[..., None, :]

    u, s, v = torch.svd(x, some=True)
    u.mul_(s[..., None, :])

    if k:
        if k > min(x.shape[-1], x.shape[-2]):
            raise ValueError('Number of components cannot be larger '
                             'than min(N,M)')
        u = u[..., k]
        v = v[..., k]

    v = v.transpose(-1, -2)

    if returns == 'latent+basis':
        return u, v
    if returns == 'basis+latent':
        return v, u
    elif returns == 'latent':
        return u
    elif returns == 'basis':
        return v
    else:
        return None


def _pca_callable(x, k, mu, returns):
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
    u, s, _ = torch.svd(cov, some=True)  # [..., n, k]
    s = s.sqrt_()
    u *= s[..., None, :]
    u = u[..., :k]
    if 'basis' not in returns:
        return u

    # build basis by projection
    iu = torch.pinverse(u)
    v = u.new_zeros([*shape, k, m])
    for n1 in range(n):
        x1 = torch.as_tensor(x[n1](), **backend)
        if not nomu:
            x1 -= mu
        v += iu[..., :, n1, None] * x1[..., None, :]

    if returns == 'latent+basis':
        return u, v
    elif returns == 'basis+latent':
        return v, u
    else:
        return v


def ppca(x, nb_components=None, mean=None, max_iter=20, tol=1e-5,
         returns='latent+basis+var', verbose=False):
    """Probabilistic Principal Component Analysis

    Parameters
    ----------
    x : (..., N, M) tensor_like or sequence[callable]
        Observed variables.
        `N` is the number of independent variables and `M` their dimension.
        If a sequence of callable, a memory efficient implementation is
        used, where tensors are loaded by calling the corresponding callable
        when needed.
    nb_components : int, default=min(N, M)-1
        Number of principal components to return.
    mean : float or (..., M) tensor_like, optional
        Mean tensor to subtract from all observations prior to SVD.
        If None, use the mean of all observations (maximum-likelihood).
        If 0, nothing is done.
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-5
        Tolerance on log model evidence for early stopping.
    returns : {'latent', 'basis', 'var'}, default='latent+basis+var'
        Which variables to return.
    verbose : bool, default=False

    Returns
    -------
    latent : (..., N, nb_components), if 'latent' in `returns`
    basis : (..., nb_components, M), if 'basis' in `returns`

    References
    ----------
    ..[1] "Probabilistic principal component analysis."
          Tipping, Michael E., and Bishop, Christopher M.
          J. R. Stat. Soc., Ser. B (1999)

    """

    if isinstance(x, (list, tuple)) and callable(x[0]):
        raise NotImplementedError
        # TODO
        # return _ppca_callable(x, nb_components, mean, max_iter, tol,
        #                       returns, verbose)
    else:
        return _ppca_tensor(x, nb_components, mean, max_iter, tol,
                            returns, verbose)


def _ppca_tensor(x, k, mu, max_iter, tol, returns, verbose=False):
    """Implementation that assumes that all the data is in memory."""

    # --- preproc ---

    x = torch.as_tensor(x)
    n, m = x.shape[-2:]
    backend = utils.backend(x)
    k = k or (min(x.shape[-1], x.shape[-2]) - 1)
    eps = 1e-6

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

    def logev(r, z, uu, s):
        """Negative log-evidence
        r  : (*batch) - squared residuals summed across N and M
        z  : (*batch, N, K) - latent variables
        uu : (*batch, K, K) - basis product (u @ u.T)
        s  : (*batch) - Residual variance
        """
        r = (r/s).sum()
        z = z.square().sum([-1, -2])
        z = z.sum()
        # uncertainty
        unc = reg_(uu, s).logdet() - s.log() * k
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
    s = x.var(unbiased=False, dim=[-1, -2]).mul_(0.1)

    # init latent with random orthogonal tensor
    z = torch.randn([*x.shape[:-1], k], **backend)
    z, _, _ = torch.svd(z, some=True)

    # init basis
    iz = rinv(z, s, 'l')
    u = iz.matmul(x)
    uu = make_sym(u.mm(t(u)))

    # init log-evidence
    r = (x - z.matmul(u)).square_().sum([-1, -2])
    l0 = l1 = logev(r, z, uu, s)
    if verbose:
        print(f'{0:3d} | {l0.item():6f}', end='\r')

    for n_iter in range(max_iter):

        # update latent
        im = inv(reg_(uu, s))
        z = x.matmul(t(u)).matmul(im)
        zz = make_sym(t(z).matmul(z))
        tiny = eps * get_diag(zz).abs().max(-1).values
        zz += im * s[..., None, None].clamp_min(tiny)

        r = (x - z.matmul(u)).square_().sum([-1, -2])
        logev(r, z, uu, s)

        # update basis
        # u = inv(zz).matmul(t(z).matmul(x))
        u = inv(zz).matmul(t(z)).matmul(x)
        uu = make_sym(u.mm(t(u)))

        r = (x - z.matmul(u)).square_().sum([-1, -2])
        logev(r, z, uu, s)

        # update sigma
        im = s * inv(reg_(uu, s))
        r = (x - z.matmul(u)).square_().sum([-1, -2])
        # residual + uncertainty term
        s = r / (n*m) + trace(im.matmul(uu)) / m

        # update log-evidence
        l = logev(r, z, uu, s)
        gain = (l1-l)/(l0 - l)
        if verbose:
            print(f'{n_iter+1:3d} | {l.item():6f} | '
                  f'{gain.item():.3e} ({"-" if l < l1 else "+"})', end='\r')
        if abs(gain) < tol:
            break
        l1 = l
    if verbose:
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



