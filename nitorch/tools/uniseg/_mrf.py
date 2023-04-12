from nitorch.core import utils, math, linalg
import itertools
import torch


def _inplace_softmax(Z, W=None, dim=0):

    Z = utils.fast_movedim(Z, dim, 0)
    mx, _ = torch.max(Z, dim=0, keepdim=True)

    Z.sub_(mx).exp_()
    lse = torch.sum(Z, dim=0, keepdim=True)
    Z /= lse

    # Compute log-sum-exp
    #   maxval = max(logit)
    #   lse = maxval + log[sum(exp(logit - maxval))]
    lse = lse.log_().add_(mx)
    if W is not None:
        lse *= W
    lse = lse.sum(dtype=torch.float64)
    return lse


def mrf_suffstat(Z, W=None, vx=1):
    """ Compute the sum of probabilities across neighbors

    Notes
    -----
    .. This function returns
            Z[k, m] = \sum_{n \in Neighbours(m)} Z[k, n] * W[n] / d(n,m)
       where Z are the input responsibilities, W are the voxels weights
       and d(n,m) is the distance between voxels n and m (i.e., voxel size).
    .. Only first order neighbors are used (4 in 2D, 6 in 3D).

    Parameters
    ----------
    Z : (K, *spatial) tensor
        Responsibilities
    W : (*spatial) tensor, optional
        Voxel weights
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    E : (K, *spatial) tensor
        Sufficient statistics

    """
    dim = Z.dim() - 1
    K = len(Z)
    vx = utils.make_vector(vx, dim, dtype=Z.dtype, device='cpu')
    ivx = vx.reciprocal().tolist()

    S = torch.zeros_like(Z)
    if W is not None:
        W = W.to(Z.dtype)

    # iterate across first order neighbors
    for d in range(dim):
        Z = Z.transpose(d+1, 1)
        S = S.transpose(d+1, 1)
        if W is not None:
            W = W.transpose(d, 0)
        ivx1 = ivx[d]

        for k in range(K):
            if W is not None:
                S[k, 1:].addcmul_(W[:-1], Z[k, :-1], value=ivx1)
                S[k, :-1].addcmul_(W[1:], Z[k, 1:], value=ivx1)
            else:
                S[k, 1:].add_(Z[k, :-1], alpha=ivx1)
                S[k, :-1].add_(Z[k, 1:], alpha=ivx1)

        Z = Z.transpose(1, d+1)
        S = S.transpose(1, d+1)
        if W is not None:
            W = W.transpose(0, d)

    return S


def mrf_logprior(Z, logP, W=None, vx=1):
    """Compute the conditional MRF log-prior

    Notes
    -----
    .. This function returns
            Z[k, m] = \sum_j logP[k, j] \sum_{n \in Neighbours(m)} Z[j, n] * W[n] / d(n,m)
       where Z are the input responsibilities, W are the voxels weights
       and d(n,m) is the distance between voxels n and m (i.e., voxel size).
    .. It is equivalent to `logP @ mrf_suffstat(Z)` (but hopefully faster)
    .. Only first order neighbors are used (4 in 2D, 6 in 3D).

    Parameters
    ----------
    Z : (K, *spatial) tensor
        Responsibilities
    logP : (K[-1], K) tensor
        MRF weights
    W : (*spatial) tensor, optional
        Voxel weights
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    L : (K, *spatial) tensor
        log of MRF prior

    """
    dim = Z.dim() - 1
    K = len(Z)
    vx = utils.make_vector(vx, dim, dtype=Z.dtype, device='cpu')
    ivx = vx.reciprocal().tolist()

    S = torch.zeros_like(Z)

    if len(logP) == K:
        # make implicit
        logP = logP[1:] - logP[:1]

    # iterate across first order neighbors
    for d in range(dim):
        Z = Z.transpose(d+1, 1)
        S = S.transpose(d+1, 1)
        if W is not None:
            W = W.transpose(d, 0)
        ivx1 = ivx[d]

        for ko in range(K):
            if len(logP) < K:
                if ko == 0:
                    continue
            for ki in range(K):
                logP1 = logP[ko-1, ki] if len(logP) < K else logP[ko, ki]
                logP1 = logP1.item() * ivx1
                if W is not None:
                    S[ko, 1:].addcmul_(W[:-1], Z[ki, :-1], value=logP1)
                    S[ko, :-1].addcmul_(W[1:], Z[ki, 1:], value=logP1)
                else:
                    S[ko, 1:].add_(Z[ki, :-1], alpha=logP1)
                    S[ko, :-1].add_(Z[ki, 1:], alpha=logP1)

        Z = Z.transpose(1, d+1)
        S = S.transpose(1, d+1)
        if W is not None:
            W = W.transpose(0, d)

    return S


def mrf_covariance(Z, W=None, vx=1):
    """Compute the covariance of the MRF term

    Notes
    -----
    .. This function returns
            V = \sum_n (diag(Z[:, n]) - Z[:, n] @ Z[:, n].T) * W[n]
                * \sum_{m \in Neighbours(n)} 1/square(d(n,m))
       where Z are the input responsibilities, W are the voxels weights
       and d(n,m) is the distance between voxels n and m (i.e., voxel size).
    .. Only first order neighbors are used (4 in 2D, 6 in 3D).

    Parameters
    ----------
    Z : (K, *spatial) tensor
        Responsibilities
    W : (*spatial) tensor, optional
        Voxel weights
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    V : (K, K) tensor
        Covariance

    """
    def reduce(P, Q):
        P = P.reshape([len(P), -1])
        Q = Q.reshape([len(Q), -1])
        return P.matmul(Q.T)

    dim = Z.dim() - 1
    vx = utils.make_vector(vx, dim, dtype=Z.dtype, device='cpu')
    ivx2 = vx.reciprocal().square_()

    # build weights responsibilities
    V = Z.new_zeros(Z.shape[1:])
    for d in range(dim):
        V = V.transpose(d, 0)
        V[1:] += ivx2[d]
        V[:-1] += ivx2[d]
        V = V.transpose(d, 0)
    if W is not None:
        V *= W
        if W.dtype is not torch.bool:
            V *= W
            V *= W
    V *= 2  # overcounting

    # Compute (weighted) covariance
    V = reduce(Z, Z*V).neg_()
    Vdiag = V.diagonal(0, -1, -2)
    if W is None:
        Vdiag += Z.reshape([len(Z), -1]).sum(-1)
    elif W.dtype is torch.bool:
        Vdiag += Z[:, W].sum(-1)
    else:
        Vdiag += Z.reshape([len(Z), -1]).matmul(W.reshape([-1, 1]))[:, 0]
    return V


def mrf(Z, logP, L=None, W=None, vx=1, max_iter=5, tol=1e-4, inplace=False):
    """Apply a Markov Random Field using a mean-field approximation
    (= variational Bayes)

    Parameters
    ----------
    Z : (K, *spatial) tensor
        Previous responsibilities
    logP: (K, K) or (K-1, K) tensor
        MRF log-probabilities (do not need to be normalized)
    L : (K, *spatial) tensor, optional
        Log-likelihood
    W : (*spatial) tensor, optional
        Observation weights
    vx : [sequence of] float, default=1
        Voxel size, used to modulate the MRF probabilities.
    max_iter : int, default=5
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for early stopping
    inplace : bool, default=False
        Write into Z in-place.

    Returns
    -------
    Z : (K, *spatial) tensor
        Updated responsibilities
    lZ : (K, *spatial) tensor
        MRF log-term
    """

    if not inplace:
        Z = Z.clone()

    dim = Z.dim() - 1

    vx = utils.make_vector(vx, dim, dtype=Z.dtype, device='cpu')
    ivx = vx.reciprocal().tolist()

    K = len(Z)
    Nw = W.sum() if W is not None else Z[0].numel()
    Z = utils.fast_movedim(Z, 0, -1)
    lZ = torch.empty_like(Z)
    if L is not None:
        L = utils.fast_movedim(L, 0, -1)
    if W is not None and W.dtype is torch.bool:
        W = W.to(Z.dtype)

    if len(logP) == K-1:
        logP0 = logP
        logP = logP0.new_zeros([K, K])
        logP[1:] = logP0

    redblack = list(itertools.product([0, 1], repeat=dim))
    red = [x for x in redblack if sum(x) % 2]
    black = [x for x in redblack if not sum(x) % 2]

    ll = -float('inf')
    for n_iter in range(max_iter):
        oll = ll

        for color in (red, black):

            # Compute prior term in-place
            ll = 0
            for offset in color:
                # extract center voxel
                slicer0 = tuple(slice(o, None, 2) for o in offset)
                Z0 = Z[slicer0]
                Z0.zero_()
                # iterate across first order neighbors
                for d in range(dim):
                    ivx1 = ivx[d]
                    slicer_pre = list(slicer0)
                    slicer_pre[d] = slice(1 - offset[d], None, 2)
                    pre = Z[tuple(slicer_pre)]
                    if W is not None:
                        Wpre = W[tuple(slicer_pre)]
                    slicer_pre = [slice(None)] * dim
                    slicer_pre[d] = slice(Z0.shape[d]+offset[d]-1)
                    pre = pre[tuple(slicer_pre)]
                    if W is not None:
                        Wpre = Wpre[tuple(slicer_pre)]
                    slicer_pre[d] = slice(1 - offset[d], None)

                    slicer_post = list(slicer0)
                    slicer_post[d] = slice(offset[d]+1, None, 2)
                    post = Z[tuple(slicer_post)]
                    if W is not None:
                        Wpost = W[tuple(slicer_post)]
                    slicer_post = [slice(None)] * dim
                    slicer_post[d] = slice(post.shape[d])

                    for ko in range(K):
                        for ki in range(K):
                            logP1 = logP[ko, ki]
                            if W is not None:
                                logP1 = logP1.item() * ivx1
                                Z0[(*slicer_pre, ko)].addcmul_(Wpre, pre[..., ki], value=logP1)
                                Z0[(*slicer_post, ko)].addcmul_(Wpost, post[..., ki], value=logP1)
                            else:
                                logP1 = logP1 * ivx1
                                Z0[(*slicer_pre, ko)].addcmul_(logP1, pre[..., ki])
                                Z0[(*slicer_post, ko)].addcmul_(logP1, post[..., ki])

                lZ[slicer0].copy_(Z0)  # could be avoided if lZ not needed

                # add likelihood
                if L is not None:
                    Z0 += L[slicer0]

                # softmax
                W0 = W[slicer0] if W is not None else None
                ll += _inplace_softmax(Z0, W0, dim)

        # gain
        if ll - oll < tol * Nw:
            break

    Z = utils.fast_movedim(Z, -1, 0)
    lZ = utils.fast_movedim(lZ, -1, 0)
    # NOTE: returning lZ could be optional (saving memory and a few copies)
    return Z, lZ


def mrf_sample(logP, shape=None, M=None, vx=1, nb_iter=100, implicit=False):
    """Sample from a MRF using Gibbs conclique sampling

    Parameters
    ----------
    logP: (K, K) tensor
        MRF log-probabilities (do not need to be normalized)
    shape : sequence[int], optional if M is provided
        Shape of the lattice
    M : (K, *shape) tensor, optional
        Voxel-wise log-probability (do not need to be normalized)
    vx : [list of] float, default=1
        Voxel size
    nb_iter : int, default=100
        Number of Gibbs iterations
    implicit : bool, default=False

    Returns
    -------
    Z : (*shape) tensor[long]
        Sample

    References
    ----------
    ..[1] "Simulating Markov random fields with a conclique-based Gibbs sampler"
          Andee Kaplan, Mark S. Kaiser, Soumendra N. Lahiri, Daniel J. Nordman
          Journal of Computational and Graphical Statistics (2020)
          https://arxiv.org/abs/1808.04739

    """
    Cat = torch.distributions.Categorical

    if shape is None and M is None:
        raise ValueError('Either M or shape must be provided')
    if shape is None:
        shape = M.shape[1:]
    dim = len(shape)
    K = len(logP)
    if implicit:
        K = K + 1
        logP_ = logP
        logP = logP_.new_zeros([K, logP_.shape[-1]])
        logP[1:] = logP_
    P = logP.new_zeros([K, *shape])             # log-prior
    if M is None:
        L = Cat(logits=logP.sum(-1)).sample(shape)
    elif len(M) == K-1:
        M0 = M.new_zeros([K, *M.shape[1:]])
        M0[1:] = M
        L = Cat(logits=utils.movedim(M0, 0, -1)).sample()
        del M0
    else:
        L = Cat(logits=utils.movedim(M, 0, -1)).sample()

    vx = utils.make_vector(vx, dim, dtype=logP.dtype, device='cpu')
    ivx = vx.reciprocal().tolist()

    redblack = list(itertools.product([0, 1], repeat=dim))
    red = [x for x in redblack if sum(x) % 2]
    black = [x for x in redblack if not sum(x) % 2]

    for n_iter in range(nb_iter):

        for color in (red, black):

            for offset in color:

                # extract center voxel
                slicer0 = tuple(slice(o, None, 2) for o in offset)
                L0 = L[slicer0]
                P0 = P[(Ellipsis, *slicer0)]
                if M is not None:
                    if implicit:
                        P0[1:].copy_(M(Ellipsis, *slicer0))
                    else:
                        P0.copy_(M(Ellipsis, *slicer0))
                else:
                    P0.zero_()

                # iterate across first order neighbors
                for d in range(dim):
                    ivx1 = ivx[d]
                    slicer_pre = list(slicer0)
                    slicer_pre[d] = slice(1 - offset[d], None, 2)
                    pre = L[tuple(slicer_pre)]
                    slicer_pre = [slice(None)] * dim
                    slicer_pre[d] = slice(L0.shape[d]+offset[d]-1)
                    pre = pre[tuple(slicer_pre)]
                    slicer_pre[d] = slice(1 - offset[d], None)

                    slicer_post = list(slicer0)
                    slicer_post[d] = slice(offset[d]+1, None, 2)
                    post = L[tuple(slicer_post)]
                    slicer_post = [slice(None)] * dim
                    slicer_post[d] = slice(post.shape[d])

                    for ko in range(K):
                        for ki in range(K):
                            logP1 = (logP[ko, ki] * ivx1).item()
                            P0[(ko, *slicer_pre)].add_(pre == ki, alpha=logP1)
                            P0[(ko, *slicer_post)].add_(post == ki, alpha=logP1)

            sampler = Cat(logits=utils.movedim(P0, 0, -1))
            L0.copy_(sampler.sample())

    return L