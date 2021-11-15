from nitorch.core import utils, math, linalg
from nitorch.spatial import spconv
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


def mrf(Z, P, L=None, W=None, max_iter=5, tol=1e-4, inplace=False):
    """Apply a Markov Random Field using a mean-field approximation
    (= variational Bayes)

    Parameters
    ----------
    Z : (K, *spatial) tensor
        Previous responsibilities
    P: (K, K) tensor
        MRF conditional probabilities
    L : (K, *spatial) tensor, optional
        Log-likelihood
    W : (*spatial) tensor, optional
        Observation weights
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

    P = P.log()
    dim = Z.dim() - 1

    K = len(Z)
    Nw = W.sum() if W is not None else Z[0].numel()
    Z = utils.fast_movedim(Z, 0, -1)
    lZ = torch.empty_like(Z)
    if L is not None:
        L = utils.fast_movedim(L, 0, -1)
    if W is not None and W.dtype is torch.bool:
        W = W.to(Z.dtype)

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
                            P1 = P[ko, ki]
                            if W is not None:
                                P1 = P1.item()
                                Z0[(*slicer_pre, ko)].addcmul_(Wpre, pre[..., ki], value=P1)
                                Z0[(*slicer_post, ko)].addcmul_(Wpost, post[..., ki], value=P1)
                            else:
                                Z0[(*slicer_pre, ko)].addcmul_(P[ko, ki], pre[..., ki])
                                Z0[(*slicer_post, ko)].addcmul_(P[ko, ki], post[..., ki])
                # normalize by number of neighbors to decrease a bit
                # the strength of the MRF (I feel that might be the
                # correct way to normalize the joint probability...)
                # Z0.div_(2*dim)
                lZ[slicer0].copy_(Z0)

                # add likelihood
                if L is not None:
                    Z0 += L[slicer0]

                # softmax
                ll += _inplace_softmax(Z0, W[slicer0], dim)

        # gain
        if ll - oll < tol * Nw:
            break

    Z = utils.fast_movedim(Z, -1, 0)
    lZ = utils.fast_movedim(lZ, -1, 0)
    return Z, lZ


def update_mrf(Z, W=None, prior=None):
    """Compute maximum-likelihood MRF prior

    Properties
    ----------
    Z : (K, *spatial) tensor
        Responsibilities
    W : (*spatial) tensor, optional
        Observation weights
    prior : ([K], K) tensor
        Dirichlet prior

    Returns
    -------
    P : (K, K) tensor
    """

    dim = Z.dim() - 1
    K = len(Z)
    P = Z.new_zeros(K, K)

    for d in range(dim):
        slicer_pre = [slice(None)] * dim
        slicer_pre[d] = slice(1, None)
        pre = Z[(Ellipsis, *slicer_pre)]
        slicer_post = [slice(None)] * dim
        slicer_post[d] = slice(-1)
        post = Z[(Ellipsis, *slicer_post)]
        if W is not None:
            pre = pre * W[tuple(slicer_pre)]
            pre *= W[tuple(slicer_post)]
        P += torch.matmul(pre.reshape([K, -1]), post.reshape([K, -1]).T)

    if prior is not None:
        P += prior

    P += 1e-8
    P /= P.sum(-1, keepdim=True)
    return P
