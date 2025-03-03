__all__ = ["optimal_svf"]
import torch
from nitorch.core.py import prod
from nitorch.core.linalg import lmdiv, sym_matvec
from nitorch.spatial import solve_grid_fmg


def optimal_svf(svf, weights=None, squared=True, **prm):
    """Compute optimal "template-to-subj" SVFs from "subj-to-subj" pairs.

    Parameters
    ----------
    svf : dict[tuple[int, int], (*shape, D) tensor]
        SVF for each pair
    weight : dict[tuple[int, int], (*shape, [D2|D|1]) tensor]
        Weight for each pair
    squared : bool
        If true, weight contains squared weights (== precision)
    **prm
        Regularization parameters

    Returns
    -------
    optimal : (N, *shape, D) tensor
        Optimal template-to-subject SVFs

    """
    backend = dict(
        dtype=prm.pop("dtype", None),
        device=prm.pop("device", None)
    )

    ndim = list(svf.values())[0].shape[-1]
    shape = list(svf.values())[0].shape[-ndim-1:-1]

    # compute list of unique labels
    labels = list(set([label for pair in svf for label in pair]))

    # compute log-optimal parameters
    N = len(labels)
    K = len(svf)
    D = ndim

    def load(d):
        return {
            k: getattr(v, 'fdata', getattr(v, 'to', None))(**backend)
            for k, v in d.items()
        }

    def fit_weighted(svf, weights):
        svf, weights = load(svf), load(weights)

        # ensure forward/backward are present
        # (avoids bias if some are and others are not)
        svf = dict(svf)
        for (a, b), v in svf.items():
            svf.setdefault((b, a), -v)

        if weights:
            weights = dict(weights)
            for (a, b), w in weights.items():
                weights.setdefault((b, a), w)
            if set(svf.keys()) - set(weights.keys()):
                meanw = sum(w.mean() for w in weights.values())
                meanw /= len(weights)
                for (a, b) in svf.keys():
                    weights.setdefault((a, b), meanw)

        # Build matrix
        wl = torch.zeros([*shape[:2], N, D, N, D])
        wr = torch.zeros([*shape[:2], N, D, K, D])
        for k, (i, j) in enumerate(svf.keys()):
            h = weights[(i, j)]
            # h = h[..., :D].mean(-1, keepdim=True)
            if not squared:
                h = h*h
            I, J = labels.index(i), labels.index(j)
            if h.shape[-1] in (1, D):
                wl[..., I, :, I, :].diagonal(0, -1, -2).add_(h)
                wl[..., J, :, J, :].diagonal(0, -1, -2).add_(h)
                wl[..., I, :, J, :].diagonal(0, -1, -2).sub_(h)
                wl[..., J, :, I, :].diagonal(0, -1, -2).sub_(h)

                wr[..., I, :, k, :].diagonal(0, -1, -2).add_(h)
                wr[..., J, :, k, :].diagonal(0, -1, -2).sub_(h)
            else:
                for m in range(D):
                    wl[..., I, m, I, m] += h[..., m]
                    wl[..., J, m, J, m] += h[..., m]
                    wl[..., I, m, J, m] -= h[..., m]
                    wl[..., J, m, I, m] -= h[..., m]

                    wr[..., I, m, k, m] += h[..., m]
                    wr[..., J, m, k, m] -= h[..., m]
                mn = D
                for m in range(D):
                    for n in range(m+1, D):
                        wl[..., I, m, I, n] += h[..., mn]
                        wl[..., J, m, J, n] += h[..., mn]
                        wl[..., I, n, J, m] -= h[..., mn]
                        wl[..., J, n, I, m] -= h[..., mn]

                        wr[..., I, m, k, n] += h[..., mn]
                        wr[..., I, n, k, m] += h[..., mn]
                        wr[..., J, m, k, n] -= h[..., mn]
                        wr[..., J, n, k, m] -= h[..., mn]

                        mn += 1

        x = torch.stack(list(svf.values()), -2)  # (*shape, K, D)
        x = x.reshape(x.shape[:-2] + (K*D,))
        wr = wr.reshape(wr.shape[:-4] + (N*D, K*D))
        wl = wl.reshape(wl.shape[:-4] + (N*D, N*D))

        # load matrix for inversion
        wd = wl.diagonal(0, -1, -2)
        wd += wd.abs().max(-1).values[..., None] * 1e-4 + 1e-16

        # solve
        z = x.unsqueeze(-1)
        z = torch.matmul(wr.to(z), z)
        z = lmdiv(wl.to(z), z).squeeze(-1)

        # reshape and zero-center
        z = z.reshape(z.shape[:-1] + (N, D))
        z -= torch.mean(z, dim=-2, keepdim=True)
        z = torch.movedim(z, -2, 0)

        # Build absolute matrix
        w = torch.zeros([N, *shape[:2], (D*(D+1))//2])
        for k, (i, j) in enumerate(svf.keys()):
            h = weights[(i, j)]
            # h = h[..., :D].mean(-1, keepdim=True)
            if not squared:
                h = h*h
            I, J = labels.index(i), labels.index(j)
            if h.shape[-1] in (1, D):
                w[I, ..., :D] += h
                w[J, ..., :D] += h
            else:
                w[I] += h
                w[J] += h
        return z, w

    def fit_unweighted(svf):
        svf = load(svf)

        # ensure forward/backward are present
        # (avoids bias if some are and others are not)
        svf = dict(svf)
        for (a, b), v in svf.items():
            svf.setdefault((b, a), -v)

        w = torch.zeros([K, N])
        for p, (i, j) in enumerate(svf.keys()):
            w[p, labels.index(i)] = +1
            w[p, labels.index(j)] = -1
        x = torch.stack(list(svf.values()), -1)
        z = lmdiv(w.to(x),  x.unsqueeze(-1)).squeeze(-1)
        z -= torch.mean(z, dim=-1, keepdim=True)  # zero-center
        z = torch.movedim(z, -1, 0)
        w = w.abs().sum(0)
        return z, w

    def fit(svf, weights):
        return fit_weighted(svf, weights) if weights else fit_unweighted(svf)

    print("fit")
    if ndim == 3:
        # slicewise
        z = torch.zeros([N, *shape, D])
        if weights:
            w = torch.zeros([N, *shape, (D*(D+1))//2])
        for k in range(shape[-1]):
            print("slice: ", k, end='\r')
            vk = dict(zip(svf.keys(), [v[:, :, k, :] for v in svf.values()]))
            if weights:
                wk = dict(zip(weights.keys(), [w[:, :, k, :] for w in weights.values()]))
                z[..., k, :], w[..., k, :] = fit(vk, wk)
            else:
                z[..., k, :], w = fit(vk, None)
        print("")
    else:
        z, w = fit(svf, weights)

    # regularize
    if prm:
        print("regularize")
        alpha0 = prm.get("factor", 1.0)
        alpha0 = alpha0 / prod(z.shape[1:-1])
        prm["factor"] = alpha0

        # load matrix for inversion
        if weights:
            wd = w[..., :D]
            wd += wd.abs().max(-1).values[..., None] * 1e-4

        for n in range(N):

            if weights:
                w[n, ~torch.isfinite(z[n]).any(-1), :] = 0
                z[n, ~torch.isfinite(z[n])] = 0
                z[n] = sym_matvec(w[n], z[n])
            else:
                z[n] *= w[n]

            foo = solve_grid_fmg(w[n], z[n], **prm)
            z[n] = foo
        z -= torch.mean(z, dim=0)  # zero-center again

    return z
