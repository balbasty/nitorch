__all__ = ["optimal_affine", "log_optimal_affine", "exp_optimal_affine"]
import torch
from nitorch.core.linalg import inv, lmdiv, _expm as expm
from ._affine import affine_basis, affine_parameters, affine_matrix


def optimal_affine(matrices, loss='exp', basis='SE'):
    """Compute optimal "template-to-subj" matrices from "subj-to-subj" pairs.

    Parameters
    ----------
    matrices : dict[tuple[int, int], (D+1, D+1) tensor]
        Affine matrix for each pair
    loss : {'exp', 'log'}
        Whether to minimize the L2 loss in the embedding matrix space
        ('exp') or in the Lie algebra ('log').
    basis : {'SE', 'CSO', 'Aff+'}, default='SE'
        Constrain matrices to belong to this group

    Returns
    -------
    optimal : (N, D+1, D+1) tensor
        Optimal template-to-subject matrices

    """
    ndim = list(matrices.values())[0].shape[-1] - 1

    # convert to Lie parameters
    basis = affine_basis(basis, ndim, dtype=list(matrices.values())[0].dtype)
    parameters = {
        k: affine_parameters(v, basis)[0]
        for k, v in matrices.items()
    }

    # ensure forward/backward are present
    # (avoids bias if some are and others are not)
    for (a, b), v in parameters.items():
        parameters.setdefault((b, a), -v)

    # compute log-optimal parameters
    optimal_parameters = log_optimal_affine(parameters)

    # compute Frobenius-optimal parameters
    if loss[0].lower() != 'l':
        optimal_parameters = exp_optimal_affine(
            matrices, basis=basis, init=optimal_parameters
        )

    # reconstruct matrices
    return affine_matrix(optimal_parameters, basis)


def log_optimal_affine(parameters):
    """Compute optimal Lie parameters wrt L2 loss in the Lie algebra

    Parameters
    ----------
    parameters : dict[tuple[int, int], (F,) tensor]
        Lie parameters for each pair

    Returns
    -------
    optimal : (N, F) tensor
        Optimal template-to-subject parameters

    """
    labels = list(set([label for pair in parameters for label in pair]))
    n = len(labels)
    w = torch.zeros([len(parameters), n])
    for p, (i, j) in enumerate(parameters.keys()):
        w[p, labels.index(i)], w[p, labels.index(j)] = 1, -1
    x = torch.stack(list(parameters.values()), -2)
    w = inv(w, method='pinv')
    z = torch.matmul(w.to(x), x)
    z -= torch.mean(z, dim=-2, keepdim=True)  # zero-center
    return z


def exp_optimal_affine(matrices, basis, init=None):
    """Compute optimal Lie parameters wrt L2 loss in the embedding space

    Parameters
    ----------
    matrices : dict[tuple[int, int], (D+1, D+1) tensor]
        Affine matrix for each pair
    basis : (F, D+1, D+1) tensor
        Constrain matrices to belong to this group
    init : (N, F) tensor
        Initial guess

    Returns
    -------
    optimal : (N, F) tensor
        Optimal template-to-subject parameters

    """

    def flat(x):
        return x.reshape(x.shape[:-2] + (-1,))

    def dot(x, y):
        return torch.matmul(x[..., None, :], y[..., None])[..., 0, 0]

    labels = list(set([label for pair in matrices for label in pair]))
    n = len(labels)
    f = len(basis)
    if init is None:
        init = basis.new_zeros([n, len(basis)])
    z = init.clone()

    x = flat(torch.stack(list(matrices.values()), -3))
    keys = [(labels.index(i), labels.index(j)) for i, j in matrices.keys()]

    loss_prev = float('inf')
    alpha = 1
    retry = False
    z0 = z.clone()
    for n_iter in range(1000):

        y, gz, hz = expm(z, basis, grad_X=True, hess_X=True)
        iy, igz, ihz = expm(-z, basis, grad_X=True, hess_X=True)

        h = z.new_zeros([n, f, n, f])
        g = z.new_zeros([n, f])
        loss = 0

        for (i, j), xij in zip(keys, x):
            yij = flat(torch.matmul(y[i], iy[j]))      # [D*D]
            r = yij - xij                              # [D*D]
            loss += dot(r, r) / len(r)

        loss = loss.item()
        loss /= len(x)
        print(f"{loss_prev:12.9f} - {loss:12.9f} = {loss_prev - loss:12.9f} | {alpha:12.9f} {'(retry)' if retry else ''}")

        if not retry:
            if abs(loss_prev - loss) < 1e-9 or alpha < 1e-9:
                break

        if loss_prev < loss:
            z = z0.clone()
            alpha *= 0.1
            retry = True
        else:
            alpha *= 1.1
            retry = False
            loss_prev = loss
            z0 = z.clone()

        for (i, j), xij in zip(keys, x):
            yij = flat(torch.matmul(y[i], iy[j]))      # [D*D]
            r = yij - xij                              # [D*D]

            gi = flat(torch.matmul(gz[i], iy[j]))      # [F, D*D]
            hi = torch.matmul(hz[i], iy[j])            # [F, F, D, D]
            hi = flat(torch.abs(hi).sum(-3))           # [F, D*D]
            hii = torch.matmul(gi, gi.T)               # [F, F]

            gj = -flat(torch.matmul(y[i], igz[j]))     # [F, D*D]
            hj = torch.matmul(y[i], ihz[j])            # [F, F, D, D]
            hj = flat(torch.abs(hj).sum(-3))           # [F, D*D]
            hjj = torch.matmul(gj, gj.T)               # [F, F]

            hij = torch.matmul(gi, gj.T)               # [F, F]

            g[i, :] += dot(gi, r)                         # [F]
            hii[range(f), range(f)] += dot(hi, r.abs())   # [F]
            hii[range(f), range(f)] *= 1.001
            h[i, :, i, :] += hii

            g[j, :] += dot(gj, r)                          # [F]
            hjj[range(f), range(f)] += dot(hj, r.abs())    # [F]
            hjj[range(f), range(f)] *= 1.001
            h[j, :, j, :] += hjj

            h[i, :, j, :] += hij
            h[j, :, i, :] += hij

        g = g.reshape([n*f])
        h = h.reshape([n*f, n*f])
        delta = lmdiv(h, g[..., None])[..., 0]    # [N*F]
        delta = delta.reshape([n, f])
        z -= delta * alpha
        z -= torch.mean(z, dim=-2, keepdim=True)  # zero-center

    return z
