from nitorch.core import utils, linalg


def slice_correct(x, dim=-1, nb_iter=20):

    n = x.shape[dim]
    x = utils.movedim(x, dim, -1)
    shape = x.shape
    x = x.reshape([-1, n])

    vmax = x.max()

    m = x > 0
    x = x.log()
    x[~m] = 0
    a = x.new_zeros([n])
    g = x.new_zeros([n])
    h = x.new_zeros([n, n])

    for i in range(nb_iter):

        # compute forward differences
        d = a + x

        d = d[:, 1:] - d[:, :-1]
        d[~(m[:, 1:] & m[:, :-1])] = 0
        w = d.abs()
        print(w.mean().item())

        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow((a+x).reshape(shape)[shape[0]//2].exp(), vmin=0, vmax=vmax)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(w.reshape([*shape[:-1], n-1]).mean(0))
        plt.colorbar()
        plt.show()

        w = w.clamp_min_(1e-5).reciprocal_()
        w[~(m[:, 1:] & m[:, :-1])] = 0

        # compute gradient
        g.zero_()
        g = g.reshape([n])
        g[1:] = (w * d).sum(0)
        g[:-1] -= (w * d).sum(0)

        # compute hessian
        h.zero_()
        h.diagonal(0, -1, -2)[1:] = w.sum(0)
        h.diagonal(0, -1, -2)[:-1] += w.sum(0)
        h.diagonal(1, -1, -2)[:] = -w.sum(0)
        h.diagonal(-1, -1, -2)[:] = h.diagonal(1, -1, -2)

        h = h.reshape([n, n])
        g = g.reshape([n, 1])
        g /= len(x)
        h /= len(x)
        h.diagonal(0, -1, -2).add_(h.diagonal(0, -1, -2).max() * 1e-6)
        a -= linalg.lmdiv(h, g).reshape([n])

        # zero center
        a -= a.mean()

    x = (a + x).exp()
    x = x.reshape(shape).movedim(-1, dim)
    return x, a


