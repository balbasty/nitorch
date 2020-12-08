import torch
from nitorch import core
from nitorch.tools.img_statistics import estimate_noise
from ._utils import extract_patches, patch_dry_run
import math


def denoise(x, dict_size=16, patch_size=5, window_size=None,
            max_iter=200, max_iter_rls=100, mode='ppca', reg=10.):
    """Patch-based denoising.

    Parameters
    ----------
    x : (...) tensor
        Input image
    dict_size : int, default=16
        Number of basis function
        (Must be less than prod(patch_size) in 'ppca' mode)
    patch_size : int or sequence[int], default=5
        Patch size. Each patch is an independent observation.
    window_size : int or sequence[int], default=None
        Window size.
        If used, one dictionary per window of patches is computed.
        Else the dictionary is shared by all patches across the image.
    mode : {'ppca', 'sca', 'vlca'}
        - 'ppca': probabilistic principal component analysis
            (Gaussian prior on the coordinates, which are marginalized)
        - 'sca': sparse component analysis
            (Laplace prior on the coordinates, Gaussian prior on the dict,
             maximum likelihood)
        - 'vlca': variational Laplace component analysis
            (Laplace prior on the coordinates, which are marginalized)

    Returns
    -------
    denoised : (...)

    """

    x = torch.as_tensor(x)
    nb_dim = x.dim()
    shape = x.shape
    backend = dict(dtype=x.dtype, device=x.device)
    eps = core.constants.eps(x.dtype)

    # estimate noise s.d. from background
    lam, _, tau, _ = estimate_noise(x)
    lam = lam.to(**backend).square_().reciprocal_()  # noise precision
    tau = tau.to(**backend).square_().reciprocal_()  # signal precision

    # prepare patch and window size
    subshape, window_size, patch_size0 = patch_dry_run(shape, patch_size, window_size)

    # extract patches
    x = extract_patches(x, patch_size, window_size)
    nb_windows = x.shape[0]
    nb_patches = x.shape[1]
    x = x.reshape((nb_windows, nb_patches, -1))
    patch_size = x.shape[-1]

    if mode == 'sca':
        w = x.new_ones((nb_windows, nb_patches, dict_size))
    else:
        w = x.new_ones(())
        max_iter_rls = 1
    z = torch.randn((nb_windows, nb_patches, dict_size), **backend)
    u = x.new_zeros((nb_windows, dict_size, patch_size))
    m = x.mean(dim=1, keepdim=True)
    x -= m

    # NOTES
    # x : (nb_windows, nb_patches, patch_size)    zero-centered observed patches
    # m : (nb_windows, 1,          patch_size)    mean patch
    # u : (nb_windows, dict_size,  patch_size)    dictionary/basis of patches
    # z : (nb_windows, nb_patches, dict_size)     coordinates in the basis
    # w : (nb_windows, nb_patches, dict_size)     weights that approximate L1

    uu = 2*torch.eye(dict_size, **backend)
    uu = uu[None, None, :, :]
    ll = torch.as_tensor(core.constants.inf, **backend)

    for n_iter_rls in range(max_iter_rls):
        # I think that the problem is not jointly convex
        # I need to reach the optimum of the L2 problem before updating
        # the weights.

        ll_prev_rls = ll

        for n_iter_em in range(max_iter):

            ll_prev = ll

            # update dictionary
            zz = torch.matmul(z.transpose(-1, -2), z)
            zz *= lam
            if mode == 'ppca':
                zz += uu[:, 0, :, :]  # inv(lam * uu + eye)
            elif mode == 'sca':
                eye = torch.eye(dict_size, **backend)
                eye *= tau
                zz += eye
            zz = core.linalg.inv(zz, method='svd')
            xz = torch.matmul(z.transpose(-1, -2), x)
            xz *= lam
            u = torch.matmul(zz, xz, out=u)
            del xz, zz

            ll = 0.5 * lam * (torch.matmul(z, u) - x).square_().sum(dtype=torch.double) \
            + 0.5 * tau * u.square().sum(dtype=torch.double) \
            + 0.5 * reg * (z.square() * w).sum(dtype=torch.double) \
            + 0.5 * w.reciprocal().sum(dtype=torch.double)
            print('{:4d} | {:4d} | {:10s}: {:12.6g}'.format(
                n_iter_rls, n_iter_em, 'update u', ll.item()))

            # update coordinates
            uu = torch.matmul(u, u.transpose(-1, -2))
            uu *= lam
            if mode == 'sca':
                uu = uu[:, None, :, :] + reg * torch.diag_embed(w)
            elif mode == 'ppca':
                uu += torch.eye(dict_size, **backend)
                uu = uu[:, None, :, :]
            uu = core.linalg.inv(uu, method='svd')
            ux = torch.matmul(x, u.transpose(-1, -2))
            ux *= lam
            z = core.linalg.matvec(uu, ux, out=z)

            ll = 0.5 * lam * (torch.matmul(z, u) - x).square_().sum(dtype=torch.double) \
            + 0.5 * tau * u.square().sum(dtype=torch.double) \
            + 0.5 * reg * (z.square() * w).sum(dtype=torch.double) \
            + 0.5 * w.reciprocal().sum(dtype=torch.double)
            print('{:4d} | {:4d} | {:10s}: {:12.6g}'.format(
                n_iter_rls, n_iter_em, 'update z', ll.item()))

            if n_iter_rls == 0:
                # orthogonalize
                uu = torch.matmul(u, u.transpose(-1, -2))
                vu, su, _ = torch.svd(uu)
                su = su.clamp_min_(eps).sqrt_()
                zz = torch.matmul(z.transpose(-1, -2), z)
                vz, sz, _ = torch.svd(zz)
                sz = sz.clamp_min_(eps).sqrt_()
                ww, ss, vv = torch.svd(torch.matmul(
                    vu.transpose(-1, -2) * su[:, :, None],
                    vz * sz[:, None, :]))
                q = torch.matmul(vv.transpose(-1, -2) * ss[:, :, None],
                                 vz.transpose(-1, -2) / sz[:, :, None])
                iq = torch.matmul(vu, ww / su[:, :, None])
                u = core.linalg.matvec(iq.transpose(-1, -2)[:, None, ...],
                                       u.transpose(-1, -2)).transpose(-1, -2)
                z = core.linalg.matvec(q[:, None, ...], z)

            # rescale
            nrmz = z.square()
            nrmz *= w
            nrmz *= reg
            nrmz = nrmz.sum(dim=1)
            nrmu = u.square().sum(dim=-1)
            nrmu *= tau
            alpha = (nrmz / nrmu).sqrt_().sqrt_()
            u *= alpha[:, :, None]
            z /= alpha[:, None, :]

            # recompute uu
            uu = torch.matmul(u, u.transpose(-1, -2))
            uu *= lam
            if mode == 'sca':
                uu = uu[:, None, :, :] + reg * torch.diag_embed(w)
            elif mode == 'ppca':
                uu += torch.eye(dict_size, **backend)
                uu = uu[:, None, :, :]
            uu = core.linalg.inv(uu, method='svd')

            ll = 0.5 * lam * (torch.matmul(z, u) - x).square_().sum(dtype=torch.double) \
             + 0.5 * tau * u.square().sum(dtype=torch.double) \
             + 0.5 * reg * (z.square() * w).sum(dtype=torch.double) \
             + 0.5 * w.reciprocal().sum(dtype=torch.double)
            print('{:4d} | {:4d} | {:10s}: {:12.6g}'.format(
                n_iter_rls, n_iter_em, 'rescale', ll.item()))

            gain = (ll_prev - ll)/ll_prev
            if abs(gain.item()) < eps:
                break

        # update weights
        if mode == 'sca':
            w = torch.abs(z, out=w)
            w *= math.sqrt(reg)
            w += 1e-3
            w = w.reciprocal_()

            ll = 0.5 * lam * (torch.matmul(z, u) - x).square_().sum(dtype=torch.double) \
             + 0.5 * tau * u.square().sum(dtype=torch.double) \
             + 0.5 * reg * (z.square() * w).sum(dtype=torch.double) \
             + 0.5 * w.reciprocal().sum(dtype=torch.double)
            print('{:4d} | {:4d} | {:10s}: {:12.6g}'.format(
                n_iter_rls, n_iter_em, 'rescale', ll.item()))

            gain = (ll_prev_rls - ll)/ll_prev_rls
            if abs(gain.item()) < eps:
                break

        y = m + torch.matmul(z, u)
        show_slice(unfold(x + m, subshape, window_size, patch_size0),
                   unfold(y, subshape, window_size, patch_size0))

    # reconstruct
    y = m + torch.matmul(z, u)
    y = unfold(y, subshape, window_size, patch_size0)

    return y


def unfold(y, subshape, window_size, patch_size):
    nb_dim = len(patch_size)
    y = y.reshape((*subshape, *window_size, *patch_size))
    perm = []
    shape = []
    for d in range(nb_dim):
        perm.extend([d, d + nb_dim, d + 2 * nb_dim])
        shape.append(subshape[d] * window_size[d] * patch_size[d])
    y = y.permute(perm)
    y = y.reshape(shape)
    return y


def show_slice(x, y):
    import matplotlib.pyplot as plt

    if x.dim() == 3:
        x = x[:, :, x.shape[-1]//2]
    if y.dim() == 3:
        y = y[:, :, x.shape[-1]//2]

    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(y)
    plt.axis('off')
    plt.show()


def showbar(z, k):
    import matplotlib.pyplot as plt
    cnt = torch.histc(z[0, :, k], 64)
    x = torch.linspace(z[0, :, k].min(), z[0, :, k].max(), 64)
    plt.bar(x, cnt, (x[-1] - x[0]) / 128.)
    plt.show()
