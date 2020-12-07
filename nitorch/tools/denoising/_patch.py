import torch
from nitorch import core
from nitorch.tools.img_statistics import estimate_noise
from ._utils import extract_patches, patch_dry_run


def denoise(x, dict_size=16, patch_size=5, window_size=None, max_iter=10,
            mode='ppca'):
    """

    Parameters
    ----------
    x : (...) tensor
    dict_size : int, default=16
    patch_size : int or sequence[int], default=5
    window_size : int or sequence[int], default=None
    mode : {'ppca', 'sca', 'vlca'}

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
    lam = estimate_noise(x).to(*backend)
    lam = lam.square_().reciprocal_()

    # prepare patch and window size
    subshape, window_size, patch_size = patch_dry_run(shape, patch_size, window_size)

    # extract patches
    x = extract_patches(x, patch_size, window_size)
    nb_windows = x.shape[0]
    nb_patches = x.shape[1]
    x = x.reshape((nb_windows, nb_patches, -1))
    patch_size = x.shape[-1]

    w = x.new_ones((nb_windows, nb_patches))
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

    uu = torch.eye(dict_size, **backend)

    for n_iter in range(max_iter):

        # update dictionary
        zz = torch.matmul(z.transpose(-1, -2), z)
        if mode == 'ppca':
            zz += uu
        elif mode == 'sca':
            zz += torch.eye(dict_size, **backend)
        zz = core.linalg.inv(zz, method='chol')
        xz = torch.matmul(z.transpose(-1, -2), x)
        u = torch.matmul(zz, xz, out=u)
        del xz, zz

        # update coordinates
        uu = torch.matmul(u, u.transpose(-1, -2))
        uu *= lam
        if mode == 'sca':
            uu = uu[:, None, :, :] + torch.diag_embed(w)
        elif mode == 'ppca':
            uu += torch.eye(dict_size, **backend)
        uu = core.linalg.inv(uu, method='chol')
        ux = torch.matmul(u, x.transpose(-1, -2))
        z = z.transpose(-1, -2)
        z = torch.matmul(uu, ux, out=z).transpose(-1, -2)

        # update weights
        w = torch.abs(z, out=w).clamp_min_(eps).reciprocal_()

    # reconstruct
    y = m + torch.matmul(z, u)

    # reshape
    y = y.reshape((*subshape, *window_size, *patch_size))
    perm = []
    shape = []
    for d in range(nb_dim):
        perm.extend([d, 2*d, 3*d])
        shape.append(subshape[d] * window_size[d] * patch_size[d])
    y = y.permute(perm)
    y = y.reshape(shape)
    return y
