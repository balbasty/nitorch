import torch
from nitorch import core


def first_to_last(x):
    perm = list(range(1, x.dim())) + [0]
    return x.permute(perm)


def last_to_first(x):
    perm = [-1] + list(range(x.dim()-1))
    return x.permute(perm)


def hessian_matvec(hess, grad, inplace=False):
    """Matrix-multiplication specialized to the ESTATICS sparse hessian.

    Parameters
    ----------
    hess : (P|P*P|P*(P+1)//2, ...) tensor
        Hessian matrix.
        If sparse, should be ordered from the main to the smallest
        diagonal.
    grad : (P, ...) tensor
    inplace: bool, default=False

    Returns
    -------
    mm : (P, ...) tensor

    """
    grad = first_to_last(torch.as_tensor(grad))
    hess = first_to_last(torch.as_tensor(hess))
    nb_param = grad.shape[-1]
    is_diag = hess.shape[-1] == nb_param
    is_full = hess.shape[-1] == nb_param**2

    if is_diag:
        if inplace:
            grad *= hess
            return last_to_first(grad)
        else:
            return last_to_first(grad * hess)
    elif is_full:
        hess = hess.reshape(hess.shape[:-1] + (nb_param, nb_param))
        grad = core.linalg.matvec(hess, grad)
        return last_to_first(grad)
    else:
       raise NotImplementedError


def patch_dry_run(shape, patch_size, window_size=None):

    nb_dim = len(shape)
    patch_size = core.pyutils.make_list(patch_size, nb_dim)
    patch_size = [pch or dim for pch, dim in zip(patch_size, shape)]
    post_patch_shape = [s // p for s, p in zip(shape, patch_size)]

    window_size = core.pyutils.make_tuple(window_size, nb_dim)
    window_size = [pch or dim for pch, dim in zip(window_size, post_patch_shape)]
    post_window_shape = [s // p for s, p in zip(post_patch_shape, window_size)]

    return tuple(post_window_shape), tuple(window_size), tuple(patch_size)


def extract_patches(x, patch_size, window_size=None):
    """

    Parameters
    ----------
    x : (*shape)
        Tensor
    patch_size : int or sequence[int or None]
        Size of patches to extract
    window_size : int or None or sequence[int or None], default=None
        Meta patch size, used to group neighbouring patches together

    Returns
    -------
    patches : (nb_windows, nb_patches, *patch_size)

    """
    # TODO: option to pad the input so that no data is lost due to rounding

    nb_dim = x.dim()
    shape = x.shape

    # Extract patches
    patch_size = core.pyutils.make_list(patch_size, nb_dim)
    patch_size = [pch or dim for pch, dim in zip(patch_size, shape)]
    for d, sz in enumerate(patch_size):
        x = x.unfold(dimension=d, size=sz, step=sz)

    # group patches within a window
    if window_size is not None:
        window_size = core.pyutils.make_tuple(window_size, nb_dim)
        window_size = [pch or dim for pch, dim in zip(window_size, shape)]
        for d, sz in enumerate(window_size):
            x = x.unfold(dimension=d, size=sz, step=sz)
        x = x.reshape((-1, *patch_size, core.pyutils.prod(window_size)))
        x = last_to_first(x).transpose(0, 1)
    else:
        x = x.reshape((1, -1, *patch_size))

    return x
