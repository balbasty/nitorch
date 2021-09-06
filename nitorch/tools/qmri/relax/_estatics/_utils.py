import torch
from nitorch import core, spatial


def hessian_matmul(hess, grad):
    """Matrix-multiplication specialized to the ESTATICS sparse hessian.

    `>>> hess @ grad`

    The Hessian of the likelihood term is sparse with structure:
    [[D, b],
     [b', r]]
    where D = diag(d) is diagonal.
    It is stored in a flattened form: [d0, b0, d1, b1, ..., dP, bP, r]

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    grad : (P+1, ...) tensor
    

    Returns
    -------
    mm : (P+1, ...) tensor

    """
    mm = torch.zeros_like(grad)
    mm[:-1] = hess[:-1:2] * grad[:-1] + hess[1:-1:2] * grad[-1:]
    mm[-1] = (hess[1:-1:2] * grad[:-1]).sum(0) + hess[-1] * grad[-1:]
    return mm


def hessian_loaddiag(hess, eps=None, eps2=None):
    """Load the diagonal of the (sparse) Hessian

    ..warning:: Modifies `hess` in place

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    eps : float, optional

    Returns
    -------
    hess

    """
    if eps is None:
        eps = core.constants.eps(hess.dtype)
    weight = hess[::2].max(dim=0, keepdim=True).values
    weight.clamp_min_(eps)
    weight *= eps
    if eps2:
        weight += eps2
    hess[::2] += weight
    return hess


def hessian_solve(hess, grad, lam=None):
    """Left matrix division specialized to the ESTATICS sparse hessian.

    The Hessian of the likelihood term is sparse with structure:
    `[[D, b], [b.T, r]]` where `D = diag(d)` is diagonal.
    It is stored in a flattened form: `[d0, b0, d1, b1, ..., dP, bP, r]`

    Because of this specific structure, the Hessian is inverted in
    closed-form using the formula for the inverse of a 2x2 block matrix.
    See: https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion

    Parameters
    ----------
    hess : (2*P+1, ...) tensor
    grad : (P+1, ...) tensor
    lam : float or (P+1,) sequence[float], optional
        Smoothing term added to the diagonal of H

    Returns
    -------
    result : (P+1, ...) tensor

    """

    backend = dict(dtype=hess.dtype, device=hess.device)
    nb_prm = len(grad)
    
    
    # H = [[diag, vec], [vec.T, scal]]
    diag = hess[:-1:2]
    vec = hess[1:-1:2]
    scal = hess[-1]

    if lam is not None:
        # add smoothing term
        lam = torch.as_tensor(lam, **backend).flatten()
        lam = torch.cat([lam, lam[-1].expand(nb_prm-len(lam))])
        lam = lam.reshape([len(lam)] + [1] * (hess.dim()-1))
        diag = diag + lam[:-1]
        scal = scal + lam[1]
                                              
    # precompute stuff
    vec_norm = vec/diag
    mini_inv = scal - (vec*vec_norm).sum(dim=0)
    result = torch.empty_like(grad)

    # top left corner
    result[:-1] = ((vec_norm * grad[:-1]).sum(dim=0) / mini_inv) * vec_norm
    result[:-1] += grad[:-1]/diag

    # top right corner:
    result[:-1] -= vec_norm * grad[-1] / mini_inv

    # bottom left corner:
    result[-1] = - (vec_norm * grad[:-1]).sum(dim=0) / mini_inv

    # bottom right corner:
    result[-1] += grad[-1] / mini_inv

    return result


def smart_grid(aff, shape, inshape=None):
    """Generate a sampling grid iff it is not the identity.
    
    Parameters
    ----------
    aff : (D+1, D+1) tensor
        Affine transformation matrix (voxels to voxels)
    shape : (D,) tuple[int]
        Output shape
    inshape : (D,) tuple[int], optional
        Input shape

    Returns
    -------
    grid : (*shape, D) tensor or None
        Sampling grid
    
    """
    backend = dict(dtype=aff.dtype, device=aff.device)
    identity = torch.eye(aff.shape[-1], **backend)
    inshape = inshape or shape
    if torch.allclose(aff, identity) and shape == inshape:
        return None
    return spatial.affine_grid(aff, shape)


def smart_pull(tensor, grid, **opt):
    """Pull iff grid is defined (+ add/remove batch dim).

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*output_shape, D) tensor or None
        Sampling grid

    Returns
    -------
    pulled : (channels, *output_shape) tensor
        Sampled volume

    """
    if grid is None:
        return tensor
    return spatial.grid_pull(tensor[None], grid[None], **opt)[0]


def smart_pull1d(tensor, grid, dim=-1, **opt):
    """Pull iff grid is defined (+ add/remove batch dim).

    This is a 1D version used for distortion correction.
    It is advised to use negative indices for `dim`.

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*output_shape) tensor or None
        Sampling grid

    Returns
    -------
    pulled : (channels, *output_shape) tensor
        Sampled volume

    """
    if grid is None:
        return tensor
    channels = len(tensor)
    out_shape = grid.shape
    out_shape_t = list(out_shape)
    out_shape_t[dim], out_shape_t[-1] = out_shape_t[-1], out_shape_t[dim]

    # tensor -> [other_spatial, channel, pulled_spatial]
    # grid   -> [other_spatial, pulled_spatial, 1]
    tensor = tensor.transpose(dim, -1).reshape([channels, -1, tensor.shape[-1]])
    tensor = tensor.transpose(0, 1)
    grid = grid.transpose(dim, -1).reshape([-1, grid.shape[-1], 1])

    tensor = spatial.grid_pull(tensor, grid, **opt)

    tensor = tensor.transpose(0, 1).reshape([channels, out_shape_t])
    tensor = tensor.transpose(dim, -1)
    return tensor


def smart_push1d(tensor, grid, dim=-1, shape=None, **opt):
    """Push iff grid is defined (+ add/remove batch dim).

    This is a 1D version used for distortion correction.
    It is advised to use negative indices for `dim`.

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*input_shape) tensor or None
        Sampling grid

    Returns
    -------
    pushed : (channels, *shape) tensor
        Pushed volumhe

    """
    if grid is None:
        return tensor
    channels = len(tensor)
    shape = list(shape or grid.shape)
    shape_t = list(shape)
    shape_t[dim], shape_t[-1] = shape_t[-1], shape_t[dim]

    # tensor -> [other_spatial, channel, pulled_spatial]
    # grid   -> [other_spatial, pulled_spatial, 1]
    tensor = tensor.transpose(dim, -1).reshape([channels, -1, tensor.shape[-1]])
    tensor = tensor.transpose(0, 1)
    grid = grid.transpose(dim, -1).reshape([-1, grid.shape[-1], 1])

    tensor = spatial.grid_push(tensor, grid, shape=shape_t[-1], **opt)

    tensor = tensor.transpose(0, 1).reshape([channels, shape_t])
    tensor = tensor.transpose(dim, -1)
    return tensor


def smart_grad1d(tensor, grid, dim=-1, **opt):
    """Pull gradient iff grid is defined (+ add/remove batch dim).

    This is a 1D version used for distortion correction.
    It is advised to use negative indices for `dim`.

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*output_shape) tensor or None
        Sampling grid

    Returns
    -------
    pulled : (channels, *output_shape) tensor
        Sampled 1d gradient

    """
    if grid is None:
        opt.pop('extrapolate', None)
        opt.pop('interpolation', None)
        return spatial.diff1d(tensor, dim=dim, **opt)

    channels = len(tensor)
    out_shape = grid.shape
    out_shape_t = list(out_shape)
    out_shape_t[dim], out_shape_t[-1] = out_shape_t[-1], out_shape_t[dim]

    # tensor -> [other_spatial, channel, pulled_spatial]
    # grid   -> [other_spatial, pulled_spatial, 1]
    tensor = tensor.transpose(dim, -1).reshape([channels, -1, tensor.shape[-1]])
    tensor = tensor.transpose(0, 1)
    grid = grid.transpose(dim, -1).reshape([-1, grid.shape[-1], 1])

    tensor = spatial.grid_grad(tensor, grid, **opt).squeeze(-1)

    tensor = tensor.transpose(0, 1).reshape([channels, out_shape_t])
    tensor = tensor.transpose(dim, -1)
    return tensor


def exp_backward_1d(vel, *g_and_h, dim=-1, **opt):
    has_hess = len(g_and_h) > 0
    g_and_h = list(g_and_h)
    g = g_and_h.pop(0)
    h = g_and_h.pop(0) if g_and_h else None

    out_shape = vel.shape
    out_shape_t = list(out_shape)
    out_shape_t[dim], out_shape_t[-1] = out_shape_t[-1], out_shape_t[dim]

    # g -> [other_spatial, 1, pulled_spatial]
    # h -> [other_spatial, 1, pulled_spatial]
    # vel   -> [other_spatial, pulled_spatial, 1]
    vel = vel.transpose(dim, -1).reshape([-1, vel.shape[-1], 1])
    g = g.transpose(dim, -1).reshape([1, -1, g.shape[-1]]).transpose(0, 1)
    if h is not None:
        h = h.transpose(dim, -1).reshape([1, -1, h.shape[-1]]).transpose(0, 1)

    g, h = spatial.exp_backward(vel, g, h, **opt)

    g = g.transpose(0, 1).reshape([1, out_shape_t]).transpose(dim, -1)
    if h is not None:
        h = h.transpose(0, 1).reshape([1, out_shape_t]).transpose(dim, -1)
    return (g, h) if has_hess else g


def smart_grad(tensor, grid, **opt):
    """Pull gradients iff grid is defined (+ add/remove batch dim).

    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*output_shape, D) tensor or None
        Sampling grid

    Returns
    -------
    pulled : (channels, *output_shape) tensor
        Sampled volume

    """
    if grid is None:
        opt.pop('extrapolate', None)
        opt.pop('interpolation', None)
        return spatial.diff(tensor, dim=3, **opt)
    return spatial.grid_grad(tensor[None], grid[None], **opt)[0]


def smart_push(tensor, grid, shape=None, **opt):
    """Pull iff grid is defined (+ add/remove batch dim).
    
    Parameters
    ----------
    tensor : (channels, *input_shape) tensor
        Input volume
    grid : (*input_shape, D) tensor or None
        Sampling grid
    shape : (D,) tuple[int], default=input_shape
        Output shape

    Returns
    -------
    pushed : (channels, *output_shape) tensor
        Sampled volume
    
    """
    if grid is None:
        return tensor
    return spatial.grid_push(tensor[None], grid[None], shape, **opt)[0]

