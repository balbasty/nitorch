import torch
from nitorch import core
from nitorch.core.utils import movedim, make_vector, unsqueeze
from nitorch.core.py import make_list
from nitorch.core.linalg import sym_matvec, sym_solve
from ._finite_differences import diff, div, diff1d, div1d
from ._conv import spconv
import itertools


def absolute(field, weights=None):
    """Precision matrix for the Absolute energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : tensor

    """
    field = torch.as_tensor(field)
    if weights is not None:
        backend = dict(dtype=field.dtype, device=field.device)
        weights = torch.as_tensor(weights, **backend)
        return field * weights
    else:
        return field


def absolute_grid(grid, voxel_size=1, weights=None):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = grid * voxel_size.square()
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)
        grid = grid * weights[..., None]
    return grid


def membrane(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Membrane energy

    Note
    ----
    .. This is exactly equivalent to SPM's membrane energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial) tensor

    """
    def mul_(x, y):
        """Smart in-place multiplication"""
        if ((torch.is_tensor(x) and x.requires_grad) or
                (torch.is_tensor(y) and y.requires_grad)):
            return x * y
        else:
            return x.mul_(y)

    field = torch.as_tensor(field)
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    voxel_size = make_vector(voxel_size, dim, **backend)
    dims = list(range(field.dim()-dim, field.dim()))
    fieldf = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    if weights is not None:
        backend = dict(dtype=fieldf.dtype, device=fieldf.device)
        weights = torch.as_tensor(weights, **backend)
        fieldf = mul_(fieldf, weights[..., None])
        # backward gradients (not needed in l2 case)
        fieldb = diff(field, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
        fieldb = mul_(fieldb, weights[..., None])
        dims = list(range(fieldb.dim() - 1 - dim, fieldb.dim() - 1))
        fieldb = div(fieldb, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    dims = list(range(fieldf.dim()-1-dim, fieldf.dim()-1))
    field = div(fieldf, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    del fieldf
    if weights is not None:
        field += fieldb
        field *= 0.5
    return field


def _membrane_l2(field, voxel_size=1, bound='dct2', dim=None):
    """Precision matrix for the Membrane energy

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's membrane energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()

    Returns
    -------
    field : (..., *spatial) tensor

    """
    field = torch.as_tensor(field)
    backend = core.utils.backend(field)
    dim = dim or field.dim()
    voxel_size = make_vector(voxel_size, dim, **backend)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = [2 * vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [-vx[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=field.device)
    kernel = torch.as_tensor(kernel, **backend)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    # perform convolution
    return spconv(field, kernel, bound=bound, dim=dim)


def membrane_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    if (voxel_size != 1).any():
        grid = grid * voxel_size
    grid = movedim(grid, -1, -(dim + 1))
    grid = membrane(grid, weights=weights, voxel_size=voxel_size,
                    bound=bound, dim=dim)
    grid = movedim(grid, -(dim + 1), -1)
    return grid


def bending(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Bending energy

    Note
    ----
    .. This is exactly equivalent to SPM's bending energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial) tensor

    """
    field = torch.as_tensor(field)
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    voxel_size = make_vector(voxel_size, dim, **backend)
    bound = make_list(bound, dim)
    dims = list(range(field.dim()-dim, field.dim()))
    if weights is not None:
        backend = dict(dtype=field.dtype, device=field.device)
        weights = torch.as_tensor(weights, **backend)

    mom = 0
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj)
                    if weights is not None:
                        dj = dj * weights
                    dj = div1d(dj, **optj)
                    dj = div1d(dj, **opti)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj = dj * 2
                    mom += dj
    mom = mom / 4.
    return mom


def bending_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    if (voxel_size != 1).any():
        grid = grid * voxel_size
    grid = movedim(grid, -1, -(dim + 1))
    grid = bending(grid, weights=weights, voxel_size=voxel_size,
                   bound=bound, dim=dim)
    grid = movedim(grid, -(dim + 1), -1)
    return grid


def lame_shear(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Shear component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the second Lame constant (the shear modulus).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the Frobenius norm of the symmetric part
       of the Jacobian (shears on the off-diagonal and zooms on the
       diagonal).
    .. Formaly: `<f, Lf> = \int \sum_i (df_i/dx_i)^2
                                + \sum_{j > i} (df_j/dx_i + df_i/dx_j)^2 dx

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    if (voxel_size != 1).any():
        grid = grid * voxel_size
    bound = make_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)

    mom = [0] * dim
    for i in range(dim):
        # symmetric part
        x_i = grid[..., i]
        for j in range(dim):
            for side_i in ('f', 'b'):
                opt_ij = dict(dim=dims[j], side=side_i, bound=bound[j],
                              voxel_size=voxel_size[j])
                diff_ij = diff1d(x_i, **opt_ij)
                if i == j:
                    # diagonal elements
                    diff_ij_w = diff_ij if weights is None else diff_ij * weights
                    mom[i] += 2 ** (dim-1) * div1d(diff_ij_w, **opt_ij)
                else:
                    # off diagonal elements
                    x_j = grid[..., j]
                    for side_j in ('f', 'b'):
                        opt_ji = dict(dim=dims[i], side=side_j, bound=bound[i],
                                      voxel_size=voxel_size[i])
                        diff_ji = diff1d(x_j, **opt_ji)
                        diff_ji = (diff_ij + diff_ji) / 2.
                        if weights is not None:
                            diff_ji = diff_ji * weights
                        mom[j] += div1d(diff_ji, **opt_ji)
                        mom[i] += div1d(diff_ji, **opt_ij)
                    del x_j
        del x_i
    del grid

    mom = torch.stack(mom, dim=-1)
    mom = mom / float(2 ** (dim-1))  # weight sides combinations
    return mom


def lame_div(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Divergence component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the first Lame constant (the divergence).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the square of the trace of the Jacobian
       (i.e., volume changes)
    .. Formaly: `<f, Lf> = \int (\sum_i df_i/dx_i)^2 dx

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    grid = torch.as_tensor(grid)
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = core.utils.make_vector(voxel_size, dim, **backend)
    if (voxel_size != 1).any():
        grid = grid * voxel_size
    bound = make_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)

    # precompute gradients
    grad = [dict(f={}, b={}) for _ in range(dim)]
    opt = [dict(f={}, b={}) for _ in range(dim)]
    for i in range(dim):
        x_i = grid[..., i]
        for side in ('f', 'b'):
            opt_i = dict(dim=dims[i], side=side, bound=bound[i],
                         voxel_size=voxel_size[i])
            grad[i][side] = diff1d(x_i, **opt_i)
            opt[i][side] = opt_i

    # compute divergence
    mom = [0] * dim
    all_sides = list(itertools.product(['f', 'b'], repeat=dim))
    for sides in all_sides:
        div = 0
        for i, side in enumerate(sides):
            div += grad[i][side]
        if weights is not None:
            div = div * weights
        for i, side in enumerate(sides):
            mom[i] += div1d(div, **(opt[i][side]))

    mom = torch.stack(mom, dim=-1)
    mom = mom / float(2 ** dim)  # weight sides combinations
    return mom


# aliases to avoid shadowing
_absolute = absolute
_membrane = membrane
_bending = bending


def regulariser_grid(v, absolute=0, membrane=0, bending=0, lame=0,
                     factor=1, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for a mixture of energies for a deformation grid.

    Parameters
    ----------
    v : (..., *spatial, dim) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1
    bound : str, default='dft'
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    v = torch.as_tensor(v)
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    voxel_size = make_vector(voxel_size, dim, **backend)
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = make_vector(lame, 2, **backend)
    lame = [l*factor for l in lame]
    fdopt = dict(bound=bound, voxel_size=voxel_size)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = make_list(wl, 2)

    y = 0
    if absolute:
        y += absolute_grid(v, weights=wa) * absolute
    if membrane:
        y += membrane_grid(v, weights=wm, **fdopt) * membrane
    if bending:
        y += bending_grid(v, weights=wb, **fdopt) * bending
    if lame[0]:
        y += lame_div(v, weights=wl[0], **fdopt) * lame[0]
    if lame[1]:
        y += lame_shear(v, weights=wl[1], **fdopt) * lame[1]

    if y is 0:
        y = torch.zeros_like(v)
    return y


def regulariser(x, absolute=0, membrane=0, bending=0, factor=1,
                voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for a mixture of energies.

    Parameters
    ----------
    x : (..., K, *spatial) tensor
    absolute : (sequence of) float, default=0
    membrane : (sequence of) float, default=0
    bending : (sequence of) float, default=0
    factor : (sequence of) float, default=1
    voxel_size : (sequence of) float, default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lx : (..., K, *spatial) tensor

    """
    x = torch.as_tensor(x)
    backend = dict(dtype=x.dtype, device=x.device)
    dim = dim or x.dim() - 1
    nb_prm = x.shape[-1]
    channel2last = lambda x: movedim(x, -(dim + 1), -1)
    last2channel = lambda x: movedim(x, -1, -(dim + 1))

    voxel_size = make_vector(voxel_size, dim, **backend)
    factor = make_vector(factor, nb_prm, **backend)
    absolute = make_vector(absolute, **backend) * factor
    membrane = make_vector(membrane, **backend) * factor
    bending = make_vector(bending, **backend) * factor
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
    else:
        wa = wm = wb = weights
    # wa = core.utils.unsqueeze(wa, 0, max(0, dim+1-wa.dim()))
    # wm = core.utils.unsqueeze(wm, 0, max(0, dim+1-wa.dim()))
    # wb = core.utils.unsqueeze(wb, 0, max(0, dim+1-wa.dim()))

    y = 0
    if any(absolute):
        y += channel2last(_absolute(x, weights=wa)) * absolute
    if any(membrane):
        y += channel2last(_membrane(x, weights=wm, **fdopt)) * membrane
    if any(bending):
        y += channel2last(_bending(x, weights=wb, **fdopt)) * bending

    if y is 0:
        y = torch.zeros_like(x)
    else:
        y = last2channel(y)
    return y


def solve_field_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                    factor=1, voxel_size=1, bound='dct2', dim=None,
                    weights=None):
    """Solve a positive-definite linear system of the form (H + L)x = g

    Parameters
    ----------
    hessian : (..., 1 or K or K*(K+1)//2, *spatial) tensor
    gradient : (..., K, *spatial) tensor
    absolute : (sequence of) float, default=0
    membrane : (sequence of) float, default=0
    bending : (sequence of) float, default=0
    factor : (sequence of) float, default=1
    voxel_size : (sequence of) float, default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.

    """
    hessian, gradient = core.utils.to_max_backend(hessian, gradient)
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = dim or gradient.dim() - 1
    ch2last = lambda x: (movedim(unsqueeze(x, 0, max(0, dim+1-x.dim())),
                                 -(dim + 1), -1)
                         if x is not None else x)
    last2ch = lambda x: (movedim(x, -1, -(dim + 1))
                         if x is not None else x)
    hessian = ch2last(hessian)
    gradient = ch2last(gradient)
    nb_prm = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    factor = make_vector(factor, nb_prm, **backend)
    absolute = make_vector(absolute, **backend) * factor
    membrane = make_vector(membrane, **backend) * factor
    bending = make_vector(bending, **backend) * factor
    no_reg = not (any(membrane) or any(bending))

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
    else:
        wa = wm = wb = weights
    if wa is not None:
        wa = unsqueeze(wa, 0, max(0, dim+1-wa.dim()))
    if wm is not None:
        wm = unsqueeze(wm, 0, max(0, dim+1-wm.dim()))
    if wb is not None:
        wb = unsqueeze(wb, 0, max(0, dim+1-wb.dim()))

    def regulariser(x):
        x = last2ch(x)
        y = 0
        if any(absolute):
            y += ch2last(_absolute(x, weights=wa)) * absolute
        if any(membrane):
            y += ch2last(_membrane(x, weights=wm, **fdopt)) * membrane
        if any(bending):
            y += ch2last(_bending(x, weights=wb, **fdopt)) * bending
        return y

    # diagonal of the regulariser
    smo = 0
    if any(absolute):
        smo += absolute * ch2last(absolute_diag(weights=wa))
    if any(membrane):
        smo += membrane * ch2last(membrane_diag(weights=wm, **fdopt))
    if any(bending):
        smo += bending * ch2last(bending_diag(weights=wb, **fdopt))

    if is_diag:
        hessian_smo = hessian + smo
    else:
        hessian_smo = hessian.clone()
        hessian_smo[..., :nb_prm] += smo
    precond = ((lambda x: x / hessian_smo) if is_diag else
               (lambda x: sym_solve(hessian_smo, x)))
    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))

    if no_reg:
        result = precond(gradient)
    else:
        result = core.optim.cg(forward, gradient, precond=precond,
                               max_iter=100)
    return last2ch(result)


def solve_grid_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                   lame=0, factor=1, voxel_size=1, bound='dft', weights=None,
                   verbose=False):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Parameters
    ----------
    hessian : (..., *spatial, 1 or D or D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    """
    hessian, gradient = core.utils.to_max_backend(hessian, gradient)
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = [l*factor for l in make_vector(lame, 2, **backend)]
    no_reg = not (membrane or bending or any(lame))

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = make_list(wl, 2)

    def regulariser(v):
        y = 0
        if absolute:
            y += absolute_grid(v, weights=wa, voxel_size=voxel_size) * absolute
        if membrane:
            y += membrane_grid(v, weights=wm, **fdopt) * membrane
        if bending:
            y += bending_grid(v, weights=wb, **fdopt) * bending
        if lame[0]:
            y += lame_div(v, weights=wl[0], **fdopt) * lame[0]
        if lame[1]:
            y += lame_shear(v, weights=wl[1], **fdopt) * lame[1]
        return y

    # diagonal of the regulariser
    vx2 = voxel_size.square()
    ivx2 = vx2.reciprocal()
    smo = 0
    if absolute:
        if wa is not None:
            smo = smo + absolute * vx2 * wa
        else:
            smo = smo + absolute * vx2
    if membrane:
        if wm is not None:
            smo = smo + 2 * membrane * ivx2.sum() * vx2 * wm
        else:
            smo = smo + 2 * membrane * ivx2.sum() * vx2
    if bending:
        val = torch.combinations(ivx2, r=2).prod(dim=-1).sum()
        if wb is not None:
            smo = smo + bending * (8 * val + 6 * ivx2.square().sum()) * vx2 * wb
        else:
            smo = smo + bending * (8 * val + 6 * ivx2.square().sum()) * vx2
    if lame[0]:
        if wl[0] is not None:
            smo = smo + 2 * lame[0] * wl[0]
        else:
            smo = smo + 2 * lame[0]
    if lame[1]:
        if wl[1] is not None:
            smo = smo + 2 * lame[1] * (ivx2.sum() + ivx2)/ivx2 * wl[1]
        else:
            smo = smo + 2 * lame[1] * (ivx2.sum() + ivx2)/ivx2

    hessian_smo = hessian.clone()
    hessian_smo[..., :dim] += smo
    # precond = ((lambda x: x / hessian_smo) if is_diag else
    #            (lambda x: sym_solve(hessian_smo, x)))
    # forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
    #            (lambda x: sym_matvec(hessian, x) + regulariser(x)))

    precond = ((lambda x, s: x / hessian_smo[s]) if is_diag else
               (lambda x, s: sym_solve(hessian_smo[s], x)))
    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))

    if no_reg:
        result = precond(gradient)
    else:
        result = core.optim.relax(forward, gradient, precond=precond,
                                  max_iter=16, verbose=verbose, stop='residuals')
    return result


# ---- WORK IN PROGRESS ----


def absolute_diag(weights=None):
    """Diagonal of the membrane regulariser.

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    if weights is None:
        return 1
    else:
        return weights


def membrane_weights(field, lam=1, voxel_size=1, bound='dct2',
                     dim=None, joint=True, return_sum=False):
    """Update the (L1) weights of the membrane energy.

    Parameters
    ----------
    field : (..., K, *spatial) tensor
        Field
    lam : float or (K,) sequence[float], default=1
        Regularisation factor
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions
    joint : bool, default=False
        Joint norm across channels.
    return_sum : bool, default=False

    Returns
    -------
    weight : (..., 1 or K, *spatial) tensor
        Weights for the reweighted least squares scheme
    """
    field = torch.as_tensor(field)
    backend = core.utils.backend(field)
    dim = dim or field.dim() - 1
    nb_prm = field.shape[-dim-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    lam = make_vector(lam, nb_prm, **backend)
    lam = core.utils.unsqueeze(lam, -1, dim+1)
    if joint:
        lam = lam * nb_prm
    dims = list(range(field.dim()-dim, field.dim()))
    fieldb = diff(field, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    field = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    field.square_().mul_(lam)
    field += fieldb.square_().mul_(lam)
    field /= 2.
    dims = [-1] + ([-dim-2] if joint else [])
    field = field.sum(dim=dims, keepdims=True)[..., 0].sqrt_()
    if return_sum:
        ll = field.sum()
        return field.clamp_min_(1e-5).reciprocal_(), ll
    else:
        return field.clamp_min_(1e-5).reciprocal_()


def bending_weights(field, lam=1, voxel_size=1, bound='dct2',
                     dim=None, joint=True, return_sum=False):
    """Update the (L1) weights of the membrane energy.

    Parameters
    ----------
    field : (..., K, *spatial) tensor
        Field
    lam : [(K,) sequence of] float, default=1
        Regularisation factor
    voxel_size : [(dim,) sequence of] float, default=1
        Voxel size
    bound : [(dim,) sequence of] str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions
    joint : bool, default=False
        Joint norm across channels.
    return_sum : bool, default=False

    Returns
    -------
    weight : (..., 1 or K, *spatial) tensor
        Weights for the reweighted least squares scheme
    """
    field = torch.as_tensor(field)
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim() - 1
    field = unsqueeze(field, 0, max(0, dim+1-field.dim()))
    nb_prm = field.shape[-dim-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    bound = make_list(bound, dim)
    lam = make_vector(lam, nb_prm, **backend)
    lam = core.utils.unsqueeze(lam, -1, dim)
    if joint:
        lam = lam * nb_prm
    dims = list(range(field.dim()-dim, field.dim()))

    diffs = []
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj).square_()
                    # normalize by number of sides
                    # + off diagonal -> x2  (upper + lower element)
                    dj /= (2. if i != j else 4.)
                    diffs.append(dj)
    field = sum(diffs).mul_(lam)
    if joint:
        field = field.sum(dim=-dim-1, keepdims=True)
    field = field.sqrt_()
    if return_sum:
        ll = field.sum()
        return field.clamp_min_(1e-5).reciprocal_(), ll
    else:
        return field.clamp_min_(1e-5).reciprocal_()


def membrane_diag(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membrane regulariser.

    If no weight map is provided, the diagonal of the membrane regulariser
    is a scaled identity with scale `2 * alpha`, where
    `alpha = vx.reciprocal().square().sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a first order "diamond" shape:
                                    b0
                                b1  a   b1
                                    b0

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = core.utils.make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = core.utils.make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
        # move spatial dimensions to the front
        spdim = list(range(weights.dim() - dim, weights.dim()))
        weights = core.utils.movedim(weights, spdim, list(range(dim)))
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()

    if weights is None:
        return 2 * vx.sum()

    from ._finite_differences import _window1d, _lincomb
    values = [[weights]]
    dims = [None] + [d for d in range(dim) for _ in range(2)]
    kernel = [2 * vx.sum()]
    for d in range(dim):
        values.extend(_window1d(weights, d, [-1, 1], bound=bound))
        kernel += [vx[d], vx[d]]
    weights = _lincomb(values, kernel, dims, ref=weights)

    # send spatial dimensions to the back
    weights = core.utils.movedim(weights, list(range(dim)), spdim)
    return weights


def membrane_diag_alt(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membane regulariser.

    This is an alternate implementation of `membrane_diag` that uses the
    sparse convolution code, which relies on linear combinations of views
    into the input tensor. It seems to be a fair bit slower than the
    implementation that relies on moving windows (maybe because it
    does a bit more indexing?).
    For higher orders, we don't have a choice and must use the sparse
    conv code.

    If no weight map is provided, the diagonal of the membrane regulariser
    is a scaled identity with scale `2 * alpha`, where
    `alpha = vx.reciprocal().square().sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a first order "diamond" shape:
                                    b0
                                b1  a   b1
                                    b0
    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()

    if weights is None:
        return 2 * vx.sum()

    # build sparse kernel
    kernel = [2 * vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [vx[d], vx[d]]
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=backend['device'])
    kernel = torch.as_tensor(kernel, **backend)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    weights = spconv(weights, kernel, bound=bound, dim=dim)
    return weights


def bending_diag(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the bending regulariser.

    If no weight map is provided, the diagonal of the bending regulariser
    is a scaled identity with scale `6 * alpha + 8 * beta`, where
        - `alpha = vx.reciprocal().square().sum()`
        - `beta = vx.combinations(r=2).prod(-1).sum()`
    However, is a weight map is provided, the diagonal of the regulariser
    is a convolved version of the weight map. In 2D, the convolution kernel
    has a second order "diamond" shape:
                                    c1
                                d   b0  d
                            c1  b1  a   b1  c1
                                d   b0  d
                                    c0

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()
    cvx = torch.combinations(vx, r=2).prod(dim=-1)
    ivx = torch.combinations(torch.arange(dim), r=2)

    if weights is None:
        return 6 * vx.sum() + 8 * cvx.sum()

    # build sparse kernel
    kernel = [10 * vx.sum() + 8 * cvx.sum()]
    center_index = [2] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        subcvx = sum(cvx1 for cvx1, ivx1 in zip(cvx, ivx) if d in ivx1)
        w1 = 6 * vx[d].square() + 4 * subcvx
        w2 = vx[d].square()
        kernel += [w1, w1, w2, w2]
        index = list(center_index)
        index[d] = 1
        indices.append(index)
        index = list(center_index)
        index[d] = 3
        indices.append(index)
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 4
        indices.append(index)
        # corners
        for dd in range(d+1, dim):
            subcvx = [cvx1 for cvx1, ivx1 in zip(cvx, ivx)
                      if ivx1[0] == d and ivx1[1] == dd][0]
            kernel += [subcvx] * 4
            index = list(center_index)
            index[d] = 1
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 1
            index[dd] = 3
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 3
            indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=backend['device'])
    kernel = torch.as_tensor(kernel, **backend)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [5] * dim)

    weights = spconv(weights, kernel, bound=bound, dim=dim)
    return weights


def membrane_diag_old(voxel_size=1, bound='dct2', dim=None, weights=None):
    """Diagonal of the membrane regulariser.

    This is an old implementation that hapeens to be slower than
    a simpler/more generic one.

    Parameters
    ----------
    weights : (..., *spatial) tensor
        Weights from the reweighted least squares scheme
    voxel_size : float or sequence[float], default=1
        Voxel size
    bound : str, default='dct2'
        Boundary condition.
    dim : int, optional
        Number of spatial dimensions.
        Default: from voxel_size

    Returns
    -------
    diag : () or (..., *spatial) tensor
        Convolved weight map if provided.
        Else, central convolution weight.

    """
    vx = core.utils.make_vector(voxel_size)
    if dim is None:
        dim = len(vx)
    vx = core.utils.make_vector(vx, dim)
    if weights is not None:
        weights = torch.as_tensor(weights)
        backend = dict(dtype=weights.dtype, device=weights.device)
        # move spatial dimensions to the front
        spdim = list(range(weights.dim() - dim, weights.dim()))
        weights = core.utils.movedim(weights, spdim, list(range(dim)))
    else:
        backend = dict(dtype=vx.dtype, device=vx.device)
    vx = vx.to(**backend)
    vx = vx.square().reciprocal()

    if weights is None:
        return 2 * vx.sum()

    w = weights  # alias for readability
    out = (2 * vx.sum()) * w

    # define out-of-bound coordinate on the left and right
    # and the eventual value transformer depending on the
    # boundary condition
    f = (1 if bound == 'dct1' else -1 if bound == 'dft' else 0)
    l = (-2 if bound == 'dct1' else 0 if bound == 'dft' else -1)
    m = ((lambda w: -w) if bound == 'dst2' else
         (lambda w: 0) if bound in ('dst1', 'zero') else
         (lambda w: w))

    if dim == 3:
        # center
        out[1:-1, 1:-1, 1:-1] += (
                (w[:-2, 1:-1, 1:-1] + w[2:, 1:-1, 1:-1]) * vx[0] +
                (w[1:-1, :-2, 1:-1] + w[1:-1, 2:, 1:-1]) * vx[1] +
                (w[1:-1, 1:-1, :-2] + w[1:-1, 1:-1, 2:]) * vx[2])
        # sides
        out[0, 1:-1, 1:-1] += (
                (m(w[f, 1:-1, 1:-1]) + w[1, 1:-1, 1:-1]) * vx[0] +
                (w[0, :-2, 1:-1] + w[0, 2:, 1:-1]) * vx[1] +
                (w[0, 1:-1, :-2] + w[0, 1:-1, 2:]) * vx[2])
        out[-1, 1:-1, 1:-1] += (
                (w[-2, 1:-1, 1:-1] + m(w[l, 1:-1, 1:-1])) * vx[0] +
                (w[-1, :-2, 1:-1] + w[-1, 2:, 1:-1]) * vx[1] +
                (w[-1, 1:-1, :-2] + w[-1, 1:-1, 2:]) * vx[2])
        out[1:-1, 0, 1:-1] += (
                (w[:-2, 0, 1:-1] + w[2:, 0, 1:-1]) * vx[0] +
                (m(w[1:-1, f, 1:-1]) + w[1:-1, 1, 1:-1]) * vx[1] +
                (w[1:-1, 0, :-2] + w[1:-1, 0, 2:]) * vx[2])
        out[1:-1, -1, 1:-1] += (
                (w[:-2, -1, 1:-1] + w[2:, -1, 1:-1]) * vx[0] +
                (w[1:-1, -2, 1:-1] + m(w[1:-1, l, 1:-1])) * vx[1] +
                (w[1:-1, -1, :-2] + w[1:-1, -1, 2:]) * vx[2])
        out[1:-1, 1:-1, 0] += (
                (w[:-2, 1:-1, 0] + w[2:, 1:-1, 0]) * vx[0] +
                (w[1:-1, :-2, 0] + w[1:-1, 2:, 0]) * vx[1] +
                (m(w[1:-1, 1:-1, f]) + w[1:-1, 1:-1, 1]) * vx[2])
        out[1:-1, 1:-1, -1] += (
                (w[:-2, 1:-1, -1] + w[2:, 1:-1, -1]) * vx[0] +
                (w[1:-1, :-2, -1] + w[1:-1, 2:, -1]) * vx[1] +
                (w[1:-1, 1:-1, -2] + m(w[1:-1, 1:-1, l])) * vx[2])
        # edges
        out[0, 0, 1:-1] += (
                (m(w[f, 0, 1:-1]) + w[1, 0, 1:-1]) * vx[0] +
                (m(w[0, f, 1:-1]) + w[0, 1, 1:-1]) * vx[1] +
                (w[0, 0, :-2] + w[0, 0, 2:]) * vx[2])
        out[0, -1, 1:-1] += (
                (m(w[f, -1, 1:-1]) + w[1, -1, 1:-1]) * vx[0] +
                (w[0, -2, 1:-1] + m(w[0, l, 1:-1])) * vx[1] +
                (w[0, -1, :-2] + w[0, -1, 2:]) * vx[2])
        out[-1, 0, 1:-1] += (
                (w[-1, 0, 1:-1] + m(w[l, 0, 1:-1])) * vx[0] +
                (m(w[-1, f, 1:-1]) + w[-1, 1, 1:-1]) * vx[1] +
                (w[-1, 0, :-2] + w[-1, 0, 2:]) * vx[2])
        out[-1, -1, 1:-1] += (
                (w[-2, -1, 1:-1] + m(w[l, -1, 1:-1])) * vx[0] +
                (w[-1, -2, 1:-1] + m(w[-1, l, 1:-1])) * vx[1] +
                (w[-1, -1, :-2] + w[-1, -1, 2:]) * vx[2])
        out[0, 1:-1, 0] += (
                (m(w[f, 1:-1, 0]) + w[1, 1:-1, 0]) * vx[0] +
                (w[0, :-2, 0] + w[0, 2:, 0]) * vx[1] +
                (m(w[0, 1:-1, f]) + w[0, 1:-1, 1]) * vx[2])
        out[0, 1:-1, -1] += (
                (m(w[f, 1:-1, -1]) + w[1, 1:-1, -1]) * vx[0] +
                (w[0, :-2, -1] + w[0, 2:, -1]) * vx[1] +
                (w[0, 1:-1, -2] + m(w[0, 1:-1, l])) * vx[2])
        out[-1, 1:-1, 0] += (
                (w[-2, 1:-1, 0] + w[l, 1:-1, 0]) * vx[0] +
                (w[-1, :-2, 0] + w[-1, 2:, 0]) * vx[1] +
                (w[-1, 1:-1, f] + w[-1, 1:-1, -1]) * vx[2])
        out[-1, 1:-1, -1] += (
                (w[-2, 1:-1, -1] + m(w[l, 1:-1, -1])) * vx[0] +
                (w[-1, :-2, -1] + w[-1, 2:, -1]) * vx[1] +
                (w[-1, 1:-1, -2] + m(w[-1, 1:-1, l])) * vx[2])
        out[1:-1, 0, 0] += (
                (w[:-2, 0, 0] + w[2:, 0, 0]) * vx[0] +
                (w[1:-1, f, 0] + w[1:-1, 1, 0]) * vx[1] +
                (w[1:-1, 0, f] + w[1:-1, 0, 1]) * vx[2])
        out[1:-1, 0, -1] += (
                (w[:-2, 0, -1] + w[2:, 0, -1]) * vx[0] +
                (w[1:-1, f, -1] + w[1:-1, 1, -1]) * vx[1] +
                (w[1:-1, 0, -2] + w[1:-1, 0, l]) * vx[2])
        out[1:-1, -1, 0] += (
                (w[:-2, 0, 0] + w[2:, -1, 0]) * vx[0] +
                (w[1:-1, -2, 0] + w[1:-1, l, 0]) * vx[1] +
                (w[1:-1, 0, f] + w[1:-1, -1, 1]) * vx[2])
        out[1:-1, -1, -1] += (
                (w[:-2, -1, -1] + w[2:, -1, -1]) * vx[0] +
                (w[1:-1, -2, -1] + m(w[1:-1, l, -1])) * vx[1] +
                (w[1:-1, -1, -2] + m(w[1:-1, -1, l])) * vx[2])

        # corners
        out[0, 0, 0] += ((m(w[f, 0, 0]) + w[1, 0, 0]) * vx[0] +
                         (m(w[0, f, 0]) + w[0, 1, 0]) * vx[1] +
                         (m(w[0, 0, f]) + w[0, 0, 1]) * vx[2])
        out[0, 0, -1] += ((m(w[f, 0, -1]) + w[1, 0, -1]) * vx[0] +
                          (m(w[0, f, -1]) + w[0, 1, -1]) * vx[1] +
                          (w[0, 0, -2] + m(w[0, 0, l])) * vx[2])
        out[0, -1, 0] += ((m(w[f, -1, 0]) + w[1, -1, 0]) * vx[0] +
                          (w[0, -2, 0] + m(w[0, l, 0])) * vx[1] +
                          (m(w[0, -1, f]) + w[0, -1, 1]) * vx[2])
        out[0, -1, -1] += ((m(w[f, -1, -1]) + w[1, -1, -1]) * vx[0] +
                           (w[0, -2, -1] + m(w[0, l, -1])) * vx[1] +
                           (w[0, -1, -2] + m(w[0, -1, l])) * vx[2])
        out[-1, 0, 0] += ((w[-2, 0, 0] + w[l, 0, 0]) * vx[0] +
                          (m(w[-1, f, 0]) + w[-1, 1, 0]) * vx[1] +
                          (m(w[-1, 0, f]) + w[-1, 0, 1]) * vx[2])
        out[-1, 0, -1] += ((w[-2, 0, -1] + m(w[l, 0, -1])) * vx[0] +
                           (m(w[-1, f, -1]) + w[-1, 1, -1]) * vx[1] +
                           (w[-1, 0, -2] + m(w[-1, 0, l])) * vx[2])
        out[-1, -1, 0] += ((w[-2, -1, 0] + m(w[l, -1, 0])) * vx[0] +
                           (w[-1, -2, 0] + m(w[-1, l, 0])) * vx[1] +
                           (m(w[-1, -1, f]) + w[-1, -1, 1]) * vx[2])
        out[-1, -1, -1] += ((w[-2, -1, -1] + m(w[l, -1, -1])) * vx[0] +
                            (w[-1, -2, -1] + m(w[-1, l, -1])) * vx[1] +
                            (w[-1, -1, -2] + m(w[-1, -1, l])) * vx[2])
    elif dim == 2:
        # center
        out[1:-1, 1:-1] += ((w[:-2, 1:-1] + w[2:, 1:-1]) * vx[0] +
                            (w[1:-1, :-2] + w[1:-1, 2:]) * vx[1])
        # edges
        out[0, 1:-1] += ((m(w[f, 1:-1]) + w[1, 1:-1]) * vx[0] +
                         (w[0, :-2] + w[0, 2:]) * vx[1])
        out[-1, 1:-1] += ((w[-2, 1:-1] + m(w[l, 1:-1])) * vx[0] +
                          (w[-1, :-2] + w[-1, 2:]) * vx[1])
        out[1:-1, 0] += ((w[:-2, 0] + w[2:, 0]) * vx[0] +
                         (m(w[1:-1, f]) + w[1:-1, 1]) * vx[1])
        out[1:-1, -1] += ((w[:-2, -1] + w[2:, -1]) * vx[0] +
                          (w[1:-1, -2] + m(w[1:-1, l])) * vx[1])
        # corners
        out[0, 0] += ((m(w[f, 0]) + w[1, 0]) * vx[0] +
                      (m(w[0, f]) + w[0, 1]) * vx[1])
        out[0, -1] += ((m(w[f, -1]) + w[1, -1]) * vx[0] +
                       (w[0, -2] + m(w[0, l])) * vx[1])
        out[-1, 0] += ((w[-1, 0] + m(w[l, 0])) * vx[0] +
                       (m(w[-1, f]) + w[-1, 1]) * vx[1])
        out[-1, -1] += ((w[-2, -1] + m(w[l, -1])) * vx[0] +
                        (w[-1, -2] + w[-1, l]) * vx[1])
    elif dim == 1:
        # center
        out[1:-1] += (w[:-2] + w[2:]) * vx[0]
        # corners
        out[0] += (m(w[f]) + w[1]) * vx[0]
        out[-1] += (w[-2] + m(w[l])) * vx[0]
    else:
        raise NotImplementedError

    # send spatial dimensions to the back
    out = core.utils.movedim(out, list(range(dim)), spdim)
    return out
