import torch
from nitorch import core
from nitorch.core.utils import movedim, make_vector
from nitorch.core.pyutils import make_list
from nitorch.core.linalg import sym_matvec, sym_solve
from ._finite_differences import diff, div, diff1d, div1d
import itertools


def absolute(field):
    """Precision matrix for the Absolute energy

    Parameters
    ----------
    field : tensor

    Returns
    -------
    field : tensor

    """
    return field


def absolute_grid(field, voxel_size=1):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    field : (..., *spatial, dim) tensor

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    dim = field.shape[-1]
    voxel_size = make_vector(voxel_size, dim)
    field = field * voxel_size.square()
    return field


def membrane(field, voxel_size=1, bound='dct2', dim=None):
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

    Returns
    -------
    field : (..., *spatial) tensor

    """
    field = torch.as_tensor(field)
    dim = dim or field.dim()
    voxel_size = make_vector(voxel_size, dim)
    dims = list(range(field.dim()-dim, field.dim()))
    field = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    field = div(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    return field


def membrane_grid(field, voxel_size=1, bound='dft'):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    field : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    field = torch.as_tensor(field)
    dim = field.shape[-1]
    field = movedim(field, -1, -(dim + 1))
    field = membrane(field, voxel_size, bound, dim)
    field = movedim(field, -(dim + 1), -1)
    return field


def bending(field, voxel_size=1, bound='dct2', dim=None):
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

    Returns
    -------
    field : (..., *spatial) tensor

    """
    field = torch.as_tensor(field)
    dim = dim or field.dim()
    voxel_size = make_vector(voxel_size, dim)
    bound = make_list(bound, dim)
    dims = list(range(field.dim()-dim, field.dim()))

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
                    dj = div1d(dj, **optj)
                    dj = div1d(dj, **opti)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj = dj * 2
                    mom += dj
    mom = mom / 4.
    return mom


def bending_grid(field, voxel_size=1, bound='dft'):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    field : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    field = torch.as_tensor(field)
    dim = field.shape[-1]
    field = movedim(field, -1, -(dim + 1))
    field = bending(field, voxel_size, bound, dim)
    field = movedim(field, -(dim + 1), -1)
    return field


def lame_shear(field, voxel_size=1, bound='dft'):
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
    field : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    field = torch.as_tensor(field)
    dim = field.shape[-1]
    voxel_size = make_vector(voxel_size, dim)
    bound = make_list(bound, dim)
    dims = list(range(field.dim()-1-dim, field.dim()-1))

    mom = [0] * dim
    for i in range(dim):
        # symmetric part
        x_i = field[..., i]
        for j in range(dim):
            for side_i in ('f', 'b'):
                opt_ij = dict(dim=dims[j], side=side_i, bound=bound[j],
                              voxel_size=voxel_size[j])
                diff_ij = diff1d(x_i, **opt_ij)
                if i == j:
                    # diagonal elements
                    mom[i] += 2 ** (dim-1) * div1d(diff_ij, **opt_ij)
                else:
                    # off diagonal elements
                    x_j = field[..., j]
                    for side_j in ('f', 'b'):
                        opt_ji = dict(dim=dims[i], side=side_j, bound=bound[i],
                                      voxel_size=voxel_size[i])
                        diff_ji = diff1d(x_j, **opt_ji)
                        diff_ji = (diff_ij + diff_ji) / 2.
                        mom[j] += div1d(diff_ji, **opt_ji)
                        mom[i] += div1d(diff_ji, **opt_ij)
                    del x_j
        del x_i
    del field

    mom = torch.stack(mom, dim=-1)
    mom = mom / float(2 ** (dim-1))  # weight sides combinations
    return mom


def lame_div(field, voxel_size=1, bound='dft'):
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
    field : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    field = torch.as_tensor(field)
    dim = field.shape[-1]
    voxel_size = make_vector(voxel_size, dim)
    bound = make_list(bound, dim)
    dims = list(range(field.dim()-1-dim, field.dim()-1))

    # precompute gradients
    grad = [dict(f={}, b={}) for _ in range(dim)]
    opt = [dict(f={}, b={}) for _ in range(dim)]
    for i in range(dim):
        x_i = field[..., i]
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
                     voxel_size=1, bound='dft'):
    """Precision matrix for a mixture of energies for a deformation grid.

    Parameters
    ----------
    v : (..., *spatial, dim) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    voxel_size : [sequence of] float, default=1
    bound : str, default='dft'

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    v = torch.as_tensor(v)
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    voxel_size = make_vector(voxel_size, dim, **backend)
    lame = make_vector(lame, 2, **backend)
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)

    y = 0
    if absolute:
        y += absolute_grid(v) * absolute
    if membrane:
        y += membrane_grid(v, **fdopt) * membrane
    if bending:
        y += bending_grid(v, **fdopt) * bending
    if lame[0]:
        y += lame_div(v, **fdopt) * lame[0]
    if lame[1]:
        y += lame_shear(v, **fdopt) * lame[1]

    if y is 0:
        y = torch.zeros_like(v)
    return y


def regulariser(x, absolute=0, membrane=0, bending=0, factor=1,
                voxel_size=1, bound='dct2', dim=None):
    """Precision matrix for a mixture of energies.

    Parameters
    ----------
    x : (..., K, *spatial) tensor
    absolute : [sequence of] float, default=0
    membrane : [sequence of] float, default=0
    bending : [sequence of] float, default=0
    factor : [sequence of] float, default=1
    voxel_size : [sequence of] float, default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`

    Returns
    -------
    Lx : (..., K, *spatial) tensor

    """
    x = torch.as_tensor(x)
    backend = dict(dtype=x.dtype, device=x.device)
    dim = dim or x.dim() - 1
    nb_prm = x.shape[-1]

    voxel_size = make_vector(voxel_size, dim, **backend)
    factor = make_vector(factor, nb_prm, **backend)
    absolute = make_vector(absolute, **backend) * factor
    membrane = make_vector(membrane, **backend) * factor
    bending = make_vector(bending, **backend) * factor
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)

    y = 0
    if any(absolute):
        y += movedim(_absolute(x), -(dim + 1), -1) * absolute
    if any(membrane):
        y += movedim(_membrane(x, **fdopt), -(dim + 1), -1) * membrane
    if any(bending):
        y += movedim(_bending(x, **fdopt), -(dim + 1), -1) * bending

    if y is 0:
        y = torch.zeros_like(x)
    else:
        y = movedim(y, -1, -(dim + 1))
    return y


def solve_field_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                    voxel_size=1, bound='dct2', dim=None):
    """Solve a positive-definite linear system of the form (H + L)x = g

    Parameters
    ----------
    hessian : (..., K or K*(K+1)//2, *spatial) tensor
    gradient : (..., K, *spatial) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`

    """
    hessian, gradient = core.utils.to_max_backend(hessian, gradient)
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = dim or gradient.dim() - 1
    hessian = movedim(hessian, -dim-1, -1)
    gradient = movedim(gradient, -dim-1, -1)
    nb_prm = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    absolute = make_vector(absolute, nb_prm, **backend)
    membrane = make_vector(membrane, nb_prm, **backend)
    bending = make_vector(bending, nb_prm, **backend)
    no_reg = not (any(membrane) or any(bending))

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)

    def regulariser(x):
        x = movedim(x, -1, -dim-1)
        y = 0
        if any(absolute):
            y += movedim(_absolute(x), -dim-1, -1) * absolute
        if any(membrane):
            y += movedim(_membrane(x, **fdopt), -dim-1, -1) * membrane
        if any(bending):
            y += movedim(_bending(x, **fdopt), -dim-1, -1) * bending
        return y

    # diagonal of the regulariser
    smo = 0
    if any(absolute):
        smo = smo + absolute
    if any(membrane):
        ivx2 = voxel_size.square().reciprocal().sum()
        smo = smo + 2 * membrane * ivx2
    if any(bending):
        ivx4 = voxel_size.square().square().reciprocal().sum()
        ivx2 = torch.combinations(voxel_size.square().reciprocal(), r=2)
        ivx2 = ivx2.prod(dim=-1).sum()
        smo = smo + bending * (8 * ivx2 + 6 * ivx4)

    hessian_smo = hessian + smo
    precond = ((lambda x: x / hessian_smo) if is_diag else
               (lambda x: sym_solve(hessian_smo, x)))
    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))

    if no_reg:
        result = precond(gradient)
    else:
        result = core.optim.cg(forward, gradient, precond=precond,
                               max_iter=100)
    return movedim(result, -1, -dim - 1)


def solve_grid_sym(hessian, gradient, absolute=0, membrane=0, bending=0,
                   lame=0, voxel_size=1, bound='dft'):
    """Solve a positive-definite linear system of the form (H + L)v = g

    Parameters
    ----------
    hessian : (..., *spatial, D or D*(D+1)//2) tensor
    gradient : (..., *spatial, D) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'

    """
    hessian, gradient = core.utils.to_max_backend(hessian, gradient)
    backend = dict(dtype=hessian.dtype, device=hessian.device)
    dim = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    is_diag = hessian.shape[-1] in (1, gradient.shape[-1])

    lame = make_vector(lame, 2, **backend)
    no_reg = not (membrane or bending or any(lame))

    # regulariser
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)

    def regulariser(v):
        y = 0
        if absolute:
            y += absolute_grid(v) * absolute
        if membrane:
            y += membrane_grid(v, **fdopt) * membrane
        if bending:
            y += bending_grid(v, **fdopt) * bending
        if lame[0]:
            y += lame_div(v, **fdopt) * lame[0]
        if lame[1]:
            y += lame_shear(v, **fdopt) * lame[1]
        return y

    # diagonal of the regulariser
    ivx2 = voxel_size.square().reciprocal()
    smo = 0
    if absolute:
        smo = smo + absolute * voxel_size.square()
    if membrane:
        smo = smo + 2 * membrane * ivx2.sum()
    if bending:
        val = torch.combinations(ivx2, r=2).prod(dim=-1).sum()
        smo = smo + bending * (8 * val + 6 * ivx2.square().sum())
    if lame[0]:
        smo = smo + 2 * lame[0]
    if lame[1]:
        smo = smo + 2 * lame[1] * (ivx2.sum() + ivx2)/ivx2

    hessian_smo = hessian + smo
    precond = ((lambda x: x / hessian_smo) if is_diag else
               (lambda x: sym_solve(hessian_smo, x)))
    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))

    if no_reg:
        result = precond(gradient)
    else:
        result = core.optim.cg(forward, gradient, precond=precond,
                               max_iter=100)
    return result

