import torch
from nitorch import core
from nitorch.core.utils import movedim, ensure_shape
from nitorch.core.pyutils import make_list
from nitorch.core.linalg import sym_matvec, sym_solve
from ._finite_differences import diff, div


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


def membrane(field, voxel_size=1, bound='dct2', dim=None):
    """Precision matrix for the Membrane energy

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
    voxel_size = make_list(voxel_size, dim)
    dims = list(range(field.dim()-dim, field.dim()))
    field = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    field = div(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    return field


def bending(field, voxel_size=1, bound='dct2', dim=None):
    """Precision matrix for the Bending energy

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
    voxel_size = make_list(voxel_size, dim)
    dims = list(range(field.dim()-dim, field.dim()))
    field = diff(field, 2, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    field = div(field, 2, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    return field


# aliases to avoid shadowing
_absolute = absolute
_membrane = membrane
_bending = bending


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

    voxel_size = torch.as_tensor(voxel_size, **backend)
    voxel_size = core.utils.ensure_shape(voxel_size, dim)
    factor = torch.as_tensor(factor, **backend)
    factor = ensure_shape(factor, nb_prm)
    absolute = torch.as_tensor(absolute, **backend) * factor
    membrane = torch.as_tensor(membrane, **backend) * factor
    bending = torch.as_tensor(bending, **backend) * factor
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
    voxel_size = torch.as_tensor(voxel_size, **backend)
    voxel_size = core.utils.ensure_shape(voxel_size, dim)
    is_diag = hessian.shape[-1] == gradient.shape[-1]

    absolute = make_list(absolute, nb_prm)
    absolute = torch.as_tensor(absolute, **backend)
    membrane = make_list(membrane, nb_prm)
    membrane = torch.as_tensor(membrane, **backend)
    bending = make_list(bending, nb_prm)
    bending = torch.as_tensor(bending, **backend)
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
        ivx2 = voxel_size.square().reciprocal().prod()
        smo = smo + bending * (8 * ivx2 + 6 * ivx4)

    hessian_smo = hessian + smo
    precond = ((lambda x: x / hessian_smo) if is_diag else
               (lambda x: sym_solve(hessian_smo, x)))
    forward = ((lambda x: x * hessian + regulariser(x)) if is_diag else
               (lambda x: sym_matvec(hessian, x) + regulariser(x)))

    if no_reg:
        result = precond(gradient)
    else:
        result = core.optim.cg(forward, gradient, precond=precond)
    return movedim(result, -1, -dim - 1)

