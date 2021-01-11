"""Various functions that relate to affine spaces.

NiTorch encodes affine matrices using their Lie algebra representation.
This
"""

# TODO:
#   * Batch ``volume_axis`` and ``volume_layout``
#     -> need to find a way to differentiate "index" from "axis" inputs.

import torch
from ast import literal_eval
from warnings import warn
from nitorch import core
from nitorch.core import utils
from nitorch.core import pyutils
from nitorch.core import itertools
from nitorch.core import linalg
from nitorch.core import constants
import math


def volume_axis(*args, **kwargs):
    """Describe an axis of a volume of voxels.

    Signature
    ---------
    * ``def volume_axis(index, flipped=False, device=None)``
    * ``def volume_axis(name, device=None)``
    * ``def volume_axis(axis, device=None)``

    Parameters
    ----------
    index : () tensor_like[int]
        Index of the axis in 'direct' space (RAS)

    flipped : () tensor_like[bool], default=False
        Whether the axis is flipped or not.

    name : {'R' or 'L', 'A' or 'P', 'S' or 'I'}
        Name of the axis, according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)

    axis : (2,) tensor_like[long]
        ax[0] = index
        ax[1] = flipped
        Description of the axis.

    device : str or torch.device
        Device.

    Returns
    -------
    axis : (2,) tensor[long]
        Description of the axis.
        * `ax[0] = index`
        * `ax[1] = flipped`

    """
    def axis_from_name(name, device=None):
        name = name.upper()
        if name == 'R':
            return utils.as_tensor([0, 0], dtype=torch.long, device=device)
        elif name == 'L':
            return utils.as_tensor([0, 1], dtype=torch.long, device=device)
        elif name == 'A':
            return utils.as_tensor([1, 0], dtype=torch.long, device=device)
        elif name == 'P':
            return utils.as_tensor([1, 1], dtype=torch.long, device=device)
        elif name == 'S':
            return utils.as_tensor([2, 0], dtype=torch.long, device=device)
        elif name == 'I':
            return utils.as_tensor([2, 1], dtype=torch.long, device=device)

    def axis_from_index(index, flipped=False, device=None):
        index = utils.as_tensor(index).reshape(())
        flipped = utils.as_tensor(flipped).reshape(())
        return utils.as_tensor([index, flipped], dtype=torch.long, device=device)

    def axis_from_axis(ax, device=None):
        ax = utils.as_tensor(ax, dtype=torch.long, device=device).flatten()
        if ax.numel() != 2:
            raise ValueError('An axis should have two elements. Got {}.'
                             .format(ax.numel()))
        return ax

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            return axis_from_name(*args, **kwargs)
        else:
            args[0] = utils.as_tensor(args[0])
            if args[0].numel() == 1:
                return axis_from_index(*args, **kwargs)
            else:
                return axis_from_axis(*args, **kwargs)
    else:
        if 'name' in kwargs.keys():
            return axis_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            return axis_from_index(*args, **kwargs)
        else:
            return axis_from_axis(*args, **kwargs)


# Mapping from (index, flipped) to axis name
_axis_names = [['R', 'L'], ['A', 'P'], ['S', 'I']]


def volume_axis_to_name(axis):
    """Return the (neuroimaging) name of an axis. Its index must be < 3.

    Parameters
    ----------
    axis : (2,) tensor_like

    Returns
    -------
    name : str

    """
    index, flipped = axis
    if index >= 3:
        raise ValueError('Index names are only defined up to dimension 3. '
                         'Got {}.'.format(index))
    return _axis_names[index][flipped]


def volume_layout(*args, **kwargs):
    """Describe the layout of a volume of voxels.

    A layout is characterized by a list of axes. See `volume_axis`.

    Signature
    ---------
    volume_layout(dim=3, device=None)
    volume_layout(name, device=None)
    volume_layout(axes, device=None)
    volume_layout(index, flipped=False, device=None)

    Parameters
    ----------
    dim : int, default=3
        Dimension of the space.
        This version of the function always returns a directed layout
        (identity permutation, no flips), which is equivalent to 'RAS'
        but in any dimension.

    name : str
        Permutation of axis names,  according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)
        The number of letters defines the dimension of the matrix
        (`ndim = len(name)`).

    axes : (ndim, 2) tensor_like[long]
        List of objects returned by `axis`.

    index : (ndim, ) tensor_like[long]
        Index of the axes in 'direct' space (RAS)

    flipped : (ndim, ) tensor_like[bool], default=False
        Whether each axis is flipped or not.

    Returns
    -------
    layout : (ndim, 2) tensor[long]
        Description of the layout.

    """
    def layout_from_dim(dim, device=None):
        return volume_layout(list(range(dim)), flipped=False, device=device)

    def layout_from_name(name, device=None):
        return volume_layout([volume_axis(a, device) for a in name])

    def layout_from_index(index, flipped=False, device=None):
        index = utils.as_tensor(index, torch.long, device).flatten()
        ndim = index.shape[0]
        flipped = utils.as_tensor(flipped, torch.long, device).flatten()
        if flipped.shape[0] == 1:
            flipped = torch.repeat_interleave(flipped, ndim, dim=0)
        return torch.stack((index, flipped), dim=-1)

    def layout_from_axes(axes, device=None):
        axes = utils.as_tensor(axes, torch.long, device)
        if axes.dim() != 2 or axes.shape[1] != 2:
            raise ValueError('A layout should have shape (ndim, 2). Got {}.'
                             .format(axes.shape))
        return axes

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            layout = layout_from_name(*args, **kwargs)
        else:
            args[0] = utils.as_tensor(args[0])
            if args[0].dim() == 0:
                layout = layout_from_dim(*args, **kwargs)
            elif args[0].dim() == 2:
                layout = layout_from_axes(*args, **kwargs)
            else:
                layout = layout_from_index(*args, **kwargs)
    else:
        if 'dim' in kwargs.keys():
            layout = layout_from_dim(*args, **kwargs)
        elif 'name' in kwargs.keys():
            layout = layout_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            layout = layout_from_index(*args, **kwargs)
        else:
            layout = layout_from_axes(*args, **kwargs)

    # Remap axes indices if not contiguous
    axes = layout[:, 0]
    new_axes = torch.argsort(axes).to(layout.dtype)
    layout = torch.stack((new_axes, layout[:, 1]), dim=-1)
    return layout


def volume_layout_to_name(layout):
    """Return the (neuroimaging) name of a layout.

    Its length must be <= 3 (else, we just return the permutation and
    flips), e.g. '[2, 3, -1, 4]'

    Parameters
    ----------
    layout : (dim, 2) tensor_like

    Returns
    -------
    name : str

    """
    layout = volume_layout(layout)
    if len(layout) > 3:
        layout = [('-' if bool(f) else '') + str(int(p)) for p, f in layout]
        return '[' + ', '.join(layout) + ']'
    names = [volume_axis_to_name(axis) for axis in layout]
    return ''.join(names)


def iter_layouts(ndim, device=None):
    """Compute all possible layouts for a given dimensionality.

    Parameters
    ----------
    ndim : () tensor_like
        Dimensionality (rank) of the space.

    Returns
    -------
    layouts : (nflip*nperm, ndim, 2) tensor[long]
        All possible layouts.
        * nflip = 2 ** ndim     -> number of flips
        * nperm = ndim!         -> number of permutations

    """
    if device is None:
        if torch.is_tensor(ndim):
            device = ndim.device

    # First, compute all possible directed layouts on one hand,
    # and all possible flips on the other hand.
    axes = torch.arange(ndim, dtype=torch.long, device=device)
    layouts = itertools.permutations(axes)           # [P, D]
    flips = itertools.product([0, 1], r=ndim)        # [F, D]

    # Now, compute combination (= cartesian product) of both
    # We replicate each tensor so that shapes match and stack them.
    nb_layouts = len(layouts)
    nb_flips = len(flips)
    layouts = layouts[None, ...]
    layouts = torch.repeat_interleave(layouts, nb_flips, dim=0)  # [F, P, D]
    flips = flips[:, None, :]
    flips = torch.repeat_interleave(flips, nb_layouts, dim=1)    # [F, P, D]
    layouts = torch.stack([layouts, flips], dim=-1)

    # Finally, flatten across repeats
    layouts = layouts.reshape([-1, ndim, 2])    # [F*P, D, 2]

    return layouts


def layout_matrix(layout, voxel_size=1., shape=0., dtype=None, device=None):
    """Compute the origin affine matrix for different voxel layouts.

    Resources
    ---------
    .. https://nipy.org/nibabel/image_orientation.html
    .. https://nipy.org/nibabel/neuro_radio_conventions.html

    Parameters
    ----------
    layout : str or (ndim, 2) tensor_like[long]
        See `affine.layout`

    voxel_size : (ndim, 1) tensor_like, default=1
        Voxel size of the lattice

    shape : (ndim, 1) tensor_like, default=0
        Shape of the lattice

    dtype : torch.dtype, optional
        Data type of the matrix

    device : torch.device, optional
        Output device.

    Returns
    -------
    mat : (ndim+1, ndim+1) tensor[dtype]
        Corresponding affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Extract info from layout
    layout = volume_layout(layout, device=device)
    device = layout.device
    dim = len(layout)
    perm = utils.invert_permutation(layout[:, 0])
    flip = layout[:, 1].bool()

    # ensure tensor
    voxel_size = torch.as_tensor(voxel_size, dtype=dtype, device=device)
    dtype = voxel_size.dtype
    shape = torch.as_tensor(shape, dtype=dtype, device=device)

    # ensure dim
    shape = utils.ensure_shape(shape, [dim], mode='replicate')
    voxel_size = utils.ensure_shape(voxel_size, [dim], mode='replicate')
    zero = torch.zeros(dim, dtype=dtype, device=device)

    # Create matrix
    mat = torch.diag(voxel_size)
    mat = mat[perm, :]
    mat = torch.cat((mat, zero[:, None]), dim=1)
    mflip = torch.ones(dim, dtype=dtype, device=device)
    mflip = torch.where(flip, -mflip, mflip)
    mflip = torch.diag(mflip)
    shift = torch.where(flip, shape[perm], zero)
    mflip = torch.cat((mflip, shift[:, None]), dim=1)
    mat = affine_matmul(mflip, mat)
    mat = affine_make_square(mat)

    return mat


def affine_to_layout(mat):
    """Find the volume layout associated with an affine matrix.

    Parameters
    ----------
    mat : (..., dim, dim+1) or (..., dim+1, dim+1) tensor_like
        Affine matrix (or matrices).

    Returns
    -------
    layout : (..., dim, 2) tensor[long]
        Volume layout(s) (see `volume_layout`).

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original idea
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Convert input
    mat = utils.as_tensor(mat)
    dtype = mat.dtype
    device = mat.device

    # Extract linear component + remove voxel scaling
    dim = mat.shape[-1] - 1
    mat = mat[..., :dim, :dim]
    vs = (mat ** 2).sum(dim=-1)
    mat = linalg.rmdiv(mat, torch.diag(vs))
    eye = torch.eye(dim, dtype=dtype, device=device)

    # Compute SOS between a layout matrix and the (stripped) affine matrix
    def check_space(space):
        layout = layout_matrix(space)[:dim, :dim]
        sos = ((linalg.rmdiv(mat, layout) - eye) ** 2).sum()
        return sos

    # Compute SOS between each layout and the (stripped) affine matrix
    all_layouts = iter_layouts(dim)
    all_sos = torch.stack([check_space(layout) for layout in all_layouts])
    argmin_layout = torch.argmin(all_sos, dim=0)
    min_layout = all_layouts[argmin_layout, ...]

    return min_layout


affine_subbasis_choices = ('T', 'R', 'Z', 'Z0', 'I', 'S', 'SC')


def affine_subbasis(mode, dim=3, sub=None, dtype=None, device=None):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group required does not require translations. To extract the
    linear part of the basis: lin = basis[:-1, :-1].

    This function focuses on very simple (and coherent) groups.

    Note that shears generated by the 'S' basis are not exactly the same
    as classical shears ('SC'). Setting one classical shear parameter to
    a non-zero value generally applies a gradient of translations along
    a direction:
    + -- +         + -- +
    |    |   =>   /    /
    + -- +       + -- +
    Setting one Lie shear parameter to a non zero value is more alike
    to performing an expansion in one (diagonal) direction and a
    contraction in the orthogonal (diagonal) direction. It is a bit
    harder to draw in ascii, but it can also be seen as a horizontal
    shear followed by a vertical shear.
    The 'S' basis is orthogonal to the 'R' basis, but the 'SC' basis is
    not.

    Parameters
    ----------
    mode : {'T', 'R', 'Z', 'Z0', 'I', 'S', 'SC'}
        Group that should be encoded by the basis set:
            * 'T'   : Translations                     [dim]
            * 'R'   : Rotations                        [dim*(dim-1)//2]
            * 'Z'   : Zooms (= anisotropic scalings)   [dim]
            * 'Z0'  : Isovolumic scalings              [dim-1]
            * 'I'   : Isotropic scalings               [1]
            * 'S'   : Shears (symmetric)               [dim*(dim-1)//2]
            * 'SC'  : Shears (classic)                 [dim*(dim-1)//2]
        If the group name is appended with a list of integers, they
        have the same use as ``sub``. For example 'R[0]' returns the
        first rotation basis only. This grammar cannot be used in
        conjunction with the ``sub`` keyword.

    dim : int, default=3
        Dimension

    sub : int or list[int], optional
        Request only subcomponents of the basis.

    dtype : torch.type, optional
        Data type of the returned array.

    device : torch.device, optional
        Device.

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Check if sub passed in mode
    mode = mode.split('[')
    if len(mode) > 1:
        if sub is not None:
            raise ValueError('Cannot use both ``mode`` and ``sub`` '
                             'to specify a sub-basis.')
        sub = '[' + mode[1]
        sub = literal_eval(sub)  # Safe eval for list of native types
    mode = mode[0]

    if mode not in affine_subbasis_choices:
        raise ValueError('mode must be one of {}.'
                         .format(affine_subbasis_choices))

    # Compute the basis

    if mode == 'T':
        basis = torch.zeros((dim, dim+1, dim+1), dtype=dtype, device=device)
        for i in range(dim):
            basis[i, i, dim] = 1

    elif mode == 'Z':
        basis = torch.zeros((dim, dim+1, dim+1), dtype=dtype, device=device)
        for i in range(dim):
            basis[i, i, i] = 1

    elif mode == 'Z0':
        basis = torch.zeros((dim-1, dim+1), dtype=dtype, device=device)
        for i in range(dim-1):
            basis[i, i] = 1
            basis[i, i+1] = -1
        # Orthogonalise numerically.
        u, s, _ = torch.svd(basis)
        s = s[..., None]
        basis = u.t().mm(basis) / s
        basis = torch.diag_embed(basis)

        # TODO:
        #   Is there an analytical form?
        #   I would say yes. It seems that we have (I only list the diagonals):
        #       2D: [[a, -a]]           with a = 1/sqrt(2)
        #       3D: [[a, 0, -a],        with a = 1/sqrt(2)
        #            [b, -2*b, b]]           b = 1/sqrt(6)
        #       4D: [[a, -b, b, -a],
        #            [c, -c, -c, c],    with c = 1/sqrt(4)
        #            [b, a, -a, -b]]         a = ?, b = ?

    elif mode == 'I':
        basis = torch.eye(dim+1, dtype=dtype, device=device)[None, ...]
        basis /= torch.sqrt(utils.as_tensor(dim, basis.dtype, device))
        basis[:, dim, dim] = 0

    elif mode == 'R':
        nb_rot = dim*(dim-1)//2
        basis = torch.zeros((nb_rot, dim+1, dim+1), dtype=dtype, device=device)
        k = 0
        isqrt2 = 1 / torch.sqrt(utils.as_tensor(2, basis.dtype, device))
        for i in range(dim):
            for j in range(i + 1, dim):
                basis[k, i, j] = isqrt2
                basis[k, j, i] = -isqrt2
                k += 1

    elif mode == 'S':
        nb_shr = dim*(dim-1)//2
        basis = torch.zeros((nb_shr, dim+1, dim+1), dtype=dtype, device=device)
        k = 0
        isqrt2 = 1 / torch.sqrt(utils.as_tensor(2, basis.dtype, device))
        for i in range(dim):
            for j in range(i + 1, dim):
                basis[k, i, j] = isqrt2
                basis[k, j, i] = isqrt2
                k += 1

    elif mode == 'SC':
        nb_shr = dim*(dim-1)//2
        basis = torch.zeros((nb_shr, dim+1, dim+1), dtype=dtype, device=device)
        k = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                basis[k, i, j] = 1
                k += 1

    else:
        # We should never reach this (a test was performed earlier)
        raise ValueError

    # Select subcomponents of the basis
    if sub is not None:
        try:
            sub = list(sub)
        except TypeError:
            sub = [sub]
        basis = torch.stack([basis[i, ...] for i in sub])

    return basis


affine_basis_choices = ('T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+')


def affine_basis(group='SE', dim=3, dtype=None, device=None):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    This function focuses on 'classic' Lie groups. Note that, while it
    is commonly used in registration software, we do not have a
    "9-parameter affine" (translations + rotations + zooms),
    because such transforms do not form a group; that is, their inverse
    may contain shears.

    Parameters
    ----------
    group : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='SE'
        Group that should be encoded by the basis set:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
    dim : {1, 2, 3}, default=3
        Dimension
    dtype : torch.dtype, optional
        Data type of the returned array
    device : torch.device, optional
        Output device

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """
    # TODO:
    # - other groups?

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    if group not in affine_basis_choices:
        raise ValueError('group must be one of {}.'
                         .format(affine_basis_choices))

    if group == 'T':
        return affine_subbasis('T', dim, dtype=dtype, device=device)
    elif group == 'SO':
        return affine_subbasis('R', dim, dtype=dtype, device=device)
    elif group == 'SE':
        return torch.cat((affine_subbasis('T', dim, dtype=dtype, device=device),
                          affine_subbasis('R', dim, dtype=dtype, device=device)))
    elif group == 'D':
        return torch.cat((affine_subbasis('T', dim, dtype=dtype, device=device),
                          affine_subbasis('I', dim, dtype=dtype, device=device)))
    elif group == 'CSO':
        return torch.cat((affine_subbasis('T', dim, dtype=dtype, device=device),
                          affine_subbasis('R', dim, dtype=dtype, device=device),
                          affine_subbasis('I', dim, dtype=dtype, device=device)))
    elif group == 'SL':
        return torch.cat((affine_subbasis('R', dim, dtype=dtype, device=device),
                          affine_subbasis('Z0', dim, dtype=dtype, device=device),
                          affine_subbasis('S', dim, dtype=dtype, device=device)))
    elif group == 'GL+':
        return torch.cat((affine_subbasis('R', dim, dtype=dtype, device=device),
                          affine_subbasis('Z', dim, dtype=dtype, device=device),
                          affine_subbasis('S', dim, dtype=dtype, device=device)))
    elif group == 'Aff+':
        return torch.cat((affine_subbasis('T', dim, dtype=dtype, device=device),
                          affine_subbasis('R', dim, dtype=dtype, device=device),
                          affine_subbasis('Z', dim, dtype=dtype, device=device),
                          affine_subbasis('S', dim, dtype=dtype, device=device)))

    
def affine_basis_size(group, dim=3):  
    """Return the number of parameters in a given group."""
    
    if group not in affine_basis_choices:
        raise ValueError('group must be one of {}.'
                         .format(affine_basis_choices))
    if group == 'T':
        return dim
    elif group == 'SO':
        return dim*(dim-1)//2
    elif group == 'SE':
        return dim + dim*(dim-1)//2
    elif group == 'D':
        return dim + 1
    elif group == 'CSO':
        return dim + dim*(dim-1)//2 + 1
    elif group == 'SL':
        return (dim+1)*(dim-1)
    elif group == 'GL+':
        return dim*dim
    elif group == 'Aff+':
        return (dim+1)*dim
    

def build_affine_basis(*basis, dim=None, dtype=None, device=None):
    """Transform Affine Lie bases into tensors.

    Signatures
    ----------
    basis = build_affine_basis(basis)
    basis1, basis2, ... = build_affine_basis(basis1, basis2, ...)

    Parameters
    ----------
    *basis : basis_like or sequence[basis_like]
        A basis_like is a str or ([F], D+1, D+1) tensor_like that
        describes a basis. If several arguments are provided, each one
        is built independently.
    dim : int, optional
        Dimensionality. If None, infer.
    dtype : torch.dtype, optional
        Output data type
    device : torch.device, optional
        Output device

    Returns
    -------
    *basis : (F, D+1, D+1) tensor
        Basis sets.

    """
    if basis and (isinstance(basis[-1], int) or basis[-1] is None):
        *basis, dim = basis
    opts = dict(dtype=dtype, device=device, dim=dim)
    bases = [_build_affine_basis(b, **opts) for b in basis]
    return bases[0] if len(bases) == 1 else tuple(bases)


def _build_affine_basis(basis, dim=None, dtype=None, device=None):
    """Actual implementation: only one basis set."""

    # Helper to convert named bases to matrices
    def name_to_basis(name, dim, dtype, device):
        basename = name.split('[')[0]
        if basename in affine_subbasis_choices:
            return affine_subbasis(name, dim, dtype=dtype, device=device)
        elif basename in affine_basis_choices:
            return affine_basis(name, dim, dtype=dtype, device=device)
        else:
            raise ValueError('Unknown basis name {}.'.format(basename))

    # make list
    basis = pyutils.make_list(basis)
    built_bases = [b for b in basis if not isinstance(b, str)]

    # check dimension
    dims = [dim] if dim else []
    dims = dims + [b.shape[-1] - 1 for b in built_bases]
    dims = set(dims)
    if not dims:
        dim = 3
    elif len(dims) > 1:
        raise ValueError('Dimension not consistent across bases.')
    else:
        dim = dims.pop()

    # max dtype and device
    dtype = dtype or utils.max_dtype(*built_bases, force_float=True)
    device = device or utils.max_device(*built_bases)
    info = dict(dtype=dtype, device=device)

    # build bases
    basis0 = basis
    basis = []
    for b in basis0:
        if isinstance(b, str):
            b = name_to_basis(b, dim, **info)
        else:
            b = utils.as_tensor(b, **info)
        b = utils.unsqueeze(b, dim=0, ndim=max(0, 3-b.dim()))
        if b.shape[-2] != b.shape[-1] or b.dim() != 3:
            raise ValueError('Expected basis with shape (B, D+1, D+1) '
                             'but got {}'.format(tuple(b.shape)))
        basis.append(b)
    basis = torch.cat(basis, dim=0)

    return basis


def affine_matrix(prm, *basis, dim=None):
    r"""Reconstruct an affine matrix from its Lie parameters.

    Affine matrices are encoded as product of sub-matrices, where
    each sub-matrix is encoded in a Lie algebra.
    ..math:: M   = exp(A_1) \times ... \times exp(A_n)
    ..math:: A_i = \sum_k = p_{ik} B_{ik}

    Examples
    --------
    ```python
    >> prm = torch.randn(6)
    >> # from a classic Lie group
    >> A = affine_matrix_lie(prm, 'SE', dim=3)
    >> # from a user-defined Lie group
    >> A = affine_matrix_lie(prm, ['Z', 'R[0]', 'T[1]', 'T[2]'], dim=3)
    >> # from two Lie groups
    >> A = affine_matrix_lie(prm, 'Z', 'R', dim=3)
    >> B = affine_matrix_lie(prm[:3], 'Z', dim=3)
    >> C = affine_matrix_lie(prm[3:], 'R', dim=3)
    >> assert torch.allclose(A, B @ C)
    >> # from a pre-built basis
    >> basis = affine_basis('SE', dim=3)
    >> A = affine_matrix_lie(prm, basis, dim=3)
    ```

    Parameters
    ----------
    prm : (..., nb_basis)
        Parameters in the Lie algebra(s).

    *basis : basis_like, default='CSO'
        A basis_like is a (sequence of) (F, D+1, D+1) tensor or string.
        The number of parameters (for each batch element) should be equal
        to the total number of bases (the sum of all bases across sub-bases).

    dim : int, default=guess or 3
        If not provided, the function tries to guess it from the shape
        of the basis matrices. If the dimension cannot be guessed
        (because all bases are named bases), the default is 3.

    Returns
    -------
    mat : (..., dim+1, dim+1) tensor
        Reconstructed affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Input parameters
    prm = utils.as_tensor(prm)
    info = dict(dtype=prm.dtype, device=prm.device)

    # Make sure basis is a vector_like of (F, D+1, D+1) tensor_like
    if len(basis) == 0:
        basis = ['CSO']
    if basis and (isinstance(basis[-1], int) or basis[-1] is None):
        *basis, dim = basis
    basis = build_affine_basis(*basis, dim, **info)
    basis = pyutils.make_list(basis)

    # Check length
    nb_basis = sum([len(b) for b in basis])
    if prm.shape[-1] != nb_basis:
        raise ValueError('Number of parameters and number of bases do '
                         'not match. Got {} and {}'
                         .format(len(prm), nb_basis))

    # Helper to reconstruct a matrix
    def recon(p, B):
        return linalg.expm(torch.sum(B*p[..., None, None], dim=-3))

    # Reconstruct each sub matrix
    n_prm = 0
    mats = []
    for a_basis in basis:
        nb_prm = len(a_basis)
        a_prm = prm[..., n_prm:(n_prm+nb_prm)]
        mats.append(recon(a_prm, a_basis))
        n_prm += nb_prm

    # Matrix product
    if len(mats) > 1:
        affine = torch.chain_matmul(mats)
    else:
        affine = mats[0]
    return affine


def affine_matrix_classic(prm=None, dim=3, *,
                          translations=None,
                          rotations=None,
                          zooms=None,
                          shears=None):
    """Build an affine matrix in the "classic" way (no Lie algebra).

    Parameters can either be provided already concatenated in the last
    dimension (`prm=...`) or as individual components (`translations=...`)

    Parameters
    ----------
    prm : (..., K) tensor_like
        Affine parameters, ordered -- in the last dimension -- as
        `[*translations, *rotations, *zooms, *shears]`
        Rotation parameters should be expressed in radians.
    dim : () tensor_like[int]
        Dimensionality.

    Alternative Parameters
    ----------------------
    translations : (..., dim) tensor_like, optional
        Translation parameters (along X, Y, Z).
    rotations : (..., dim*(dim-1)//2) tensor_like, optional
        Rotation parameters, in radians (about X, Y, Z)
    zooms : (..., dim|1) tensor_like, optional
        Zoom parameters (along X, Y, Z).
    shears : (..., dim*(dim-1)//2) tensor_like, optional
        Shear parameters (about XY, XZ, YZ).


    Returns
    -------
    mat : (..., dim+1, dim+1) tensor
        Reconstructed affine matrix `mat = T @ Rx @ Ry @ Rz @ Z @ S`

    """
    def mat_last(mat):
        """Move matrix from first to last dimensions"""
        return mat.permute([*range(2, mat.dim()), 0, 1])

    def vec_first(mat):
        """Move vector from last to first dimension"""
        return mat.permute([-1, *range(mat.dim()-1)])

    # Prepare parameter vector
    if prm is not None:
        # A stacked tensor was provided: we must unstack it and extract
        # each component. We expect them to be ordered as [*T, *R, *Z, *S].
        prm = vec_first(utils.as_tensor(prm))
        nb_prm, *batch_shape = prm.shape
        nb_t = dim
        nb_r = dim*(dim-1) // 2
        nb_z = dim
        nb_s = dim*(dim-1) // 2
        idx = 0
        prm_t = prm[idx:idx+nb_t] if nb_prm > idx else None
        idx = idx + nb_t
        prm_r = prm[idx:idx+nb_r] if nb_prm > idx else None
        idx = idx + nb_r
        prm_z = prm[idx:idx+nb_z] if nb_prm > idx else None
        idx = idx + nb_z
        prm_s = prm[idx:idx+nb_s] if nb_prm > idx else None
    else:
        # Individual components were provided, but some may be None, and
        # they might not have exactly the same batch shape (we only
        # require that their batch shapes can be broadcasted together).
        prm_t = translations if translations is not None else [0] * dim
        prm_r = rotations if rotations is not None else [0] * (dim*(dim-1)//2)
        prm_z = zooms if zooms is not None else [1] * dim
        prm_s = shears if shears is not None else [0] * (dim*(dim-1)//2)
        # Convert and move to a common backend (dtype + device)
        prm_t, prm_r, prm_z, prm_s = utils.to_max_backend(prm_t, prm_r, prm_z, prm_s)
        # Broadcast all batch shapes
        batch_shape = utils.expand(prm_t[..., 0], prm_r[..., 0],
                                   prm_z[..., 0], prm_s[..., 0],
                                   shape=[], dry_run=True)
        prm_t = vec_first(prm_t.expand(batch_shape + prm_t.shape[-1:]))
        prm_r = vec_first(prm_r.expand(batch_shape + prm_r.shape[-1:]))
        prm_z = vec_first(prm_z.expand(batch_shape + prm_z.shape[-1:]))
        prm_s = vec_first(prm_s.expand(batch_shape + prm_s.shape[-1:]))

    backend = dict(dtype=prm_t.dtype, device=prm_t.device)

    if dim == 2:

        def make_affine(t, r, z, sh, o, i):
            if t is not None:
                T = [[i, o, t[0]],
                     [o, i, t[1]],
                     [o, o, i]]
                T = mat_last(utils.as_tensor(T, **backend))
            else:
                T = torch.eye(3, **backend)
                T = utils.expand(T, [*batch_shape, 1, 1])
            if r is not None:
                c = torch.cos(r)
                s = torch.sin(r)
                R = [[c[0],  s[0], o],
                     [-s[0], c[0], o],
                     [o,     o,    i]]
                R = mat_last(utils.as_tensor(R, **backend))
            else:
                R = torch.eye(3, **backend)
                R = utils.expand(R, [*batch_shape, 1, 1])
            if z is not None:
                Z = [[z[0], o,    o],
                     [o,    z[1], o],
                     [o,    o,    i]]
                Z = mat_last(utils.as_tensor(Z, **backend))
            else:
                Z = torch.eye(3, **backend)
                Z = utils.expand(Z, [*batch_shape, 1, 1])
            if sh is not None:
                S = [[i, sh[0], o],
                     [o, i,     o],
                     [o, o,     i]]
                S = mat_last(utils.as_tensor(S, **backend))
            else:
                S = torch.eye(3, **backend)
                S = utils.expand(S, [*batch_shape, 1, 1])

            return T.matmul(R.matmul(Z.matmul(S)))
    else:
        def make_affine(t, r, z, sh, o, i):
            if t is not None:
                T = [[i, o, o, t[0]],
                     [o, i, o, t[1]],
                     [o, o, i, t[2]],
                     [o, o, o, i]]
                T = mat_last(utils.as_tensor(T, **backend))
            else:
                T = torch.eye(4, **backend)
                T = utils.expand(T, [*batch_shape, 1, 1])
            if r is not None:
                c = torch.cos(r)
                s = torch.sin(r)
                Rx = [[i, o,     o,    o],
                      [o, c[0],  s[0], o],
                      [o, -s[0], c[0], o],
                      [o, o,     o,    i]]
                Ry = [[c[1],  o, s[1], o],
                      [o,     i, o,    o],
                      [-s[1], o, c[1], o],
                      [o,     o,    o, i]]
                Rz = [[c[2],  s[2], o, o],
                      [-s[2], c[2], o, o],
                      [o,     o,    i, o],
                      [o,     o,    o, i]]
                Rx = mat_last(utils.as_tensor(Rx, **backend))
                Ry = mat_last(utils.as_tensor(Ry, **backend))
                Rz = mat_last(utils.as_tensor(Rz, **backend))
                R = Rx.matmul(Ry.matmul(Rz))
            else:
                R = torch.eye(4, **backend)
                R = utils.expand(R, [*batch_shape, 1, 1])
            if z is not None:
                Z = [[z[0], o,    o,    o],
                     [o,    z[1], o,    o],
                     [o,    o,    z[2], o],
                     [o,    o,    o,    i]]
                Z = mat_last(utils.as_tensor(Z, **backend))
            else:
                Z = torch.eye(4, **backend)
                Z = utils.expand(Z, [*batch_shape, 1, 1])
            if sh is not None:
                S = [[i, sh[0], sh[1], o],
                     [o, i,     sh[2], o],
                     [o, o,     i,     o],
                     [o, o,     o,     i]]
                S = mat_last(utils.as_tensor(S, **backend))
            else:
                S = torch.eye(4, **backend)
                S = utils.expand(S, [*batch_shape, 1, 1])

            return T.matmul(R.matmul(Z.matmul(S)))

    zero = torch.zeros([], **backend)
    zero = utils.expand(zero, batch_shape)
    one = torch.ones([], **backend)
    one = utils.expand(one, batch_shape)

    # Build affine matrix
    mat = make_affine(prm_t, prm_r, prm_z, prm_s, zero, one)

    return mat


def affine_parameters(mat, *basis, max_iter=10000, tol=None,
                      max_line_search=6):
    """Compute the parameters of an affine matrix in a basis of the algebra.

    This function finds the matrix closest to ``mat`` (in the least squares
    sense) that can be encoded in the specified basis.

    Parameters
    ----------
    mat : (dim+1, dim+1) tensor_like
        Affine matrix

    basis : vector_like[basis_like]
        Basis of the Lie algebra(s).

    max_iter : int, default=10000
        Maximum number of Gauss-Newton iterations in the least-squares fit.

    tol : float, optional
        Tolerance criterion for convergence.
        Use the data type's machine epsilon by default.
        It is based on the squared norm of the GN step divided by the
        squared norm of the input matrix.

    max_line_search: int, default=6
        Maximum number of line search steps.
        If zero: no line-search is performed.

    Returns
    -------
    prm : tensor
        Parameters in the specified basis

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original GN fit in Matlab
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Format mat
    mat = utils.as_tensor(mat)
    dtype = mat.dtype
    dim = mat.shape[-1] - 1

    if tol is None:
        tol = core.dtypes.dtype(dtype).eps

    # Format basis
    basis = build_affine_basis(*basis, dim)
    basis = pyutils.make_list(basis)
    nb_basis = sum([len(b) for b in basis])

    def gauss_newton():
        # Predefine these values in case max_iter == 0
        n_iter = -1
        # Gauss-Newton optimisation
        prm = torch.zeros(nb_basis, dtype=dtype)
        M = torch.eye(dim+1, dtype=dtype)
        sos = ((M - mat) ** 2).sum()
        norm = (mat ** 2).sum()
        crit = constants.inf
        for n_iter in range(max_iter):

            # Compute derivative of each submatrix with respect to its basis
            # * Mi
            # * dMi/dBi
            Ms = []
            dMs = []
            n_basis = 0
            for a_basis in basis:
                nb_a_basis = a_basis.shape[0]
                a_prm = prm[n_basis:(n_basis+nb_a_basis)]
                M, dM = linalg._expm(a_prm, a_basis, grad_X=True)
                Ms.append(M)
                dMs.append(dM)
                n_basis += nb_a_basis
            M = torch.stack(Ms)

            # Compute derivative of the full matrix with respect to each basis
            # * M = mprod(M[:, ...])
            # * dM/dBi = mprod(M[:i, ...]) @ dMi/dBi @ mprod(M[i+1:, ...])
            for n_mat, dM in enumerate(dMs):
                if n_mat > 0:
                    pre = torch.chain_matmul(*M[:n_mat, ...])
                    dM = pre.mm(dM)
                if n_mat < M.shape[0]-1:
                    post = torch.chain_matmul(*M[(n_mat+1):, ...])
                    dM = dM.mm(post)
                dMs[n_mat] = dM
            dM = torch.cat(dMs)
            M = torch.chain_matmul(*M)

            # Compute gradient/Hessian of the loss (squared residuals)
            diff = M - mat
            diff = diff.flatten()
            dM = dM.reshape((nb_basis, -1))
            gradient = linalg.matvec(dM, diff)
            hessian = torch.matmul(dM, dM.t())
            delta_prm = hessian.inverse().matmul(gradient)

            crit = delta_prm.square().sum() / norm
            if crit < tol:
                break

            if max_line_search == 0:
                # We trust the Gauss-Newton step
                prm -= delta_prm
            else:
                # Line Search
                sos0 = sos
                prm0 = prm
                M0 = M
                armijo = 1
                success = False
                for _ in range(max_line_search):
                    prm = prm0 - armijo * delta_prm
                    M = affine_matrix(prm, *basis)
                    sos = ((M - mat) ** 2).sum()
                    if sos < sos0:
                        success = True
                        break
                    else:
                        armijo /= 2
                if not success:
                    prm = prm0
                    M = M0
                    break

        if crit >= tol:
            warn('Gauss-Newton optimisation did not converge: '
                 'n_iter = {}, sos = {}.'.format(n_iter + 1, crit),
                 RuntimeWarning)

        return prm, M

    prm, M = gauss_newton()

    # TODO: should I stack parameters per basis?
    return prm, M


def affine_parameters_classic(mat, return_stacked=True):
    """Compute the parameters of an affine matrix.

    This functions decomposes the input matrix into a product of
    simpler matrices (translation, rotation, ...) and extracts their
    parameters, so that the input matrix can be (approximately)
    reconstructed by `mat = T @ Rx @ Ry @ Rz @ Z @ S`

    This function only works in 2D and 3D.

    Parameters
    ----------
    mat : (..., dim+1, dim+1) tensor_like
        Affine matrix
    return_stacked : bool, default=True
        Return all parameters stacked in a vector

    Returns
    -------
    prm : (..., dim*(dim+1)) tensor, if return_stacked
        Individual parameters, ordered as
        [*translations, *rotations, *zooms, *shears].

    translations : (..., dim) tensor, if not return_stacked
        Translation parameters.
    rotations : (..., dim*(dim-1)//2) tensor, if not return_stacked
        Rotation parameters, in radian.
    zooms : (..., dim*) tensor, if not return_stacked
        Zoom parameters.
    shears : (..., dim*(dim-1)//2) tensor, if not return_stacked
        Shear parameters.

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original code (SPM12)
    # .. Stefan Kiebel <stefan.kiebel@tu-dresden.de> : original code (SPM12)
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Batching + Autograd

    mat = torch.as_tensor(mat)
    dim = mat.shape[-1] - 1

    if dim not in (2, 3):
        raise ValueError(f'Expected dimension 2 or 3, but got {dim}.')

    # extract linear part + cholesky decomposition
    # (note that matlab's chol is upper-triangular by default while
    #  pytorch's is lower-triangular by default).
    #
    # > the idea is that M = R @ Z @ S and
    #   M.T @ M = (S.T @ Z.T @ R.T) @ (R @ Z @ S)
    #   M.T @ M = (S.T @ Z.T) @ (Z @ S)
    #           = U.T @ U
    #  where U is upper-triangular, such that the diagonal of U contains
    #  zooms and the off-diagonal elements of Z\U are the shears.
    lin = mat[..., :dim, :dim]
    chol = torch.cholesky(lin.transpose(-1, -2).matmul(lin), upper=True)
    diag_chol = torch.diagonal(chol, dim1=-1, dim2=-2)

    # Translations
    prm_t = mat[..., :dim, -1]

    # Zooms (with fix for negative determinants)
    # > diagonal of the cholesky factor
    prm_z = diag_chol
    prm_z0 = torch.where(lin.det() < 0, -prm_z[..., 0], prm_z[..., 0])
    prm_z0 = prm_z0[..., None]
    prm_z = torch.cat((prm_z0, prm_z[..., 1:]), dim=-1)

    # Shears
    # > off-diagonal of the normalized cholesky factor
    chol = chol / diag_chol[..., None]
    upper_ind = torch.triu_indices(chol.shape[-2], chol.shape[-1],
                                   device=mat.device, offset=1)
    prm_s = chol[..., upper_ind[0], upper_ind[1]]

    # Rotations
    # > we know the zooms and shears and therefore `Z @ S`.
    #   If the problem is well conditioned, we can recover the pure
    #   rotation (orthogonal) matrix as `R = M / (Z @ S)`.
    lin0 = affine_matrix_classic(zooms=prm_z, shears=prm_s, dim=dim)
    lin0 = lin0[..., :dim, :dim]
    rot = lin.matmul(lin0.inverse())          # `R = M / (Z @ S)`
    clamp = lambda x: x.clamp(min=-1, max=1)  # correct rounding errors

    xz = rot[..., 0, -1]
    rot_y = torch.asin(clamp(xz))
    if dim == 2:
        prm_r = rot_y[..., None]
    else:
        xy = rot[..., 0, 1]
        yx = rot[..., 1, 0]
        yz = rot[..., 1, -1]
        xx = rot[..., 0, 0]
        zz = rot[..., -1, -1]
        zx = rot[..., -1, 0]
        cos_y = torch.cos(rot_y)

        # find matrices for which the first rotation is 90 deg
        # (we cannot divide by its cos in that case)
        cos_zero = (torch.abs(rot_y) - constants.pi/2)**2 < 1e-9
        zero = xy.new_zeros([]).expand(cos_zero.shape)

        rot_x = torch.where(cos_zero.bool(), zero,
                            torch.atan2(clamp(yz/cos_y), clamp(zz/cos_y)))
        rot_z = torch.where(cos_zero,
                            torch.atan2(-clamp(yx), clamp(-zx/xz)),
                            torch.atan2(clamp(xy/cos_y), clamp(xx/cos_y)))
        prm_r = torch.stack((rot_x, rot_y, rot_z), dim=-1)

    if return_stacked:
        return torch.cat((prm_t, prm_r, prm_z, prm_s), dim=-1)
    else:
        return prm_t, prm_r, prm_z, prm_s


def as_homogeneous(affine):
    """Mark a tensor as having homogeneous coordinates.

    We set the attribute `is_homogeneous` to True.
    Note that this attribute is not persistent and will be lost
    after any operation on the tensor.
    """
    affine.is_homogeneous = True
    return affine


def as_euclidean(affine):
    """Mark a tensor as having non-homogeneous coordinates.

    We set the attribute `is_homogeneous` to True.
    Note that this attribute is not persistent and will be lost
    after any operation on the tensor.
    """
    affine.is_homogeneous = False
    return affine


def affine_is_square(affine):
    """Return True if the matrix is square"""
    affine = torch.as_tensor(affine)
    return affine.shape[-1] == affine.shape[-2]


def affine_is_rect(affine):
    """Return False if the matrix is rectangular"""
    affine = torch.as_tensor(affine)
    return affine.shape[-1] == affine.shape[-2] + 1


def affine_is_homogeneous(affine, sym=False):
    """Return true is the last row is [*zeros, 1]"""
    if getattr(affine, 'is_homogeneous', False):
        return True
    if not getattr(affine, 'is_homogeneous', True):
        return False
    if sym:
        return affine_is_square(affine)
    with torch.no_grad():
        affine = torch.as_tensor(affine)
        info = dict(dtype=affine.dtype, device=affine.device)
        last_row = affine[..., -1, :]
        template_row = torch.zeros(last_row.shape[-1], **info)
        template_row[-1] = 1
        check = (last_row == template_row).all().item()
    return check


def affine_make_square(affine):
    """Transform a rectangular affine into a square affine.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor

    Returns
    -------
    affine : (..., ndim+1, ndim+1) tensor

    """
    affine = torch.as_tensor(affine)
    device = affine.device
    dtype = affine.dtype
    ndims = affine.shape[-1]-1
    if affine.shape[-2] not in (ndims, ndims+1):
        raise ValueError('Input affine matrix should be of shape\n'
                         '(..., ndims+1, ndims+1) or (..., ndims, ndims+1).')
    if affine.shape[-1] != affine.shape[-2]:
        bottom_row = torch.cat((torch.zeros(ndims, device=device, dtype=dtype),
                                torch.ones(1, device=device, dtype=dtype)), dim=0)
        bottom_row = bottom_row.unsqueeze(0)
        bottom_row = bottom_row.expand(affine.shape[:-2] + bottom_row.shape)
        affine = torch.cat((affine, bottom_row), dim=-2)
    return affine


def affine_make_rect(affine):
    """Transform a square affine into a rectangular affine.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor

    Returns
    -------
    affine : (..., ndim, ndim+1) tensor

    """
    affine = torch.as_tensor(affine)
    ndims = affine.shape[-1]-1
    return affine[..., :ndims, :]


def affine_make_homogeneous(affine, sym=False, force=False):
    """Ensure that the last row of the matrix is (*zeros, 1).

    This function is more generic than `make_square` because it
    works with images where the dimension of the output space differs
    from the dimension of the input space.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix
    sym : bool, default=False
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    affine : (..., ndim_out+1, ndim_in+1) tensor

    """
    if sym:
        return as_homogeneous(affine_make_square(affine))

    affine = torch.as_tensor(affine)
    if force or not affine_is_homogeneous(affine):
        info = dict(dtype=affine.dtype, device=affine.device)
        ndims_in = affine.shape[-1] - 1
        bottom_row = torch.cat((torch.zeros(ndims_in, **info),
                                torch.ones(1, **info)), dim=0)
        bottom_row = bottom_row.unsqueeze(0)
        bottom_row = bottom_row.expand(affine.shape[:-2] + bottom_row.shape)
        affine = torch.cat((affine, bottom_row), dim=-2)
    return as_homogeneous(affine)


def affine_del_homogeneous(affine, sym=False, force=False):
    """Ensure that the last row of the matrix is _not_ (*zeros, 1).

    This function is more generic than `make_rect` because it
    works with images where the dimension of the output space differs
    from the dimension of the input space.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix
    sym : bool, default=False
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    affine : (..., ndim_out, ndim_in+1) tensor

    """
    if sym:
        return as_euclidean(affine_make_rect(affine))
    affine = torch.as_tensor(affine)
    if force or affine_is_homogeneous(affine):
        affine = affine[..., :-1, :]
    return as_euclidean(affine)


def voxel_size(mat, sym=True):
    """ Compute voxel sizes from affine matrices.

    Parameters
    ----------
    mat :  (..., ndim_out[+1], ndim_in+1) tensor
        Affine matrix
    sym : bool, default=True
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    vx :  (..., ndim_in) tensor
        Voxel size

    """
    mat = torch.as_tensor(mat)
    dim = mat.shape[-1] - 1
    if sym:
        mat = mat[..., :dim, :dim]
    elif affine_is_homogeneous(mat):
        mat = mat[..., :-1, :-1]
    else:
        mat = mat[..., :, :-1]
    return as_euclidean(mat.square().sum(-2).sqrt())


def affine_matvec(affine, vector, sym=True):
    """Matrix-vector product of an affine and a (possibly homogeneous) vector.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
    vector : (..., ndim_in[+1]) tensor
    sym : bool, default=True
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    affine_times_vector : (..., ndim_out[+1]) tensor
        The returned vector is homogeneous iff `vector` is too.

    """
    affine = affine_del_homogeneous(affine, sym)
    vector = torch.as_tensor(vector)
    info = dict(dtype=vector.dtype, device=vector.device)
    ndims_in = affine.shape[-1] - 1
    is_h = vector.shape[-1] == ndims_in + 1
    zoom = affine[..., :, :-1]
    translation = affine[..., :, -1]
    out = linalg.matvec(zoom, vector[..., :ndims_in]) + translation
    if is_h:
        one = torch.ones(out.shape[:-1] + (1,), **info)
        out = torch.cat((out, one), dim=-1)
        out = as_homogeneous(out)
    else:
        out = as_euclidean(out)
    return out


def affine_matmul(a, b, sym=False, symb=None):
    """Matrix-matrix product of affine matrices.

    Parameters
    ----------
    a : (..., ndim_out[+1], ndim_inter+1) tensor
        Affine matrix
    b : (..., ndim_inter[+1], ndim_in+1) tensor
        Affine matrix
    sym : bool, default=True
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.
    symb : bool, default=sym
        Same, but for `b`.

    Returns
    -------
    affine_times_matrix : (..., ndim_out[+1], ndim_in+1) tensor
        The returned matrix is homogeneous iff `a` is too.

    """
    symb = sym if symb is None else symb
    is_h = affine_is_homogeneous(a, sym)
    a = affine_del_homogeneous(a, sym)
    b = affine_del_homogeneous(b, symb)
    Za = a[..., :, :-1]
    Ta = a[..., :, -1]
    Zb = b[..., :, :-1]
    Tb = b[..., :, -1]
    Z = torch.matmul(Za, Zb)
    T = linalg.matvec(Za, Tb) + Ta
    out = torch.cat((Z, T[..., None]), dim=-1)
    if is_h:
        out = affine_make_homogeneous(out, force=True)
    else:
        out = as_euclidean(out)
    return out


def affine_inv(affine, sym=True):
    """Inverse of an affine matrix.

    If the input matrix is not symmetric with respect to its input
    and output spaces, a pseudo-inverse is returned instead.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine
    sym : bool, default=True
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    inv_affine : (..., ndim_in[+1], ndim_out+1) tensor
        The returned matrix is homogeneous iff `affine` is too.

    """
    is_h = affine_is_homogeneous(affine, sym)
    affine = affine_del_homogeneous(affine, sym)
    ndims_in = affine.shape[-1] - 1
    ndims_out = affine.shape[-2]
    inverse = torch.inverse if ndims_in == ndims_out else torch.pinverse
    zoom = inverse(affine[..., :, :-1])
    translation = -linalg.matvec(zoom, affine[..., :, -1])
    out = torch.cat((zoom, translation[..., None]), dim=-1)
    if is_h:
        out = affine_make_homogeneous(out, force=True)
    else:
        out = as_euclidean(out)
    return out


def affine_lmdiv(a, b, sym=True):
    """Left matrix division of affine matrices: inv(a) @ b

    Parameters
    ----------
    a : (..., ndim_inter[+1], ndim_out+1) tensor
        Affine matrix that will be inverted.
        A peudo-inverse is used if `ndim_inter != ndim_out`
    b : (..., ndim_inter[+1], ndim_in+1) tensor
        Affine matrix.
    sym : bool, default=True
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    output_affine : (..., ndim_out[+1], ndim_in+1) tensor
        The returned matrix is homogeneous iff `a` is too.

    """
    return affine_matmul(affine_inv(a, sym), b, sym)


def affine_rmdiv(a, b, sym=True):
    """Right matrix division of affine matrices: a @ inv(b)

    Parameters
    ----------
    a : (..., ndim_out[+1], ndim_inter+1) tensor
        Affine matrix.
    b : (..., ndim_in[+1], ndim_inter+1) tensor
        Affine matrix that will be inverted.
        A peudo-inverse is used if `ndim_inter != ndim_in`
    sym : bool, default=True
        Assume that the affine is symmetric with respect to the
        input and output space dimensions, _i.e._, it maps from
        ND coordinates to ND coordinates (`ndim_out == ndim_in`).
        Knowing this allows the function to be more efficient.

    Returns
    -------
    output_affine : (..., ndim_out[+1], ndim_in+1) tensor
        The returned matrix is homogeneous iff `a` is too.

    """
    return affine_matmul(a, affine_inv(b, sym), sym)


def affine_resize(affine, shape, factor, anchor='c'):
    """Update an affine matrix according to a resizing of the lattice.

    Notes
    -----
    This function is related to the `resize` function, which allows the
    user to choose between modes:
        * 'c' or 'centers': align centers
        * 'e' or 'edges':   align edges
        * 'f' or 'first':   align center of first voxel
        * 'l' or 'last':    align center of last voxel
    In cases 'c' and 'e', the volume shape is multiplied by the zoom
    factor (and eventually truncated), and two anchor points are used
    to determine the voxel size (neurite's behavior corresponds to 'c').
    In cases 'f' and 'l', a single anchor point is used so that the voxel
    size is exactly divided by the zoom factor. This case with an integer
    factor corresponds to subslicing the volume (e.g., vol[::f, ::f, ::f]).

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    factor : float or sequence[float]
        Resizing factor.
        * > 1 : larger image <-> smaller voxels
        * < 1 : smaller image <-> larger voxels
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        Anchor points.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Resized affine matrix.
    shape : (ndim,) tuple[int]
        Resized shape.

    """

    # read parameters
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    factor = utils.make_list(factor, nb_dim)
    anchor = [a[0].lower() for a in utils.make_list(anchor, nb_dim)]
    info = {'dtype': affine.dtype, 'device': affine.device}

    # compute output shape
    shape_out = [int(s * f) for s, f in zip(shape, factor)]

    # compute shift and scale in each dimension
    shifts = []
    scales = []
    for anch, f, inshp, outshp in zip(anchor, factor, shape, shape_out):
        if anch == 'c':
            shifts.append(0)
            scales.append((inshp - 1) / (outshp - 1))
        elif anch == 'e':
            shifts.append((inshp * (1 / outshp - 1) + (inshp - 1)) / 2)
            scales.append(inshp/outshp)
        elif anch == 'f':
            shifts.append(0)
            scales.append(1/f)
        elif anch == 'l':
            shifts.append((inshp - 1) - (outshp - 1) / f)
            scales.append(1/f)
        else:
            raise ValueError('Unknown anchor {}'.format(anch))

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scales, **info))
    trl = torch.as_tensor(shifts, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1)

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(shape_out)


def affine_sub(affine, shape, indices):
    """Update an affine matrix according to a sub-indexing of the lattice.

    Notes
    -----
    .. Only sub-indexing that *keep an homogeneous voxel size* are allowed.
       Therefore, indices must be `None` or of type `int`, `slice`, `ellipsis`.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    shape : (ndim_in,) sequence[int]
        Input shape.
    indices : tuple[slice or ellipsis]
        Subscripting indices.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_new+1) tensor
        Updated affine matrix.
    shape : (ndim_new,) tuple[int]
        Updated shape.

    """
    def is_int(elem):
        if torch.is_tensor(elem):
            return elem.dtype in (torch.int32, torch.int64)
        elif isinstance(elem, int):
            return True
        else:
            return False

    def to_int(elem):
        if torch.is_tensor(elem):
            return elem.item()
        else:
            assert isinstance(elem, int)
            return elem

    # check types
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    info = {'dtype': affine.dtype, 'device': affine.device}
    shape = list(shape)
    if len(shape) != nb_dim:
        raise ValueError('Expected shape of length {}. Got {}'
                         .format(nb_dim, len(shape)))
    if not isinstance(indices, tuple):
        raise TypeError('Indices should be a tuple.')
    indices = list(indices)

    # compute the number of input dimension that correspond to each index
    #   > slice index one dimension but eliipses index multiple dimension
    #     and their number must be computed.
    nb_dims_in = []
    ind_ellipsis = None
    for n_ind, ind in enumerate(indices):
        if isinstance(ind, slice):
            nb_dims_in.append(1)
        elif ind is Ellipsis:
            if ind_ellipsis is not None:
                raise ValueError('Cannot have more than one ellipsis.')
            ind_ellipsis = n_ind
            nb_dims_in.append(-1)
        elif is_int(ind):
            nb_dims_in.append(1)
        elif ind is None:
            nb_dims_in.append(0)
        else:
            raise TypeError('Indices should be None, integers, slices or '
                            'ellipses. Got {}.'.format(type(ind)))
    nb_known_dims = sum(nb_dims for nb_dims in nb_dims_in if nb_dims > 0)
    if ind_ellipsis is not None:
        nb_dims_in[ind_ellipsis] = max(0, nb_dim - nb_known_dims)

    # transform each index into a slice
    # note that we don't need to know "stop" to update the affine matrix
    nb_ind = 0
    indices0 = indices
    indices = []
    for d, ind in enumerate(indices0):
        if isinstance(ind, slice):
            start = ind.start
            step = ind.step
            step = 1 if step is None else step
            start = 0 if (start is None and step > 0) else \
                    shape[nb_ind] - 1 if (start is None and step < 0) else \
                    shape[nb_ind] + start if start < 0 else \
                    start
            indices.append(slice(start, None, step))
            nb_ind += 1
        elif ind is Ellipsis:
            for dd in range(nb_ind, nb_ind + nb_dims_in[d]):
                start = 0
                step = 1
                indices.append(slice(start, None, step))
                nb_ind += 1
        elif is_int(ind):
            indices.append(to_int(ind))
        elif ind is None:
            assert (ind is None), "Strange index of type {}".format(type(ind))
            indices.append(None)

    # Extract shift and scale in each dimension
    shifts = []
    scales = []
    slicer = []
    shape_out = []
    for d, ind in enumerate(indices):
        # translation + scale
        if isinstance(ind, slice):
            shifts.append(ind.start)
            scales.append(ind.step)
            shape_out.append(shape[d] // abs(ind.step))
            slicer.append(slice(None))
        elif isinstance(ind, int):
            scales.append(0)
            shifts.append(ind)
            slicer.append(0)
        else:
            slicer.append(None)
            assert (ind is None), "Strange index of type {}".format(type(ind))

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scales, **info))
    if any(not isinstance(s, slice) for s in slicer):
        # drop/add columns
        lin = torch.unbind(lin, dim=-1)
        zero = torch.zeros(len(shifts), **info)
        new_lin = []
        for s in slicer:
            if isinstance(s, slice):
                col, *lin = lin
                new_lin.append(col)
            elif isinstance(s, int):
                col, *lin = lin
            elif s is None:
                new_lin.append(zero)
        lin = torch.stack(new_lin, dim=-1) if new_lin else []
    trl = torch.as_tensor(shifts, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1) if len(lin) else trl

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(shape_out)


def affine_permute(affine, perm=None, shape=None):
    """Update an affine matrix according to a permutation of the lattice dims.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    perm : sequence[int], optional
        Permutation of the lattice dimensions.
        By default, reverse dimension order.
    shape : (ndim_in,) sequence[int], optional
        Input shape.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Updated affine matrix.
    shape : (ndim_in,) tuple[int], optional
        Updated shape.
    """
    nb_dim = affine.shape[-1] - 1
    if perm is None:
        perm = list(range(nb_dim-1, -1, -1))
    if torch.is_tensor(perm):
        perm = perm.tolist()
    if len(perm) != nb_dim:
        raise ValueError('Expected perm to have {} elements. Got {}.'
                         .format(nb_dim, len(perm)))
    affine = affine[..., :, perm + [-1]]
    if shape is not None:
        shape = tuple(shape[p] for p in perm)
        return affine, shape
    else:
        return affine


def affine_transpose(affine, dim0, dim1, shape):
    """Update an affine matrix according to a transposition of the lattice.

    A transposition is a permutation that only impacts two dimensions.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    dim0 : int
        Index of the first dimension
    dim1 : int
        Index of the second dimension
    shape : (ndim,) sequence[int], optional
        Input shape.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int], optional
        Updated shape.
    """
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    perm = list(range(nb_dim))
    perm[dim0] = dim1
    perm[dim1] = dim0
    return affine_permute(affine, perm, shape)


def affine_conv(affine, shape, kernel_size, stride=1, padding=0,
                output_padding=0, dilation=1, transposed=False):
    """Update an affine matrix according to a convolution of the lattice.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    kernel_size : int or list[int]
        Kernel size
    stride : int or list[int], default=1
        Strides (= step size when moving the kernel)
    padding : int or list[int], default=0
        Amount of padding added to (both sides of) the input
    output_padding : int or list[int], default=0
        Additional size added to (the bottom/right) side of each
        dimension in the output shape. Only used if `transposed is True`.
    dilation : int or list[int], default=1
        Dilation (= step size between elements of the kernel)
    transposed : bool, default=False
        Transposed convolution.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int]
        Updated shape.

    """
    affine = torch.as_tensor(affine)
    info = {'dtype': affine.dtype, 'device': affine.device}
    ndim = affine.shape[-1] - 1
    if len(shape) != ndim:
        raise ValueError('Affine and shape not consistant. Found dim '
                         '{} and {}.'.format(ndim, len(shape)))
    kernel_size = pyutils.make_list(kernel_size, ndim)
    stride = pyutils.make_list(stride, ndim)
    padding = pyutils.make_list(padding, ndim)
    output_padding = pyutils.make_list(output_padding, ndim)
    dilation = pyutils.make_list(dilation, ndim)

    # compute new shape and scale/offset that transform the
    # new lattice into the old lattice
    oshape = []
    scale = []
    offset = []
    for L, S, Pi, D, K, Po in zip(shape, stride, padding,
                                  dilation, kernel_size, output_padding):
        if transposed:
            oshape += [(L - 1) * S - 2 * Pi + D * (K - 1) + Po + 1]
            scale += [1/S]
            offset += [(Pi - (K-1)/2)/S]
        else:
            oshape += [math.floor((L + 2 * Pi - D * (K - 1) - 1) / S + 1)]
            scale += [S]
            offset += [(K-1)/2 - Pi]

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scale, **info))
    trl = torch.as_tensor(offset, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1)

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(oshape)


def affine_default(shape, voxel_size=1., layout=None, dtype=None, device=None):
    """Generate an orientation matrix with the origin in the center of the FOV.

    Parameters
    ----------
    shape : sequence[int]
        Lattice shape.
    voxel_size : sequence[float], default=1
        Lattice voxel size
    layout : str or layout_like, default='RAS'
        Lattice layout (see `volume_layout`).
    dtype : dtype, optional
    device : device, optional

    Returns
    -------
    affine : (ndim, ndim+1) tensor
        Orientation matrix

    """
    shape = list(shape)
    nb_dim = len(shape)
    voxel_size = pyutils.make_list(voxel_size, nb_dim)

    # build affine matrix
    voxel_size = torch.as_tensor(voxel_size, dtype=dtype, device=device)
    dtype = voxel_size.dtype
    device = voxel_size.device
    shape = torch.as_tensor(shape, dtype=dtype, device=device)

    # build layout
    if layout is None:
        layout = volume_layout(index=list(range(nb_dim)))
    else:
        layout = volume_layout(layout)
    layout = layout_matrix(layout, voxel_size=voxel_size, shape=shape,
                           dtype=dtype, device=device)

    # compute shift
    lin = layout[:nb_dim, :nb_dim]
    shift = -linalg.matvec(lin, shape/2.)
    affine = torch.cat((lin, shift[:, None]), dim=1)

    return affine_make_homogeneous(as_euclidean(affine))


def max_bb(all_mat, all_dim, vx=None):
    """Specify mean space voxel-to-world matrix (mat) and dimensions (dm),
    from a collection of images, as the maximum bounding-box containing
    all of the images.

    Parameters
    ----------
    all_mat : (N, 4, 4) tensor
    all_dim (N, 3) tensor
    vx : (3, ), default=(1, 1, 1)

    Returns
    -------
    mat : (4, 4) tensor
    dm : (3, ) tensor

    """
    dtype = all_mat.dtype
    device = all_mat.device
    if vx is None:
        vx = (1, ) * 3
    vx = torch.as_tensor(vx, device=device, dtype=dtype)
    # Number of subjects
    N = all_mat.shape[0]
    # Get all images field of view
    mx = torch.zeros([3, N], device=device, dtype=dtype)
    mn = torch.zeros([3, N], device=device, dtype=dtype)
    for n in range(N):
        dm = all_dim[n, ...]
        corners = torch.tensor([[1, dm[0], 1, dm[0], 1, dm[0], 1, dm[0]],
                                [1, 1, dm[1], dm[1], 1, 1, dm[1], dm[1]],
                                [1, 1, 1, 1, dm[2], dm[2], dm[2], dm[2]],
                                [1, 1, 1, 1, 1, 1, 1, 1]],
                               device=device, dtype=dtype)
        M = all_mat[n, ...]
        pos = M[:-1,:].mm(corners)
        mx[..., n] = torch.max(pos, dim=1)[0]
        mn[..., n] = torch.min(pos, dim=1)[0]
    # Get bounding box
    mx = torch.sort(mx, dim=1, descending=False)[0]
    mn = torch.sort(mn, dim=1, descending=True)[0]
    mx = mx[..., -1]
    mn = mn[..., -1]
    bb = torch.stack((mn, mx))
    # Change voxel size
    vx = torch.tensor([-1, 1, 1], dtype=dtype, device=device)*vx.abs()
    mn = vx*torch.min(bb/vx, dim=0)[0]
    mx = vx*torch.max(bb/vx, dim=0)[0]
    # Make output affine matrix and image dimensions
    mat = affine_matrix_classic(torch.cat((mn, torch.zeros(3, dtype=dtype, device=device), vx)))\
        .mm(affine_matrix_classic(-torch.ones(3, dtype=dtype, device=device)))
    dm = torch.cat((mx, torch.ones(1, dtype=dtype, device=device)))[..., None].solve(mat)[0]
    dm = dm[:3].round().flatten()

    return mat, dm


def affine_reorient(mat, shape_or_tensor=None, layout=None):
    """Reorient an affine matrix / a tensor to match a target layout.

    Parameters
    ----------
    mat : (dim[+1], dim+1) tensor_like
        Orientation matrix
    shape_or_tensor : (dim,) sequence[int] or (..., *shape) tensor_like, optional
        Shape or Volume
    layout : layout_like, optional
        Target layout. Defaults to a directed layout (equivalent to 'RAS').

    Returns
    -------
    mat : (dim[+1], dim+1) tensor
        Reoriented orientation matrix
    shape_or_tensor : (dim,) tuple or (..., *permuted_shape) tensor, optional
        Reoriented shape or volume

    """
    # parse inputs
    mat = torch.as_tensor(mat)
    dim = mat.shape[-1] - 1
    shape = tensor = None
    if shape_or_tensor is not None:
        shape_or_tensor = torch.as_tensor(shape_or_tensor)
        if shape_or_tensor.dim() > 1:
            tensor = shape_or_tensor
            shape = torch.as_tensor(tensor.shape[-dim:])
        else:
            shape = shape_or_tensor

    # find current layout and target layout
    #   layouts are (dim, 2) tensors where
    #       - the first column stores indices of the axes in RAS space
    #         (and is therefore a permutation that transforms a RAS
    #          space into this layout)
    #       - the second column stores 0|1 values that indicate whether
    #         this axis (after permutation) is flipped.
    current_layout = affine_to_layout(mat)
    target_layout = volume_layout(dim if layout is None else layout)
    ras_to_current, current_flips = current_layout.unbind(-1)
    ras_to_target, target_flips = target_layout.unbind(-1)
    current_to_ras = utils.invert_permutation(ras_to_current)

    # compose flips (xor)
    current_flips = current_flips.bool()
    target_flips = target_flips.bool()
    ras_flips = current_flips[current_to_ras]
    target_flips = target_flips ^ ras_flips[ras_to_target]

    # compose permutations
    current_to_target = current_to_ras[ras_to_target]

    # apply permutation and flips
    mat, shape = affine_permute(mat, current_to_target, shape)
    index = tuple(slice(None, None, -1) if flip else slice(None)
                  for flip in target_flips)
    mat, _ = affine_sub(mat, shape, index)

    if tensor is not None:
        # we need to append stuff to take into account batch dimensions
        nb_dim_left = tensor.dim() - len(index)
        current_to_target = current_to_target + nb_dim_left
        current_to_target = list(range(nb_dim_left)) + current_to_target.tolist()
        tensor = tensor.permute(current_to_target)
        index = (slice(None),) * nb_dim_left + index
        tensor = tensor[index]
        return mat, tensor
    else:
        return mat, shape


def affine_mean(mats, shapes):
    """Compute a mean orientation matrix.

    Gradient *do not* propagate through this function.

    Parameters
    ----------
    mats : (N, dim+1, dim+1) tensor_like
        Input orientation matrices
    shapes : (N, dim) tensor_like
        Input shape

    Returns
    -------
    mat : (dim+1, dim+1) np.ndarray
        Mean orientation matrix, with an RAS layout

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2019-2020 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    # Convert to (N, D+1, D+1) tensor
    device = utils.max_device(mats, shapes)
    shapes = torch.as_tensor(shapes, device=device).detach().clone()
    mats = torch.as_tensor(mats, device=device).detach().clone()
    dim = mats.shape[-1] - 1

    # STEP 1: Reorient to RAS layout
    # ------
    # Computing an exponential mean only works if all matrices are
    # "close". In particular, if the voxel layout associated with these
    # matrices is different (e.g., RAS vs LAS vs RSA), the exponential
    # mean will fail. The first step is therefore to reorient all
    # matrices so that they map to a common voxel layout.
    # We choose RAS as the common layout, as it makes further steps
    # easier and matches the world space orientation.
    for mat, shape in zip(mats, shapes):
        mat1, shape1 = affine_reorient(mat, shape)
        mat[:, :] = torch.as_tensor(mat1)
        shape[:] = torch.as_tensor(shape1)

    # STEP 2: Compute exponential barycentre
    # ------
    mat = linalg.meanm(mats)

    # STEP 3: Remove spurious shears
    # ------
    # We want the matrix to be "rigid" = the combination of a
    # rotation+translation (T*R) in world space and of a "voxel size"
    # scaling (Z), i.e., M = T*R*Z.
    # We look for the matrix that can be encoded without shears
    # that is the closest to the original matrix (in terms of the
    # Frobenius norm of the residual matrix)
    _, M = affine_parameters(mat, ['R', 'Z'])
    mat[:dim, :dim] = M[:dim, :dim]

    return mat


_voxel_size = voxel_size  # little alias to avoid the function being shadowed


def mean_space(mats, shapes, voxel_size=None, layout=None, fov='bb', crop=0):
    """Compute a mean space from a set of spaces (= affine + shape).

    Gradient *do not* propagate through this function.

    Parameters
    ----------
    mats : (N, dim+1, dim+1) tensor_like
        Input affine matrices
    shapes : (N, dim) tensor_like
        Input shapes
    voxel_size : (dim,) tensor_like, optional
        Output voxel size.
        Uses the mean voxel size of all input matrices by default.
    layout : str or (dim+1, dim+1) array_like, default=None
        Output layout.
        Uses the majority layout of all input matrices by default
    fov : {'bb'}, default='bb'
        Method for determining the output field-of-view:
            * 'bb': Bounding box of all input field-of-views, minus
              some optional cropping.
    crop : [0..1], default=0
        Amount of cropping applied to the field-of-view.

    Returns
    -------
    mat : (dim+1, dim+1) tensor
        Mean affine matrix
    shape : (dim,) tuple
        Corresponding shape

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2019-2020 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    def hashable_layout(layout):
        layout = layout.tolist()
        layout = tuple((int(index), bool(flip)) for index, flip in layout)
        return layout

    device = utils.max_device(mats, shapes)
    shapes = utils.as_tensor(shapes, device=device).detach()
    mats = utils.as_tensor(mats, device=device).detach()
    info = dict(dtype=mats.dtype, device=mats.device)
    dim = mats.shape[-1] - 1

    # Compute mean affine
    mat = affine_mean(mats, shapes)

    # Majority layout
    # (we must make layouts hashable so that they can be counted)
    if layout is None:
        all_layouts = [hashable_layout(affine_to_layout(mat)) for mat in mats]
        layout = pyutils.majority(all_layouts)
        # print('Output layout: {}'.format(volume_layout_to_name(layout)))
    else:
        layout = volume_layout(layout)

    # Switch layout
    layout = layout_matrix(layout, **info)
    mat = torch.matmul(mat, layout)

    # Voxel size
    if voxel_size is not None:
        vs0 = torch.as_tensor(voxel_size, **info)
        voxel_size = _voxel_size(mat)
        vs0[~torch.isfinite(vs0)] = voxel_size[~torch.isfinite(vs0)]
        one = torch.ones([1], **info)
        mat = mat * torch.diag(torch.cat((vs0 / voxel_size, one)))

    # Field of view
    if fov == 'bb':
        mn = torch.full([dim], constants.inf, **info)
        mx = torch.full([dim], constants.ninf, **info)
        for a_mat, a_shape in zip(mats, shapes):
            corners = itertools.product([False, True], r=dim)
            corners = [[a_shape[i] if top else 1 for i, top in enumerate(c)] + [1]
                       for c in corners]
            corners = torch.as_tensor(corners, **info).T
            M = linalg.lmdiv(mat, a_mat)
            corners = torch.matmul(M[:dim, :], corners)
            mx = torch.max(mx, torch.max(corners, dim=1).values)
            mn = torch.min(mn, torch.min(corners, dim=1).values)
        mx = torch.ceil(mx).long()
        mn = torch.floor(mn).long()
        offset = -crop * (mx - mn)
        shape = (mx - mn + 2*offset + 1)
        M = mn - (offset + 1)
        M = torch.cat((torch.eye(dim, **info), M[:, None]), dim=1)
        pad = torch.as_tensor([[0] * dim + [1]], **info)
        M = torch.cat((M, pad), dim=0)
        mat = torch.matmul(mat, M)
    else:
        raise NotImplementedError('method {} not implemented'.format(fov))

    return mat, tuple(shape.tolist())
