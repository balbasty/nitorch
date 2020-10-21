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
import functools
from ..core import utils
from ..core import itertools
from ..core import linalg
from ..core import constants


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
    volume_layout(name='RAS', device=None)
    volume_layout(axes, device=None)
    volume_layout(index, flipped=False, device=None)

    Parameters
    ----------
    name : str, default='RAS'
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
            return layout_from_name(*args, **kwargs)
        else:
            args[0] = utils.as_tensor(args[0])
            if args[0].dim() == 2:
                return layout_from_axes(*args, **kwargs)
            else:
                return layout_from_index(*args, **kwargs)
    else:
        if 'name' in kwargs.keys():
            return layout_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            return layout_from_index(*args, **kwargs)
        else:
            return layout_from_axes(*args, **kwargs)


def volume_layout_to_name(layout):
    """Return the (neuroimaging) name of a layout. Its length must be < 3.

    Parameters
    ----------
    layout : (dim, 2) tensor_like

    Returns
    -------
    name : str

    """
    if len(layout) > 3:
        raise ValueError('Layout names are only defined up to dimension 3. '
                         'Got {}.'.format(len(layout)))
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


def layout_matrix(layout, dtype=None, device=None):
    """Compute the origin affine matrix for different voxel layouts.

    Resources
    ---------
    .. https://nipy.org/nibabel/image_orientation.html
    .. https://nipy.org/nibabel/neuro_radio_conventions.html

    Parameters
    ----------
    layout : str or (ndim, 2) tensor_like[long]
        See `affine.layout`

    dtype : torch.dtype, optional
        Data type of the matrix

    device : torch.device, optional
        Output device.

    Returns
    -------
    mat : (ndim+1, ndim+1) tensor[dtype]
        Corresponding affine matrix.

    """
    # TODO: shape argument to include the translation induced by flips.

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Extract info from layout
    layout = volume_layout(layout, device=device)
    device = layout.device
    dim = len(layout)
    perm = utils.invert_permutation(layout[:, 0])
    flip = layout[:, 1].bool()

    # Create matrix
    mat = torch.eye(dim+1, dtype=dtype, device=device)
    last = utils.as_tensor([dim], dtype=torch.long, device=device)
    mat = mat[torch.cat((perm, last)), :]
    mflip = torch.ones(dim+1, dtype=dtype, device=device)
    false = utils.as_tensor([False], dtype=torch.bool, device=device)
    mflip = torch.where(torch.cat((flip, false)), -mflip, mflip)
    mflip = torch.diag(mflip)
    mat = torch.matmul(mflip, mat)

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


def _format_basis(basis, dim=None):
    """Transform an Outter/Inner Lie basis into a list of tensors.

    Parameters
    ----------
    basis : basis_like or list[basis_like] or list[list[basis_like]]
        A basis_like is a str or (F, D+1, D+1) tensor_like that
        describes a basis.
    dim : int, optional
        Dimensionality. If None, infer.

    Returns
    -------
    basis : list[tensor]
        A list of basis sets.

    """

    if isinstance(basis, str):
        basis = [basis]

    # Guess dimension
    if dim is None:
        if torch.is_tensor(basis):
            dim = basis.shape[-1] - 1
        elif isinstance(basis, str):
            dim = len(basis)
        else:
            for outer_basis in basis:
                if torch.is_tensor(outer_basis):
                    dim = outer_basis.shape[0] - 1
                    break
                elif isinstance(outer_basis, str):
                    dim = len(outer_basis)
                    break
                else:
                    for inner_basis in outer_basis:
                        if torch.is_tensor(inner_basis):
                            dim = inner_basis.shape[-1] - 1
                            break
                        elif isinstance(inner_basis, str):
                            dim = len(inner_basis)
                            break
                        else:
                            inner_basis = utils.as_tensor(inner_basis)
                            dim = inner_basis.shape[0] - 1
                            break
                    break
    if dim is None:
        # Guess failed
        dim = 3

    # Helper to convert named bases to matrices
    def name_to_basis(name):
        basename = name.split('[')[0]
        if basename in affine_subbasis_choices:
            return affine_subbasis(name, dim)
        elif basename in affine_basis_choices:
            return affine_basis(name, dim)
        else:
            raise ValueError('Unknown basis name {}.'.format(basename))

    # Convert 'named' bases to matrix bases
    if not torch.is_tensor(basis):
        basis = list(basis)
        for n_outer, outer_basis in enumerate(basis):
            if isinstance(outer_basis, str):
                basis[n_outer] = name_to_basis(outer_basis)
            elif not torch.is_tensor(outer_basis):
                outer_basis = list(outer_basis)
                for n_inner, inner_basis in enumerate(outer_basis):
                    if isinstance(inner_basis, str):
                        outer_basis[n_inner] = name_to_basis(inner_basis)
                    else:
                        outer_basis[n_inner] = utils.as_tensor(inner_basis)
                outer_basis = torch.cat(outer_basis)
                basis[n_outer] = outer_basis

    return basis, dim


def affine_matrix(prm, basis, dim=None, layout=None):
    r"""Reconstruct an affine matrix from its Lie parameters.

    Affine matrices are encoded as product of sub-matrices, where
    each sub-matrix is encoded in a Lie algebra. Finally, the right
    most matrix is a 'layout' matrix (see affine_layout).
    ..math:: M   = exp(A_1) \times ... \times exp(A_n) \times L
    ..math:: A_i = \sum_k = p_{ik} B_{ik}

    An SPM-like construction (as in ``spm_matrix``) would be:
    >>> M = affine_matrix(prm, ['T', 'R[0]', 'R[1]', 'R[2]', 'Z', 'SC'])
    Rotations need to be split by axis because they do not commute.

    Parameters
    ----------
    prm : vector_like or vector_like[vector_like]
        Parameters in the Lie algebra(s).

    basis : list[basis_like]
        The outer level corresponds to matrices in the product (*i.e.*,
        exponentiated matrices), while the inner level corresponds to
        Lie algebras.

    dim : int, default=guess or 3
        If not provided, the function tries to guess it from the shape
        of the basis matrices. If the dimension cannot be guessed
        (because all bases are named bases), the default is 3.

    layout : str or matrix_like, default=None
        A layout matrix.

    Returns
    -------
    mat : (dim+1, dim+1) tensor
        Reconstructed affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Make sure basis is a vector_like of (F, D+1, D+1) tensor_like
    basis, dim = _format_basis(basis, dim)

    # Check length
    nb_basis = sum([len(b) for b in basis])
    prm = utils.as_tensor(prm).flatten()
    dtype = prm.dtype
    device = prm.device
    if len(prm) != nb_basis:
        raise ValueError('Number of parameters and number of bases do '
                         'not match. Got {} and {}'
                         .format(len(prm), nb_basis))

    # Helper to reconstruct a log-matrix
    def recon(p, B):
        p = utils.as_tensor(p, dtype=dtype, device=device)
        B = utils.as_tensor(B, dtype=dtype, device=device)
        return linalg.expm(torch.sum(B*p[:, None, None], dim=0))

    # Reconstruct each sub matrix
    n_prm = 0
    mats = []
    for a_basis in basis:
        nb_prm = len(a_basis)
        a_prm = prm[n_prm:(n_prm+nb_prm)]
        mats.append(recon(a_prm, a_basis))
        n_prm += nb_prm

    # Add layout matrix
    if layout is not None:
        layout = layout_matrix(layout)
        mats.append(layout)

    # Matrix product
    return torch.chain_matmul(mats)


def affine_matrix_classic(prm, dim=3, layout=None):
    """Build an affine matrix in the "classic" way (no Lie algebra).

    Parameters
    ----------
    prm : (K,) tensor_like
        Affine parameters, ordered as
        `[*translations, *rotations, *zooms, *shears]`
        Rotation parameters should be expressed in radians.
    dim : () tensor_like[int]
        Dimensionality.
    layout : str or matrix_like, default=None
        Volume layout.

    Returns
    -------
    mat : (dim+1, dim+1) tensor
        Reconstructed affine matrix `mat = T @ Rx @ Ry @ Rz @ Z @ S`

    """

    def affine_2d(t, r, z, sh, dtype, device):
        if t is not None:
            T = [[1, 0, t[0]],
                 [0, 1, t[1]],
                 [0, 0, 1]]
        else:
            T = torch.eye(3, dtype=dtype, device=device)
        if r is not None:
            c = torch.cos(r)
            s = torch.sin(r)
            R = [[c[0],  s[0], 0],
                 [-s[0], c[0], 0],
                 [0,     0,    1]]
        else:
            R = torch.eye(3, dtype=dtype, device=device)
        if z is not None:
            Z = [[z[0], 0,    0],
                 [0,    z[1], 0],
                 [0,    0,    1]]
        else:
            Z = torch.eye(3, dtype=dtype, device=device)
        if sh is not None:
            S = [[1, sh[0], 0],
                 [0, 1,     0],
                 [0, 0,     1]]
        else:
            S = torch.eye(3, dtype=dtype, device=device)

        T = utils.as_tensor(T, dtype=dtype, device=device)
        R = utils.as_tensor(R, dtype=dtype, device=device)
        Z = utils.as_tensor(Z, dtype=dtype, device=device)
        S = utils.as_tensor(S, dtype=dtype, device=device)
        return T.mm(R.mm(Z.mm(S)))

    def affine_3d(t, r, z, sh, dtype, device):
        if t is not None:
            T = [[1, 0, 0, t[0]],
                 [0, 1, 0, t[1]],
                 [0, 0, 1, t[2]],
                 [0, 0, 0, 1]]
        else:
            T = torch.eye(4, dtype=dtype, device=device)
        if r is not None:
            c = torch.cos(r)
            s = torch.sin(r)
            Rx = [[1, 0,     0,    0],
                  [0, c[0],  s[0], 0],
                  [0, -s[0], c[0], 0],
                  [0, 0,     0,    1]]
            Ry = [[c[1],  0, s[1], 0],
                  [0,     1, 0,    0],
                  [-s[1], 0, c[1], 0],
                  [0,     0,    0, 1]]
            Rz = [[c[2],  s[2], 0, 0],
                  [-s[2], c[2], 0, 0],
                  [0,     0,    1, 0],
                  [0,     0,    0, 1]]
            Rx = utils.as_tensor(Rx, dtype=dtype, device=device)
            Ry = utils.as_tensor(Ry, dtype=dtype, device=device)
            Rz = utils.as_tensor(Rz, dtype=dtype, device=device)
            R = Rx.mm(Ry.mm(Rz))
        else:
            R = torch.eye(4, dtype=dtype, device=device)
        if z is not None:
            Z = [[z[0], 0,    0,    0],
                 [0,    z[1], 0,    0],
                 [0,    0,    z[2], 0],
                 [0,    0,    0,    1]]
        else:
            Z = torch.eye(4, dtype=dtype, device=device)
        if sh is not None:
            S = [[1, sh[0], sh[1], 0],
                 [0, 1, sh[2], 0],
                 [0, 0,    1,    0],
                 [0, 0,    0,    1]]
        else:
            S = torch.eye(4, dtype=dtype, device=device)

        T = utils.as_tensor(T, dtype=dtype, device=device)
        R = utils.as_tensor(R, dtype=dtype, device=device)
        Z = utils.as_tensor(Z, dtype=dtype, device=device)
        S = utils.as_tensor(S, dtype=dtype, device=device)
        return T.mm(R.mm(Z.mm(S)))

    def affine_2d_or_3d(t, r, z, s, dim, dtype, device):
        if dim == 2:
            return affine_2d(t, r, z, s, dtype, device)
        else:
            return affine_3d(t, r, z, s, dtype, device)

    # Unstack
    prm = utils.as_tensor(prm).flatten()
    dtype = prm.dtype
    device = prm.device
    nb_prm = prm.numel()
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
    idx = idx + nb_s
    prm_s = prm[idx:idx+nb_s] if nb_prm > idx else None

    # Build affine matrix
    mat = affine_2d_or_3d(prm_t, prm_r, prm_z, prm_s, dim, dtype, device)

    # Multiply with RAS
    if layout is not None:
        mat = mat.mm(layout_matrix(layout))

    return mat


def affine_parameters(mat, basis, layout='RAS', max_iter=10000, tol=1e-16,
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

    layout : str or (D+1, D+1) tensor_like, default='RAS'
        "Point" at which to take the matrix exponential
        (see affine_layout)

    max_iter : int, default=10000
        Maximum number of Gauss-Newton iterations in the least-squares fit.

    tol : float, default = 1e-16
        Tolerance criterion for convergence.
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

    # Format basis
    basis, _ = _format_basis(basis, dim)
    nb_basis = sum([len(b) for b in basis])

    # Create layout matrix
    # TODO: check that it works with new layout
    if isinstance(layout, str):
        layout = layout_matrix(layout)

    def gauss_newton():
        # Predefine these values in case max_iter == 0
        n_iter = -1
        # Gauss-Newton optimisation
        prm = torch.zeros(nb_basis, dtype=dtype)
        M = torch.eye(nb_basis, dtype=dtype)
        M = torch.mm(M, layout)
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

            # Multiply with layout
            M = M.mm(layout)
            dM = dM.mm(layout)

            # Compute gradient/Hessian of the loss (squared residuals)
            diff = M - mat
            diff = diff.flatten()
            dM = dM.reshape((nb_basis, -1))
            gradient = dM.mm(diff)
            hessian = dM.mm(dM.t())
            delta_prm = linalg.lmdiv(hessian, gradient)

            crit = (delta_prm ** 2).sum() / norm
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
                    M = affine_matrix(prm, basis)
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


def voxel_size(mat):
    """ Compute voxel sizes from affine matrices.

    Parameters
    ----------
    mat :  (..., ndim[+1], ndim+1) tensor
        Affine matrix

    Returns
    -------
    vx :  (..., ndim) tensor
        Voxel size

    """
    dim = mat.shape[-1] - 1
    return mat[..., :dim, :dim].square().sum(-2).sqrt()


voxsize = functools.wraps(voxel_size)  # backward compatibility


def affine_matvec(affine, vector):
    """Matrix-vector product of a rectangular affine.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
    vector : (..., ndim[+1]) tensor

    Returns
    -------
    affine_times_vector : (..., ndim) tensor

    """
    affine = torch.as_tensor(affine)
    vector = torch.as_tensor(vector)
    ndims = affine.shape[-1] - 1
    zoom = affine[..., :ndims, :ndims]
    translation = affine[..., :ndims, ndims]
    return linalg.matvec(zoom, vector[..., :ndims]) + translation


def affine_matmul(a, b):
    """Matrix-matrix product of rectangular affine matrices.

    Parameters
    ----------
    a : (..., ndim[+1], ndim+1) tensor
        Affine matrix
    b : (..., ndim[+1], ndim+1) tensor
        Affine matrix

    Returns
    -------
    affine_times_matrix : (..., ndim, ndim+1) tensor

    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    ndims = a.shape[-1] - 1
    Za = a[..., :ndims, :ndims]
    Ta = a[..., :ndims, ndims]
    Zb = b[..., :ndims, :ndims]
    Tb = b[..., :ndims, ndims]
    Z = torch.matmul(Za, Zb)
    T = linalg.matvec(Za, Tb) + Ta
    return torch.cat((Z, T[..., None]), dim=-1)


def affine_inv(affine):
    """Inverse of an affine matrix.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor

    Returns
    -------
    inv_affine : (..., ndim, ndim+1) tensor

    """
    affine = torch.as_tensor(affine)
    ndims = affine.shape[-1] - 1
    zoom = torch.inverse(affine[..., :ndims, :ndims])
    translation = -linalg.matvec(zoom, affine[..., :ndims, ndims])
    return torch.cat((zoom, translation[..., None]), dim=-1)


def affine_lmdiv(a, b):
    """inv(a) @ b"""
    return affine_matmul(affine_inv(a), b)


def affine_rmdiv(a, b):
    """a @ inv(b)"""
    return affine_matmul(a, affine_inv(b))


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
    ndims = affine.shape[-1]-1
    if affine.shape[-2] not in (ndims, ndims+1):
        raise ValueError('Input affine matrix should be of shape\n'
                         '(..., ndims+1, ndims+1) or (..., ndims, ndims+1).')
    if affine.shape[-1] != affine.shape[-2]:
        bottom_row = torch.cat((torch.zeros(ndims), torch.ones(1)), dim=0)
        bottom_row = utils.unsqueeze(bottom_row, 0, ndim=affine.dim()-1)
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
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    factor : float or sequence[float]
        Resizing factor.
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        Anchor points.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
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
    affine = affine_matmul(affine, trf)
    return affine, tuple(shape_out)


def affine_sub(affine, shape, indices):
    """Update an affine matrix according to a sub-indexing of the lattice.

    Notes
    -----
    .. Only sub-indexing that *do not change the number of dimensions*
       and that *keep an homogeneous voxel size* are allowed. Therefore,
       indices must be of type `slice` or `ellipsis`.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    indices : tuple[slice or ellipsis]
        Subscripting indices.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int]
        Updated shape.

    """
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
        else:
            raise TypeError('Indices should be slices or ellipses. '
                            'Got {}.'.format(type(ind)))
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
        else:
            raise TypeError('Indices should be slices or ellipses. '
                            'Got {}.'.format(type(ind)))

    # Extract shift and scale in each dimension
    shifts = []
    scales = []
    shape_out = []
    for d, ind in enumerate(indices):
        # translation + scale
        shifts.append(ind.start)
        scales.append(ind.step)
        shape_out.append(shape[d] // abs(ind.step))

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scales, **info))
    trl = torch.as_tensor(shifts, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1)

    # compose
    affine = affine_matmul(affine, trf)
    return affine, tuple(shape_out)


def affine_permute(affine, shape, perm=None):
    """Update an affine matrix according to a permutation of the lattice dims.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    perm : sequence[int], optional
        Permutation of the lattice dimensions.
        By default, reverse dimension order.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int]
        Updated shape.
    """
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    if perm is None:
        perm = list(range(nb_dim-1, -1, -1))
    if len(perm) != nb_dim:
        raise ValueError('Expected perm to have {} elements. Got {}.'
                         .format(nb_dim, len(perm)))
    affine = affine[..., perm, :]
    shape = [shape[p] for p in perm]
    return affine, tuple(shape)


def affine_transpose(affine, shape, dim0, dim1):
    """Update an affine matrix according to a transposition of the lattice.

    A transposition is a permutation that only impacts two dimensions.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    dim0 : int
        Index of the first dimension
    dim1 : int
        Index of the second dimension

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int]
        Updated shape.
    """
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    perm = list(range(nb_dim))
    perm[dim0] = dim1
    perm[dim1] = dim0
    return affine_permute(affine, shape, perm)

