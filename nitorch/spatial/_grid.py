# -*- coding: utf-8 -*-
"""Spatial deformations (i.e., grids)."""

import torch
from nitorch.core import utils, linalg
from nitorch.core.utils import expand, make_vector
from nitorch.core.py import make_list, prod
from nitorch._C.grid import (GridPull, GridPush, GridCount, GridGrad,
                             BoundType, InterpolationType,
                             SplineCoeff, SplineCoeffND)
from ._affine import affine_resize, affine_lmdiv, voxel_size
from ._finite_differences import diff


__all__ = ['grid_pull', 'grid_push', 'grid_count', 'grid_grad',
           'identity_grid', 'affine_grid', 'resize', 'resize_grid', 'reslice',
           'add_identity_grid', 'add_identity_grid_',
           'sub_identity_grid', 'sub_identity_grid_',
           'grid_jacobian', 'grid_jacdet', 'BoundType', 'InterpolationType',
           'spline_coeff', 'spline_coeff_nd']

_doc_interpolation = \
"""`interpolation` can be an int, a string or an InterpolationType.
    Possible values are:
        - 0 or 'nearest'    or InterpolationType.nearest
        - 1 or 'linear'     or InterpolationType.linear
        - 2 or 'quadratic'  or InterpolationType.quadratic
        - 3 or 'cubic'      or InterpolationType.cubic
        - 4 or 'fourth'     or InterpolationType.fourth
        - etc.
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders."""

_doc_bound = \
"""`bound` can be an int, a string or a BoundType.
    Possible values are:
        - 'replicate'  or BoundType.replicate
        - 'dct1'       or BoundType.dct1
        - 'dct2'       or BoundType.dct2
        - 'dst1'       or BoundType.dst1
        - 'dst2'       or BoundType.dst2
        - 'dft'        or BoundType.dft
        - 'zero'       or BoundType.zero
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    Note that
    - `dft` corresponds to circular padding
    - `dct2` corresponds to Neumann boundary conditions (symmetric)
    - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
    See https://en.wikipedia.org/wiki/Discrete_cosine_transform
        https://en.wikipedia.org/wiki/Discrete_sine_transform"""

_doc_bound_coeff = \
"""`bound` can be an int, a string or a BoundType.
    Possible values are:
        - 'replicate'  or BoundType.replicate
        - 'dct1'       or BoundType.dct1
        - 'dct2'       or BoundType.dct2
        - 'dst1'       or BoundType.dst1
        - 'dst2'       or BoundType.dst2
        - 'dft'        or BoundType.dft
        - 'zero'       or BoundType.zero
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    Note that
    - `dft` corresponds to circular padding
    - `dct1` corresponds to mirroring about the center of hte first/last voxel
    See https://en.wikipedia.org/wiki/Discrete_cosine_transform
        https://en.wikipedia.org/wiki/Discrete_sine_transform

    /!\ Only 'dct1', 'dct2' and 'dft' are implemented for interpolation
        orders >= 6."""


_ref_coeff = \
"""..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).
"""


def _preproc(grid, input=None, mode=None):
    """Preprocess tensors for pull/push/count/grad

    Low level C bindings expect inputs of shape
    [batch, channel, *spatial] and [batch, *spatial, dim], whereas
    the high level python API accepts inputs of shape
    [..., [channel], *spatial] and [..., *spatial, dim].

    This function broadcasts and reshapes the input tensors accordingly.
            /!\\ This *can* trigger large allocations /!\\
    """
    dim = grid.shape[-1]
    if input is None:
        spatial = grid.shape[-dim-1:-1]
        batch = grid.shape[:-dim-1]
        grid = grid.reshape([-1, *spatial, dim])
        info = dict(batch=batch, channel=[1] if batch else [], dim=dim)
        return grid, info

    grid_spatial = grid.shape[-dim-1:-1]
    grid_batch = grid.shape[:-dim-1]
    input_spatial = input.shape[-dim:]
    channel = 0 if input.dim() == dim else input.shape[-dim-1]
    input_batch = input.shape[:-dim-1]

    if mode == 'push':
        grid_spatial = input_spatial = utils.expanded_shape(grid_spatial, input_spatial)

    # broadcast and reshape
    batch = utils.expanded_shape(grid_batch, input_batch)
    grid = grid.expand([*batch, *grid_spatial, dim])
    grid = grid.reshape([-1, *grid_spatial, dim])
    input = input.expand([*batch, channel or 1, *input_spatial])
    input = input.reshape([-1, channel or 1, *input_spatial])

    out_channel = [channel] if channel else ([1] if batch else [])
    info = dict(batch=batch, channel=out_channel, dim=dim)
    return grid, input, info


def _postproc(out, shape_info, mode):
    """Postprocess tensors for pull/push/count/grad"""
    dim = shape_info['dim']
    if mode != 'grad':
        spatial = out.shape[-dim:]
        feat = []
    else:
        spatial = out.shape[-dim-1:-1]
        feat = [out.shape[-1]]
    batch = shape_info['batch']
    channel = shape_info['channel']

    out = out.reshape([*batch, *channel, *spatial, *feat])
    return out


def grid_pull(input, grid, interpolation='linear', bound='zero',
              extrapolate=False, prefilter=False):
    """Sample an image with respect to a deformation field.

    Notes
    -----
    {interpolation}

    {bound}

    If the input dtype is not a floating point type, the input image is
    assumed to contain labels. Then, unique labels are extracted
    and resampled individually, making them soft labels. Finally,
    the label map is reconstructed from the individual soft labels by
    assigning the label with maximum soft value.

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *outshape, dim) tensor
        Transformation field.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.
    prefilter : bool, default=False
        Apply spline pre-filter (= interpolates the input)

    Returns
    -------
    output : (..., [channel], *outshape) tensor
        Deformed image.

    """
    grid, input, shape_info = _preproc(grid, input)
    batch, channel = input.shape[:2]
    dim = grid.shape[-1]

    if not input.dtype.is_floating_point:
        # label map -> specific processing
        out = input.new_zeros([batch, channel, *grid.shape[1:-1]])
        pmax = grid.new_zeros([batch, channel, *grid.shape[1:-1]])
        for label in input.unique():
            soft = (input == label).to(grid.dtype)
            if prefilter:
                input = spline_coeff_nd(soft, interpolation=interpolation,
                                        bound=bound, dim=dim, inplace=True)
            soft = GridPull.apply(soft, grid, interpolation, bound, extrapolate)
            out[soft > pmax] = label
            pmax = torch.max(pmax, soft)
    else:
        if prefilter:
            input = spline_coeff_nd(input, interpolation=interpolation,
                                    bound=bound, dim=dim)
        out = GridPull.apply(input, grid, interpolation, bound, extrapolate)

    return _postproc(out, shape_info, mode='pull')


def grid_push(input, grid, shape=None, interpolation='linear', bound='zero',
              extrapolate=False, prefilter=False):
    """Splat an image with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.
    prefilter : bool, default=False
        Apply spline pre-filter.

    Returns
    -------
    output : (..., [channel], *shape) tensor
        Spatted image.

    """
    grid, input, shape_info = _preproc(grid, input, mode='push')
    dim = grid.shape[-1]

    if shape is None:
        shape = tuple(input.shape[2:])

    out = GridPush.apply(input, grid, shape, interpolation, bound, extrapolate)
    if prefilter:
        out = spline_coeff_nd(out, interpolation=interpolation, bound=bound,
                              dim=dim, inplace=True)
    return _postproc(out, shape_info, mode='push')


def grid_count(grid, shape=None, interpolation='linear', bound='zero',
               extrapolate=False):
    """Splatting weights with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [1], *shape) tensor
        Splatted weights.

    """
    grid, shape_info = _preproc(grid)
    if shape is None:
        shape = tuple(grid.shape[1:-1])
    out = GridCount.apply(grid, shape, interpolation, bound, extrapolate)
    return _postproc(out, shape_info, mode='count')


def grid_grad(input, grid, interpolation='linear', bound='zero',
              extrapolate=False, prefilter=False):
    """Sample spatial gradients of an image with respect to a deformation field.

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.
    prefilter : bool, default=False
        Apply spline pre-filter (= interpolates the input)

    Returns
    -------
    output : (..., [channel], *shape, dim) tensor
        Sampled gradients.

    """
    grid, input, shape_info = _preproc(grid, input)
    dim = grid.shape[-1]
    if prefilter:
        input = spline_coeff_nd(input, interpolation, bound, dim)
    out = GridGrad.apply(input, grid, interpolation, bound, extrapolate)
    return _postproc(out, shape_info, mode='grad')


def spline_coeff(input, interpolation='linear', bound='dct2', dim=-1,
                 inplace=False):
    """Compute the interpolating spline coefficients, for a given spline order
    and boundary conditions, along a single dimension.

    Notes
    -----
    {interpolation}

    {bound}

    References
    ----------
    {ref}


    Parameters
    ----------
    input : tensor
        Input image.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='dct1'
        Boundary conditions.
    dim : int, default=-1
        Dimension along which to process
    inplace : bool, default=False
        Process the volume in place.

    Returns
    -------
    output : tensor
        Coefficient image.

    """
    # This implementation is based on the file bsplines.c in SPM12, written
    # by John Ashburner, which is itself based on the file coeff.c,
    # written by Philippe Thevenaz: http://bigwww.epfl.ch/thevenaz/interpolation
    # . DCT1 boundary conditions were derived by Thevenaz and Unser.
    # . DFT boundary conditions were derived by John Ashburner.
    # SPM12 is released under the GNU-GPL v2 license.
    # Philippe Thevenaz's code does not have an explicit license as far
    # as we know.
    out = SplineCoeff.apply(input, bound, interpolation, dim, inplace)
    return out


def spline_coeff_nd(input, interpolation='linear', bound='dct2', dim=None,
                    inplace=False):
    """Compute the interpolating spline coefficients, for a given spline order
    and boundary conditions, along the last `dim` dimensions.

    Notes
    -----
    {interpolation}

    {bound}

    References
    ----------
    {ref}

    Parameters
    ----------
    input : (..., *spatial) tensor
        Input image.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='dct1'
        Boundary conditions.
    dim : int, default=-1
        Number of spatial dimensions
    inplace : bool, default=False
        Process the volume in place.

    Returns
    -------
    output : (..., *spatial) tensor
        Coefficient image.

    """
    # This implementation is based on the file bsplines.c in SPM12, written
    # by John Ashburner, which is itself based on the file coeff.c,
    # written by Philippe Thevenaz: http://bigwww.epfl.ch/thevenaz/interpolation
    # . DCT1 boundary conditions were derived by Thevenaz and Unser.
    # . DFT boundary conditions were derived by John Ashburner.
    # SPM12 is released under the GNU-GPL v2 license.
    # Philippe Thevenaz's code does not have an explicit license as far
    # as we know.
    out = SplineCoeffND.apply(input, bound, interpolation, dim, inplace)
    return out


grid_pull.__doc__ = grid_pull.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_push.__doc__ = grid_push.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_count.__doc__ = grid_count.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_grad.__doc__ = grid_grad.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
spline_coeff.__doc__ = spline_coeff.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound_coeff, ref=_ref_coeff)
spline_coeff_nd.__doc__ = spline_coeff_nd.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound_coeff, ref=_ref_coeff)

# aliases
pull = grid_pull
push = grid_push
count = grid_count


def identity_grid(shape, dtype=None, device=None, jitter=False):
    """Returns an identity deformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.
    dtype : torch.dtype, default=`get_default_dtype()`
        Data type.
    device torch.device, optional
        Device.
    jitter : bool or 'reproducible', default=False
        Jitter identity grid.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = utils.meshgrid_ij(*mesh1d)
    grid = torch.stack(grid, dim=-1)
    if jitter:
        reproducible = jitter == 'reproducible'
        device_ids = [grid.device.index] if grid.device.type == 'cuda' else []
        with torch.random.fork_rng(device_ids, enabled=reproducible):
            if reproducible:
                torch.manual_seed(0)
            jitter = torch.rand_like(grid).sub_(0.5).mul_(0.1)
            grid += jitter
    return grid


@torch.jit.script
def _movedim1(x, source: int, destination: int):
    dim = x.dim()
    source = dim + source if source < 0 else source
    destination = dim + destination if destination < 0 else destination
    permutation = [d for d in range(dim)]
    permutation = permutation[:source] + permutation[source+1:]
    permutation = permutation[:destination] + [source] + permutation[destination:]
    return x.permute(permutation)


@torch.jit.script
def add_identity_grid_(disp):
    """Adds the identity grid to a displacement field, inplace.

    Parameters
    ----------
    disp : (..., *spatial, dim) tensor
        Displacement field

    Returns
    -------
    grid : (..., *spatial, dim) tensor
        Transformation field

    """
    dim = disp.shape[-1]
    spatial = disp.shape[-dim-1:-1]
    mesh1d = [torch.arange(s, dtype=disp.dtype, device=disp.device)
              for s in spatial]
    grid = utils.meshgrid_script_ij(mesh1d)
    disp = _movedim1(disp, -1, 0)
    for i, grid1 in enumerate(grid):
        disp[i].add_(grid1)
    disp = _movedim1(disp, 0, -1)
    return disp


@torch.jit.script
def sub_identity_grid_(disp):
    """Subtracts the identity grid to a displacement field, inplace.

    Parameters
    ----------
    disp : (..., *spatial, dim) tensor
        Transformation field

    Returns
    -------
    grid : (..., *spatial, dim) tensor
        Displacement field

    """
    dim = disp.shape[-1]
    spatial = disp.shape[-dim-1:-1]
    mesh1d = [torch.arange(s, dtype=disp.dtype, device=disp.device)
              for s in spatial]
    grid = utils.meshgrid_script_ij(mesh1d)
    disp = _movedim1(disp, -1, 0)
    for i, grid1 in enumerate(grid):
        disp[i].sub_(grid1)
    disp = _movedim1(disp, 0, -1)
    return disp


@torch.jit.script
def add_identity_grid(disp):
    """Adds the identity grid to a displacement field.

    Parameters
    ----------
    disp : (..., *spatial, dim) tensor
        Displacement field

    Returns
    -------
    grid : (..., *spatial, dim) tensor
        Transformation field

    """
    return add_identity_grid_(disp.clone())


@torch.jit.script
def sub_identity_grid(disp):
    """Subtracts the identity grid to a displacement field.

    Parameters
    ----------
    disp : (..., *spatial, dim) tensor
        Transformation field

    Returns
    -------
    grid : (..., *spatial, dim) tensor
        Displacement field

    """
    return sub_identity_grid_(disp.clone())


def affine_grid(mat, shape, jitter=False):
    """Create a dense transformation grid from an affine matrix.

    Parameters
    ----------
    mat : (..., D[+1], D[+1]) tensor
        Affine matrix (or matrices).
    shape : (D,) sequence[int]
        Shape of the grid, with length D.
    jitter : bool or 'reproducible', default=False
        Jitter identity grid.

    Returns
    -------
    grid : (..., *shape, D) tensor
        Dense transformation grid

    """
    mat = torch.as_tensor(mat)
    shape = list(shape)
    nb_dim = mat.shape[-1] - 1
    if nb_dim != len(shape):
        raise ValueError('Dimension of the affine matrix ({}) and shape ({}) '
                         'are not the same.'.format(nb_dim, len(shape)))
    if mat.shape[-2] not in (nb_dim, nb_dim+1):
        raise ValueError('First argument should be matrces of shape '
                         '(..., {0}, {1}) or (..., {1], {1}) but got {2}.'
                         .format(nb_dim, nb_dim+1, mat.shape))
    batch_shape = mat.shape[:-2]
    grid = identity_grid(shape, mat.dtype, mat.device, jitter=jitter)
    if batch_shape:
        grid = utils.unsqueeze(grid, dim=0, ndim=len(batch_shape))
        mat = utils.unsqueeze(mat, dim=-3, ndim=nb_dim)
    lin = mat[..., :nb_dim, :nb_dim]
    off = mat[..., :nb_dim, -1]
    grid = linalg.matvec(lin, grid) + off
    return grid


def resize(image, factor=None, shape=None, affine=None, anchor='c',
           dim=None, *args, **kwargs):
    """Resize an image by a factor or to a specific shape.

    Notes
    -----
    .. A least one of `factor` and `shape` must be specified
    .. If `anchor in ('centers', 'edges')`, and both `factor` and `shape`
       are specified, `factor` is discarded.
    .. If `anchor in ('first', 'last')`, `factor` must be provided even
       if `shape` is specified.
    .. Because of rounding, it is in general not assured that
       `resize(resize(x, f), 1/f)` returns a tensor with the same shape as x.

        edges          centers          first           last
    e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
    | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
    e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +

    Parameters
    ----------
    image : (batch, channel, ...) tensor
        Image to resize
    factor : float or list[float], optional
        Resizing factor
        * > 1 : larger image <-> smaller voxels
        * < 1 : smaller image <-> larger voxels
    shape : (ndim,) sequence[int], optional
        Output shape
    affine : (batch, ndim[+1], ndim+1), optional
        Orientation matrix of the input image.
        If provided, the orientation matrix of the resized image is
        returned as well.
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        * In cases 'c' and 'e', the volume shape is multiplied by the
          zoom factor (and eventually truncated), and two anchor points
          are used to determine the voxel size.
        * In cases 'f' and 'l', a single anchor point is used so that
          the voxel size is exactly divided by the zoom factor.
          This case with an integer factor corresponds to subslicing
          the volume (e.g., `vol[::f, ::f, ::f]`).
        * A list of anchors (one per dimension) can also be provided.
    dim : int, optional
        Number of spatial dimensions.
        By default, it is guessed from the length of `shape` (if provided)
        or from `image.dim() - 2`.
    **kwargs : dict
        Parameters of `grid_pull`.
        Note that here, `prefilter=True` by default, whereas in
        `grid_pull`, `prefilter=False` by default.

    Returns
    -------
    resized : (batch, channel, ...) tensor
        Resized image.
    affine : (batch, ndim[+1], ndim+1) tensor, optional
        Orientation matrix

    """
    # TODO: we could also use dft/dct/dst to resize, which correspond
    #   to some sort of sinc interpolation.

    # read parameters
    image = torch.as_tensor(image)
    if shape is not None:
        nb_dim = dim or len(shape)
    else:
        nb_dim = dim or (image.dim() - 2)
    inshape = image.shape[-nb_dim:]
    info = {'dtype': image.dtype, 'device': image.device}
    factor = make_vector(factor or 0., nb_dim).tolist()
    outshape = make_list(shape or [None], nb_dim)
    anchor = [a[0].lower() for a in make_list(anchor, nb_dim)]
    return_trf = kwargs.pop('_return_trf', False)  # hidden option

    # compute output shape
    outshape = [o or max(1, int(i*f)) for i, o, f in zip(inshape, outshape, factor)]
    if any(not s for s in outshape):
        raise ValueError('Either factor or shape must be set in '
                         'all dimensions')
    factor = [f or o/i for o, i, f in zip(outshape, inshape, factor)]

    # compute transformation grid
    # there is an affine relationship between the input and output grid:
    #    input_grid = scale * output_grid + shift
    lin = []
    scales = []
    shifts = []
    for anch, f, inshp, outshp in zip(anchor, factor, inshape, outshape):
        if inshp == 1 or outshp == 1:
            # anchor must be "edges"
            anch = 'e'
        if anch == 'c':    # centers
            lin.append(torch.linspace(0, inshp - 1, outshp, **info))
            scales.append((inshp - 1) / (outshp - 1))
            shifts.append(0)
        elif anch == 'e':  # edges
            scale = inshp / outshp
            shift = 0.5 * (scale - 1)
            lin.append(torch.arange(0., outshp, **info) * scale + shift)
            scales.append(scale)
            shifts.append(shift)
        elif anch == 'f':  # first voxel
            lin.append(torch.arange(0., outshp, **info) / f)
            scales.append(1 / f)
            shifts.append(0)
        elif anch == 'l':  # last voxel
            shift = (inshp - 1) - (outshp - 1) / f
            lin.append(torch.arange(0., outshp, **info) / f + shift)
            scales.append(1 / f)
            shifts.append(shift)
        else:
            raise ValueError('Unknown anchor {}'.format(anch))
        if inshp == outshp == 1:
            scales[-1] = 1/f
    grid = torch.stack(utils.meshgrid_ij(*lin), dim=-1)

    # resize input image
    kwargs.setdefault('prefilter', True)
    resized = grid_pull(image, grid, *args, **kwargs)

    # compute orientation matrix
    if affine is not None:
        affine, _ = affine_resize(affine, inshape, factor, anchor, shape_out=outshape)
        if return_trf:
            return resized, affine, (scales, shifts)
        else:
            return resized, affine

    if return_trf:
        return resized, (scales, shifts)
    else:
        return resized


def resize_grid(grid, factor=None, shape=None, type='grid',
                affine=None, *args, **kwargs):
    """Resize a displacement grid by a factor.

    The displacement grid is resized *and* rescaled, so that
    displacements are expressed in the new voxel referential.

    Notes
    -----
    .. A least one of `factor` and `shape` must be specified.
    .. If `anchor in ('centers', 'edges')`, and both `factor` and `shape`
       are specified, `factor` is discarded.
    .. If `anchor in ('first', 'last')`, `factor` must be provided even
       if `shape` is specified.
    .. Because of rounding, it is in general not assured that
       `resize(resize(x, f), 1/f)` returns a tensor with the same shape as x.

    Parameters
    ----------
    grid : (batch, ..., ndim) tensor
        Grid to resize
    factor : float or list[float], optional
        Resizing factor
        * > 1 : larger image <-> smaller voxels
        * < 1 : smaller image <-> larger voxels
    shape : (ndim,) sequence[int], optional
        Output shape
    type : {'grid', 'displacement'}, default='grid'
        Grid type:
        * 'grid' correspond to dense grids of coordinates.
        * 'displacement' correspond to dense grid of relative displacements.
        Both types are not rescaled in the same way.
    affine : (batch, ndim[+1], ndim+1), optional
        Orientation matrix of the input grid.
        If provided, the orientation matrix of the resized image is
        returned as well.
    anchor : {'centers', 'edges', 'first', 'last'}, default='centers'
        * In cases 'c' and 'e', the volume shape is multiplied by the
          zoom factor (and eventually truncated), and two anchor points
          are used to determine the voxel size.
        * In cases 'f' and 'l', a single anchor point is used so that
          the voxel size is exactly divided by the zoom factor.
          This case with an integer factor corresponds to subslicing
          the volume (e.g., `vol[::f, ::f, ::f]`).
        * A list of anchors (one per dimension) can also be provided.
    **kwargs
        Parameters of `grid_pull`.
        Note that here, `prefilter=True` by default, whereas in
        `grid_pull`, `prefilter=False` by default.

    Returns
    -------
    resized : (batch, ..., ndim) tensor
        Resized grid.
    affine : (batch, ndim[+1], ndim+1) tensor, optional
        Orientation matrix

    """
    # resize grid
    kwargs['_return_trf'] = True
    kwargs.setdefault('prefilter', True)
    ndim = grid.shape[-1]
    *spatial, _ = grid.shape[-ndim-1:-1]
    batch = grid.shape[:-ndim-1]
    if len(batch) == 0:
        grid = grid[None]
    elif len(batch) > 1:
        grid = grid.reshape([-1, *spatial, ndim])
    if affine is not None:
        affine = affine.expand([*batch, *affine.shape[-2:]])
        affine = affine.reshape([-1, *affine.shape[-2:]])
    grid = utils.last2channel(grid)
    outputs = resize(grid, factor, shape, affine, *args, **kwargs)
    if affine is not None:
        grid, affine, (scales, shifts) = outputs
    else:
        grid, (scales, shifts) = outputs
    grid = utils.channel2last(grid)

    # rescale each component
    # scales and shifts map resized coordinates to original coordinates:
    #   original = scale * resized + shift
    # here we want to transform original coordinates into resized ones:
    #   resized = (original - shift) / scale
    grids = []
    for d, (scl, shft) in enumerate(zip(scales, shifts)):
        grid1 = utils.slice_tensor(grid, d, dim=-1)
        if type[0].lower() == 'g':
            grid1 = grid1 - shft
        grid1 = grid1 / scl
        grids.append(grid1)
    grid = torch.stack(grids, -1)

    grid = grid.reshape([*batch, *grid.shape[-ndim-1:]])

    # return
    if affine is not None:
        affine = affine.reshape([*batch, *affine.shape[-2:]])
        return grid, affine
    else:
        return grid


def reslice(image, affine, affine_to, shape_to=None, **kwargs):
    """Reslice a spatial image to a different space (shape + affine).

    Parameters
    ----------
    image : (*batch, *channels, *spatial)
        Input image
    affine : (*batch, dim[+1], dim+1)
        Input affine
    affine_to : (*batch, dim[+1], dim+1)
        Target affine
    shape_to : (dim,) sequence[int], optional
        Target shape. Default: same as input shape

    Other Parameters
    ----------------
    Parameters of `grid_pull`
    Note that here, `prefilter=True` by default, whereas in
    `grid_pull`, `prefilter=False` by default.

    Returns
    -------
    resliced : (*batch, *channels, *shape_to)
        Resliced image.

    """
    kwargs.setdefault('prefilter', True)

    # prepare tensors
    image = torch.as_tensor(image)
    backend = dict(dtype=image.dtype, device=image.device)
    if not backend['dtype'].is_floating_point:
        backend['dtype'] = torch.get_default_dtype()
    affine = torch.as_tensor(affine, **backend)
    affine_to = torch.as_tensor(affine_to, **backend)

    # compute shape components
    dim = affine.shape[-1] - 1
    batch = affine.shape[:-2]
    channels = image.shape[len(batch):-dim]
    shape = image.shape[-dim:]
    if shape_to is None:
        shape_to = shape

    # perform reslicing
    #   > image must be reshaped to (B, C, *spatial) for grid_pull
    transformation = affine_lmdiv(affine, affine_to)        # (*batch, d+1, d+1)
    grid = affine_grid(transformation, shape_to)            # (*batch, *shape_to, d)
    if not prod(batch):
        grid = grid[None]                                   # (*batch|1, *shape_to, d)
    squeeze_shape = [prod(batch) or 1, prod(channels) or 1, *shape]
    image = image.reshape(squeeze_shape)                    # (b, c, *spatial)
    image = grid_pull(image, grid, **kwargs)                # (b, c, *shape_to)
    image = image.reshape([*batch, *channels, *shape_to])   # (*batch, *channels, *shape_to)
    return image


def grid_jacobian(grid, sample=None, bound='dft', voxel_size=1, type='grid',
                  add_identity=True, extrapolate=True):
    """Compute the Jacobian of a transformation field

    Notes
    -----
    .. If `add_identity` is True, we compute the Jacobian
       of the transformation field (identity + displacement), even if
       a displacement is provided, by adding ones to the diagonal.
    .. If `sample` is not used, this function uses central finite
       differences to estimate the Jacobian.
    .. If 'sample' is provided, `grid_grad` is used to sample derivatives.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation or displacement field
    sample : (..., *spatial, dim) tensor, optional
        Coordinates to sample in the input grid.
    bound : str, default='dft'
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    type : {'grid', 'disp'}, default='grid'
        Whether the input is a transformation ('grid') or displacement
        ('disp') field.
    add_identity : bool, default=True
        Adds the identity to the Jacobian of the displacement, making it
        the jacobian of the transformation.
    extrapolate : bool, default=True
        Extrapolate out-of-boudn data (only useful is `sample` is used)

    Returns
    -------
    jac : (..., *spatial, dim, dim) tensor
        Jacobian. In each matrix: jac[i, j] = d psi[i] / d xj

    """
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    shape = grid.shape[-dim-1:-1]
    if type == 'grid':
        grid = grid - identity_grid(shape, **utils.backend(grid))
    if sample is None:
        dims = list(range(-dim-1, -1))
        jac = diff(grid, dim=dims, bound=bound, voxel_size=voxel_size, side='c')
    else:
        grid = utils.movedim(grid, -1, -dim-1)
        jac = grid_grad(grid, sample, bound=bound, extrapolate=extrapolate)
        jac = utils.movedim(jac, -dim-2, -2)
    if add_identity:
        torch.diagonal(jac, 0, -1, -2).add_(1)
    return jac


def grid_jacdet(grid, sample=None, bound='dft', voxel_size=1, type='grid',
                add_identity=True, extrapolate=True):
    """Compute the determinant of the Jacobian of a transformation field

    Notes
    -----
    .. If `add_identity` is True, we compute the Jacobian
       of the transformation field (identity + displacement), even if
       a displacement is provided, by adding ones to the diagonal.
    .. If `sample` is not used, this function uses central finite
       differences to estimate the Jacobian.
    .. If 'sample' is provided, `grid_grad` is used to sample derivatives.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation or displacement field
    sample : (..., *spatial, dim) tensor, optional
        Coordinates to sample in the input grid.
    bound : str, default='dft'
        Boundary condition
    voxel_size : [sequence of] float, default=1
        Voxel size
    type : {'grid', 'disp'}, default='grid'
        Whether the input is a transformation ('grid') or displacement
        ('disp') field.
    add_identity : bool, default=True
        Adds the identity to the Jacobian of the displacement, making it
        the jacobian of the transformation.
    extrapolate : bool, default=True
        Extrapolate out-of-boudn data (only useful is `sample` is used)

    Returns
    -------
    det : (..., *spatial) tensor
        Jacobian determinant.

    """
    jac = grid_jacobian(grid, sample=sample, bound=bound,
                        voxel_size=voxel_size, type=type,
                        add_identity=add_identity, extrapolate=extrapolate)
    return jac.det()


# def transform_points(points, grid, type='grid',
#                      affine=None, points_unit='mm', grid_unit='voxels',
#                      bound='zero', interpolation=1, extrapolate=0):
#     """
#
#     Parameters
#     ----------
#     points : ([batch], collection, dim) tensor
#         Collection of points (in
#     grid : ([batch], *spatial, dim) tensor
#         Transformation of displacement grid
#     type : {'grid', 'displacement'}
#     affine
#     bound
#     interpolation
#     extrapolate
#
#     Returns
#     -------
#
#     """
#
#     vertices = torch.as_tensor(vertices)
#     disp = torch.as_tensor(disp)
#     aff = torch.as_tensor(aff)
#     # convert vertices to voxels to sample
#     grid = ni.spatial.affine_matvec(ni.spatial.affine_inv(aff),
#                                          vertices)
#     grid = grid.reshape([1, 1, -1, 3])  # make 3d
#     # sample displacement
#     wdisp = ni.spatial.grid_pull(disp.movedim(-1, 0), grid,
#                                       bound='zero', extrapolate=False)
#     wdisp = wdisp.movedim(0, -1)
#     grid = grid + wdisp
#     grid = grid.reshape([-1, 3])
#     # convert voxels to mm
#     wvertices = ni.spatial.affine_matvec(aff, grid)
