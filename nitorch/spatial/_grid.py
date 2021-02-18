# -*- coding: utf-8 -*-
"""Spatial deformations (i.e., grids)."""

import torch
import torch.nn.functional as _F
from nitorch.core import kernels, utils, linalg
from nitorch.core.utils import expand, make_vector
from nitorch.core.py import make_list, prod
from nitorch._C import spatial as _Cspatial
from nitorch._C.spatial import BoundType, InterpolationType
from ._affine import affine_resize, affine_lmdiv
from ._regularisers import solve_grid_sym


__all__ = ['grid_pull', 'grid_push', 'grid_count', 'grid_grad',
           'identity_grid', 'affine_grid', 'compose', 'jacobian',
           'BoundType', 'InterpolationType', 'resize', 'resize_grid',
           'reslice', 'grid_inv']

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
    https://en.wikipedia.org/wiki/Discrete_sine_transform
"""


class _GridPull(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        opt = (bound, interpolation, extrapolate)

        # Pull
        output = _Cspatial.grid_pull(input, grid, *opt)

        # Context
        if input.requires_grad or grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_variables
        opt = ctx.opt
        grad_input = grad_grid = None
        grads = _Cspatial.grid_pull_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            grad_input = grads[0]
            if ctx.needs_input_grad[1]:
                grad_grid = grads[1]
        elif ctx.needs_input_grad[1]:
            grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


def grid_pull(input, grid, interpolation='linear', bound='zero', extrapolate=True):
    """Sample an image with respect to a deformation field.

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (batch, channel, *inshape) tensor
        Input image.
    grid : (batch, *outshape, dim) tensor
        Transformation field.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (batch, channel, *outshape) tensor
        Deformed image.

    """.format(interpolation=_doc_interpolation, bound=_doc_bound)
    # Convert parameters
    if not isinstance(bound, list) and \
       not isinstance(bound, tuple):
        bound = [bound]
    if not isinstance(interpolation, list) and \
       not isinstance(interpolation, tuple):
        interpolation = [interpolation]
    bound = [BoundType.__members__[b] if type(b) is str else BoundType(b)
             for b in bound]
    interpolation = [InterpolationType.__members__[i] if type(i) is str
                     else InterpolationType(i) for i in interpolation]

    # Broadcast
    batch = max(input.shape[0], grid.shape[0])
    input = expand(input, [batch, *input.shape[1:]])
    grid = expand(grid, [batch, *grid.shape[1:]])

    return _GridPull.apply(input, grid, interpolation, bound, extrapolate)


class _GridPush(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        opt = (bound, interpolation, extrapolate)

        # Push
        output = _Cspatial.grid_push(input, grid, shape, *opt)

        # Context
        if input.requires_grad or grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_variables
        opt = ctx.opt
        grad_input = grad_grid = None
        grads = _Cspatial.grid_push_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            grad_input = grads[0]
            if ctx.needs_input_grad[1]:
                grad_grid = grads[1]
        elif ctx.needs_input_grad[1]:
            grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None, None


def grid_push(input, grid, shape=None, interpolation='linear', bound='zero',
              extrapolate=True):
    """Splat an image with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (batch, channel, *inshape) tensor
        Input image.
    grid : (batch, *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (batch, channel, *shape) tensor
        Spatted image.

    """.format(interpolation=_doc_interpolation, bound=_doc_bound)
    # Convert parameters
    if not isinstance(bound, list) and \
       not isinstance(bound, tuple):
        bound = [bound]
    bound = list(bound)
    if not isinstance(interpolation, list) and \
       not isinstance(interpolation, tuple):
        interpolation = [interpolation]
    bound = [BoundType.__members__[b] if type(b) is str else BoundType(b)
             for b in bound]
    interpolation = [InterpolationType.__members__[i] if type(i) is str
                     else InterpolationType(i) for i in interpolation]

    # Broadcast
    batch = max(input.shape[0], grid.shape[0])
    channel = input.shape[1]
    ndims = grid.shape[-1]
    input_shape = input.shape[2:]
    grid_shape = grid.shape[1:-1]
    spatial = [max(sinp, sgrd) for sinp, sgrd in zip(input_shape, grid_shape)]
    input = expand(input, [batch, channel, *spatial])
    grid = expand(grid, [batch, *spatial, ndims])

    if shape is None:
        shape = tuple(input.shape[2:])

    return _GridPush.apply(input, grid, shape, interpolation, bound, extrapolate)


class _GridCount(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        opt = (bound, interpolation, extrapolate)

        # Push
        output = _Cspatial.grid_count(grid, shape, *opt)

        # Context
        if grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_variables
        opt = ctx.opt
        grad_grid = None
        if ctx.needs_input_grad[0]:
            grad_grid = _Cspatial.grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


def grid_count(grid, shape=None, interpolation='linear', bound='zero',
               extrapolate=True):
    """Splatting weights with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    grid : (batch, *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (batch, channel, *shape) tensor
        Spatting weights.

    """.format(interpolation=_doc_interpolation, bound=_doc_bound)
    # Convert parameters
    if not isinstance(bound, list) and \
       not isinstance(bound, tuple):
        bound = [bound]
    bound = list(bound)
    if not isinstance(interpolation, list) and \
       not isinstance(interpolation, tuple):
        interpolation = [interpolation]
    bound = [BoundType.__members__[b] if type(b) is str else BoundType(b)
             for b in bound]
    interpolation = [InterpolationType.__members__[i] if type(i) is str
                     else InterpolationType(i) for i in interpolation]

    if shape is None:
        shape = tuple(grid.shape[1:-1])

    return _GridCount.apply(grid, shape, interpolation, bound, extrapolate)


class _GridGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        opt = (bound, interpolation, extrapolate)

        # Pull
        output = _Cspatial.grid_grad(input, grid, *opt)

        # Context
        if input.requires_grad or grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_variables
        opt = ctx.opt
        grad_input = grad_grid = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = _Cspatial.grid_grad_backward(grad, *var, *opt)
            if ctx.needs_input_grad[0]:
                grad_input = grads[0]
                if ctx.needs_input_grad[1]:
                    grad_grid = grads[1]
            elif ctx.needs_input_grad[1]:
                grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


def grid_grad(input, grid, interpolation='linear', bound='zero', extrapolate=True):
    """Sample spatial gradients of an image with respect to a deformation field.
    
    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : (batch, channel, *inshape) tensor
        Input image.
    grid : (batch, *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (batch, channel, *shape, dim) tensor
        Sampled gradients.

    """.format(interpolation=_doc_interpolation, bound=_doc_bound)
    # Convert parameters
    if not isinstance(bound, list) and \
       not isinstance(bound, tuple):
        bound = [bound]
    if not isinstance(interpolation, list) and \
       not isinstance(interpolation, tuple):
        interpolation = [interpolation]
    bound = [BoundType.__members__[b] if type(b) is str else BoundType(b)
             for b in bound]
    interpolation = [InterpolationType.__members__[i] if type(i) is str
                     else InterpolationType(i) for i in interpolation]

    # Broadcast
    batch = max(input.shape[0], grid.shape[0])
    input = expand(input, [batch, *input.shape[1:]])
    grid = expand(grid, [batch, *grid.shape[1:]])

    return _GridGrad.apply(input, grid, interpolation, bound, extrapolate)


def _vox2fov(shape, align_corners=True):
    """Nifti to Torch coordinates.

    Returns an affine matrix that transforms nifti volume coordinates
    (in [0, len-1]) into pytorch volume coordinates (in [-1, 1]).

    Args:
        shape (tuple[int]): Shape of the volume grid (W, H, D).
        align_corners (bool, optional): Torch coordinate type.
            Defaults to True.

    Returns:
        mat (matrix): Affine conversion matrix (:math:`D+1 \times D+1`)

    """
    shape = torch.as_tensor(shape).to(torch.float)
    dim = shape.numel()
    if align_corners:
        offset = -1.
        scale = 2./(shape - 1.)
    else:
        offset = 1./shape-1.
        scale = 2./shape
    mat = torch.diag(torch.cat((scale, torch.ones(1))))
    mat[:dim, -1] = offset
    return mat


def _fov2vox(shape, align_corners=True):
    """Torch to Nifti coordinates."""
    shape = torch.as_tensor(shape).to(torch.float)
    dim = shape.numel()
    if align_corners:
        offset = (shape-1.)/2.
        scale = (shape - 1.)/2.
    else:
        offset = (shape-1.)/2.
        scale = shape/2.
    mat = torch.diag(torch.cat((scale, torch.ones(1))))
    mat[:dim, -1] = offset
    return mat


def _make_square(mat):
    """Transform a compact affine matrix into a square affine matrix."""
    mat = torch.as_tensor(mat)
    shape = mat.size()
    if mat.dim() != 2 or not shape[0] in (shape[1], shape[1] - 1):
        raise ValueError('Input matrix should be Dx(D+1) or (D+1)x(D+1).')
    if shape[0] < shape[1]:
        addrow = torch.zeros(1, shape[1], dtype=mat.dtype, device=mat.device)
        addrow[0, -1] = 1
        mat = torch.cat((mat, addrow), dim=0)
    return mat


def _make_compact(mat):
    """Transform a square affine matrix into a compact affine matrix."""
    mat = torch.as_tensor(mat)
    shape = mat.size()
    if mat.dim() != 2 or not shape[0] in (shape[1], shape[1] - 1):
        raise ValueError('Input matrix should be Dx(D+1) or (D+1)x(D+1).')
    if shape[0] == shape[1]:
        mat = mat[:-1, :]
    return mat


def identity_grid(shape, dtype=None, device=None):
    """Returns an identity deformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.
    dtype : torch.dtype, default=`get_default_dtype()`
        Data type.
    device torch.device, optional
        Device.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = torch.meshgrid(*mesh1d)
    grid = torch.stack(grid, dim=-1)
    return grid


def affine_grid(mat, shape):
    """Create a dense transformation grid from an affine matrix.

    Parameters
    ----------
    mat : (..., D[+1], D[+1]) tensor
        Affine matrix (or matrices).
    shape : (D,) sequence[int]
        Shape of the grid, with length D.

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
    grid = identity_grid(shape, mat.dtype, mat.device)
    grid = utils.unsqueeze(grid, dim=0, ndim=len(batch_shape))
    mat = utils.unsqueeze(mat, dim=-3, ndim=nb_dim)
    lin = mat[..., :nb_dim, :nb_dim]
    off = mat[..., :nb_dim, -1]
    grid = linalg.matvec(lin, grid) + off
    return grid


def compose(*args, interpolation='linear', bound='dft'):
    """Compose multiple spatial deformations (affine matrices or flow fields).
    """
    # TODO:
    # . add shape/dim argument to generate (if needed) an identity field
    #   at the end of the chain.
    # . possibility to provide fields that have an orientation matrix?
    #   (or keep it the responsibility of the user?)
    # . For higher order (> 1) interpolation: convert to spline coeficients.

    def ismatrix(x):
        """Check that a tensor is a matrix (ndim == 2)."""
        x = torch.as_tensor(x)
        shape = torch.as_tensor(x.shape)
        return shape.numel() == 2

    # Pre-pass: check dimensionality
    dim = None
    last_affine = False
    at_least_one_field = False
    for arg in args:
        if ismatrix(arg):
            last_affine = True
            dim1 = arg.shape[1]
        else:
            last_affine = False
            at_least_one_field = True
            dim1 = arg.dim() - 2
        if dim is not None and dim != dim1:
            raise ValueError("All deformations should have the same "
                             "dimensionality (2D/3D).")
        elif dim is None:
            dim = dim1
    if at_least_one_field and last_affine:
        raise ValueError("The last deformation cannot be an affine matrix. "
                         "Use affine_field to transform it first.")

    # First pass: compose all sequential affine matrices
    args1 = []
    last_affine = None
    for arg in args:
        if ismatrix(arg):
            if last_affine is None:
                last_affine = _make_square(arg)
            else:
                last_affine = last_affine.matmul(_make_square(arg))
        else:
            if last_affine is not None:
                args1.append(last_affine)
                last_affine = None
            args1.append(arg)

    if not at_least_one_field:
        return last_affine

    # Second pass: perform all possible "field x matrix" compositions
    args2 = []
    last_affine = None
    for arg in args1:
        if ismatrix(arg):
            last_affine = arg
        else:
            if last_affine is not None:
                new_field = arg.matmul(
                    last_affine[:dim, :dim].transpose(0, 1)) \
                  + last_affine[:dim, dim].reshape((1,)*(dim+1) + (dim,))
                args2.append(new_field)
            else:
                args2.append(arg)
    if last_affine is not None:
        args2.append(last_affine)

    # Third pass: compose all flow fields
    field = args2[-1]
    for arg in args2[-2::-1]:  # args2[-2:0:-1]
        arg = arg - identity_grid(arg.shape[1:-1], arg.dtype, arg.device)
        arg = utils.last2channel(arg)
        field = field + utils.channel2last(grid_pull(arg, field, interpolation, bound))

    # /!\ (TODO) The very first field (the first one being interpolated)
    # potentially contains a multiplication with an affine matrix (i.e.,
    # it might not be expressed in voxels). This affine transformation should
    # be removed prior to subtracting the identity, and added back at the end.
    # However, I don't know how to 'guess' this matrix.
    #
    # After further though, I think we can find the matrix that minimizes in
    # the least-square sense (F*M-I), where F is NbVox*D and contains the
    # deformation field, I is NbVox*D and contains the identity field
    # (expressed in voxels) and M is the inverse of the unknown matrix.
    # This problem has a closed form solution: (F'*F)\(F'*I).
    # For better stability, We could encode M in gl(D), the Lie
    # algebra of invertible matrices, and use gauss-newton to optimise
    # the problem.
    #
    # Below is a tentative implementatin of the linear version
    # > Needs F'F to be invertible and well-conditioned

    # # For the last field, we factor out a possible affine transformation
    # arg = args2[0]
    # shape = arg.shape
    # N = shape[0]                                     # Batch size
    # D = shape[-1]                                    # Dimension
    # V = torch.as_tensor(shape[1:-1]).prod()          # Nb of voxels
    # Id = identity(arg.shape[-2:0:-1], arg.dtype, arg.device).reshape(V, D)
    # arg = arg.reshape(N, V, D)                       # Field as a matrix
    # one = torch.ones((N, V, 1), dtype=arg.dtype, device=arg.device)
    # arg = cat((arg, one), 2)
    # Id  = cat((Id, one))
    # AA = arg.transpose(1, 2).bmm(arg)                # LHS of linear system
    # AI = arg.transpose(1, 2).bmm(arg)                # RHS of linear system
    # M, _ = torch.solve(AI, AA)                       # Solution
    # arg = arg.bmm(M) - Id                            # Closest displacement
    # arg = arg[..., :-1].reshape(shape)
    # arg = utils.last2channel(arg)
    # field = grid_pull(arg, field, interpolation, bound)     # Interpolate
    # field = field + channel2grid(grid_pull(arg, field, interpolation, bound))
    # shape = field.shape
    # V = torch.as_tensor(shape[1:-1]).prod()
    # field = field.reshape(N, V, D)
    # one = torch.ones((N, V, 1), dtype=field.dtype, device=field.device)
    # field, _ = torch.solve(field.transpose(1, 2), M.transpose(1, 2))
    # field = field.transpose(1, 2)[..., :-1].reshape(shape)

    return field


def jacobian(warp, bound='circular'):
    """Compute the jacobian of a 'vox' warp.

    This function estimates the field of Jacobian matrices of a deformation
    field using central finite differences: (next-previous)/2.

    Note that for Neumann boundary conditions, symmetric padding is usuallly
    used (symmetry w.r.t. voxel edge), when computing Jacobian fields,
    reflection padding is more adapted (symmetry w.r.t. voxel centre), so that
    derivatives are zero at the edges of the FOV.

    Note that voxel sizes are not considered here. The flow field should be
    expressed in voxels and so will the Jacobian.

    Args:
        warp (torch.Tensor): flow field (N, W, H, D, 3).
        bound (str, optional): Boundary conditions. Defaults to 'circular'.

    Returns:
        jac (torch.Tensor): Field of Jacobian matrices (N, W, H, D, 3, 3).
            jac[:,:,:,:,i,j] contains the derivative of the i-th component of
            the deformation field with respect to the j-th axis.

    """
    warp = torch.as_tensor(warp)
    shape = warp.size()
    dim = shape[-1]
    ker = kernels.imgrad(dim, device=warp.device, dtype=warp.dtype)
    ker = kernels.make_separable(ker, dim)
    warp = utils.last2channel(warp)
    if bound in ('circular', 'fft'):
        warp = utils.pad(warp, (1,)*dim, mode='circular', side='both')
        pad = 0
    elif bound in ('reflect1', 'dct1'):
        warp = utils.pad(warp, (1,)*dim, mode='reflect1', side='both')
        pad = 0
    elif bound in ('reflect2', 'dct2'):
        warp = utils.pad(warp, (1,)*dim, mode='reflect2', side='both')
        pad = 0
    elif bound in ('constant', 'zero', 'zeros'):
        pad = 1
    else:
        raise ValueError('Unknown bound {}.'.format(bound))
    if dim == 1:
        conv = _F.conv1d
    elif dim == 2:
        conv = _F.conv2d
    elif dim == 3:
        conv = _F.conv3d
    else:
        raise ValueError('Warps must be of dimension 1, 2 or 3. Got {}.'
                         .format(dim))
    jac = conv(warp, ker, padding=pad, groups=dim)
    jac = jac.reshape((shape[0], dim, dim) + shape[1:])
    jac = jac.permute((0,) + tuple(range(3, 3+dim)) + (1, 2))
    return jac


def resize(image, factor=None, shape=None, affine=None, anchor='c',
           *args, **kwargs):
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
    **kwargs : dict
        Parameters of `grid_pull`.

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
    nb_dim = image.dim() - 2
    inshape = image.shape[2:]
    info = {'dtype': image.dtype, 'device': image.device}
    factor = make_vector(factor, nb_dim).tolist()
    outshape = make_list(shape, nb_dim)
    anchor = [a[0].lower() for a in make_list(anchor, nb_dim)]
    return_trf = kwargs.pop('_return_trf', False)  # hidden option

    # compute output shape
    outshape = [int(inshp*f) if outshp is None else outshp
                for inshp, outshp, f in zip(inshape, outshape, factor)]

    # compute transformation grid
    # there is an affine relationship between the input and output grid:
    #    input_grid = scale * output_grid + shift
    lin = []
    scales = []
    shifts = []
    for anch, f, inshp, outshp in zip(anchor, factor, inshape, outshape):
        if anch == 'c':    # centers
            lin.append(torch.linspace(0, inshp - 1, outshp, **info))
            scales.append((inshp - 1) / (outshp - 1))
            shifts.append(0)
        elif anch == 'e':  # edges
            shift = (inshp * (1 / outshp - 1) + (inshp - 1)) / 2
            scale = inshp/outshp
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
    grid = torch.stack(torch.meshgrid(*lin), dim=-1)[None, ...]

    # resize input image
    resized = grid_pull(image, grid, *args, **kwargs)

    # compute orientation matrix
    if affine is not None:
        affine, _ = affine_resize(affine, inshape, factor, anchor)
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

    Returns
    -------
    resized : (batch, ..., ndim) tensor
        Resized grid.
    affine : (batch, ndim[+1], ndim+1) tensor, optional
        Orientation matrix

    """
    # resize grid
    kwargs['_return_trf'] = True
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

    # return
    if affine is not None:
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

    Returns
    -------
    resliced : (*batch, *channels, *shape_to)
        Resliced image.

    """
    # prepare tensors
    image = torch.as_tensor(image)
    backend = dict(dtype=image.dtype, device=image.device)
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


def grid_inv(grid, type='grid', lam=0.1, bound='dft',
             extrapolate=True):
    """Invert a dense deformation (or displacement) grid
    
    Notes
    -----
    The deformation/displacement grid must be expressed in 
    voxels, and map from/to the same lattice.
    
    Let `f = id + d` be the transformation. The inverse 
    is obtained as `id - (f.T @ 1 + L) \ (f.T @ d)`
    where `L` is a regulariser, `f.T @ _` is the adjoint 
    operation ("push") of `f @ _` ("pull"). and `1` is an 
    image of ones. 
    
    The idea behind this is that `f.T @ _` is approximately
    the inverse transformation weighted by the determinant
    of the Jacobian of the tranformation so, in the (theoretical)
    continuous case, `inv(f) @ _ = f.T @ _ / f.T @ 1`.
    However, in the (real) discrete case, this leads to 
    lots of holes in the inverse. The solution we use 
    therefore fills these holes using a regularised 
    least-squares scheme, where the regulariser penalizes
    the spatial gradients of the inverse field.
    
    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    lam : float, default=0.1
        Regularisation
    bound : str, default='dft'
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate the transformation field when
        it is sampled out-of-bounds.
        
    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid
    
    """
    # get shape components
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    shape = grid.shape[-(dim+1):-1]
    batch = grid.shape[:-(dim+1)]
    grid = grid.reshape([-1, *shape, dim])
    backend = dict(dtype=grid.dtype, device=grid.device)
    
    # get displacement
    identity = identity_grid(shape, **backend)
    if type == 'grid':
        disp = grid - identity
    else:
        disp = grid
        grid = disp + identity
    
    # push displacement
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = utils.movedim(disp, -1, 1)
    disp = grid_push(disp, grid, **push_opt)
    count = grid_count(grid, **push_opt)
    disp = utils.movedim(disp, 1, -1)
    count = utils.movedim(count, 1, -1)
    
    # Fill missing values using regularised least squares
    disp = solve_grid_sym(count, disp, bound=bound,
                          membrane=lam)
    disp = disp.reshape([*batch, *shape, dim])
    
    if type == 'grid':
        return identity - disp
    else:
        return -disp
