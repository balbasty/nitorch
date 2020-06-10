# -*- coding: utf-8 -*-
"""Tools related to spatial sampling (displacement, warps, etc.).

Spatial ordering conventions in nitorch
---------------------------------------

NiTorch uses consistent ordering conventions throughout its API:
. We use abbreviations (B[atch], C[hannel], D[ephth], H[eight], W[idth])
  to name dimensions.
. Tensors that represent series (resp. images or volumes) should always be
  ordered as (B, C, W) (resp. (B, C, H, W) or (B, C, D, H, W)).
. We use x, y, z to denote axis/coordinates along the (W, H, D)
  dimensions.
. Conversely, displacement or deformation fields are ordered as
  (B, H, D, W, C). There, the Channel dimension contains displacements or
  deformations along the x, y, z axes (i.e., W, H, D dimensions).
. Similarly, Jacobian fields are stored as (B, H, D, W, C, C). The second
  to last dimension corresponds to x, y, z components and the last
  dimension corresponds to derivatives along the x, y, z axes.
  I.e., jac[..., i, j] = d{u_i}/d{x_j}
. This means that we usually do not store deformations per *imaging*
  channel (they are assumed to lie in the same space).
. Arguments that relate to spatial dimensions are ordered as
  (x, y, z) or (W, H, D). This means: lattice dimensions, voxel sizes,
  affine transformation matrices, etc. It can be a little bit
  counterintuitive at first, as this order is opposite to the data
  storage order. However, it is consistant with the ordering of spatial
  components.

These conventions are mostly consistent with those used in PyTorch
(conv, grid_sampler, etc.). However, some care must be taken when, e.g.,
convolving deformation fields or allocating tensors based on grid
dimensions.

TODO:
    . What about time series?
    . Should we always have a batch dimension for affine matrices?
    . Should the default storage be compact or square for affine matrices?

"""

import torch
import torch.nn.functional as _F
from nitorch import kernels, utils
from nitorch._C import spatial as _Cspatial
from nitorch._C.spatial import BoundType, InterpolationType


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
            grad_grid = grads[1]
        return grad_input, grad_grid, None, None, None


def grid_pull(input, grid, interpolation='linear', bound='zero', extrapolate=True):
    """Sample an image with respect to a deformation field.

        `interpolation` can be an int, a string or an InterpolationType.
        Possible values are:
            - 0 or 'nearest'    or InterpolationType.nearest
            - 1 or 'linear'     or InterpolationType.linear
            - 2 or 'quadratic'  or InterpolationType.quadratic
            - 3 or 'cubic'      or InterpolationType.cubic
            - 4 or 'fourth'     or InterpolationType.fourth
            - etc.
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific interpoaltion orders.

        `bound` can be an int, a string or a BoundType.
        Possible values are:
            - 0 or 'replicate'  or BoundType.replicate
            - 1 or 'dct1'       or BoundType.dct1
            - 2 or 'dct2'       or BoundType.dct2
            - 3 or 'dst1'       or BoundType.dst1
            - 4 or 'dst2'       or BoundType.dst2
            - 4 or 'dft'        or BoundType.dft
            - 4 or 'sliding'    or BoundType.sliding
            - 4 or 'zero'       or BoundType.zero
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific boundary conditions.
        `sliding` is a specific condition than only applies to flow fields
        (with as many channels as dimensions). It cannot be dimension-specific.
        Note that
        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
        See https://en.wikipedia.org/wiki/Discrete_cosine_transform
            https://en.wikipedia.org/wiki/Discrete_sine_transform

    Args:
        input (torch.Tensor): Input image. (B, C, Di, Hi, Wi).
        grid (torch.Tensor): Deformation field. (B, Do, Ho, Wo, 2|3).
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to 1.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to 'zero'.
        extrapolate (bool, optional): Extrapolate out-of-bound data.
            Defaults to True.

    Returns:
        output (torch.Tensor): Deformed image (B, C, Do, Ho, Wo).

    """
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
            grad_grid = grads[1]
        return grad_input, grad_grid, None, None, None, None


def grid_push(input, grid, shape=None, interpolation='linear', bound='zero',
              extrapolate=True):
    """Splat an image with respect to a deformation field (pull adjoint).

        `interpolation` can be an int, a string or an InterpolationType.
        Possible values are:
            - 0 or 'nearest'    or InterpolationType.nearest
            - 1 or 'linear'     or InterpolationType.linear
            - 2 or 'quadratic'  or InterpolationType.quadratic
            - 3 or 'cubic'      or InterpolationType.cubic
            - 4 or 'fourth'     or InterpolationType.fourth
            - etc.
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific interpoaltion orders.

        `bound` can be an int, a string or a BoundType.
        Possible values are:
            - 0 or 'replicate'  or BoundType.replicate
            - 1 or 'dct1'       or BoundType.dct1
            - 2 or 'dct2'       or BoundType.dct2
            - 3 or 'dst1'       or BoundType.dst1
            - 4 or 'dst2'       or BoundType.dst2
            - 4 or 'dft'        or BoundType.dft
            - 4 or 'sliding'    or BoundType.sliding
            - 4 or 'zero'       or BoundType.zero
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific boundary conditions.
        `sliding` is a specific condition than only applies to flow fields
        (with as many channels as dimensions). It cannot be dimension-specific.
        Note that
        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
        See https://en.wikipedia.org/wiki/Discrete_cosine_transform
            https://en.wikipedia.org/wiki/Discrete_sine_transform

    Args:
        input (torch.Tensor): Input image (B, C, Di, Hi, Wi).
        grid (torch.Tensor): Deformation field (B, Di, Hi, Wi, 2|3).
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to 1.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to 'zero'.
        extrapolate (bool, optional): Extrapolate out-of-bound data.
            Defaults to True.

    Returns:
        output (torch.Tensor): Splatted image (B, C, Do, Ho, Wo).

    """
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
        shape = tuple(input.shape[-1:-(input.dim()-1):-1])

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

        This function is equivalent to applying grid_push to an image of ones.

        `interpolation` can be an int, a string or an InterpolationType.
        Possible values are:
            - 0 or 'nearest'    or InterpolationType.nearest
            - 1 or 'linear'     or InterpolationType.linear
            - 2 or 'quadratic'  or InterpolationType.quadratic
            - 3 or 'cubic'      or InterpolationType.cubic
            - 4 or 'fourth'     or InterpolationType.fourth
            - etc.
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific interpoaltion orders.

        `bound` can be an int, a string or a BoundType.
        Possible values are:
            - 0 or 'replicate'  or BoundType.replicate
            - 1 or 'dct1'       or BoundType.dct1
            - 2 or 'dct2'       or BoundType.dct2
            - 3 or 'dst1'       or BoundType.dst1
            - 4 or 'dst2'       or BoundType.dst2
            - 4 or 'dft'        or BoundType.dft
            - 4 or 'sliding'    or BoundType.sliding
            - 4 or 'zero'       or BoundType.zero
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific boundary conditions.
        `sliding` is a specific condition than only applies to flow fields
        (with as many channels as dimensions). It cannot be dimension-specific.
        Note that
        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
        See https://en.wikipedia.org/wiki/Discrete_cosine_transform
            https://en.wikipedia.org/wiki/Discrete_sine_transform

    Args:
        grid (torch.Tensor): Deformation field (B, Di, Hi, Wi, 2|3).
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to 1.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to 'zero'.
        extrapolate (bool, optional): Extrapolate out-of-bound data.
            Defaults to True.

    Returns:
        output (torch.Tensor): Splat weights (B, C, Do, Ho, Wo).

    """
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
        shape = tuple(grid.shape[-2:-(input.dim()-1):-1])

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
                grad_grid = grads[1]
        return grad_input, grad_grid, None, None, None


def grid_grad(input, grid, interpolation='linear', bound='zero', extrapolate=True):
    """Sample an image with respect to a deformation field.

        `interpolation` can be an int, a string or an InterpolationType.
        Possible values are:
            - 0 or 'nearest'    or InterpolationType.nearest
            - 1 or 'linear'     or InterpolationType.linear
            - 2 or 'quadratic'  or InterpolationType.quadratic
            - 3 or 'cubic'      or InterpolationType.cubic
            - 4 or 'fourth'     or InterpolationType.fourth
            - etc.
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific interpoaltion orders.

        `bound` can be an int, a string or a BoundType.
        Possible values are:
            - 0 or 'replicate'  or BoundType.replicate
            - 1 or 'dct1'       or BoundType.dct1
            - 2 or 'dct2'       or BoundType.dct2
            - 3 or 'dst1'       or BoundType.dst1
            - 4 or 'dst2'       or BoundType.dst2
            - 4 or 'dft'        or BoundType.dft
            - 4 or 'sliding'    or BoundType.sliding
            - 4 or 'zero'       or BoundType.zero
        A list of values can be provided, in the order [W, H, D],
        to specify dimension-specific boundary conditions.
        `sliding` is a specific condition than only applies to flow fields
        (with as many channels as dimensions). It cannot be dimension-specific.
        Note that
        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
        See https://en.wikipedia.org/wiki/Discrete_cosine_transform
            https://en.wikipedia.org/wiki/Discrete_sine_transform

    Args:
        input (torch.Tensor): Input image. (B, C, Di, Hi, Wi).
        grid (torch.Tensor): Deformation field. (B, Do, Ho, Wo, 2|3).
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to 1.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to 'zero'.
        extrapolate (bool, optional): Extrapolate out-of-bound data.
            Defaults to True.

    Returns:
        output (torch.Tensor): Sampled gradients (B, C, Do, Ho, Wo, 2|3).

    """
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

    return _GridGrad.apply(input, grid, interpolation, bound, extrapolate)


def vox2fov(shape, align_corners=True):
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


def fov2vox(shape, align_corners=True):
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


def make_square(mat):
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


def make_compact(mat):
    """Transform a square affine matrix into a compact affine matrix."""
    mat = torch.as_tensor(mat)
    shape = mat.size()
    if mat.dim() != 2 or not shape[0] in (shape[1], shape[1] - 1):
        raise ValueError('Input matrix should be Dx(D+1) or (D+1)x(D+1).')
    if shape[0] == shape[1]:
        mat = mat[:-1, :]
    return mat


def channel2grid(warp):
    """Warps: Channel to Grid dimension order.

    . Channel ordering is: Batch x Direction x Depth x Height x Width
    . Grid ordering is: Batch x Depth x Height x Width x Direction
    """
    warp = torch.as_tensor(warp)
    warp = warp.permute((0,) + tuple(range(2, warp.dim())) + (1,))
    return warp


def grid2channel(warp):
    """Warps: Grid to Channel dimension order.

    . Channel ordering is: Batch x Direction x Depth x Height x Width
    . Grid ordering is: Batch x Depth x Height x Width x Direction
    """
    warp = torch.as_tensor(warp)
    warp = warp.permute((0, - 1) + tuple(range(1, warp.dim()-1)))
    return warp


def ismatrix(x):
    """Check that a tensor is a matrix (ndim == 2)."""
    x = torch.as_tensor(x)
    shape = torch.as_tensor(x.shape)
    return shape.numel() == 2


def identity(dim, dtype=None, device=None):
    """Returns an identity deformation field.

    Args:
        dim (tuple): Spatial dimension of the field, ordered as (W, H, [D]).
        dtype (torch.dtype, optional): Data type. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.

    Returns:
        g (torch.Tensor): Deformation field with shape (1, 1, [D], H, W, 2|3).

    """
    mat = torch.tensor([[[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.]]],
                       dtype=dtype, device=device)
    f2v = fov2vox(dim, False).to(device)
    g = _F.affine_grid(mat, (1, 1) + dim[::-1], align_corners=False).to(device)
    g = g.matmul(f2v[:3, :3].transpose(0, 1)) \
        + f2v[:3, 3].reshape((1, 1, 1, 1, 3))
    return g


# def affine_field(mat, dim, dtype=None, device=None):
#     if dtype is None:
#         dtype = mat.dtype
#     if device is None:
#         device = mat.device
#     mat = mat.to(dtype=dtype, device=device)
#     g = identity(dim, dtype, device)

#     arg.matmul(
#                     last_affine[:dim, :dim].transpose(0, 1)) \
#                   + last_affine[:dim, dim].reshape((1, 1, 1, 1, dim))


def compose(*args, interpolation='linear', bound='dft'):
    """Compose multiple spatial deformations (affine matrices or flow fields).
    """
    # TODO:
    # . add shape/dim argument to generate (if needed) an identity field
    #   at the end of the chain.
    # . possibility to provide fields that have an orientation matrix?
    #   (or keep it the responsibility of the user?)
    # . For higher order (> 1) interpolation: convert to spline coeficients.

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
                last_affine = make_square(arg)
            else:
                last_affine = last_affine.matmul(make_square(arg))
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
                  + last_affine[:dim, dim].reshape((1, 1, 1, 1, dim))
                args2.append(new_field)
            else:
                args2.append(arg)
    if last_affine is not None:
        args2.append(last_affine)

    # Third pass: compose all flow fields
    field = args2[-1]
    for arg in args2[-2::-1]:  # args2[-2:0:-1]
        arg = arg - identity(arg.shape[-2:0:-1], arg.dtype, arg.device)
        arg = grid2channel(arg)
        field = field + channel2grid(grid_pull(arg, field, interpolation, bound))

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
    # arg = grid2channel(arg)
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
        warp (torch.Tensor): flow field (N, D, H, W, 3).
        bound (str, optional): Boundary conditions. Defaults to 'circular'.

    Returns:
        jac (torch.tensor): Field of Jacobian matrices (N, D, H, W, 3, 3).
            jac[:,:,:,:,i,j] contains the derivative of the i-th component of
            the deformation field with respect to the j-th axis.

    """
    warp = torch.as_tensor(warp)
    shape = warp.size()
    dim = shape[-1]
    ker = kernels.imgrad(dim, device=warp.device, dtype=warp.dtype)
    ker = kernels.make_separable(ker, dim)
    warp = grid2channel(warp)
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


def voxsize(mat):
    """ Compute voxel size from affine matrix.

    Args:
        mat (torch.tensor()): Affine matrix (4, 4).

    Returns:
        vx (torch.tensor()): Voxel size (3,).

    """
    return (mat[:3, :3] ** 2).sum(0).sqrt()
