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
import torch.nn.functional as F
from nitorch import kernels, utils
from nitorch.C.cpu import spatial as Cspatial

_interpolation = {'bilinear': 0, 'nearest': 1}
_padding = {'zeros': 0, 'border': 1, 'reflection': 2}


class _Pull(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='zeros',
                align_corners=None):
        # Convert parameters
        mode = _interpolation[mode]
        padding_mode = _padding[padding_mode]
        if align_corners is None:
            align_corners = False
        opt = (mode, padding_mode, align_corners)

        # Pull
        if input.dim() == 4:
            if input.device == 'cpu':
                output = Cspatial.pull2d_cpu(input, grid, *opt)
            else:
                output = Cspatial.pull2d_cuda(input, grid, *opt)
        elif input.dim() == 5:
            if input.device == 'cpu':
                output = Cspatial.pull3d_cpu(input, grid, *opt)
            else:
                output = Cspatial.pull3d_cuda(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.info['requires_grad'] = [input.requires_grad, grid.requires_grad]
        var = ([grid, input] if input.requires_grad or grid.requires_grad
               else [])
        ctx.save_for_backward(*var)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_variables
        opt = ctx.opt

        if ctx.info['requires_grad'][0] or ctx.info['requires_grad'][1]:
            if grad.dim() == 4:
                if grad.device == 'cpu':
                    g_input, g_grid = Cspatial.push2d_cpu(grad, *var, *opt)
                else:
                    g_input, g_grid = Cspatial.push2d_cuda(grad, *var, *opt)
            elif grad.dim() == 5:
                if grad.device == 'cpu':
                    g_input, g_grid = Cspatial.push3d_cpu(grad, *var, *opt)
                else:
                    g_input, g_grid = Cspatial.push3d_cuda(grad, *var, *opt)
        else:
            g_input = None
            g_grid = None
        return g_input, g_grid, None, None, None


def pull(input, grid, mode='bilinear', padding_mode='zeros',
         align_corners=None):
    """Sample an image with respect to a deformation field.

    Args:
        input (torch.Tensor): Input image. (B, C, Di, Hi, Wi).
        grid (torch.Tensor): Deformation field. (B, Do, Ho, Wo, 2|3).
        mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        padding_mode (str, optional): 'zeros' or 'borders' or 'reflection'.
            Defaults to 'zeros'.
        align_corners (bool optional): Defaults to False.

    Returns:
        output (torch.Tensor): Deformed image (B, C, Do, Ho, Wo).

    """
    return _Pull.apply(input, grid, mode, padding_mode, align_corners)


class _Push(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, shape=None, mode='bilinear',
                padding_mode='zeros', align_corners=None):
        # Convert parameters
        mode = _interpolation[mode]
        padding_mode = _padding[padding_mode]
        if align_corners is None:
            align_corners = False
        opt = (mode, padding_mode, align_corners)
        if shape is None:
            shape = input.shape
        else:
            shape = input.shape[0:2] + shape[::-1]

        # Forward pass
        empty = torch.zeros(*shape, dtype=input.dtype, device=input.device)
        if input.dim() == 4:
            if input.device == 'cpu':
                output, _ = Cspatial.push2d_cpu(input, empty, grid, *opt)
            else:
                output, _ = Cspatial.push2d_cuda(input, empty, grid, *opt)
        elif input.dim() == 5:
            if input.device == 'cpu':
                output, _ = Cspatial.push3d_cpu(input, empty, grid, *opt)
            else:
                output, _ = Cspatial.push3d_cuda(input, empty, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.info = {}
        ctx.info['requires_grad'] = [input.requires_grad, grid.requires_grad]
        var = ([grid, input] if grid.requires_grad
               else [grid] if input.requires_grad else [])
        ctx.save_for_backward(*var)

        return output

    @staticmethod
    def backward(ctx, grad):
        opt = ctx.opt

        if ctx.info['requires_grad'][0]:
            # Gradient with respect to the pushed image.
            # That's just pull again, but applied to the output gradient.
            grid = ctx.saved_variables[0]
            if grad.dim() == 4:
                if grad.device == 'cpu':
                    g_input = Cspatial.pull2d_cpu(grad, grid, *opt)
                else:
                    g_input = Cspatial.pull2d_cuda(grad, grid, *opt)
            elif grad.dim() == 5:
                if grad.device == 'cpu':
                    g_input = Cspatial.pull3d_cpu(grad, grid, *opt)
                else:
                    g_input = Cspatial.pull3d_cuda(grad, grid, *opt)
        else:
            g_input = None

        if ctx.info['requires_grad'][1]:
            # The derivative of push with respect to the deformation field is
            # similar to the derivative of pull, except that the roles of grad
            # and input and inverted:
            #   Jac(pull(inp,grid)) x ograd = grad(inp)(grid) * ograd
            #   Jac(push(inp,grid)) x ograd = grad(ograd)(grid) * inp
            grid = ctx.saved_variables[0]
            input = ctx.saved_variables[1]
            if grad.dim() == 4:
                if grad.device == 'cpu':
                    _, g_grid = Cspatial.push2d_cpu(input, grad, grid, *opt)
                else:
                    _, g_grid = Cspatial.push2d_cuda(input, grad, grid, *opt)
            elif grad.dim() == 5:
                if grad.device == 'cpu':
                    _, g_grid = Cspatial.push3d_cpu(input, grad, grid, *opt)
                else:
                    _, g_grid = Cspatial.push3d_cuda(input, grad, grid, *opt)
        else:
            g_grid = None

        return g_input, g_grid, None, None, None, None


def push(input, grid, shape=None, mode='bilinear', padding_mode='zeros',
         align_corners=None):
    """Splat an image with respect to a deformation field (pull adjoint).

    Args:
        input (torch.Tensor): Input image (B, C, Di, Hi, Wi).
        grid (torch.Tensor): Deformation field (B, Di, Hi, Wi, 2|3).
        shape (tuple[int]): Ouput shape (Wo, Ho, Do).
        mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        padding_mode (str, optional): 'zeros' or 'borders' or 'reflection'.
            Defaults to 'zeros'.
        align_corners (bool optional): Defaults to False.

    Returns:
        output (torch.Tensor): Splatted image (B, C, Do, Ho, Wo).

    """
    return _Push.apply(input, grid, shape, mode, padding_mode, align_corners)


def vox2fov(shape, align_corners=True):
    r"""Nifti to Torch coordinates.

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
    return vox2fov(shape, align_corners).inverse()


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


def identity(shape):
    shape = torch.as_tensor(shape, dtype=torch.int64)
    dim = shape.numel()
    shape = torch.Size([1, 1] + shape.tolist())
    mat = make_compact(torch.eye(dim+1))
    grid = F.affine_grid(mat, shape, align_corners=True)
    return grid


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
        conv = F.conv1d
    elif dim == 2:
        conv = F.conv2d
    elif dim == 3:
        conv = F.conv3d
    else:
        raise ValueError('Warps must be of dimension 1, 2 or 3. Got {}.'
                         .format(dim))
    jac = conv(warp, ker, padding=pad, groups=dim)
    jac = jac.reshape((shape[0], dim, dim) + shape[1:])
    jac = jac.permute((0,) + tuple(range(3, 3+dim)) + (1, 2))
    return jac
