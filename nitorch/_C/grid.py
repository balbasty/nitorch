"""AutoGrad version of pull/push/count/grad"""

import torch

try:
    from nitorch._C.spatial import (
        grid_pull, grid_pull_backward,
        grid_push, grid_push_backward,
        grid_count, grid_count_backward,
        grid_grad, grid_grad_backward,
        InterpolationType, BoundType)
    COMPILED_BACKEND = 'nitorch'
except ImportError:
    try:
        from monai._C import (
        grid_pull, grid_pull_backward,
        grid_push, grid_push_backward,
        grid_count, grid_count_backward,
        grid_grad, grid_grad_backward,
        InterpolationType, BoundType)
        COMPILED_BACKEND = 'monai'
    except ImportError:
        grid_pull = grid_pull_backward = grid_push = grid_push_backward = None
        grid_count = grid_count_backward = grid_grad = grid_grad_backward = None
        InterpolationType = BoundType = None
        COMPILED_BACKEND = None


def make_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


def bound_to_nitorch(bound, as_enum=False):
    """Convert boundary type to NITorch's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention
    as_enum : bool, default=False
        Return BoundType rather than str

    Returns
    -------
    bound : [list of] str or BoundType
        Boundary condition in NITorch's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower() if isinstance(b, str) else b
        if b in ('replicate', 'repeat', 'border', 'nearest', BoundType.replicate):
            obound.append('replicate')
        elif b in ('zero', 'zeros', 'constant', BoundType.zero):
            obound.append('zero')
        elif b in ('dct2', 'reflect', 'reflection', 'neumann', BoundType.dct2):
            obound.append('dct2')
        elif b in ('dct1', 'mirror', BoundType.dct1):
            obound.append('dct1')
        elif b in ('dft', 'wrap', 'circular', BoundType.dft):
            obound.append('dft')
        elif b in ('dst2', 'antireflect', 'dirichlet', BoundType.dst2):
            obound.append('dst2')
        elif b in ('dst1', 'antimirror', BoundType.dst1):
            obound.append('dst1')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if as_enum:
        obound = list(map(lambda b: getattr(BoundType, b), obound))
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def inter_to_nitorch(inter, as_enum=False):
    """Convert interpolation order to NITorch's convention.

    Parameters
    ----------
    inter : [sequence of] int or str or InterpolationType
    as_enum : bool, default=False

    Returns
    -------
    inter : [sequence of] int or InterpolationType

    """
    intype = type(inter)
    if not isinstance(inter, (list, tuple)):
        inter = [inter]
    ointer = []
    for o in inter:
        o = o.lower() if isinstance(o, str) else o
        if o in (0, 'nearest', InterpolationType.nearest):
            ointer.append(0)
        elif o in (1, 'linear', InterpolationType.linear):
            ointer.append(1)
        elif o in (2, 'quadratic', InterpolationType.quadratic):
            ointer.append(2)
        elif o in (3, 'cubic', InterpolationType.cubic):
            ointer.append(3)
        elif o in (4, 'fourth', InterpolationType.fourth):
            ointer.append(4)
        elif o in (5, 'fifth', InterpolationType.fifth):
            ointer.append(5)
        elif o in (6, 'sixth', InterpolationType.sixth):
            ointer.append(6)
        elif o in (7, 'seventh', InterpolationType.seventh):
            ointer.append(7)
        else:
            raise ValueError(f'Unknown interpolation order {o}')
    if as_enum:
        ointer = list(map(InterpolationType, ointer))
    if issubclass(intype, (list, tuple)):
        ointer = intype(ointer)
    else:
        ointer = ointer[0]
    return ointer


class GridPull(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_enum=True)
        interpolation = inter_to_nitorch(make_list(interpolation), as_enum=True)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_pull(input, grid, *opt)

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
        grads = grid_pull_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            grad_input = grads[0]
            if ctx.needs_input_grad[1]:
                grad_grid = grads[1]
        elif ctx.needs_input_grad[1]:
            grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


class GridPush(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_enum=True)
        interpolation = inter_to_nitorch(make_list(interpolation), as_enum=True)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_push(input, grid, shape, *opt)

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
        grads = grid_push_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            grad_input = grads[0]
            if ctx.needs_input_grad[1]:
                grad_grid = grads[1]
        elif ctx.needs_input_grad[1]:
            grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None, None


class GridCount(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_enum=True)
        interpolation = inter_to_nitorch(make_list(interpolation), as_enum=True)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_count(grid, shape, *opt)

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
            grad_grid = grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


class GridGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_enum=True)
        interpolation = inter_to_nitorch(make_list(interpolation), as_enum=True)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_grad(input, grid, *opt)

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
            grads = grid_grad_backward(grad, *var, *opt)
            if ctx.needs_input_grad[0]:
                grad_input = grads[0]
                if ctx.needs_input_grad[1]:
                    grad_grid = grads[1]
            elif ctx.needs_input_grad[1]:
                grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None
