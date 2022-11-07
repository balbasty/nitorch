"""AutoGrad version of pull/push/count/grad"""
import torch
import os
from interpol.coeff import spline_coeff_nd, spline_coeff
from interpol.pushpull import grid_push as ts_grid_push
from interpol.bounds import BoundType as _TSBoundType
from interpol.splines import InterpolationType as _TSInterpolationType

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ImportError:
    custom_fwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)
    custom_bwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)

_compiled_backend = os.environ.get('NI_COMPILED_BACKEND', None)
COMPILED_BACKEND = None

if _compiled_backend == 'C':
    try:
        from .spatial import (
            grid_pull, grid_pull_backward,
            grid_push, grid_push_backward,
            grid_count, grid_count_backward,
            grid_grad, grid_grad_backward,
            InterpolationType, BoundType)
        COMPILED_BACKEND = 'C'
        # grid_push = ts_grid_push
    except ImportError:
        pass
elif _compiled_backend == 'TS':
    try:
        from ._ts import (
            grid_pull, grid_pull_backward,
            grid_push, grid_push_backward,
            grid_count, grid_count_backward,
            grid_grad, grid_grad_backward,
            InterpolationType, BoundType)
        COMPILED_BACKEND = 'TS'
    except ImportError:
        pass
elif _compiled_backend == 'MONAI':
    try:
        from monai._C import (
            grid_pull, grid_pull_backward,
            grid_push, grid_push_backward,
            grid_count, grid_count_backward,
            grid_grad, grid_grad_backward,
            InterpolationType, BoundType)
        COMPILED_BACKEND = 'MONAI'
    except ImportError:
        pass

if not COMPILED_BACKEND:
    try:
        from .spatial import (
            grid_pull, grid_pull_backward,
            grid_push, grid_push_backward,
            grid_count, grid_count_backward,
            grid_grad, grid_grad_backward,
            InterpolationType, BoundType)
        COMPILED_BACKEND = 'C'
        # grid_push = ts_grid_push
    except ImportError:
        try:
            from monai._C import (
            grid_pull, grid_pull_backward,
            grid_push, grid_push_backward,
            grid_count, grid_count_backward,
            grid_grad, grid_grad_backward,
            InterpolationType, BoundType)
            COMPILED_BACKEND = 'MONAI'
        except ImportError:
            try:
                from ._ts import (
                    grid_pull, grid_pull_backward,
                    grid_push, grid_push_backward,
                    grid_count, grid_count_backward,
                    grid_grad, grid_grad_backward,
                    InterpolationType, BoundType)
                COMPILED_BACKEND = 'TS'
            except ImportError:
                grid_pull = grid_pull_backward = grid_push = grid_push_backward = None
                grid_count = grid_count_backward = grid_grad = grid_grad_backward = None
                InterpolationType = BoundType = None
                COMPILED_BACKEND = None


if bool(int(os.environ.get('NI_SHOW_COMPILED_BACKEND', '0') or '0')):
    print('COMPILED_BACKEND:', COMPILED_BACKEND)


def make_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


def bound_to_nitorch(bound, as_type='str'):
    """Convert boundary type to niTorch's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention
    as_type : {'str', 'enum', 'int'}, default='str'
        Return BoundType or int rather than str

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
    if as_type in ('enum', 'int', int):
        obound = list(map(lambda b: getattr(BoundType, b), obound))
        if as_type in ('int', int):
            obound = [getattr(_TSBoundType, b.name).value for b in obound]
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def inter_to_nitorch(inter, as_type='str'):
    """Convert interpolation order to NITorch's convention.

    Parameters
    ----------
    inter : [sequence of] int or str or InterpolationType
    as_type : {'str', 'enum', 'int'}, default='int'

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
    if as_type in ('enum', 'str', str):
        ointer = list(map(InterpolationType, ointer))
        if as_type in ('str', str):
            ointer = [o.name for o in ointer]
    if issubclass(intype, (list, tuple)):
        ointer = intype(ointer)
    else:
        ointer = ointer[0]
    return ointer


enum_type = 'int' if COMPILED_BACKEND == 'TS' else 'enum'


class GridPull(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_type=enum_type)
        interpolation = inter_to_nitorch(make_list(interpolation), as_type=enum_type)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_pull(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        grads = grid_pull_backward(grad, *var, *opt)
        if COMPILED_BACKEND == 'TS':
            grad_input, grad_grid = grads
        else:
            if ctx.needs_input_grad[0]:
                grad_input = grads[0]
                if ctx.needs_input_grad[1]:
                    grad_grid = grads[1]
            elif ctx.needs_input_grad[1]:
                grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


class GridPush(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_type=enum_type)  # enum_type
        interpolation = inter_to_nitorch(make_list(interpolation), as_type=enum_type)  # enum_type
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)
        # opt = (interpolation, bound, extrapolate)

        # Push
        output = grid_push(input, grid, shape, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        grads = grid_push_backward(grad, *var, *opt)
        if COMPILED_BACKEND == 'TS':
            grad_input, grad_grid = grads
        else:
            if ctx.needs_input_grad[0]:
                grad_input = grads[0]
                if ctx.needs_input_grad[1]:
                    grad_grid = grads[1]
            elif ctx.needs_input_grad[1]:
                grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None, None


class GridCount(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_type=enum_type)
        interpolation = inter_to_nitorch(make_list(interpolation), as_type=enum_type)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_count(grid, shape, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_grid = None
        if ctx.needs_input_grad[0]:
            grad_grid = grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


class GridGrad(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_to_nitorch(make_list(bound), as_type=enum_type)
        interpolation = inter_to_nitorch(make_list(interpolation), as_type=enum_type)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_grad(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = grid_grad_backward(grad, *var, *opt)
            if COMPILED_BACKEND == 'TS':
                grad_input, grad_grid = grads
            else:
                if ctx.needs_input_grad[0]:
                    grad_input = grads[0]
                    if ctx.needs_input_grad[1]:
                        grad_grid = grads[1]
                elif ctx.needs_input_grad[1]:
                    grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


class SplineCoeff(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, bound, interpolation, dim, inplace):

        bound = bound_to_nitorch(bound, as_type='int')
        interpolation = inter_to_nitorch(interpolation, as_type='int')
        opt = (bound, interpolation, dim, inplace)

        # Pull
        output = spline_coeff(input, *opt)

        # Context
        if input.requires_grad:
            ctx.opt = opt

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # symmetric filter -> backward == forward
        # (I don't know if I can write into grad, so inplace=False to be safe)
        grad = spline_coeff(grad, *ctx.opt[:-1], inplace=False)
        return [grad] + [None] * 4


class SplineCoeffND(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, bound, interpolation, dim, inplace):

        bound = bound_to_nitorch(make_list(bound), as_type='int')
        interpolation = inter_to_nitorch(make_list(interpolation), as_type='int')
        opt = (bound, interpolation, dim, inplace)

        # Pull
        output = spline_coeff_nd(input, *opt)

        # Context
        if input.requires_grad:
            ctx.opt = opt

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # symmetric filter -> backward == forward
        # (I don't know if I can write into grad, so inplace=False to be safe)
        grad = spline_coeff_nd(grad, *ctx.opt[:-1], inplace=False)
        return grad, None, None, None, None
