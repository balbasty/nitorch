"""AutoGrad version of pull/push/count/grad"""
import torch
import os
from .coeff import spline_coeff_nd, spline_coeff
from interpol.pushpull import grid_push as ts_grid_push
from interpol.bounds import BoundType as _TSBoundType
from interpol.splines import InterpolationType as _TSInterpolationType

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
        elif b == 'sliding' or getattr(b, 'name', None) == 'sliding':
            # displacement fields: along axis d, component d is dst2 and
            # the other components dct2 (see `component_bounds`)
            obound.append('sliding')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if as_type in ('enum', 'int', int):
        obound = list(map(lambda b: getattr(BoundType, b), obound))
        if as_type in ('int', int):
            obound = [getattr(_TSBoundType, b.name).value
                      if b.name != 'sliding' else _SLIDING_INT
                      for b in obound]
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

# integer code of 'sliding' in the TorchScript backend
_SLIDING_INT = 7


def component_bounds(bound, dim, nb_comp):
    """Boundary conditions of each component (channel) of a displacement
    field, if any axis has a 'sliding' boundary condition (else None).

    Along a sliding axis `d`, the component `d` (normal to the border) is
    antisymmetric ('dst2') and the other components are symmetric
    ('dct2'). Channel `c` is the component along axis `c` (channels
    beyond the number of axes are symmetric).
    """
    names = bound_to_nitorch(make_list(bound), 'str')
    names = names + names[-1:] * max(0, dim - len(names))
    if 'sliding' not in names:
        return None
    return [[('dst2' if d == c else 'dct2') if b == 'sliding' else b
             for d, b in enumerate(names)] for c in range(nb_comp)]


def _even_bounds(bound):
    """Sliding -> dct2 (for quantities without components, e.g. counts)"""
    return ['dct2' if b == 'sliding' else b
            for b in bound_to_nitorch(make_list(bound), 'str')]


def _unpack(grads, needs):
    """(grad_input, grad_grid) from the output of a backward function"""
    if COMPILED_BACKEND == 'TS':
        grad_input, grad_grid = grads
        return (grad_input if needs[0] else None,
                grad_grid if needs[1] else None)
    grad_input = grad_grid = None
    if needs[0]:
        grad_input = grads[0]
        if needs[1]:
            grad_grid = grads[1]
    elif needs[1]:
        grad_grid = grads[0]
    return grad_input, grad_grid


def _forward_components(fn, input, grid, opts, *extra):
    """Apply `fn` to each channel, with its own options"""
    return torch.cat([fn(input[:, c:c+1], grid, *extra, *opt)
                      for c, opt in enumerate(opts)], 1)


def _backward_components(fn, grad, input, grid, opts, needs, *extra):
    """Backward of `_forward_components`: the gradients of the channels
    are concatenated, those of the grid summed"""
    grad_inputs, grad_grid = [], None
    with torch.no_grad():
        grid_ = grid.detach().requires_grad_(needs[1])
        for c, opt in enumerate(opts):
            input_ = input[:, c:c+1].detach().requires_grad_(needs[0])
            gi, gg = _unpack(fn(grad[:, c:c+1], input_, grid_, *extra, *opt),
                             needs)
            grad_inputs.append(gi)
            if gg is not None:
                grad_grid = gg if grad_grid is None else grad_grid + gg
    grad_input = torch.cat(grad_inputs, 1) if needs[0] else None
    return grad_input, grad_grid


def _options(bound, interpolation, extrapolate, dim, nb_comp):
    """Options of each channel (one set if they share them)"""
    interpolation = inter_to_nitorch(make_list(interpolation), as_type=enum_type)
    extrapolate = int(extrapolate)
    cbound = component_bounds(bound, dim, nb_comp)
    if cbound is None:
        bound = bound_to_nitorch(make_list(bound), as_type=enum_type)
        return [(bound, interpolation, extrapolate)], False
    return [(bound_to_nitorch(b, as_type=enum_type), interpolation, extrapolate)
            for b in cbound], True


class GridPull(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        opts, split = _options(bound, interpolation, extrapolate,
                               grid.shape[-1], input.shape[1])

        # Pull (each component with its own boundary conditions if sliding)
        if split:
            output = _forward_components(grid_pull, input, grid, opts)
        else:
            output = grid_pull(input, grid, *opts[0])

        # Context
        ctx.opt = opts[0]
        ctx.opts = opts if split else None
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        if ctx.opts:
            return (*_backward_components(grid_pull_backward, grad, *var,
                                          ctx.opts, ctx.needs_input_grad[:2]),
                    None, None, None)
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
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        opts, split = _options(bound, interpolation, extrapolate,
                               grid.shape[-1], input.shape[1])

        # Push (each component with its own boundary conditions if sliding)
        if split:
            output = _forward_components(grid_push, input, grid, opts, shape)
        else:
            output = grid_push(input, grid, shape, *opts[0])

        # Context
        ctx.opt = opts[0]
        ctx.opts = opts if split else None
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        if ctx.opts:
            return (*_backward_components(grid_push_backward, grad, *var,
                                          ctx.opts, ctx.needs_input_grad[:2]),
                    None, None, None, None)
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
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        # counts have no components: sliding is symmetric (dct2)
        bound = bound_to_nitorch(_even_bounds(bound), as_type=enum_type)
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
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_grid = None
        if ctx.needs_input_grad[0]:
            grad_grid = grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


class GridGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        opts, split = _options(bound, interpolation, extrapolate,
                               grid.shape[-1], input.shape[1])

        # Pull gradients (each component with its own boundary conditions
        # if sliding)
        if split:
            output = _forward_components(grid_grad, input, grid, opts)
        else:
            output = grid_grad(input, grid, *opts[0])

        # Context
        ctx.opt = opts[0]
        ctx.opts = opts if split else None
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        if ctx.opts:
            if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
                return None, None, None, None, None
            return (*_backward_components(grid_grad_backward, grad, *var,
                                          ctx.opts, ctx.needs_input_grad[:2]),
                    None, None, None)
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
    def forward(ctx, input, bound, interpolation, dim, inplace):

        if bound_to_nitorch(bound, 'str') == 'sliding':
            raise ValueError("'sliding' needs all spatial dimensions "
                             "(use spline_coeff_nd)")
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
    def backward(ctx, grad):
        # transposed filter (the filter is not symmetric for dct1 and
        # replicate)
        grad = spline_coeff(grad, *ctx.opt[:-1], inplace=False,
                            transpose=True)
        return grad, None, None, None, None


class SplineCoeffND(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bound, interpolation, dim, inplace):

        interpolation = inter_to_nitorch(make_list(interpolation), as_type='int')
        if input.dim() > dim:
            cbound = component_bounds(bound, dim, input.shape[-dim-1])
        else:
            # no components: sliding is symmetric (dct2)
            bound, cbound = _even_bounds(bound), None
        if cbound is None:
            bound = bound_to_nitorch(make_list(bound), as_type='int')
            opts = [(bound, interpolation, dim, inplace)]
        else:
            # sliding: each component (channel) with its own boundary
            # conditions
            opts = [(bound_to_nitorch(b, as_type='int'), interpolation, dim,
                     inplace) for b in cbound]

        output = _spline_coeff_components(input, opts, dim)

        # Context
        if input.requires_grad:
            ctx.opts = opts
            ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx, grad):
        # transposed filter (the filter is not symmetric for dct1 and
        # replicate)
        # (I don't know if I can write into grad, so inplace=False to be safe)
        opts = [(*opt[:-1], False) for opt in ctx.opts]
        grad = _spline_coeff_components(grad, opts, ctx.dim, transpose=True)
        return grad, None, None, None, None


def _spline_coeff_components(input, opts, dim, transpose=False):
    if len(opts) == 1:
        return spline_coeff_nd(input, *opts[0], transpose=transpose)
    ax = input.dim() - dim - 1
    return torch.cat([spline_coeff_nd(input.narrow(ax, c, 1), *opt,
                                      transpose=transpose)
                      for c, opt in enumerate(opts)], ax)
