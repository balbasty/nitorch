"""Optional TorchScript / `torch.compile` for leaf kernels (#157).

Elementwise chains over large tensors (per-voxel 3x3 algebra, ...) are
memory bound in eager mode; fusing them helps a lot (e.g. 13x for the
per-voxel J H J^T product of `exp_backward` with `torch.compile` on CPU).
Which fuser is available depends on the version of torch:

- torch < 2: TorchScript. It fuses on CUDA; on CPU its fuser is off by
  default (torch >= 1.10), so it mostly removes Python overhead.
- torch >= 2: `torch.compile` (TorchScript is deprecated). It needs a
  working backend (a C++ compiler on CPU, triton on CUDA), compiles
  lazily (5-30 s per kernel and shape) and must not be nested around
  opaque ops (e.g. nitorch's C++ extension), so it is reserved for leaf
  kernels.

Decorators
----------
maybe_script(fn)
    `torch.jit.script(fn)` on torch < 2, else `fn` itself.
maybe_compile(fn)
    `torch.compile(fn)` on torch >= 2, else `fn` itself.
maybe_jit(fn)
    `maybe_compile` on torch >= 2, `maybe_script` otherwise.

Rules
-----
- Decorate *leaf* kernels only: a decorated kernel must not call another
  decorated kernel (call the undecorated implementation instead), nor
  code that calls the C++ extension.
- Decorated kernels are called from Python (not from TorchScript code).
- Compilation is lazy (first qualifying call). If scripting/compiling or
  running the fused kernel fails where the eager function succeeds, a
  warning is emitted once and the eager function is used from then on.
  Errors that the eager function raises too propagate unchanged.

Environment variables
---------------------
NI_TORCHSCRIPT : {'auto', '0', '1'}, default 'auto'
    '1': always script (when possible); '0': never;
    'auto': script on torch < 2.
NI_COMPILE : {'auto', '0', '1'}, default 'auto'
    '1': always compile (when possible); '0': never;
    'auto': compile on torch >= 2, for calls whose largest tensor
    argument has at least NI_COMPILE_MIN_NUMEL elements.
NI_COMPILE_MIN_NUMEL : int, default 2**20
NI_TORCHSCRIPT_CPU_FUSION : {'0', '1'}, default '0'
    '1': let TorchScript fuse on CPU (process-wide setting: it also
    affects other TorchScript code), if a probe shows that it works.
    TorchScript's CPU fuser (NNC) generates code with LLVM, which some
    builds do not include (e.g. the torch 2.15 pip wheels:
    "LLVM Backend not found"). Off by default.
"""
import functools
import os
import warnings
import torch
from .version import torch_version

__all__ = ['maybe_script', 'maybe_compile', 'maybe_jit']

_TORCH2 = torch_version('>=', (2, 0))
_HAS_COMPILE = _TORCH2 and callable(getattr(torch, 'compile', None))


def _mode(name):
    mode = os.environ.get(name, 'auto').strip().lower()
    if mode in ('1', 'true', 'yes', 'on'):
        return 'on'
    if mode in ('0', 'false', 'no', 'off'):
        return 'off'
    return 'auto'


def _min_numel():
    try:
        return int(os.environ.get('NI_COMPILE_MIN_NUMEL', 2 ** 20))
    except ValueError:
        return 2 ** 20


def _largest_numel(args, kwargs):
    numel = 0
    for x in list(args) + list(kwargs.values()):
        if torch.is_tensor(x):
            numel = max(numel, x.numel())
    return numel


def _warn_fallback(fn, what, err):
    warnings.warn(
        f'nitorch: could not {what} `{getattr(fn, "__qualname__", fn)}` '
        f'({type(err).__name__}: {str(err).splitlines()[0][:200]}). '
        f'Falling back to eager mode.', RuntimeWarning, stacklevel=3)


_CPU_FUSION = None   # result of the probe (None: not probed yet)


def _probe_cpu_fusion():
    """Whether TorchScript can fuse elementwise ops on CPU here.

    Functional probe rather than version or build checks: the private
    switches are looked up defensively, a tiny function is scripted and
    run with CPU fusion enabled, and fusion must show in its optimised
    graph. The previous settings are restored whatever happens.
    """
    get = getattr(torch._C, '_jit_can_fuse_on_cpu', None)
    override = getattr(torch._C, '_jit_override_can_fuse_on_cpu', None)
    last_graph = getattr(torch.jit, 'last_executed_optimized_graph', None)
    if not (get and override and last_graph):
        return False
    previous = get()
    try:
        override(True)

        def probe(x, y):
            return (x * y + x) * y - x

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scripted = torch.jit.script(probe)
            x, y = torch.rand(64), torch.rand(64)
            for _ in range(3):  # profiling runs, then optimised graph
                out = scripted(x, y)
            fused = 'TensorExprGroup' in str(last_graph())
        return bool(fused and torch.allclose(out, probe(x, y)))
    except Exception:
        return False
    finally:
        override(previous)


def _maybe_enable_cpu_fusion():
    """Opt-in (`NI_TORCHSCRIPT_CPU_FUSION=1`): enable TorchScript's CPU
    fuser if the probe succeeds."""
    global _CPU_FUSION
    if os.environ.get('NI_TORCHSCRIPT_CPU_FUSION', '0') != '1':
        return
    if _CPU_FUSION is None:
        _CPU_FUSION = _probe_cpu_fusion()
        if _CPU_FUSION:
            torch._C._jit_override_can_fuse_on_cpu(True)
        else:
            warnings.warn('nitorch: NI_TORCHSCRIPT_CPU_FUSION=1, but '
                          'TorchScript cannot fuse on CPU with this build '
                          'of torch: not enabled.', RuntimeWarning)


def maybe_script(fn):
    """`torch.jit.script(fn)` on torch < 2 (see module docstring), else
    `fn`. Scripting happens at the first call."""
    state = dict(impl=None)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        impl = state['impl']
        if impl is None:
            mode = _mode('NI_TORCHSCRIPT')
            impl = fn
            if mode == 'on' or (mode == 'auto' and not _TORCH2):
                _maybe_enable_cpu_fusion()
                try:
                    with warnings.catch_warnings():
                        # deprecated on recent torch
                        warnings.simplefilter('ignore', FutureWarning)
                        warnings.simplefilter('ignore', DeprecationWarning)
                        impl = torch.jit.script(fn)
                except Exception as e:
                    _warn_fallback(fn, 'script', e)
            state['impl'] = impl
        if impl is fn:
            return fn(*args, **kwargs)
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            # genuine errors (eager fails too) propagate, without fallback
            out = fn(*args, **kwargs)
            _warn_fallback(fn, 'run the scripted', e)
            state['impl'] = fn
            return out

    wrapper._jit_state = state
    return wrapper


def maybe_compile(fn=None, **compile_options):
    """`torch.compile(fn, **compile_options)` on torch >= 2 (see module
    docstring), else `fn`. Compilation happens at the first qualifying
    call. Can be used as `@maybe_compile` or `@maybe_compile(...)`."""
    if fn is None:
        return functools.partial(maybe_compile, **compile_options)
    state = dict(impl=None, failed=False)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mode = _mode('NI_COMPILE')
        if state['failed'] or mode == 'off' or not _HAS_COMPILE:
            return fn(*args, **kwargs)
        if mode == 'auto' and _largest_numel(args, kwargs) < _min_numel():
            return fn(*args, **kwargs)
        if state['impl'] is None:
            try:
                state['impl'] = torch.compile(fn, **compile_options)
            except Exception as e:
                _warn_fallback(fn, 'compile', e)
                state['failed'] = True
                return fn(*args, **kwargs)
        try:
            return state['impl'](*args, **kwargs)
        except Exception as e:
            # Compilation errors surface at the first call. Genuine errors
            # (eager fails too) propagate, without disabling compilation.
            out = fn(*args, **kwargs)
            _warn_fallback(fn, 'compile or run the compiled', e)
            state['failed'] = True
            return out

    wrapper._jit_state = state
    return wrapper


def maybe_jit(fn):
    """`maybe_compile(fn)` on torch >= 2, `maybe_script(fn)` otherwise."""
    return maybe_compile(fn) if _TORCH2 else maybe_script(fn)
