"""`nitorch.core.jit_policy`: optional TorchScript / torch.compile (#157)."""
import warnings
import torch
import pytest
from nitorch.core import jit_policy as J


def _kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y + x) * y - x


def _fake_compile(calls, fail=False):
    def compile(fn, **opts):
        calls.append(opts)
        if fail:
            raise RuntimeError('no backend')

        def compiled(*a, **k):
            calls.append('run')
            return fn(*a, **k)
        return compiled
    return compile


@pytest.fixture
def torch2(monkeypatch):
    monkeypatch.setattr(J, '_TORCH2', True)
    monkeypatch.setattr(J, '_HAS_COMPILE', True)


@pytest.mark.parametrize('mode,numel,compiled', [
    ('auto', 10, False),          # small: eager
    ('auto', 2 ** 20, True),      # large: compiled
    ('1', 10, True),              # forced on
    ('0', 2 ** 20, False),        # forced off
])
def test_maybe_compile_dispatch(mode, numel, compiled, monkeypatch, torch2):
    calls = []
    monkeypatch.setattr(torch, 'compile', _fake_compile(calls), raising=False)
    monkeypatch.setenv('NI_COMPILE', mode)
    f = J.maybe_compile(_kernel)
    x = torch.rand(numel)
    assert torch.equal(f(x, x), _kernel(x, x))
    assert ('run' in calls) == compiled
    if compiled:                   # compiled once, lazily
        f(x, x)
        assert sum(isinstance(c, dict) for c in calls) == 1


def test_maybe_compile_falls_back_once(monkeypatch, torch2):
    calls = []
    monkeypatch.setattr(torch, 'compile', _fake_compile(calls, fail=True), raising=False)
    monkeypatch.setenv('NI_COMPILE', '1')
    f = J.maybe_compile(_kernel)
    x = torch.rand(8)
    with pytest.warns(RuntimeWarning, match='Falling back'):
        assert torch.equal(f(x, x), _kernel(x, x))
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # no second warning
        assert torch.equal(f(x, x), _kernel(x, x))
    assert len(calls) == 1                      # not retried


def test_maybe_compile_runtime_failure_falls_back(monkeypatch, torch2):
    def compile(fn, **opts):
        def compiled(*a, **k):
            raise RuntimeError('inductor: no C++ compiler')
        return compiled
    monkeypatch.setattr(torch, 'compile', compile, raising=False)
    monkeypatch.setenv('NI_COMPILE', '1')
    f = J.maybe_compile(_kernel)
    x = torch.rand(8)
    with pytest.warns(RuntimeWarning, match='Falling back'):
        assert torch.equal(f(x, x), _kernel(x, x))
    assert f._jit_state['failed']


def test_maybe_compile_eager_errors_propagate(monkeypatch, torch2):
    # a genuine error (eager fails too) is not swallowed by the fallback
    monkeypatch.setattr(torch, 'compile', _fake_compile([]), raising=False)
    monkeypatch.setenv('NI_COMPILE', '1')
    f = J.maybe_compile(_kernel)
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # no fallback warning
        with pytest.raises(RuntimeError):
            f(torch.rand(3), torch.rand(4))
    assert not f._jit_state['failed']           # still compiled


def test_maybe_compile_unavailable(monkeypatch):
    monkeypatch.setattr(J, '_HAS_COMPILE', False)
    monkeypatch.setenv('NI_COMPILE', '1')
    f = J.maybe_compile(_kernel)
    x = torch.rand(8)
    assert torch.equal(f(x, x), _kernel(x, x))
    assert f._jit_state['impl'] is None


@pytest.mark.parametrize('mode,torch2,scripted', [
    ('auto', False, True), ('auto', True, False),
    ('1', True, True), ('0', False, False),
])
def test_maybe_script_dispatch(mode, torch2, scripted, monkeypatch):
    monkeypatch.setattr(J, '_TORCH2', torch2)
    monkeypatch.setenv('NI_TORCHSCRIPT', mode)
    f = J.maybe_script(_kernel)
    x = torch.rand(8)
    assert torch.allclose(f(x, x), _kernel(x, x))
    impl = f._jit_state['impl']
    assert (impl is not _kernel) == scripted


def test_maybe_script_unscriptable_falls_back(monkeypatch):
    monkeypatch.setenv('NI_TORCHSCRIPT', '1')

    def unscriptable(x):
        return x * len({1: 2, 3: object()})     # not TorchScript

    f = J.maybe_script(unscriptable)
    with pytest.warns(RuntimeWarning, match='Falling back'):
        assert torch.equal(f(torch.ones(2)), 2 * torch.ones(2))
    assert f._jit_state['impl'] is unscriptable


def test_maybe_jit_picks_by_version(monkeypatch):
    monkeypatch.setattr(J, '_TORCH2', True)
    assert 'failed' in J.maybe_jit(_kernel)._jit_state       # compile
    monkeypatch.setattr(J, '_TORCH2', False)
    assert 'failed' not in J.maybe_jit(_kernel)._jit_state   # script


def test_cpu_fusion_probe_restores_setting():
    before = torch._C._jit_can_fuse_on_cpu()
    J._probe_cpu_fusion()
    assert torch._C._jit_can_fuse_on_cpu() == before


def test_cpu_fusion_is_opt_in(monkeypatch):
    monkeypatch.setattr(J, '_CPU_FUSION', None)
    monkeypatch.delenv('NI_TORCHSCRIPT_CPU_FUSION', raising=False)
    J._maybe_enable_cpu_fusion()
    assert J._CPU_FUSION is None                  # not even probed


@pytest.mark.slow
def test_maybe_compile_real(monkeypatch):
    if not J._HAS_COMPILE:
        pytest.skip('torch.compile not available')
    monkeypatch.setenv('NI_COMPILE', '1')
    f = J.maybe_compile(_kernel)
    x, y = torch.rand(1000), torch.rand(1000)
    assert torch.allclose(f(x, y), _kernel(x, y))
