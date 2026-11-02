"""Compatibility helpers for older PyTorch versions (torch >= 1.4).

Each helper is checked against the native torch function (when the
current torch has it), and its fallback branch is exercised by
monkeypatching the feature-detection flag, so that the code paths used
by old torch versions are tested on recent ones as well.

Fallbacks that rely on functions that were removed from recent torch
versions (`torch.symeig`, `torch.eig`) use the real function when it
exists, else a stand-in built on `torch.linalg` that only checks that
the arguments are translated correctly.
"""
import pytest
import torch

from nitorch.core import math as nimath, linalg as nilinalg, utils, jit


def _works(fn, *args, **kwargs):
    """Check that a function exists and is not a "removed" stub"""
    if fn is None:
        return False
    try:
        fn(*args, **kwargs)
        return True
    except RuntimeError:
        return False


def _has(mod, *names):
    for name in names:
        if not hasattr(mod, name):
            return False
        mod = getattr(mod, name)
    return True


@pytest.fixture
def fallback(monkeypatch):
    """Force the "old torch" branch by unsetting feature flags."""
    def force(module, *flags):
        for flag in flags:
            assert hasattr(module, flag), flag
            monkeypatch.setattr(module, flag, False)
    return force


def _assert_equal(x, y, **kwargs):
    assert x.shape == y.shape, (x.shape, y.shape)
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    if kwargs:
        torch.testing.assert_close(x, y, **kwargs)
    else:
        assert torch.equal(x, y)


# ----------------------------------------------------------------------
#                       maximum / minimum
# ----------------------------------------------------------------------

@pytest.mark.parametrize('old', [False, True])
@pytest.mark.parametrize('fn', ['maximum', 'minimum'])
def test_maximum_minimum(fn, old, fallback):
    if old:
        fallback(nimath, '_torch_has_maximum')
    g = torch.Generator().manual_seed(0)
    a = torch.randn([3, 1, 5], generator=g)
    b = torch.randn([4, 1], generator=g)
    ref = (torch.max if fn == 'maximum' else torch.min)(a, b)
    if _has(torch, fn):
        ref = getattr(torch, fn)(a, b)
    out = getattr(nimath, fn)(a, b)
    _assert_equal(out, ref)
    assert out.shape == (3, 4, 5)


# ----------------------------------------------------------------------
#                           amax / amin
# ----------------------------------------------------------------------

def _amax_ref(fn, x, dim, keepdim):
    # reference that does not use amax: reduce one dim at a time
    reduce = torch.max if fn == 'amax' else torch.min
    dims = list(range(x.dim())) if dim in ((), None) else dim
    dims = [dims] if isinstance(dims, int) else dims
    dims = sorted([d % x.dim() for d in dims], reverse=True)
    for d in dims:
        x = reduce(x, d, keepdim=True).values
    if not keepdim:
        for d in dims:
            x = x.squeeze(d)
    return x


@pytest.mark.parametrize('old', [False, True])
@pytest.mark.parametrize('fn', ['amax', 'amin'])
@pytest.mark.parametrize('dim', [(), 1, -1, [0, 2], (-1, -2)])
@pytest.mark.parametrize('keepdim', [False, True])
def test_amax_amin(fn, dim, keepdim, old, fallback):
    if old:
        fallback(nimath, '_torch_has_amax')
    g = torch.Generator().manual_seed(1)
    x = torch.randn([3, 4, 5], generator=g)
    out = getattr(nimath, fn)(x, dim=dim, keepdim=keepdim)
    _assert_equal(out, _amax_ref(fn, x, dim, keepdim))
    if _has(torch, fn):
        _assert_equal(out, getattr(torch, fn)(x, dim, keepdim))


@pytest.mark.parametrize('old', [False, True])
def test_amax_scalar(old, fallback):
    if old:
        fallback(nimath, '_torch_has_amax')
    x = torch.tensor(3.)
    _assert_equal(nimath.amax(x), x)
    _assert_equal(nimath.amin(x, keepdim=True), x)


# ----------------------------------------------------------------------
#                           binary_logit
# ----------------------------------------------------------------------

@pytest.mark.parametrize('old', [False, True])
@pytest.mark.parametrize('eps', [None, 1e-3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_binary_logit(eps, dtype, old, fallback):
    if old:
        fallback(nimath, '_torch_has_logit')
    x = torch.linspace(0, 1, 101, dtype=dtype)
    out = nimath.binary_logit(x, eps)
    xc = x if eps is None else x.clamp(eps, 1 - eps)
    ref = torch.log(xc / (1 - xc))
    _assert_equal(out, ref, rtol=1e-6, atol=1e-6)
    if _has(torch, 'logit'):
        _assert_equal(out, torch.logit(x, eps), rtol=1e-6, atol=1e-6)
    # inverse of the sigmoid
    inner = slice(1, -1)
    _assert_equal(torch.sigmoid(out[inner]), x[inner],
                  rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------
#                            nan_to_num
# ----------------------------------------------------------------------

def _special_values(dtype=torch.float32):
    return torch.tensor([0., float('nan'), float('inf'), -float('inf'),
                         -1.5, 2.], dtype=dtype)


@pytest.mark.parametrize('old', [False, True])
@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(nan=1.),
    dict(nan=0., posinf=0., neginf=0.),
    dict(nan=-2., posinf=10., neginf=-10.),
])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nan_to_num(kwargs, dtype, old, fallback):
    if old:
        fallback(nimath, '_torch_has_nan_to_num')
    x = _special_values(dtype)
    info = torch.finfo(dtype)
    ref = torch.tensor([0., kwargs.get('nan', 0.),
                        kwargs.get('posinf', info.max),
                        kwargs.get('neginf', info.min),
                        -1.5, 2.], dtype=dtype)

    out = nimath.nan_to_num(x, **kwargs)
    _assert_equal(out, ref)
    assert torch.isnan(x).any()  # input left untouched
    if _has(torch, 'nan_to_num'):
        _assert_equal(out, torch.nan_to_num(x, **kwargs))

    y = x.clone()
    out = nimath.nan_to_num_(y, **kwargs)
    _assert_equal(out, ref)
    _assert_equal(y, ref)  # in-place


@pytest.mark.parametrize('old', [False, True])
def test_nan_to_num_int(old, fallback):
    if old:
        fallback(nimath, '_torch_has_nan_to_num')
    x = torch.arange(5)
    _assert_equal(nimath.nan_to_num(x), x)
    _assert_equal(nimath.nan_to_num_(x.clone()), x)


# ----------------------------------------------------------------------
#                   broadcast_shapes / floor_div / movedim
# ----------------------------------------------------------------------

@pytest.mark.parametrize('shapes', [
    [(3, 1, 5), (4, 1)],
    [(1,), (2, 3)],
    [(), (2, 3)],
    [(2, 1, 1), (1, 3, 1), (1, 1, 4)],
    [(0, 1), (1, 5)],
])
def test_expanded_shape(shapes):
    """`utils.expanded_shape` replaces `torch.broadcast_shapes`"""
    out = utils.expanded_shape(*shapes)
    ref = torch.broadcast_tensors(*[torch.empty(s) for s in shapes])[0].shape
    assert tuple(out) == tuple(ref)
    if _has(torch, 'broadcast_shapes'):
        assert tuple(out) == tuple(torch.broadcast_shapes(*shapes))
        assert torch.Size(out) == torch.broadcast_shapes(*shapes)


def test_expanded_shape_error():
    with pytest.raises(ValueError):
        utils.expanded_shape((2, 3), (4, 3))


def test_floor_div():
    """`math.floor_div` replaces `torch.div(..., rounding_mode='floor')`"""
    x = torch.tensor([-7., -6., -1., 0., 1., 6., 7.])
    ref = torch.tensor([-4., -3., -1., 0., 0., 3., 3.])
    _assert_equal(nimath.floor_div(x, 2), ref)
    # fallback expression used when torch < 1.8
    _assert_equal((x / 2).floor_(), ref)


@pytest.mark.parametrize('source,destination', [
    (0, -1), (-1, 0), (1, 2), ([0, 1], [-1, -2]), ([2, 0], [0, 1]),
])
def test_movedim(source, destination):
    x = torch.randn([2, 3, 4, 5])
    out = utils.movedim(x, source, destination)
    if _has(torch, 'movedim'):
        _assert_equal(out, torch.movedim(x, source, destination))
    if isinstance(source, int):
        _assert_equal(jit.movedim1(x, source, destination), out)


# ----------------------------------------------------------------------
#                        eigh / eigvalsh
# ----------------------------------------------------------------------

def _sym_upper(n=4, batch=(3,), seed=0, dtype=torch.float64):
    """Symmetric matrices with garbage in the strict lower triangle."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn([*batch, n, n], generator=g, dtype=dtype)
    a = a + a.transpose(-1, -2)
    garbage = torch.randn([*batch, n, n], generator=g, dtype=dtype) * 100
    lower = torch.ones(n, n, dtype=torch.bool).tril(-1)
    full = a.clone()
    a = torch.where(lower, garbage, a)
    return a, full


@pytest.fixture
def symeig_fallback(monkeypatch, fallback):
    """Force the `torch.symeig` branch. Use the real `torch.symeig` if it
    still exists (torch < 2), else a stand-in that records `upper`."""
    fallback(nilinalg, '_torch_has_linalg_eigh', '_torch_has_linalg_eigvalsh')
    calls = []
    if not _works(getattr(torch, 'symeig', None), torch.eye(2)):
        def symeig(a, eigenvectors=False, upper=True):
            calls.append(upper)
            val, vec = torch.linalg.eigh(a, UPLO='U' if upper else 'L')
            if not eigenvectors:
                vec = a.new_empty([0])
            return val, vec
        monkeypatch.setattr(torch, 'symeig', symeig, raising=False)
    return calls


def _reference_eigh(full):
    import numpy as np
    val, vec = np.linalg.eigh(full.numpy())
    return torch.as_tensor(val), torch.as_tensor(vec)


@pytest.mark.parametrize('UPLO', ['L', 'U'])
def test_eigh_native(UPLO):
    a, full = _sym_upper()
    if UPLO == 'L':
        a = a.transpose(-1, -2)
    refval, refvec = _reference_eigh(full)
    val, vec = nilinalg.eigh(a, UPLO=UPLO)
    _assert_equal(val, refval, rtol=1e-10, atol=1e-10)
    # eigenvectors are defined up to a sign
    _assert_equal(vec.abs(), refvec.abs(), rtol=1e-8, atol=1e-8)
    _assert_equal(nilinalg.eigvalsh(a, UPLO=UPLO), refval,
                  rtol=1e-10, atol=1e-10)
    if UPLO == 'L':
        # default is lower, as in torch.linalg.eigh
        _assert_equal(nilinalg.eigvalsh(a), refval, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('UPLO', ['L', 'U'])
def test_eigh_fallback(UPLO, symeig_fallback):
    a, full = _sym_upper()
    if UPLO == 'L':
        a = a.transpose(-1, -2)
    refval, refvec = _reference_eigh(full)
    val, vec = nilinalg.eigh(a, UPLO=UPLO)
    _assert_equal(val, refval, rtol=1e-10, atol=1e-10)
    _assert_equal(vec.abs(), refvec.abs(), rtol=1e-8, atol=1e-8)
    val = nilinalg.eigvalsh(a, UPLO=UPLO)
    _assert_equal(val, refval, rtol=1e-10, atol=1e-10)
    if symeig_fallback:  # stand-in was used
        assert symeig_fallback == [UPLO == 'U'] * 2


# ----------------------------------------------------------------------
#                          eigvals / det / kron
# ----------------------------------------------------------------------

def _sorted_complex(x):
    x = torch.view_as_real(x)  # (..., N, 2)
    key = x[..., 0] * 1e6 + x[..., 1]
    return x.gather(-2, key.argsort(-1)[..., None].expand_as(x))


@pytest.fixture
def eig_fallback(monkeypatch, fallback):
    fallback(nilinalg, '_torch_has_linalg_eigvals')
    calls = []
    if not _works(getattr(torch, 'eig', None), torch.eye(2)):
        def eig(a, eigenvectors=False):
            calls.append(eigenvectors)
            assert a.dim() == 2  # torch.eig was not batched
            val = torch.view_as_real(torch.linalg.eigvals(a))
            return val, a.new_empty([0])
        monkeypatch.setattr(torch, 'eig', eig, raising=False)
    return calls


def _general_matrices(dtype=torch.float64):
    g = torch.Generator().manual_seed(3)
    a = torch.randn([2, 3, 4, 4], generator=g, dtype=dtype)
    a[0, 0] = torch.tensor([[0., -1., 0., 0.], [1., 0., 0., 0.],
                            [0., 0., 2., 0.], [0., 0., 0., 3.]])
    return a


@pytest.mark.parametrize('old', [False, True])
def test_eigvals(old, request):
    if old:
        request.getfixturevalue('eig_fallback')
    a = _general_matrices()
    import numpy as np
    ref = torch.as_tensor(np.linalg.eigvals(a.numpy()))
    out = nilinalg.eigvals(a)
    assert out.is_complex()
    assert out.shape == (2, 3, 4)
    _assert_equal(_sorted_complex(out), _sorted_complex(ref),
                  rtol=1e-8, atol=1e-8)
    # rotation by 90 degrees -> eigenvalues +/- i
    _assert_equal(_sorted_complex(out)[0, 0, :2],
                  torch.tensor([[0., -1.], [0., 1.]], dtype=torch.float64),
                  rtol=0, atol=1e-12)


@pytest.mark.parametrize('old', [False, True])
def test_eigvals_empty(old, request):
    if old:
        request.getfixturevalue('eig_fallback')
    out = nilinalg.eigvals(torch.zeros([0, 3, 3], dtype=torch.float64))
    assert out.shape == (0, 3) and out.is_complex()


@pytest.mark.parametrize('old', [False, True])
def test_det(old, fallback):
    if old:
        fallback(nilinalg, '_torch_has_linalg_det')
    a = _general_matrices()
    import numpy as np
    ref = torch.as_tensor(np.linalg.det(a.numpy()))
    _assert_equal(nilinalg.det(a), ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('old', [False, True])
@pytest.mark.parametrize('shapes', [
    [(2, 3), (4, 5)],
    [(3,), (4,)],
    [(5, 3, 3), (1, 3, 3)],     # batched kron(Y, eye[None])
    [(1, 3, 3), (5, 3, 3)],     # batched kron(eye[None], Y)
    [(2, 3), (2, 2, 2)],        # different number of dimensions
    [(), (3,)],
])
def test_kron(shapes, old, fallback):
    if old:
        fallback(nilinalg, '_torch_has_kron')
    g = torch.Generator().manual_seed(4)
    a = torch.randn(shapes[0], generator=g, dtype=torch.float64)
    b = torch.randn(shapes[1], generator=g, dtype=torch.float64)
    out = nilinalg.kron(a, b)
    import numpy as np
    ref = torch.as_tensor(np.kron(a.numpy(), b.numpy()))
    _assert_equal(out, ref, rtol=1e-12, atol=1e-12)
    if _has(torch, 'kron'):
        _assert_equal(out, torch.kron(a, b))


def test_kron_vs_kron2():
    """`kron2(x, y)` is `kron(y, x)` over the last two dimensions."""
    g = torch.Generator().manual_seed(5)
    x = torch.randn([2, 3], generator=g, dtype=torch.float64)
    y = torch.randn([4, 5], generator=g, dtype=torch.float64)
    _assert_equal(nilinalg.kron2(x, y), nilinalg.kron(y, x))

