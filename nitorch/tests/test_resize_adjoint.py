"""Adjointness of the compiled (C++) resize pull and its adjoint ("restrict").

`restriction`, `resize_backward` and `restriction_backward` rely on the
adjoint path of `resize` (do_adjoint=True) being the exact transpose of the
pull (do_adjoint=False). The adjoint is implemented as a gather over output
voxels; the legacy scatter implementation can be selected with the
environment variable NI_RESIZE_ADJOINT_SCATTER=1 and is compared here too.
"""
import math

import pytest
import torch

spatial = pytest.importorskip('nitorch._C.spatial')
if not hasattr(spatial, 'resize'):
    pytest.skip('compiled backend not available', allow_module_level=True)

from nitorch._C.grid import bound_to_nitorch, inter_to_nitorch  # noqa: E402
from nitorch._C.resize import align_to_nitorch  # noqa: E402

SCATTER_ENV = 'NI_RESIZE_ADJOINT_SCATTER'

# coarse shapes per dimensionality (odd/even mix, large enough so that a
# 0.7 factor still yields >= 2 voxels, needed by 'center' alignment)
COARSE = {1: (9,), 2: (7, 6), 3: (5, 6, 4)}


def _enums(dim, bound, interpolation, mode):
    bound = bound_to_nitorch([bound] * dim if isinstance(bound, str)
                             else list(bound), 'enum')
    if isinstance(interpolation, int):
        interpolation = [interpolation] * dim
    interpolation = inter_to_nitorch(list(interpolation), 'enum')
    mode = align_to_nitorch([mode] * dim, 'enum')
    return bound, interpolation, mode


def _fine_shape(coarse, factor):
    return tuple(int(math.floor(s * factor)) for s in coarse)


def pull(x, fine, factor, bound, interpolation, mode, normalize=False):
    """Resize coarse `x` to the `fine` shape (forward operator A)."""
    dim = x.dim() - 2
    b, i, m = _enums(dim, bound, interpolation, mode)
    out = x.new_zeros([*x.shape[:2], *fine])
    return spatial.resize(x, out, [factor] * dim, b, i, m, False, normalize)


def push(y, coarse, factor, bound, interpolation, mode, normalize=False):
    """Adjoint of `pull` (A^T), from the `fine` shape back to `coarse`.
    The factor is inverted, as in `resize_backward`."""
    dim = y.dim() - 2
    b, i, m = _enums(dim, bound, interpolation, mode)
    # non-zero garbage: the adjoint must overwrite it
    out = y.new_full([*y.shape[:2], *coarse], 3.)
    return spatial.resize(y, out, [1 / factor] * dim, b, i, m, True, normalize)


def _dot(a, b):
    return (a * b).sum().item()


BOUNDS = ['dct2', 'dct1', 'dft', 'dst2', 'zero']
FACTORS = [2, 3, 1.5, 0.7]
MODES = ['edge', 'center']
# 0: nearest, 1: linear, 2: quadratic, 3: cubic (generic path)
INTERPOLATIONS = [0, 1, 2, 3]


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('interpolation', INTERPOLATIONS)
@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('factor', FACTORS)
@pytest.mark.parametrize('mode', MODES)
def test_resize_adjoint(dim, interpolation, bound, factor, mode):
    g = torch.Generator().manual_seed(1234)
    coarse = COARSE[dim]
    fine = _fine_shape(coarse, factor)
    x = torch.randn([2, 3, *coarse], dtype=torch.double, generator=g)
    y = torch.randn([2, 3, *fine], dtype=torch.double, generator=g)
    Ax = pull(x, fine, factor, bound, interpolation, mode)
    Aty = push(y, coarse, factor, bound, interpolation, mode)
    assert Aty.shape == x.shape
    lhs, rhs = _dot(Ax, y), _dot(x, Aty)
    scale = Ax.norm().item() * y.norm().item() + 1
    assert abs(lhs - rhs) <= 1e-12 * scale, (lhs, rhs)


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('interpolation', INTERPOLATIONS + [5])
@pytest.mark.parametrize('bound', BOUNDS + ['dst1', 'replicate'])
@pytest.mark.parametrize('factor', FACTORS)
@pytest.mark.parametrize('mode', MODES + ['first'])
def test_resize_adjoint_dense(dim, interpolation, bound, factor, mode):
    """Stronger check: A^T == transpose of the dense matrix of A."""
    coarse = COARSE[dim]
    fine = _fine_shape(coarse, factor)
    ncoarse, nfine = math.prod(coarse), math.prod(fine)
    eye = torch.eye(ncoarse, dtype=torch.double).reshape([ncoarse, 1, *coarse])
    A = pull(eye, fine, factor, bound, interpolation, mode)
    A = A.reshape([ncoarse, nfine]).T                       # (fine, coarse)
    eye = torch.eye(nfine, dtype=torch.double).reshape([nfine, 1, *fine])
    At = push(eye, coarse, factor, bound, interpolation, mode)
    At = At.reshape([nfine, ncoarse]).T                     # (coarse, fine)
    torch.testing.assert_close(At, A.T, rtol=0, atol=1e-12)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('bound', [('dct2', 'dft', 'dst2'),
                                   ('zero', 'dct1', 'dct2')])
@pytest.mark.parametrize('interpolation', [(1, 2, 3), (0, 1, 2)])
@pytest.mark.parametrize('factor', [2, 0.7])
def test_resize_adjoint_anisotropic(dim, bound, interpolation, factor):
    """Per-dimension bounds/orders (non-isotropic -> generic pull path)."""
    g = torch.Generator().manual_seed(0)
    coarse = COARSE[dim]
    fine = _fine_shape(coarse, factor)
    bound, interpolation = bound[:dim], interpolation[:dim]
    x = torch.randn([1, 2, *coarse], dtype=torch.double, generator=g)
    y = torch.randn([1, 2, *fine], dtype=torch.double, generator=g)
    Ax = pull(x, fine, factor, bound, interpolation, 'edge')
    Aty = push(y, coarse, factor, bound, interpolation, 'edge')
    lhs, rhs = _dot(Ax, y), _dot(x, Aty)
    assert abs(lhs - rhs) <= 1e-12 * (Ax.norm().item() * y.norm().item() + 1)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('interpolation', INTERPOLATIONS)
@pytest.mark.parametrize('bound', BOUNDS + ['dst1', 'replicate'])
@pytest.mark.parametrize('factor', FACTORS)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [torch.double, torch.float])
def test_gather_matches_scatter(monkeypatch, dim, interpolation, bound,
                                factor, mode, dtype):
    g = torch.Generator().manual_seed(42)
    coarse = COARSE[dim]
    fine = _fine_shape(coarse, factor)
    y = torch.randn([2, 2, *fine], dtype=dtype, generator=g)
    monkeypatch.delenv(SCATTER_ENV, raising=False)
    gather = push(y, coarse, factor, bound, interpolation, mode)
    monkeypatch.setenv(SCATTER_ENV, '1')
    scatter = push(y, coarse, factor, bound, interpolation, mode)
    tol = 1e-12 if dtype is torch.double else 1e-5
    torch.testing.assert_close(gather, scatter, rtol=tol, atol=tol)


def test_gather_large_kernel():
    """Rows longer than the largest unrolled K (runtime-K fallback)."""
    g = torch.Generator().manual_seed(0)
    coarse, factor = (6, 5), 7
    fine = _fine_shape(coarse, factor)
    x = torch.randn([1, 1, *coarse], dtype=torch.double, generator=g)
    y = torch.randn([1, 1, *fine], dtype=torch.double, generator=g)
    for interpolation in (1, 3):
        Ax = pull(x, fine, factor, 'dct2', interpolation, 'edge')
        Aty = push(y, coarse, factor, 'dct2', interpolation, 'edge')
        assert abs(_dot(Ax, y) - _dot(x, Aty)) <= 1e-10


def test_gather_strided_input():
    """Non-contiguous input/output tensors."""
    g = torch.Generator().manual_seed(0)
    coarse, fine = (5, 6, 4), (10, 12, 8)
    y = torch.randn([2, 3, *fine[::-1]], dtype=torch.double, generator=g)
    y = y.permute(0, 1, 4, 3, 2)
    assert not y.is_contiguous()
    ref = push(y.contiguous(), coarse, 2, 'dct2', 1, 'edge')
    out = push(y, coarse, 2, 'dct2', 1, 'edge')
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize('dim', [3])
@pytest.mark.parametrize('bound', ['dct2', 'dft'])
def test_restriction_is_normalized_adjoint(dim, bound):
    """`restriction` == (prod of Edge scales) * A^T, A = edge linear pull."""
    g = torch.Generator().manual_seed(0)
    coarse = COARSE[dim]
    fine = _fine_shape(coarse, 2)
    y = torch.randn([1, 2, *fine], dtype=torch.double, generator=g)
    b, i, _ = _enums(dim, bound, 1, 'edge')
    out = spatial.restriction(y, y.new_zeros([1, 2, *coarse]), b, i)
    ref = push(y, coarse, 2, bound, 1, 'edge')
    ref = ref * math.prod(s / f for s, f in zip(coarse, fine))
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
