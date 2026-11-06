"""FMG restriction of the compiled backend (`fmg_restriction`).

The restriction kernel uses a fixed 4-tap (per dimension) stencil,
specialised for the factor-2 restriction used by FMG (coarse size =
ceil(fine size / 2), so that the fine/coarse scale is s <= 2). The legacy
kernels compute the same operator and can be selected with the environment
variable NI_MULTIRES_RESTRICT_LEGACY=1.
"""
import itertools

import numpy as np
import pytest
import torch

spatial = pytest.importorskip('nitorch._C.spatial')
if not hasattr(spatial, 'fmg_restriction'):
    pytest.skip('compiled backend not available', allow_module_level=True)

from nitorch._C.grid import bound_to_nitorch  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
DEVICES = ['cpu', pytest.param('cuda', marks=CUDA)]


def exact_tol(x):
    """Bitwise on CPU. On CUDA, nvcc may contract multiply-adds
    differently in the two kernels."""
    if x.device.type == 'cpu':
        return dict(rtol=0, atol=0)
    tol = 1e-12 if x.dtype is torch.double else 1e-6
    return dict(rtol=tol, atol=tol)

LEGACY_ENV = 'NI_MULTIRES_RESTRICT_LEGACY'
BOUNDS = ['replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft', 'zero']
SHAPES = {
    1: [(12,), (13,), (1,), (2,), (3,)],
    2: [(12, 10), (13, 9), (7, 1), (2, 5)],
    3: [(10, 8, 6), (11, 7, 9), (5, 1, 3)],
}


def coarse_shape(shape):
    # fmg_common.cpp: restrict_shape1(s) = ceil(s / 2)
    return tuple(1 + (s - 1) // 2 for s in shape)


def restrict(x, bound, shape=None):
    dim = x.dim() - 2
    shape = shape or coarse_shape(x.shape[2:])
    out = x.new_full([*x.shape[:2], *shape], 7.)  # must be overwritten
    bound = bound_to_nitorch([bound] * dim, 'enum')
    return spatial.fmg_restriction(x, out, bound)


# --- dense reference ---------------------------------------------------

def fold(bound, i, n):
    """(index, sign) of the (possibly out-of-bound) voxel i, following
    bounds_common.h."""
    if bound == 'replicate':
        return min(max(i, 0), n - 1), 1
    if bound in ('dct2', 'dst2'):
        n2 = 2 * n
        c = (n2 - ((-i - 1) % n2) - 1) if i < 0 else i % n2
        c = n2 - c - 1 if c >= n else c
        if bound == 'dct2':
            return c, 1
        cs = n - i - 1 if i < 0 else i
        return c, (-1 if (cs // n) % 2 else 1)
    if bound == 'dct1':
        if n == 1:
            return 0, 1
        n2 = 2 * (n - 1)
        c = abs(i) % n2
        return (n2 - c if c >= n else c), 1
    if bound == 'dst1':
        if n == 1:
            return 0, 1
        n2 = 2 * (n + 1)
        c = 0 if i == -1 else (-i - 2 if i < 0 else i)
        c = c % n2
        c = n - 1 if c == n else (n2 - c - 2 if c > n else c)
        cs = n - i - 1 if i < 0 else i
        cs = cs % n2
        if cs % (n + 1) == n:
            sign = 0
        else:
            sign = -1 if (cs // (n + 1)) % 2 else 1
        return c, sign
    if bound == 'dft':
        return i % n, 1
    if bound == 'zero':
        return i, (1 if 0 <= i < n else 0)
    raise ValueError(bound)


def dense_restriction_1d(n_fine, n_coarse, bound):
    """R[j, i] = sum over unfolded taps of Lin((x_j - i)/s)/s * sign."""
    s = n_fine / n_coarse
    R = np.zeros([n_coarse, n_fine])
    for j in range(n_coarse):
        x = (j + 0.5) * s - 0.5
        lo, hi = int(np.floor(x - s)) - 1, int(np.ceil(x + s)) + 1
        for i in range(lo, hi + 1):
            w = max(0., 1. - abs(x - i) / s) / s
            if w == 0:
                continue
            k, sign = fold(bound, i, n_fine)
            if sign:
                R[j, k] += sign * w
    return R


def dense_restrict(x, bound, shape):
    x = x.numpy()
    dim = x.ndim - 2
    for d in range(dim):
        R = dense_restriction_1d(x.shape[2 + d], shape[d], bound)
        x = np.moveaxis(np.tensordot(R, np.moveaxis(x, 2 + d, 0), 1), 0, 2 + d)
    return torch.as_tensor(np.ascontiguousarray(x))


# --- tests -------------------------------------------------------------

def _cases():
    for dim, shapes in SHAPES.items():
        for shape in shapes:
            for bound in BOUNDS:
                yield dim, shape, bound


@pytest.mark.parametrize('dim,shape,bound', list(_cases()))
@pytest.mark.parametrize('dtype', [torch.double, torch.float])
@pytest.mark.parametrize('device', DEVICES)
def test_restriction_matches_legacy(monkeypatch, dim, shape, bound, dtype,
                                    device):
    """New kernel == legacy kernel (factor 2, even and odd sizes)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn([2, 3, *shape], generator=g).to(dtype).to(device)
    monkeypatch.delenv(LEGACY_ENV, raising=False)
    new = restrict(x, bound)
    monkeypatch.setenv(LEGACY_ENV, '1')
    old = restrict(x, bound)
    # same operations in the same order: bitwise identical on CPU
    torch.testing.assert_close(new, old, **exact_tol(x))


@pytest.mark.parametrize('dim,shape,bound', list(_cases()))
@pytest.mark.parametrize('device', DEVICES)
def test_restriction_dense_reference(dim, shape, bound, device):
    """Kernel == independent dense implementation of the operator,
    including boundary folding at odd sizes."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn([1, 2, *shape], dtype=torch.double, generator=g)
    cshape = coarse_shape(shape)
    out = restrict(x.to(device), bound, cshape).cpu()
    ref = dense_restrict(x, bound, cshape)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('device', DEVICES)
def test_restriction_strided(dim, device):
    g = torch.Generator().manual_seed(2)
    shape = SHAPES[dim][1]
    x = torch.randn([2, 3, *shape[::-1]], dtype=torch.double, generator=g)
    x = x.to(device).permute(0, 1, *range(x.dim() - 1, 1, -1))
    if dim > 1:
        assert not x.is_contiguous()
    out = restrict(x, 'dct2')
    ref = restrict(x.contiguous(), 'dct2')
    torch.testing.assert_close(out, ref, **exact_tol(x))


def test_restriction_preserves_constants():
    """With an integer scale (even sizes: s = 2, size 1: s = 1), the
    samples of Lin(./s)/s sum to one -> constants are preserved
    (everywhere with dct2 / replicate)."""
    for shape in itertools.chain(*SHAPES.values()):
        if any(s % 2 and s != 1 for s in shape):
            continue
        x = torch.ones([1, 1, *shape], dtype=torch.double)
        for bound in ('dct2', 'replicate'):
            out = restrict(x, bound)
            torch.testing.assert_close(out, torch.ones_like(out))


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('legacy', [False, True])
def test_restriction_factor3_raises(monkeypatch, dim, legacy):
    """The 4-tap stencil only covers factors <= 2: refuse larger ones."""
    if legacy:
        monkeypatch.setenv(LEGACY_ENV, '1')
    else:
        monkeypatch.delenv(LEGACY_ENV, raising=False)
    x = torch.randn([1, 1, *([12] * dim)], dtype=torch.double)
    with pytest.raises(RuntimeError, match='factor of'):
        restrict(x, 'dct2', [4] * dim)
    # factor 2 (and the fmg shapes of odd / unit sizes) are fine
    restrict(x, 'dct2', [6] * dim)
    for n in (1, 2, 3, 5):
        restrict(torch.randn([1, 1, *([n] * dim)], dtype=torch.double), 'dct2')
