"""Spline prefilter (`spatial.spline_coeff[_nd]`) for all boundary
conditions, and the antisymmetric (dst) boundaries of `grid_pull`.

The prefilter is a port of the one of brainhops (`_ext.interpol`): its
coefficients are exact for every boundary condition, i.e., sampling
them at the integer coordinates with the same order and boundary gives
back the input."""
import numpy as np
import pytest
import torch
from nitorch import spatial
from nitorch._C import coeff as C

dt = torch.double
BOUNDS = ['zero', 'replicate', 'dct1', 'dct2', 'dst1', 'dst2', 'dft']
ORDERS = [2, 3, 4, 5, 6, 7]
LENGTHS = [1, 2, 3, 5, 7, 12, 90]  # 90: longer than the pole horizons


def sampling_matrix(n, order, bound):
    """Dense matrix that samples the coefficients at the integers"""
    return C._sampling_rows(n, order, bound, range(n))


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('n', LENGTHS)
def test_prefilter_is_inverse(bound, order, n):
    """The prefilter (and its transpose) applies inv(A) (and inv(A).T)"""
    A = sampling_matrix(n, order, bound)
    if abs(np.linalg.det(A)) < 1e-12:
        pytest.skip('singular')
    gen = torch.Generator().manual_seed(0)
    f = torch.randn([n, 3], dtype=dt, generator=gen)
    ref = torch.as_tensor(np.linalg.solve(A, f.numpy()))
    reft = torch.as_tensor(np.linalg.solve(A.T, f.numpy()))
    c = C.spline_coeff(f, bound, order, dim=0)
    ct = C.spline_coeff(f, bound, order, dim=0, transpose=True)
    assert torch.allclose(c, ref, rtol=0, atol=1e-10 * ref.abs().max())
    assert torch.allclose(ct, reft, rtol=0, atol=1e-10 * reft.abs().max())


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('n', [1, 2, 3, 5, 12, 40])
def test_prefilter_round_trip(bound, order, n):
    """Sampling the coefficients at the integers with `grid_pull` gives
    back the input (so the prefilter and `grid_pull` agree on the
    boundary conditions)"""
    gen = torch.Generator().manual_seed(1)
    f = torch.randn([2, n, 4], dtype=dt, generator=gen)
    c = spatial.spline_coeff_nd(f, interpolation=order, bound=bound, dim=2)
    grid = torch.stack(torch.meshgrid(
        torch.arange(n, dtype=dt), torch.arange(4, dtype=dt),
        indexing='ij'), -1)
    out = spatial.grid_pull(c, grid, interpolation=order, bound=bound,
                            extrapolate=True)
    assert torch.allclose(out, f, rtol=0, atol=1e-10)


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('order', [2, 3, 5])
def test_prefilter_gradcheck(bound, order):
    """The backward pass of the prefilter is its transpose"""
    gen = torch.Generator().manual_seed(2)
    x = torch.randn([2, 9, 7], dtype=dt, generator=gen, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: spatial.spline_coeff_nd(x, interpolation=order,
                                          bound=bound, dim=2), (x,))
    assert torch.autograd.gradcheck(
        lambda x: spatial.spline_coeff(x, interpolation=order,
                                       bound=bound, dim=1), (x,))


@pytest.mark.parametrize('bound, expected', [
    # extensions of [1, 2, 3, 4], at the indices -4 .. 7
    ('dst1', [-3, -2, -1, 0, 1, 2, 3, 4, 0, -4, -3, -2]),
    ('dst2', [-4, -3, -2, -1, 1, 2, 3, 4, -4, -3, -2, -1]),
    ('dct1', [3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2]),
    ('dct2', [4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1]),
    ('dft', [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]),
])
@pytest.mark.parametrize('order', [0, 1])
def test_pull_extension(bound, expected, order):
    """`grid_pull` extends the input as the boundary condition says,
    and leaves it unchanged inside (the TorchScript backend used to
    flip signs inside the field of view with dst1/dst2)"""
    f = torch.tensor([1., 2., 3., 4.], dtype=dt)[None, None]
    grid = torch.arange(-4, 8, dtype=dt)[None, :, None]
    out = spatial.grid_pull(f, grid, interpolation=order, bound=bound,
                            extrapolate=True)
    assert out.flatten().tolist() == [float(e) for e in expected]


def test_prefilter_matches_previous():
    """For the boundaries that torch-interpol handled (dct1, dct2, dft),
    the coefficients are unchanged, up to the accuracy of torch-interpol.

    torch-interpol truncates its initial conditions at a horizon of
    `|pole|**k < exp(-30)` (~1e-13), and its coefficients are ~1e-13
    (relative) away from a dense solve, whereas ours are ~1e-15 away.
    With order 6-7 in 3D the coefficients reach ~1e3, so the absolute
    difference reaches ~1e-10 and depends on the rounding of the torch
    build: the tolerance must be relative to the coefficients. A change
    of boundary condition would change the coefficients near the
    borders by a relative O(1e-2) or more."""
    interpol = pytest.importorskip('interpol.coeff')
    gen = torch.Generator().manual_seed(3)
    x = torch.randn([2, 30, 20, 9], dtype=dt, generator=gen)
    for name, code in (('dct1', 2), ('dct2', 3), ('dft', 6)):
        for order in ORDERS:
            old = interpol.spline_coeff_nd(x, [code] * 3, [order] * 3, 3)
            new = C.spline_coeff_nd(x, name, order, 3)
            atol = 1e-11 * new.abs().max()
            assert torch.allclose(old, new, rtol=0, atol=atol), \
                (name, order, (old - new).abs().max().item())


def test_prefilter_inplace():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn([3, 11, 6], dtype=dt, generator=gen)
    ref = C.spline_coeff_nd(x, 'dst2', 3, 2)
    y = x.clone()
    out = C.spline_coeff_nd(y, 'dst2', 3, 2, inplace=True)
    assert out is y
    assert torch.allclose(y, ref)
    assert not torch.allclose(x, ref)


# ----------------------------------------------------------------------
#   dst1 on axes of one voxel
# ----------------------------------------------------------------------
# The dst1 extension of a single sample x0 is `0, x0, 0, -x0` (period 4),
# not a constant: pull, push and grad must sample that extension.

def _dense_fold(y, shape, bound, margin):
    """Transpose of `utils.pad(x, margin, mode=bound, side='both')` along
    the last `len(shape)` axes of y"""
    from nitorch.core import bounds as B
    dim = len(shape)
    for d, (n, b) in enumerate(zip(shape, bound)):
        axis = y.dim() - dim + d
        idx, sign = getattr(B, b)(torch.arange(-margin, n + margin), n)
        sign = torch.as_tensor(sign).to(y.dtype)
        sign = sign.reshape([-1] + [1] * (y.dim() - axis - 1))
        out_shape = list(y.shape)
        out_shape[axis] = n
        y = y.new_zeros(out_shape).index_add_(axis, idx, y * sign)
    return y


@pytest.mark.parametrize('shape, bound', [
    ([1], ['dst1']),
    ([1, 4], ['dst1', 'dct2']),
    ([3, 1], ['dst2', 'dst1']),
    ([1, 1], ['dst1', 'dst1']),
])
@pytest.mark.parametrize('order', [0, 1, 2, 3, 5])
def test_dst1_singleton_matches_dense_extension(shape, bound, order):
    from nitorch.core import utils
    dim = len(shape)
    margin, m = 16, 6
    gen = torch.Generator().manual_seed(3)
    f = torch.randn([1, 2, *shape], dtype=dt, generator=gen)
    grid = torch.stack(torch.meshgrid(
        *[torch.linspace(-m, n - 1 + m, 4 * (n + 2 * m) + 1, dtype=dt)
          for n in shape], indexing='ij'), -1)[None]
    # (avoid the knots of the even orders, where the derivative jumps)
    grid = grid + 0.0123
    fp = utils.pad(f, [margin] * dim, mode=bound, side='both')
    opt = dict(interpolation=order, extrapolate=True)
    pull = spatial.grid_pull(f, grid, bound=bound, **opt)
    ref = spatial.grid_pull(fp, grid + margin, bound='zero', **opt)
    assert torch.allclose(pull, ref, atol=1e-12)
    if order > 0:
        grad = spatial.grid_grad(f, grid, bound=bound, **opt)
        ref = spatial.grid_grad(fp, grid + margin, bound='zero', **opt)
        assert torch.allclose(grad, ref, atol=1e-10)
    z = torch.randn(pull.shape, dtype=dt, generator=gen)
    push = spatial.grid_push(z, grid, shape, bound=bound, **opt)
    ref = spatial.grid_push(z, grid + margin, [n + 2 * margin for n in shape],
                            bound='zero', **opt)
    ref = _dense_fold(ref, shape, bound, margin)
    assert torch.allclose(push, ref, atol=1e-10)


@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('shape', [[1], [1, 4], [5, 1]])
def test_prefilter_round_trip_dst1_singleton(order, shape):
    """On an axis of one voxel, the dst1 prefilter inverts the sampling
    of `grid_pull` (which reads the extension `0, x0, 0, -x0`)"""
    gen = torch.Generator().manual_seed(4)
    dim = len(shape)
    f = torch.randn([2, *shape], dtype=dt, generator=gen)
    c = spatial.spline_coeff_nd(f, interpolation=order, bound='dst1', dim=dim)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(n, dtype=dt) for n in shape], indexing='ij'), -1)
    out = spatial.grid_pull(c, grid, interpolation=order, bound='dst1',
                            extrapolate=True)
    assert torch.allclose(out, f, rtol=0, atol=1e-10)
    if shape == [1]:
        # b[0] - 2 b[2] + 2 b[4] - ... (the odd neighbours are zeros)
        h = order // 2
        b = [float(v) for v in C._stencil(order)][h:]  # offsets 0..h
        a = b[0] + 2 * sum((-1) ** (k // 2) * b[k] for k in range(2, h + 1, 2))
        assert torch.allclose(c, f / a, rtol=0, atol=1e-12)
