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
@pytest.mark.parametrize('n', [2, 3, 5, 12, 40])
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
    """For the boundaries that torch-interpol handled exactly (dct1,
    dct2, dft), the coefficients are unchanged"""
    interpol = pytest.importorskip('interpol.coeff')
    gen = torch.Generator().manual_seed(3)
    x = torch.randn([2, 30, 20, 9], dtype=dt, generator=gen)
    for name, code in (('dct1', 2), ('dct2', 3), ('dft', 6)):
        for order in ORDERS:
            old = interpol.spline_coeff_nd(x, [code] * 3, [order] * 3, 3)
            new = C.spline_coeff_nd(x, name, order, 3)
            assert torch.allclose(old, new, rtol=0, atol=1e-10)


def test_prefilter_inplace():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn([3, 11, 6], dtype=dt, generator=gen)
    ref = C.spline_coeff_nd(x, 'dst2', 3, 2)
    y = x.clone()
    out = C.spline_coeff_nd(y, 'dst2', 3, 2, inplace=True)
    assert out is y
    assert torch.allclose(y, ref)
    assert not torch.allclose(x, ref)
