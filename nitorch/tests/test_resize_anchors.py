"""Output size / sampling of `spatial.resize` anchors (issue #171)."""
import math

import pytest
import torch

from nitorch import spatial


@pytest.mark.parametrize('anchor', ['first', 'last'])
@pytest.mark.parametrize('N,f,expected', [
    (10, 2, 19),        # last output voxel lands on input voxel 9
    (10, 0.5, 5),
    (10, 1 / 3, 4),     # 0, 3, 6, 9 == vol[::3]
    (10, 1, 10),
    (1, 2, 1),
    (10, 0.9, 9),       # (N-1)*f = 8.1
    (11, 0.1, 2),       # (N-1)*f = 1.0000000000000002 (float error)
])
def test_first_last_shape(anchor, N, f, expected):
    img = torch.randn(1, 1, N)
    out = spatial.resize(img, f, anchor=anchor)
    assert out.shape[-1] == expected
    aff, shape = spatial.affine_resize(torch.eye(2), [N], f, anchor=anchor)
    assert shape == (expected,)


def test_float_error_guard():
    # (N-1)*f just below an integer must not lose a voxel
    N, f = 50, 0.7  # 49 * 0.7 = 34.29999...
    img = torch.randn(1, 1, N)
    out = spatial.resize(img, f, anchor='f')
    assert out.shape[-1] == math.floor(49 * 0.7 + 1e-6) + 1
    N, f = 4, 1 / 3 * 3 / 1  # = 1.0 exactly-ish
    out = spatial.resize(torch.randn(1, 1, N), f, anchor='f')
    assert out.shape[-1] == N
    N, f = 30, 0.1 * 3  # 29 * 0.30000000000000004 = 8.700000000000001
    out = spatial.resize(torch.randn(1, 1, N), f, anchor='f')
    assert out.shape[-1] == 9
    N, f = 101, 0.07  # 100 * 0.07 = 7.000000000000001
    out = spatial.resize(torch.randn(1, 1, N), f, anchor='l')
    assert out.shape[-1] == 8
    N, f = 8, 1 / 7 * 1.0  # 7 * (1/7) = 1.0 or 0.9999999999999999
    out = spatial.resize(torch.randn(1, 1, N), f, anchor='f')
    assert out.shape[-1] == 2


def test_float_error_rounds_down_to_integer():
    # 3 * 0.1 * 10 style error: (N-1)*f slightly below an integer
    f = 0.57
    N = 101   # 100 * 0.57 = 56.99999999999999
    assert (N - 1) * f < 57
    out = spatial.resize(torch.randn(1, 1, N), f, anchor='f')
    assert out.shape[-1] == 58


@pytest.mark.parametrize('anchor', ['first', 'last'])
@pytest.mark.parametrize('N', [1, 2, 5, 10, 17])
@pytest.mark.parametrize('f', [0.3, 1 / 3, 0.5, 0.9, 1, 1.5, 2, 3, 2.7])
def test_first_last_samples_inside_fov(anchor, N, f):
    img = torch.randn(1, 1, N)
    _, (scales, shifts) = spatial.resize(img, f, anchor=anchor,
                                         _return_trf=True)
    M = spatial.resize(img, f, anchor=anchor).shape[-1]
    first = shifts[0]
    last = scales[0] * (M - 1) + shifts[0]
    assert first >= -1e-6
    assert last <= N - 1 + 1e-6


@pytest.mark.parametrize('anchor', ['first', 'last'])
@pytest.mark.parametrize('N,f', [(10, 2), (10, 1 / 3), (17, 1.5), (9, 0.5)])
def test_affine_consistency(anchor, N, f):
    vx, origin = 1.5, -7.
    affine = torch.tensor([[vx, origin], [0., 1.]], dtype=torch.double)
    img = torch.randn(1, 1, N, dtype=torch.double)
    out, oaff = spatial.resize(img, f, affine=affine, anchor=anchor)
    M = out.shape[-1]
    world = lambda aff, i: (aff[0, 0] * i + aff[0, 1]).item()
    if anchor == 'first':
        assert world(oaff, 0) == pytest.approx(world(affine, 0))
        exp_last = (M - 1) / f
        assert world(oaff, M - 1) == pytest.approx(world(affine, exp_last))
    else:
        assert world(oaff, M - 1) == pytest.approx(world(affine, N - 1))
        exp_first = (N - 1) - (M - 1) / f
        assert world(oaff, 0) == pytest.approx(world(affine, exp_first))
    # affine_resize alone agrees with resize
    aff2, shp2 = spatial.affine_resize(affine, [N], f, anchor=anchor)
    assert shp2 == (M,)
    assert torch.allclose(aff2, oaff)
    # world coordinates of the output stay inside the input FOV
    lo, hi = world(affine, 0), world(affine, N - 1)
    assert lo - 1e-6 <= world(oaff, 0) <= hi + 1e-6
    assert lo - 1e-6 <= world(oaff, M - 1) <= hi + 1e-6


@pytest.mark.parametrize('anchor', ['first', 'last'])
def test_first_vol_slicing(anchor):
    img = torch.arange(10.)[None, None]
    out = spatial.resize(img, 1 / 3, anchor=anchor, interpolation=1)
    assert torch.allclose(out[0, 0], img[0, 0, ::3], atol=1e-5)


@pytest.mark.parametrize('anchor', ['centers', 'edges'])
@pytest.mark.parametrize('N,f', [(10, 2), (10, 0.5), (10, 1 / 3), (7, 1.5)])
def test_centers_edges_shape(anchor, N, f):
    out = spatial.resize(torch.randn(1, 1, N), f, anchor=anchor)
    assert out.shape[-1] == max(1, int(N * f))


@pytest.mark.parametrize('anchor', ['first', 'last', 'centers', 'edges'])
def test_explicit_shape_wins(anchor):
    out = spatial.resize(torch.randn(1, 1, 10), 2, shape=[20], anchor=anchor)
    assert out.shape[-1] == 20


@pytest.mark.parametrize('anchor', ['edges', 'first', 'last'])
def test_displacement_no_zero_border(anchor):
    d = torch.ones(1, 10, 10, 2)
    out = spatial.resize_grid(d, 2, anchor=anchor, type='displacement')
    # constant displacement of 1 input voxel == 2 output voxels
    assert torch.allclose(out, torch.full_like(out, 2.), atol=1e-4)


@pytest.mark.parametrize('anchor', ['first', 'last'])
def test_identity_grid_first_last(anchor):
    d = spatial.identity_grid([10, 10])[None]
    out = spatial.resize_grid(d, 2, anchor=anchor, type='grid')
    idt = spatial.identity_grid(out.shape[1:-1])[None]
    assert torch.allclose(out, idt, atol=1e-4)


def test_grid_edges_explicit_bound_respected():
    d = torch.ones(1, 10, 10, 2)
    out = spatial.resize_grid(d, 2, anchor='e', type='displacement',
                              bound='zero', extrapolate=False)
    assert out.abs().min() == 0


def _smooth_disp(shape):
    """Smooth (affine-free) displacement, in voxels."""
    x = spatial.identity_grid(shape, dtype=torch.double)
    d = torch.stack([0.3 * torch.sin(x[..., 1] / 3),
                     0.2 * torch.cos(x[..., 0] / 4)], -1)
    return d


@pytest.mark.parametrize('anchor', ['centers', 'edges', 'first', 'last'])
@pytest.mark.parametrize('f', [2, 0.5])
def test_identity_grid_all_anchors(anchor, f):
    """No reflection / zero border on identity transformation grids."""
    idt = spatial.identity_grid([10, 12], dtype=torch.double)[None]
    out = spatial.resize_grid(idt, f, anchor=anchor, type='grid')
    exp = spatial.identity_grid(out.shape[1:-1], dtype=torch.double)[None]
    assert torch.allclose(out, exp, atol=1e-6)


@pytest.mark.parametrize('anchor', ['centers', 'edges', 'first', 'last'])
def test_identity_plus_disp_grid(anchor):
    """grid = identity + disp resizes to identity + resized disp."""
    shape = [10, 12]
    disp = _smooth_disp(shape)[None]
    idt = spatial.identity_grid(shape, dtype=torch.double)[None]
    out_g = spatial.resize_grid(idt + disp, 2, anchor=anchor, type='grid')
    out_d = spatial.resize_grid(disp, 2, anchor=anchor, type='displacement')
    exp = spatial.identity_grid(out_g.shape[1:-1], dtype=torch.double)[None]
    assert torch.allclose(out_g, exp + out_d, atol=1e-6)
    # and the resized displacement matches the analytic one (up to
    # interpolation error) everywhere, borders included
    _, (scales, shifts) = spatial.resize(
        disp.movedim(-1, 1), 2, anchor=anchor, _return_trf=True)
    y = spatial.identity_grid(out_d.shape[1:-1], dtype=torch.double)
    x = torch.stack([y[..., i] * scales[i] + shifts[i] for i in range(2)], -1)
    ref = torch.stack([0.3 * torch.sin(x[..., 1] / 3) / scales[0],
                       0.2 * torch.cos(x[..., 0] / 4) / scales[1]], -1)
    assert torch.allclose(out_d[0], ref, atol=5e-2)
