"""Sliding boundary conditions in pull/push/grad/count.

Along a sliding axis `d`, the component `d` of a displacement field
(normal to the border) is antisymmetric (dst2) and the other components
are symmetric (dct2): the field is reflected geometrically about the
border. Each test runs on the backend in use (TorchScript or compiled).
"""
import pytest
import torch
from torch.autograd import gradcheck
from nitorch import spatial
from nitorch._C.grid import bound_to_nitorch, BoundType

dt = torch.double


def _field(shape, seed=0):
    torch.manual_seed(seed)
    dim = len(shape)
    return torch.randn([2, dim, *shape], dtype=dt)


def _grid(shape, n=7, seed=1):
    """(2, n, 1, [1], dim) sampling coordinates inside the field of view"""
    torch.manual_seed(seed)
    dim = len(shape)
    points = [n] + [1] * (dim - 1)
    return torch.stack([torch.rand([2, *points], dtype=dt) * (s - 1)
                        for s in shape], -1)


def test_bound_to_nitorch():
    assert bound_to_nitorch('sliding') == 'sliding'
    assert bound_to_nitorch(['sliding', 'dct2'], 'enum')[0] == BoundType.sliding
    assert spatial.BoundType.sliding is not None


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 5)])
@pytest.mark.parametrize('order', [0, 1, 3])
@pytest.mark.parametrize('prefilter', [False, True])
def test_pull_reflects_the_field(shape, order, prefilter):
    """Sampling at the mirror image of a point (about the border of a
    sliding axis) returns the mirror image of the displacement"""
    if prefilter and order < 2:
        pytest.skip('no prefilter for orders < 2')
    if prefilter:
        # (the normal component is prefiltered with dst2)
        try:
            spatial.spline_coeff_nd(torch.ones([1, 1, 4, 4], dtype=dt),
                                    interpolation=order, bound='dst2', dim=2)
        except NotImplementedError:
            pytest.skip('spline prefilter not implemented for dst2')
    dim = len(shape)
    v = _field(shape)
    g = _grid(shape)
    opt = dict(interpolation=order, bound='sliding', extrapolate=True,
               prefilter=prefilter)
    inside = spatial.grid_pull(v, g, **opt)
    for d in range(dim):
        for side in ('low', 'high'):
            gd = g.clone()
            n = shape[d]
            # half-sample mirrors: about -1/2 and n - 1/2
            gd[..., d] = (-1 - g[..., d]) if side == 'low' else (2 * n - 1 - g[..., d])
            outside = spatial.grid_pull(v, gd, **opt)
            target = inside.clone()
            target[:, d].neg_()
            assert torch.allclose(outside, target, atol=1e-8), (d, side)


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 5)])
@pytest.mark.parametrize('order', [1, 3])
def test_push_is_the_adjoint(shape, order):
    dim = len(shape)
    v = _field(shape)
    g = _grid(shape) * 1.6 - 1.5            # some samples outside
    opt = dict(interpolation=order, bound='sliding', extrapolate=True)
    y = torch.randn([2, dim, *g.shape[1:-1]], dtype=dt)
    lhs = (spatial.grid_pull(v, g, **opt) * y).sum()
    rhs = (v * spatial.grid_push(y, g, shape, **opt)).sum()
    assert torch.allclose(lhs, rhs)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('fn', ['pull', 'push', 'grad'])
def test_gradcheck(dim, fn):
    shape = [3] * dim
    torch.manual_seed(0)
    v = torch.randn([1, dim, *shape], dtype=dt, requires_grad=True)
    g = (torch.rand([1, 4] + [1] * (dim - 1) + [dim], dtype=dt) * 4 - 1)
    g.requires_grad_()
    opt = dict(interpolation=1, bound='sliding', extrapolate=True)
    if fn == 'pull':
        f = lambda v, g: spatial.grid_pull(v, g, **opt)
    elif fn == 'grad':
        f = lambda v, g: spatial.grid_grad(v, g, **opt)
    else:
        v = torch.randn([1, dim, *g.shape[1:-1]], dtype=dt,
                        requires_grad=True)
        f = lambda v, g: spatial.grid_push(v, g, shape, **opt)
    assert gradcheck(f, (v, g), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('shape', [(5, 4), (4, 3, 5)])
def test_count_is_symmetric(shape):
    g = _grid(shape) * 1.6 - 1.5
    opt = dict(interpolation=1, extrapolate=True)
    assert torch.allclose(spatial.grid_count(g, shape, bound='sliding', **opt),
                          spatial.grid_count(g, shape, bound='dct2', **opt))
