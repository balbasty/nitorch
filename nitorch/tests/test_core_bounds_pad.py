"""Boundary conditions of `core.bounds`, and the tools built on them:
`core.utils.pad` (and its adjoint), `core.utils.roll`, and the second
central differences of `spatial.diff1d`/`div1d`."""
import pytest
import torch
from nitorch.core import bounds, utils
from nitorch.spatial import diff1d, div1d
from nitorch.tests._regulariser_reference import extend_index

BOUNDS = ['dft', 'dct2', 'dst2', 'dct1', 'dst1', 'zero', 'replicate']
SIZES = [1, 2, 3, 5]
dt = torch.double


def _expected(n, bound, js):
    return [extend_index(j, n, bound) for j in js]


def _same(got, ref):
    # where the sign is zero, the index does not matter
    return all((gs == rs == 0) or (gi == ri and gs == rs)
               for (gi, gs), (ri, rs) in zip(got, ref))


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('n', SIZES)
def test_bounds_index_and_sign(bound, n):
    js = list(range(-3 * n - 3, 4 * n + 4))
    ref = _expected(n, bound, js)
    got = [getattr(bounds, bound)(j, n) for j in js]
    assert _same(got, ref)
    for fn in (getattr(bounds, bound), getattr(bounds, bound + '_')):
        i, s = fn(torch.as_tensor(js), n)
        s = torch.as_tensor(s).expand(i.shape)
        assert _same(list(zip(i.tolist(), s.tolist())), ref)


def test_even():
    assert bounds.even(['dst2', 'dst1', 'zero', 'dct1', 'reflect', 'dft']) \
        == ['dct2', 'replicate', 'replicate', 'dct1', 'dct2', 'dft']


def _dense_pad(n, m, bound):
    """(n + 2m, n) extension matrix"""
    E = torch.zeros(n + 2 * m, n, dtype=dt)
    for p, (k, s) in enumerate(_expected(n, bound, range(-m, n + m))):
        if s:
            E[p, k] = s
    return E


@pytest.mark.parametrize('shape', [(5,), (4, 3), (3, 4, 2), (1, 3)])
@pytest.mark.parametrize('bound', BOUNDS + ['mixed'])
@pytest.mark.parametrize('margin', [1, 2, 3])
def test_pad_and_adjoint(shape, bound, margin):
    dim = len(shape)
    bound = BOUNDS[-dim:] if bound == 'mixed' else [bound] * dim
    torch.manual_seed(0)
    x = torch.randn([2, *shape], dtype=dt)
    E = _dense_pad(shape[0], margin, bound[0])
    for n, b in zip(shape[1:], bound[1:]):
        E = torch.kron(E, _dense_pad(n, margin, b))
    y = utils.pad(x, [margin] * dim, mode=bound, side='both')
    assert torch.allclose(y.reshape(2, -1), x.reshape(2, -1) @ E.T)
    z = torch.randn_like(y)
    w = utils.pad(z, [margin] * dim, mode=bound, side='both', adjoint=True)
    assert torch.allclose(w.reshape(2, -1), z.reshape(2, -1) @ E)


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('shift', [-2, -1, 1, 3])
def test_roll(bound, shift):
    torch.manual_seed(0)
    x = torch.randn(3, 5, dtype=dt)
    y = utils.roll(x, shift, dims=1, bound=bound)
    ref = torch.stack([s * x[:, k] for k, s in
                       _expected(5, bound, [j - shift for j in range(5)])], 1)
    assert torch.allclose(y, ref)


@pytest.mark.parametrize('bound', BOUNDS)
@pytest.mark.parametrize('n', [1, 2, 3, 6])
def test_second_central_difference(bound, n):
    # x[i+1] - 2 x[i] + x[i-1] on the extension, and its adjoint
    torch.manual_seed(0)
    x = torch.randn(3, n, 2, dtype=dt)
    ext = _dense_pad(n, 1, bound)
    K = torch.zeros(n, n + 2, dtype=dt)
    for i in range(n):
        K[i, i], K[i, i + 1], K[i, i + 2] = 1, -2, 1
    D = (K @ ext) / 1.7 ** 2
    y = diff1d(x, order=2, side='c', dim=1, bound=bound, voxel_size=1.7)
    assert torch.allclose(y, torch.einsum('ij,ajb->aib', D, x))
    z = div1d(x, order=2, side='c', dim=1, bound=bound, voxel_size=1.7)
    assert torch.allclose(z, torch.einsum('ji,ajb->aib', D, x))
