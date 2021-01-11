import pytest
import torch
from nitorch.spatial import diff, div, diff1d, div1d

bounds = ('dct2', 'dst2', 'dct1', 'dst1', 'zero', 'replicate')
orders = (1, 2, 3, 4)
sides = ('f', 'b', 'c')


@pytest.mark.parametrize('bound', bounds)
@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('side', sides)
def test_adjoint_1d(order, bound, side):
    u = torch.randn([64, 64, 64], dtype=torch.double)
    v = torch.randn([64, 64, 64], dtype=torch.double)
    Lv = diff1d(v, side=side, order=order, bound=bound)
    Ku = div1d(u, side=side, order=order, bound=bound)
    assert torch.allclose((Lv*u).sum(), (Ku*v).sum())


@pytest.mark.parametrize('bound', bounds)
@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('side', sides)
def test_adjoint_3d(order, bound, side):
    u = torch.randn([64, 64, 64, 3], dtype=torch.double)
    v = torch.randn([64, 64, 64], dtype=torch.double)
    Lv = diff(v, dim=[0, 1, 2], side=side, order=order, bound=bound)
    Ku = div(u, dim=[0, 1, 2], side=side, order=order, bound=bound)
    assert torch.allclose((Lv*u).sum(), (Ku*v).sum())
