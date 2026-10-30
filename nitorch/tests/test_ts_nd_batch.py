"""TorchScript nd (spline order >= 2) routines on batched inputs/grids."""
import itertools

import pytest
import torch

from nitorch._C._ts import nd
from nitorch._C._ts.pushpull import make_bound, make_spline

# dct2, dft, zero
BOUNDS = [3, 6, 0]


@pytest.mark.parametrize('fn', ['pull', 'grad', 'hess', 'push', 'pushgrad'])
@pytest.mark.parametrize('dim,order,bound,inp_batch',
                         list(itertools.product([2, 3], [2, 3], BOUNDS, [1, 3])))
def test_nd_batch_matches_loop(fn, dim, order, bound, inp_batch):
    torch.manual_seed(0)
    batch, shape = 3, [4] * dim
    grid = torch.rand([batch, *shape, dim], dtype=torch.double) * 5 - 0.5
    extra = [dim] if fn == 'pushgrad' else []
    inp = torch.randn([inp_batch, 2, *shape, *extra], dtype=torch.double)
    opt = (make_bound([bound] * dim), make_spline([order] * dim), 1)
    f = getattr(nd, fn)

    def call(x, g):
        return f(x, g, shape, *opt) if fn.startswith('push') else f(x, g, *opt)

    out = call(inp, grid)
    ref = torch.cat([call(inp[min(b, inp_batch - 1)][None], grid[b][None])
                     for b in range(batch)])
    assert torch.allclose(out, ref)


def test_slicewise_mi_spline3():
    # joint histogram of a batch of slices goes through nd.push
    from nitorch.tools.registration import losses
    torch.manual_seed(0)
    x = torch.rand([1, 8, 8, 6], dtype=torch.double)
    y = torch.rand([1, 8, 8, 6], dtype=torch.double)
    loss = losses.SliceWiseLoss(losses.MI(spline=3), -1)
    ll = loss.loss(x, y, dim=3)
    ll2, g = loss.loss_grad(x, y, dim=3)
    assert torch.isfinite(ll) and torch.allclose(ll, ll2)
    assert g.shape == x.shape and torch.isfinite(g).all()
