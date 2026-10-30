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


@pytest.mark.parametrize('dim,order,bound',
                         list(itertools.product([2, 3], [1, 2, 3], BOUNDS)))
def test_nd_hess_matches_fd_of_grad(dim, order, bound):
    torch.manual_seed(0)
    shape = [6] * dim
    grid = torch.rand([2, *([3] * dim), dim], dtype=torch.double) * 3 + 1
    inp = torch.randn([2, 2, *shape], dtype=torch.double)
    opt = (make_bound([bound] * dim), make_spline([order] * dim), 1)
    hess = nd.hess(inp, grid, *opt)
    fd = torch.zeros_like(hess)
    eps = 1e-6
    for d in range(dim):
        gp, gm = grid.clone(), grid.clone()
        gp[..., d] += eps
        gm[..., d] -= eps
        fd[..., d] = (nd.grad(inp, gp, *opt) - nd.grad(inp, gm, *opt)) / (2 * eps)
    assert torch.allclose(hess, fd, atol=1e-6)


def test_nd_grad_order1_matches_iso1():
    from nitorch._C._ts import iso1
    torch.manual_seed(0)
    grid = torch.rand([1, 6, 2], dtype=torch.double) * 3 + 1
    inp = torch.randn([1, 2, 6, 6], dtype=torch.double)
    bnd = make_bound([3, 3])
    out = nd.grad(inp, grid, bnd, make_spline([1, 1]), 1)
    assert torch.allclose(out, iso1.grad2d(inp, grid, bnd, 1))


def test_slicewise_mi_spline3_hess_matches_per_slice():
    from nitorch.tools.registration import losses
    torch.manual_seed(0)
    x = torch.rand([1, 8, 8, 6], dtype=torch.double)
    y = torch.rand([1, 8, 8, 6], dtype=torch.double)
    # no `dim` anywhere: inferred from the input
    ll, g, h = losses.SliceWiseLoss(losses.MI(spline=3), -1).loss_grad_hess(x, y)
    base = losses.MI(spline=3, dim=2)
    ref = [base.loss_grad_hess(x[..., z], y[..., z]) for z in range(x.shape[-1])]
    assert torch.allclose(ll, sum(r[0] for r in ref))
    assert torch.allclose(g, torch.stack([r[1] for r in ref], -1))
    assert torch.allclose(h, torch.stack([r[2] for r in ref], -1))
