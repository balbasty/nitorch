"""Finite-difference checks of the (normalized) mutual information loss
used by `nitorch register` (`nitorch.tools.registration.losses.mi`)."""
import torch
import pytest
from nitorch.core._hist import JointHistCount
from nitorch.spatial import grid_count
from nitorch.tools.registration.losses.mi import mi, MI
from nitorch.tools.registration.losses.base import SliceWiseLoss

norms = ['studholme', 'arithmetic', 'geometric', None, 'no']


def make_data(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = torch.rand([1, *shape], dtype=torch.double, generator=g)
    mov = 0.7 * fix + 0.3 * torch.rand([1, *shape], dtype=torch.double,
                                       generator=g)
    return mov, fix


def fd_check(fn, mov, eps=1e-6):
    """Return (relative error, cosine) between analytical and FD gradients.
    `fn(mov, grad)` returns `loss` if not grad else `(loss, grad, hess)`."""
    _, g, h = fn(mov, True)
    assert g.shape == mov.shape and h.shape == mov.shape
    fd = torch.zeros_like(mov)
    for i in range(mov.numel()):
        m1 = mov.clone().view(-1)
        m1[i] += eps
        m2 = mov.clone().view(-1)
        m2[i] -= eps
        fd.view(-1)[i] = (fn(m1.view_as(mov), False) -
                          fn(m2.view_as(mov), False)) / (2 * eps)
    err = (fd - g).norm() / fd.norm()
    cos = (fd * g).sum() / (fd.norm() * g.norm())
    return err.item(), cos.item()


@pytest.mark.parametrize('norm', norms)
def test_mi_grad(norm):
    mov, fix = make_data([12, 12])
    kw = dict(dim=2, bins=8, order=1, fwhm=0, norm=norm)
    _, mn, mx = mi(mov, fix, grad=False, hess=False, minmax=True, **kw)

    def fn(m, grad):
        return mi(m, fix, grad=grad, hess=grad, minmax=(mn, mx), **kw)

    err, cos = fd_check(fn, mov)
    assert cos > 0.9999, (norm, err, cos)
    assert err < 1e-5, (norm, err, cos)


@pytest.mark.parametrize('norm', norms)
def test_mi_value(norm):
    """Value matches the documented definition (1 - NMI, or -MI)"""
    mov, fix = make_data([12, 12])
    kw = dict(bins=8, order=1, fwhm=0)
    ll = mi(mov, fix, dim=2, norm=norm, grad=False, hess=False, **kw)

    idx = torch.stack([mov.reshape([1, -1]), fix.reshape([1, -1])], -1)
    pxy = JointHistCount(**kw).forward(idx).clamp(1e-8) / mov.numel()
    px, py = pxy.sum(-1), pxy.sum(-2)
    hxy = -(pxy * pxy.log()).sum()
    hx = -(px * px.log()).sum()
    hy = -(py * py.log()).sum()
    mutual = hx + hy - hxy
    if norm == 'studholme':
        ref = 1 - mutual / hxy
    elif norm == 'arithmetic':
        ref = 1 - mutual / (hx + hy)
    elif norm == 'geometric':
        ref = 1 - mutual / (hx * hy).sqrt()
    else:
        ref = -mutual
    assert torch.allclose(ll, ref, rtol=1e-10, atol=0), (norm, ll, ref)


def test_mi_unknown_norm():
    mov, fix = make_data([4, 4])
    with pytest.raises(ValueError):
        mi(mov, fix, dim=2, bins=4, fwhm=0, norm='geometrical')


@pytest.mark.parametrize('norm', ['studholme', 'arithmetic', 'geometric', None])
@pytest.mark.parametrize('masked', [False, True])
def test_mi_slicewise_grad(norm, masked):
    mov, fix = make_data([8, 8, 3], seed=1)
    mask = None
    if masked:
        mask = torch.ones_like(fix)
        mask[:, :2] = 0
        mask[:, :, -1, 1] = 0
    base = MI(dim=3, bins=8, spline=1, fwhm=0, norm=norm)
    base.minmax = None  # compute min/max once, then keep them fixed
    loss = SliceWiseLoss(base, -1)
    loss.loss(mov, fix, mask=mask)

    def fn(m, grad):
        if grad:
            return loss.loss_grad_hess(m, fix, mask=mask)
        return loss.loss(m, fix, mask=mask)

    err, cos = fd_check(fn, mov)
    assert cos > 0.9999, (norm, masked, err, cos)
    assert err < 1e-5, (norm, masked, err, cos)


def test_mi_slicewise_empty_slice():
    mov, fix = make_data([8, 8, 3], seed=2)
    mask = torch.ones_like(fix)
    mask[..., 0] = 0
    loss = SliceWiseLoss(MI(dim=3, bins=8, spline=1, fwhm=0), -1)
    ll, g, h = loss.loss_grad_hess(mov, fix, mask=mask)
    assert torch.isfinite(ll)
    assert torch.isfinite(g).all() and torch.isfinite(h).all()
    assert (g[..., 0] == 0).all()


@pytest.mark.parametrize('order', [0, 1])
def test_grid_count_batched(order):
    g = torch.Generator().manual_seed(0)
    grid = 4 * torch.rand([3, 1, 10, 2], dtype=torch.double, generator=g)
    count = grid_count(grid, [5, 5], interpolation=order)
    assert count.shape == (3, 1, 5, 5)
    for b in range(3):
        ref = grid_count(grid[b:b+1], [5, 5], interpolation=order)
        assert torch.allclose(count[b:b+1], ref)
