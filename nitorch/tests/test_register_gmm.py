"""Scaling and finite-difference checks of the GMM entropy loss used by
`nitorch register` (`nitorch.tools.registration.losses.gmm`)."""
import math
import torch
from nitorch.tools.registration.losses.gmm import gmmh, GMMH


def make_data(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = torch.rand([1, *shape], dtype=torch.double, generator=g)
    mov = 0.7 * fix + 0.3 * torch.rand([1, *shape], dtype=torch.double,
                                       generator=g)
    return mov, fix


def repeat2(x):
    """Duplicate each voxel 2x2: same intensity distribution, 4x voxels."""
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)


def test_gmm_value_is_mean_nll():
    """Loss equals the *average* negative log-likelihood under the fit."""
    mov, fix = make_data([12, 12])
    ll, theta = gmmh(mov, fix, dim=2, bins=3, grad=False, hess=False,
                     return_theta=True)
    x = mov[..., None] - theta['xmean']
    y = fix[..., None] - theta['ymean']
    sx, sy = theta['xvar'].sqrt(), theta['yvar'].sqrt()
    r = theta['corr']
    q = (x / sx) ** 2 + (y / sy) ** 2 - 2 * r * x * y / (sx * sy)
    q = q / (1 - r * r)
    logp = (-0.5 * q - torch.log(sx * sy * (1 - r * r).sqrt())
            + theta['prior'].log())
    # (the loss omits the 2*pi constant)
    nll = -torch.logsumexp(logp, -1).mean()
    assert torch.allclose(ll, nll, rtol=1e-4), (ll.item(), nll.item())


def test_gmm_scaling_consistent():
    """Value, gradient and Hessian are all normalised by the voxel count:
    duplicating voxels leaves the value unchanged and divides the voxel-wise
    gradient and Hessian by the duplication factor (as for MSE)."""
    mov, fix = make_data([12, 12])
    ll1, g1, h1 = gmmh(mov, fix, dim=2, bins=3)
    ll2, g2, h2 = gmmh(repeat2(mov), repeat2(fix), dim=2, bins=3)
    assert math.isclose(ll1.item(), ll2.item(), rel_tol=1e-3), (ll1, ll2)
    assert torch.allclose(repeat2(g1) / 4, g2, rtol=1e-2, atol=1e-6)
    assert torch.allclose(repeat2(h1) / 4, h2, rtol=1e-2, atol=1e-6)


def _fd(fn, mov, eps=1e-5):
    out = torch.zeros_like(mov)
    for i in range(mov.numel()):
        m1 = mov.clone().view(-1)
        m1[i] += eps
        m2 = mov.clone().view(-1)
        m2[i] -= eps
        out.view(-1)[i] = (fn(m1.view_as(mov), i) -
                           fn(m2.view_as(mov), i)) / (2 * eps)
    return out


def test_gmm_grad_hess_fd():
    """FD check at fixed (fitted) GMM parameters.

    The GMM is fitted once; each FD evaluation then reuses those parameters
    with `max_iter=0` (single E-step), instead of re-running up to 128 EM
    iterations ~4*N times, which took minutes. With fixed parameters the
    analytical gradient matches FD much more tightly than with a refit."""
    mov, fix = make_data([8, 8])
    kw = dict(dim=2, bins=3, max_iter=0)
    _, theta = gmmh(mov, fix, dim=2, bins=3, grad=False, hess=False,
                    return_theta=True)
    theta = {k: theta[k] for k in
             ('xmean', 'ymean', 'xvar', 'yvar', 'corr', 'prior')}

    def fn(m, **k):
        # `theta` is consumed (popped) by the fit: pass a copy
        return gmmh(m, fix, theta=dict(theta), **kw, **k)

    _, g, h = fn(mov)
    fd = _fd(lambda m, i: fn(m, grad=False, hess=False), mov)
    cos = (fd * g).sum() / (fd.norm() * g.norm())
    ratio = fd.norm() / g.norm()
    assert cos > 0.99999, cos
    assert abs(ratio - 1) < 0.01, ratio

    # The Hessian is a Fisher-scoring approximation of the diagonal:
    # check that it has the right magnitude.
    hfd = _fd(lambda m, i: fn(m, hess=False)[1].view(-1)[i], mov)
    hratio = hfd.sum() / h.sum()
    assert 0.5 < hratio < 2, hratio


def test_gmm_class_loss_matches_function():
    mov, fix = make_data([12, 12])
    loss = GMMH(dim=2, bins=3, cache=False)
    ll = loss.loss(mov, fix)
    ll2, g, h = loss.loss_grad_hess(mov, fix)
    assert math.isclose(ll.item(), ll2.item(), rel_tol=1e-6)
    assert ll.abs() > 0.1  # not divided twice by the voxel count
