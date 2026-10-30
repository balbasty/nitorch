"""Scaling and finite-difference checks of the GMM entropy loss used by
`nitorch register` (`nitorch.tools.registration.losses.gmm`)."""
import math
import torch
from nitorch.tools.registration.losses.gmm import gmmh, GMMH, lgmmh


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


def test_gmm_em_stops_at_fixed_point():
    """Warm-starting EM at its own fixed point must stop immediately.

    With one class the M-step is closed-form, so re-fitting from the fitted
    parameters gives exactly the same nll; the convergence ratio used to be
    0/0 = NaN, which never satisfied `gain < tol`, so EM ran all `max_iter`
    iterations (~11 s for 500 iterations on an 8x8 image)."""
    import time
    mov, fix = make_data([8, 8])
    kw = dict(dim=2, bins=1, grad=False, hess=False)
    _, theta = gmmh(mov, fix, **kw, return_theta=True)
    theta = {k: theta[k] for k in
             ('xmean', 'ymean', 'xvar', 'yvar', 'corr', 'prior')}
    gmmh(mov, fix, **kw, theta=dict(theta), max_iter=1)  # warm-up
    tic = time.time()
    ll1 = gmmh(mov, fix, **kw, theta=dict(theta), max_iter=1)
    t1 = time.time() - tic
    tic = time.time()
    ll500 = gmmh(mov, fix, **kw, theta=dict(theta), max_iter=500)
    t500 = time.time() - tic
    assert torch.allclose(ll1, ll500)
    assert t500 < 10 * t1 + 1, (t1, t500)


# --- local GMM -------------------------------------------------------------


def test_lgmm_scaling_consistent():
    """Duplicating voxels 2x2 (and doubling the patch size so that local
    statistics are preserved) leaves the value unchanged and divides the
    voxel-wise gradient and Hessian by 4: everything is normalised by the
    voxel count exactly once."""
    mov, fix = make_data([10, 10])
    ll1, g1, h1 = lgmmh(mov, fix, dim=2, bins=3, patch=5, max_iter=8)
    ll2, g2, h2 = lgmmh(repeat2(mov), repeat2(fix), dim=2, bins=3, patch=10,
                         max_iter=8)
    assert math.isclose(ll1.item(), ll2.item(), rel_tol=1e-2), (ll1, ll2)
    assert math.isclose(g1.abs().sum().item(), g2.abs().sum().item(),
                        rel_tol=2e-2)
    assert math.isclose(h1.sum().item(), h2.sum().item(), rel_tol=2e-2)


def test_lgmm_grad_hess_fd():
    """Directional FD check at fixed (fitted) local GMM parameters.

    Each local-GMM evaluation is slow (local smoothing), so instead of a
    voxel-wise FD we check directional derivatives along a few random
    directions v: <g, v> vs (L(m + eps v) - L(m - eps v)) / 2 eps, and the
    curvature <v, h v> against the FD of <g, v>."""
    mov, fix = make_data([8, 8])
    _, theta = lgmmh(mov, fix, dim=2, bins=3, patch=4, max_iter=8,
                     grad=False,
                     hess=False, return_theta=True)
    theta = {k: theta[k] for k in
             ('xmean', 'ymean', 'xvar', 'yvar', 'corr', 'prior')}
    kw = dict(dim=2, bins=3, patch=4, max_iter=0)

    def fn(m, **k):
        # `theta` is consumed (popped) by the fit: pass a copy
        return lgmmh(m, fix, theta=dict(theta), **kw, **k)

    ll, g, h = fn(mov)
    assert ll.abs() > 0.1  # per-voxel average, not divided twice

    gen = torch.Generator().manual_seed(1)
    eps = 1e-6
    for _ in range(3):
        # random directions correlated with g, so that <g, v> is not
        # dominated by cancellation (the gradient is only ~1% accurate)
        v = g * (1 + torch.randn(mov.shape, dtype=mov.dtype, generator=gen))
        v = v / v.norm()
        fd = (fn(mov + eps * v, grad=False, hess=False) -
              fn(mov - eps * v, grad=False, hess=False)) / (2 * eps)
        an = (g * v).sum()
        assert abs(fd / an - 1) < 0.05, (fd.item(), an.item())

        # Hessian: Fisher-scoring diagonal approximation, check magnitude
        gp = fn(mov + eps * v, hess=False)[1]
        gm = fn(mov - eps * v, hess=False)[1]
        hfd = ((gp - gm) / (2 * eps) * v).sum()
        han = (h * v * v).sum()
        assert 0.4 < hfd / han < 2.5, (hfd.item(), han.item())
