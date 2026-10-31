"""Scaling and finite-difference checks of the GMM entropy loss used by
`nitorch register` (`nitorch.tools.registration.losses.gmm`)."""
import math
import pytest
import torch
from nitorch.tools.registration.losses.gmm import gmmh, GMMH, lgmmh, LGMMH


def make_data(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = torch.rand([1, *shape], dtype=torch.double, generator=g)
    mov = 0.7 * fix + 0.3 * torch.rand([1, *shape], dtype=torch.double,
                                       generator=g)
    return mov, fix


def repeat2(x):
    """Duplicate each voxel 2x2: same intensity distribution, 4x voxels."""
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)


def _gmm_logp(mov, fix, theta):
    """Joint and moving-marginal log-likelihoods under the fitted mixture
    (without the 2*pi constants)"""
    x = mov[..., None] - theta['xmean']
    y = fix[..., None] - theta['ymean']
    sx, sy = theta['xvar'].sqrt(), theta['yvar'].sqrt()
    r, logpi = theta['corr'], theta['prior'].log()
    q = (x / sx) ** 2 + (y / sy) ** 2 - 2 * r * x * y / (sx * sy)
    q = q / (1 - r * r)
    lxy = -0.5 * q - torch.log(sx * sy * (1 - r * r).sqrt()) + logpi
    lx = -0.5 * (x / sx) ** 2 - sx.log() + logpi
    return [torch.logsumexp(l, -1).mean() for l in (lxy, lx)]


@pytest.mark.parametrize('measure', ['joint', 'mi'])
def test_gmm_value_is_mean_nll(measure):
    """Loss equals the *average* negative log-likelihood under the fit:
    joint, or conditional (fixed | moving) = minus the mutual information
    up to the (constant) fixed entropy."""
    mov, fix = make_data([12, 12])
    ll, theta = gmmh(mov, fix, dim=2, bins=3, grad=False, hess=False,
                     measure=measure, return_theta=True)
    lxy, lx = _gmm_logp(mov, fix, theta)
    expected = {'joint': -lxy, 'mi': lx - lxy}[measure]
    assert torch.allclose(ll, expected, rtol=1e-4), (ll.item(), expected.item())


def test_gmm_mi_default_and_one_class_is_correlation():
    """The default measure is MI (i.e., the conditional entropy). With one
    class, it is 0.5 * log(1 - corr^2) + 0.5 * log(var(fixed)) + 0.5."""
    mov, fix = make_data([12, 12])
    ll = gmmh(mov, fix, dim=2, bins=1, grad=False, hess=False)
    llc = gmmh(mov, fix, dim=2, bins=1, grad=False, hess=False,
               measure='mi')
    assert torch.allclose(ll, llc)
    x, y = mov.flatten(), fix.flatten()
    r = torch.corrcoef(torch.stack([x, y]))[0, 1]
    vy = y.var(unbiased=False)
    expected = 0.5 * torch.log(1 - r * r) + 0.5 * vy.log() + 0.5
    assert torch.allclose(ll, expected, rtol=1e-6), (ll.item(), expected.item())


def test_gmm_mi_invariant_to_moving_scale():
    """The joint entropy can be decreased by shrinking the moving
    intensities (by log(s)), irrespective of the fixed image. The
    mutual information cannot."""
    mov, fix = make_data([12, 12])
    kw = dict(dim=2, bins=3, grad=False, hess=False)
    for measure, shift in (('joint', math.log(0.1)), ('mi', 0)):
        ll1 = gmmh(mov, fix, measure=measure, **kw)
        ll2 = gmmh(0.1 * mov, fix, measure=measure, **kw)
        assert math.isclose(ll2.item(), ll1.item() + shift, abs_tol=1e-6), \
            (measure, ll1.item(), ll2.item())


def test_gmm_mi_not_degenerate_under_shrinking():
    """Shrinking the moving image into a zero background (zoom out)
    lowers the joint entropy, but not the mutual information."""
    from nitorch import spatial
    n = 48
    yy, xx = torch.meshgrid(torch.arange(n).double(),
                            torch.arange(n).double(), indexing='ij')
    c = (n - 1) / 2
    r = ((yy - c) ** 2 + (xx - c) ** 2).sqrt()
    lab = (r < 18).long() + (r < 9).long() + ((xx - c).abs() < 3).long()
    g = torch.Generator().manual_seed(0)
    noise = lambda: 0.05 * torch.randn([n, n], dtype=torch.double, generator=g)
    fix = (torch.tensor([0., 1., 2., 3.]).double()[lab] + noise())[None]
    mov = (torch.tensor([0., 2.5, 1., 0.3]).double()[lab] + noise())[None]
    grid = spatial.identity_grid([n, n], dtype=torch.double)
    grid = (grid - c) * 2 + c
    shrunk = spatial.grid_pull(mov[None], grid[None], bound='zero',
                               extrapolate=True)[0]
    kw = dict(dim=2, bins=5, grad=False, hess=False)
    assert gmmh(shrunk, fix, measure='joint', **kw) < \
        gmmh(mov, fix, measure='joint', **kw)
    assert gmmh(shrunk, fix, measure='mi', **kw) > \
        gmmh(mov, fix, measure='mi', **kw)


@pytest.mark.parametrize('measure', ['joint', 'mi'])
def test_gmm_scaling_consistent(measure):
    """Value, gradient and Hessian are all normalised by the voxel count:
    duplicating voxels leaves the value unchanged and divides the voxel-wise
    gradient and Hessian by the duplication factor (as for MSE)."""
    mov, fix = make_data([12, 12])
    ll1, g1, h1 = gmmh(mov, fix, dim=2, bins=3, measure=measure)
    ll2, g2, h2 = gmmh(repeat2(mov), repeat2(fix), dim=2, bins=3,
                       measure=measure)
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


@pytest.mark.slow
@pytest.mark.parametrize('measure', ['joint', 'mi'])
def test_gmm_grad_hess_fd(measure):
    """FD check at fixed (fitted) GMM parameters.

    The GMM is fitted once; each FD evaluation then reuses those parameters
    with `max_iter=0` (single E-step), instead of re-running up to 128 EM
    iterations ~4*N times, which took minutes. With fixed parameters the
    analytical gradient is the exact derivative of the value."""
    mov, fix = make_data([8, 8])
    kw = dict(dim=2, bins=3, max_iter=0, measure=measure)
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
    assert cos > 1 - 1e-8, cos
    assert abs(ratio - 1) < 1e-6, ratio

    # The Hessian is a Fisher-scoring approximation of the diagonal:
    # check that it has the right magnitude.
    hfd = _fd(lambda m, i: fn(m, hess=False)[1].view(-1)[i], mov)
    hratio = hfd.sum() / h.sum()
    assert 0.5 < hratio < 2, hratio


def test_gmm_class_loss_matches_function():
    mov, fix = make_data([12, 12])
    for measure in ('joint', 'mi'):
        loss = GMMH(dim=2, bins=3, measure=measure, cache=False)
        ll = loss.loss(mov, fix)
        ll2, g, h = loss.loss_grad_hess(mov, fix)
        ll3 = gmmh(mov, fix, dim=2, bins=3, measure=measure,
                   grad=False, hess=False)
        assert math.isclose(ll.item(), ll2.item(), rel_tol=1e-6)
        assert math.isclose(ll.item(), ll3.item(), rel_tol=1e-6)
        if measure == 'joint':
            assert ll.abs() > 0.1  # not divided twice by the voxel count


def test_gmm_class_defaults_match_function():
    import inspect
    for fn, cls in ((gmmh, GMMH), (lgmmh, LGMMH)):
        fdef = inspect.signature(fn).parameters
        cdef = inspect.signature(cls).parameters
        for key in cdef:
            if key in fdef and key not in ('dim',):
                assert fdef[key].default == cdef[key].default, \
                    (cls.__name__, key)


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


@pytest.mark.parametrize('measure', ['joint', 'mi'])
def test_gmm_masked_fit_ignores_masked_voxels(measure):
    """Masked-out voxels have no influence: value, gradient and Hessian
    equal those of the fit on the cropped in-mask data, and the gradient
    and Hessian are zero outside the mask. (Used to return NaN.)"""
    mov, fix = make_data([8, 8])
    mask = torch.ones_like(fix)
    mask[..., :2] = 0
    m, f = mov.clone(), fix.clone()
    m[..., :2] = 5.  # outliers in the masked-out region
    f[..., :2] = -3.
    ll, g, h = gmmh(m, f, dim=2, bins=3, mask=mask, measure=measure)
    llc, gc, hc = gmmh(mov[..., 2:], fix[..., 2:], dim=2, bins=3,
                       measure=measure)
    assert torch.isfinite(ll)
    assert torch.allclose(ll, llc, rtol=1e-6), (ll.item(), llc.item())
    assert (g[..., :2] == 0).all() and (h[..., :2] == 0).all()
    assert torch.allclose(g[..., 2:], gc, rtol=1e-6, atol=1e-12)
    assert torch.allclose(h[..., 2:], hc, rtol=1e-6, atol=1e-12)


# --- local GMM -------------------------------------------------------------


@pytest.mark.parametrize('measure', ['joint', 'mi'])
def test_lgmm_scaling_consistent(measure):
    """Duplicating voxels 2x2 (and doubling the patch size so that local
    statistics are preserved) leaves the value unchanged and divides the
    voxel-wise gradient and Hessian by 4: everything is normalised by the
    voxel count exactly once."""
    mov, fix = make_data([10, 10])
    kw = dict(dim=2, bins=3, max_iter=8, measure=measure)
    ll1, g1, h1 = lgmmh(mov, fix, patch=5, **kw)
    ll2, g2, h2 = lgmmh(repeat2(mov), repeat2(fix), patch=10, **kw)
    assert math.isclose(ll1.item(), ll2.item(), rel_tol=1e-2), (ll1, ll2)
    assert math.isclose(g1.abs().sum().item(), g2.abs().sum().item(),
                        rel_tol=2e-2)
    assert math.isclose(h1.sum().item(), h2.sum().item(), rel_tol=2e-2)


@pytest.mark.parametrize('measure', ['joint', 'mi'])
def test_lgmm_grad_hess_fd(measure):
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
    kw = dict(dim=2, bins=3, patch=4, max_iter=0, measure=measure)

    def fn(m, **k):
        # `theta` is consumed (popped) by the fit: pass a copy
        return lgmmh(m, fix, theta=dict(theta), **kw, **k)

    ll, g, h = fn(mov)
    if measure == 'joint':
        assert ll.abs() > 0.1  # per-voxel average, not divided twice

    gen = torch.Generator().manual_seed(1)
    eps = 1e-6
    for _ in range(3):
        v = torch.randn(mov.shape, dtype=mov.dtype, generator=gen)
        v = v / v.norm()
        fd = (fn(mov + eps * v, grad=False, hess=False) -
              fn(mov - eps * v, grad=False, hess=False)) / (2 * eps)
        an = (g * v).sum()
        assert abs(fd / an - 1) < 1e-5, (fd.item(), an.item())

        # Hessian: Fisher-scoring diagonal approximation, check magnitude
        gp = fn(mov + eps * v, hess=False)[1]
        gm = fn(mov - eps * v, hess=False)[1]
        hfd = ((gp - gm) / (2 * eps) * v).sum()
        han = (h * v * v).sum()
        assert 0.4 < hfd / han < 2.5, (hfd.item(), han.item())


def test_lgmm_one_class_hessian_not_zero():
    """With a single class, the Hessian used to be identically zero."""
    mov, fix = make_data([10, 10])
    for measure in ('joint', 'mi'):
        _, _, h = lgmmh(mov, fix, dim=2, bins=1, patch=5, max_iter=8,
                        measure=measure)
        assert (h > 0).all(), measure


def test_gmm_hessian_mi_one_class():
    """With one class and fixed parameters, the MI Hessian is
    exact: corr^2 / (var(moving) * (1 - corr^2)) / N."""
    mov, fix = make_data([12, 12])
    _, _, h, theta = gmmh(mov, fix, dim=2, bins=1, return_theta=True)
    r2, vx = theta['corr'] ** 2, theta['xvar']
    expected = (r2 / (vx * (1 - r2))).squeeze() / mov.numel()
    assert torch.allclose(h, expected.expand_as(h))


# --- performance helpers ---------------------------------------------------


@pytest.mark.parametrize('shape, fwhm, stride, mode', [
    ([20, 17, 9], [5., 3., 2.], [1], 'g'),   # stride 1: conv path
    ([20, 17], [4.], [1], 'g'),
    ([20, 17, 9], [5.], [2, 1, 1], 'g'),      # mixed: conv + conv_transpose
    ([31], [6.], [1], 'g'),
    pytest.param([12, 10], [4.], [2], 'c', marks=pytest.mark.xfail(
        reason='square-kernel backward: output buffer lacks batch dims and '
               'overlapping patches are copied instead of accumulated',
        strict=True)),
])
def test_local_mean_backward_is_adjoint(shape, fwhm, stride, mode):
    from nitorch.tools.registration.losses.utils_local import local_mean
    g = torch.Generator().manual_seed(0)
    d = len(shape)
    a = torch.randn([2, 3, *shape], dtype=torch.double, generator=g)
    fa = local_mean(a, fwhm, stride, mode, d)
    b = torch.randn(fa.shape, dtype=torch.double, generator=g)
    bb = local_mean(b, fwhm, stride, mode, d, backward=True, shape=shape)
    assert torch.allclose((fa * b).sum(), (a * bb).sum(), rtol=1e-10)


def test_sum_outer_matches_direct():
    from nitorch.tools.registration.losses.utils_gmm2 import sum_outer
    g = torch.Generator().manual_seed(0)
    f = torch.randn([2, 1000, 6], generator=g)
    z = torch.rand([2, 1000, 4], generator=g)
    ref = f.double().transpose(-1, -2) @ z.double()
    out = sum_outer(f, z, chunk=64)  # 1000 = 15 * 64 + 40
    assert out.dtype == torch.double
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-4)


def test_gmm_class_state_restore_keeps_warm_start():
    """The registration loop restores the state saved before line-search
    evaluations. The fit used to consume (pop) the parameters of the saved
    state, so that EM was cold-started after each line search."""
    mov, fix = make_data([12, 12])
    loss = GMMH(dim=2, bins=3)
    loss.loss_grad_hess(mov, fix)
    state = loss.get_state()
    loss.loss(mov + 0.01, fix)
    loss.set_state(state)
    assert set(loss.theta) >= {'xmean', 'ymean', 'xvar', 'yvar', 'corr',
                               'prior'}


def test_lgmm_class_warm_start():
    """LGMMH warm-starts EM when shapes are unchanged, and cold-starts
    (rather than failing) when they change."""
    mov, fix = make_data([10, 10])
    loss = LGMMH(dim=2, bins=3, patch=4, max_iter=16)
    ll1 = loss.loss(mov, fix)
    ll2 = loss.loss(mov, fix)          # warm start: (nearly) same fit
    assert math.isclose(ll1.item(), ll2.item(), rel_tol=1e-3, abs_tol=1e-3)
    mov2, fix2 = make_data([12, 8])
    ll3, g3, h3 = loss.loss_grad_hess(mov2, fix2)  # new shape: cold start
    assert g3.shape == mov2.shape and torch.isfinite(ll3)


@pytest.mark.parametrize('local', [False, True])
def test_gmm_float32_more_classes_than_tissues(local):
    """With more classes than tissues (and a constant background), the
    responsibilities of a class can underflow in single precision. The
    M-step used to compute 0/0 = NaN; empty classes now revert to the
    global statistics."""
    n = 16
    g = torch.Generator().manual_seed(0)
    lab = torch.randint(0, 4, [1, 1, 2, 2, 2], generator=g).float()
    lab = torch.nn.functional.interpolate(lab, size=[n] * 3)[0].round().long()
    noise = lambda: 0.05 * torch.randn([1, n, n, n], generator=g) * (lab > 0)
    fix = torch.tensor([0., 1, 2, 3])[lab] + noise()
    mov = torch.tensor([0., 2.5, 1, .3])[lab] + noise()
    if local:
        fn = lambda m, f: lgmmh(m, f, dim=3, bins=8, patch=6, max_iter=16)
    else:
        fn = lambda m, f: gmmh(m, f, dim=3, bins=8)
    out32 = fn(mov, fix)
    out64 = fn(mov.double(), fix.double())
    for o32, o64 in zip(out32, out64):
        assert torch.isfinite(o32).all()
    assert torch.allclose(out32[0].double(), out64[0], rtol=1e-3, atol=1e-3)
