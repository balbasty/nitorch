"""Gauss-Newton Hessian of (joint) histograms for spline orders != 1
(order-1 kernel rescaled by position-averaged curvature constants)."""
import torch
import pytest
from nitorch.core import _hist
from nitorch.core._hist import (
    _spline_sq_integrals, _hess_order_scale, _jhistc_forward,
    _jhistc_backward2, _histc_forward, _histc_backward2,
    HistCount, iso1_grad2_diag_2d)
from nitorch._C._ts.pushpull import make_bound

BINS = 8


def test_constants():
    ref = {1: (2 / 3, 2.), 2: (0.55, 1.), 3: (151 / 315, 2 / 3)}
    for k, (w, d) in ref.items():
        w_, d_ = _spline_sq_integrals(k)
        assert abs(w_ - w) < 1e-10 and abs(d_ - d) < 1e-10, (k, w_, d_)
    for k in range(1, 8):
        w, d = _spline_sq_integrals(k)
        assert 0 < w < 1 and d > 0
    assert _hess_order_scale([1, 1]) == [1., 1.]
    assert abs(_hess_order_scale(3) - 1 / 3) < 1e-10


def rand_points(n, ndim, seed=0):
    g = torch.Generator().manual_seed(seed)
    # keep away from the edges so that boundary effects do not matter
    x = 2.5 + (BINS - 6) * torch.rand([1, n, ndim], dtype=torch.double,
                                      generator=g)
    return x.squeeze(-1) if ndim == 1 else x


def rand_h(shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand([1, *shape], dtype=torch.double, generator=g) + 0.5


def test_order1_unchanged():
    x = rand_points(50, 2)
    h = rand_h([BINS, BINS])
    out = _jhistc_backward2(h, x, order=1, bound='replicate',
                            extrapolate=True)
    ref = iso1_grad2_diag_2d(h.unsqueeze(1), x.unsqueeze(1),
                             make_bound([_hist.bound_to_nitorch('replicate', 'int')] * 2),
                             1).squeeze(1).squeeze(1)
    assert torch.equal(out, ref)


def test_order0_raises():
    with pytest.raises(NotImplementedError, match='order 0'):
        _jhistc_backward2(rand_h([BINS, BINS]), rand_points(5, 2), order=[0, 1])


@pytest.mark.parametrize('order', [[2, 2], [3, 3], [2, 3], [3, 1]])
def test_joint_matches_exact_gn(order):
    n = 400
    x = rand_points(n, 2)
    h = rand_h([BINS, BINS])

    def fwd(x):
        return _jhistc_forward(x, BINS, order=order, bound='replicate',
                               extrapolate=True)
    jac = torch.autograd.functional.jacobian(fwd, x)  # (1,B,B,1,n,2)
    jac = jac.reshape([BINS, BINS, n, 2])
    exact = (h[0, :, :, None, None] * jac.square()).sum([0, 1])  # (n, 2)
    approx = _jhistc_backward2(h, x, order=order, bound='replicate',
                               extrapolate=True)[0]
    ratio = approx.mean(0) / exact.mean(0)
    assert ((ratio - 1).abs() < 0.1).all(), ratio


@pytest.mark.parametrize('order', [2, 3])
def test_1d_matches_exact_gn(order):
    n = 400
    x = rand_points(n, 1)
    h = rand_h([BINS])

    def fwd(x):
        return _histc_forward(x, BINS, order=order, bound='replicate',
                              extrapolate=True)
    jac = torch.autograd.functional.jacobian(fwd, x).reshape([BINS, n])
    exact = (h[0, :, None] * jac.square()).sum(0)
    approx = _histc_backward2(h, x, order=order, bound='replicate',
                              extrapolate=True)[0]
    assert approx.shape == exact.shape
    ratio = approx.mean() / exact.mean()
    assert abs(ratio - 1) < 0.1, ratio


def test_1d_order1_exact():
    # order 1: B1' = +/-1, so the GN term is exactly H[x0] + H[x1]
    x = rand_points(50, 1)
    h = rand_h([BINS])
    jac = torch.autograd.functional.jacobian(
        lambda x: _histc_forward(x, BINS, order=1, bound='replicate',
                                 extrapolate=True), x).reshape([BINS, 50])
    exact = (h[0, :, None] * jac.square()).sum(0)
    approx = _histc_backward2(h, x, order=1, bound='replicate',
                              extrapolate=True)[0]
    assert torch.allclose(approx, exact)


def test_histcount_backward2():
    """HistCount.backward2 runs and matches the chain rule
    (bins/(max-min))^2 applied to the bin-index Hessian."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand([2, 30], dtype=torch.double, generator=g) * 10 + 3
    h = rand_h([BINS]).expand([2, BINS]).clone()
    hc = HistCount(bins=BINS, order=1)
    mn, mx = 3., 13.
    out = hc.backward2(h, x, min=mn, max=mx)
    assert out.shape == x.shape
    idx = x * (BINS / (mx - mn)) + BINS / (1 - mx / mn) - 0.5
    ref = _histc_backward2(h, idx, order=1, bound=hc.bound,
                           extrapolate=hc.extrapolate)
    assert torch.allclose(out, ref * (BINS / (mx - mn)) ** 2)
    # default min/max
    assert hc.backward2(h, x).shape == x.shape


def test_register_gn_mi_order3(tmp_path):
    from nitorch.tests.test_register_optim import make_pair, check_affine
    from nitorch.cli.registration.register.cli import cli as register
    f, expected = make_pair(tmp_path, shift=(1.5, -1., 0.5))
    register(['@loss', 'mi', '--order', '3', '@@fix', f['fix'], '-o', 'false',
              '@@mov', f['mov'], '-o', 'false',
              '@affine', 'translation', '-o', f['out'], '@@optim', 'gn'])
    check_affine(f['out'], expected, atol_lin=1e-5, atol_shift=0.15)


@pytest.mark.parametrize('order', [1, 3])
@pytest.mark.parametrize('weighted', [False, True])
def test_histcount_backward(order, weighted):
    """HistCount.backward matches autograd through the intensity->bin map.
    (Batch 1: batched order > 1 needs the nd backend fix from #117.)"""
    g = torch.Generator().manual_seed(0)
    x = torch.rand([1, 30], dtype=torch.double, generator=g) * 10 + 3
    w = torch.rand([1, 30], dtype=torch.double, generator=g) if weighted else None
    gr = torch.rand([1, BINS], dtype=torch.double, generator=g)
    hc = HistCount(bins=BINS, order=order)
    mn, mx = 3., 13.
    out = hc.backward(gr.clone(), x, w, min=mn, max=mx)

    xg = x.clone().requires_grad_()
    idx = xg * (BINS / (mx - mn)) + BINS / (1 - mx / mn) - 0.5
    hist = _histc_forward(idx, BINS, w, order=order, bound=hc.bound,
                          extrapolate=hc.extrapolate)
    ref, = torch.autograd.grad((hist * gr).sum(), xg)
    assert out.shape == x.shape
    assert torch.allclose(out, ref)
    # default min/max
    assert hc.backward(gr.clone(), x, w).shape == x.shape
