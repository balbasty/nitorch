"""Gradient and Hessian of the (normalized) mutual information loss
(`nitorch.tools.registration.losses.mi`).

With `p` the (smoothed) joint histogram and `J = dp/dx`, the Hessian of
the loss with respect to the moving voxels is
    J^T F J + diag_n(sum_ab g_ab d2 p_ab / dx_n^2),     F = d2L/dp2, g = dL/dp
Along moving-image directions, every norm is a function of H(Y|X) and Hx
with non-negative coefficients, both concave in `p`: the first (Gauss-
Newton) term without the rank-2 terms of the normalisation is negative
semi-definite. The returned Hessian is therefore the absolute value of the
second (voxel-diagonal) term.
"""
import torch
import pytest
from nitorch.core._hist import JointHistCount
from nitorch.tools.registration.losses.mi import mi

BINS = 8
NORMS = ['studholme', 'arithmetic', 'geometric', None]


def make_data(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = torch.rand([1, *shape], dtype=torch.double, generator=g)
    mov = 0.7 * fix + 0.3 * torch.rand([1, *shape], dtype=torch.double,
                                       generator=g)
    return mov, fix


def entropies(p):
    p = p.clamp(1e-8)
    px, py = p.sum(-1, keepdim=True), p.sum(-2, keepdim=True)
    hxy = -(p * p.log()).sum()
    hx = -(px * px.log()).sum()
    hy = -(py * py.log()).sum()
    return hxy, hx, hy


def loss_of_entropies(hxy, hx, hy, norm):
    if norm == 'studholme':
        return 2 - (hx + hy) / hxy
    if norm == 'arithmetic':
        return hxy / (hx + hy) - 1
    if norm == 'geometric':
        return 1 - (hx + hy - hxy) / (hx * hy).sqrt()
    return hxy - hx - hy


def loss_from_p(p, norm):
    return loss_of_entropies(*entropies(p), norm)


def linearized_loss(p0, norm):
    """Loss with its entropies' coefficients frozen at `p0` (the
    Dinkelbach surrogate, up to a constant and a positive scale)."""
    h0 = [h.detach().requires_grad_() for h in entropies(p0)]
    coeff = torch.autograd.grad(loss_of_entropies(*h0, norm), h0)
    return lambda p: sum(c * h for c, h in zip(coeff, entropies(p)))


def per_slice_reference(mov, fix, mn, mx, fwhm, norm, order):
    """Autograd gradient, Jacobian of the histogram and its Hessians."""
    n = mov.numel()
    hist = JointHistCount(bins=BINS, order=order, fwhm=fwhm)
    fixf = fix.reshape(-1)

    def p_of(m):
        return hist.forward(torch.stack([m, fixf], -1)[None],
                            min=mn, max=mx)[0] / n

    m0 = mov.reshape(-1).clone()
    J = torch.autograd.functional.jacobian(p_of, m0).reshape(BINS**2, n)
    p0 = p_of(m0)
    F = torch.autograd.functional.hessian(
        lambda q: loss_from_p(q.reshape(BINS, BINS), norm), p0.reshape(-1))
    lin = linearized_loss(p0, norm)
    F1 = torch.autograd.functional.hessian(
        lambda q: lin(q.reshape(BINS, BINS)), p0.reshape(-1))
    m = m0.clone().requires_grad_()
    grad, = torch.autograd.grad(loss_from_p(p_of(m), norm), m)
    return grad, J, F, F1


def run_mi(shape, fwhm, norm, slicewise, order=3, seed=0):
    mov, fix = make_data(shape, seed)
    dim = len(shape)
    if slicewise:  # batch of 2D slices along the last axis
        mov, fix = mov.movedim(-1, 0), fix.movedim(-1, 0)
        dim = dim - 1
    kw = dict(dim=dim, bins=BINS, order=order, fwhm=fwhm, norm=norm)
    _, mn, mx = mi(mov, fix, grad=False, hess=False, minmax=True, **kw)
    _, g, h = mi(mov, fix, minmax=(mn, mx), **kw)
    return mov, fix, mn.reshape(-1, 2), mx.reshape(-1, 2), g, h, kw


def fd_hess_diag(mov, fix, mn, mx, kw, eps=1e-5):
    """Per-voxel diagonal of the true Hessian: finite differences of the
    (analytic) gradient, one voxel at a time."""
    flat = mov.reshape(-1)
    out = torch.empty_like(flat)
    for i in range(flat.numel()):
        gp, gm = [mi((flat + s * eps * (torch.arange(flat.numel()) == i)
                      ).reshape(mov.shape), fix, minmax=(mn, mx),
                     hess=False, **kw)[1].reshape(-1)[i]
                  for s in (1, -1)]
        out[i] = (gp - gm) / (2 * eps)
    return out


# Tolerance of the loss-level checks: `mi` clamps empty bins of the counts
# (not of the probabilities, as done here), which moves the results by
# ~1e-6.
TOL = 1e-4


def relerr(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


cases = [([6, 6, 6], False), ([6, 6, 3], True)]


@pytest.mark.parametrize('norm', ['studholme', None])
@pytest.mark.parametrize('shape,slicewise', cases)
def test_mi_grad_smoothed(norm, shape, slicewise):
    """The backward smoothing must be the adjoint of the forward one
    (it is not self-adjoint near the histogram edges)."""
    mov, fix, mn, mx, g, _, _ = run_mi(shape, 2, norm, slicewise, order=1)
    if slicewise:
        grad = torch.cat([per_slice_reference(mov[s], fix[s], mn[s], mx[s],
                                              2, norm, 1)[0]
                          for s in range(len(mov))])
    else:
        grad = per_slice_reference(mov, fix, mn[0], mx[0], 2, norm, 1)[0]
    g = g.reshape(-1)
    assert relerr(g, grad) < TOL, relerr(g, grad)


@pytest.mark.parametrize('fwhm', [0, 2])
@pytest.mark.parametrize('norm', NORMS)
def test_gn_term_nsd(norm, fwhm):
    """Without the rank-2 terms of the normalisation, the Gauss-Newton
    term J^T F J is negative semi-definite (voxel coupling included):
    bounding it by zero is a valid Loewner bound."""
    mov, fix, mn, mx, _, _, _ = run_mi([6, 6, 6], fwhm, norm, False)
    _, J, F, F1 = per_slice_reference(mov, fix, mn[0], mx[0], fwhm, norm, 3)
    A = J.T @ F1 @ J
    eig = torch.linalg.eigvalsh(A)
    # With smoothing, the 'replicate' bound does not preserve mass near
    # the histogram edges, so moving the moving image changes py a
    # little and the bound is only approximate (largest eigenvalue
    # ~1e-3 of the spectrum).
    tol = 1e-8 if fwhm == 0 else 1e-2
    assert eig.max() <= tol * eig.abs().max(), (eig.max(), eig.min())
    if norm is None:
        # plain MI has no rank-2 term
        assert torch.allclose(F1, F, rtol=1e-10, atol=0)


@pytest.mark.parametrize('order', [2, 3])
@pytest.mark.parametrize('fwhm', [0, 2])
@pytest.mark.parametrize('norm', NORMS)
def test_mi_hess_is_abs_newton_term(norm, fwhm, order):
    """The returned Hessian is |true diagonal - diag(J^T F J)|, i.e. the
    absolute value of the term that Gauss-Newton drops."""
    mov, fix, mn, mx, _, h, kw = run_mi([5, 5, 5], fwhm, norm, False,
                                        order=order)
    _, J, F, _ = per_slice_reference(mov, fix, mn[0], mx[0], fwhm, norm,
                                     order)
    exact = fd_hess_diag(mov, fix, mn, mx, kw)
    gn = torch.einsum('in,ij,jn->n', J, F, J)
    ref = (exact - gn).abs()
    h = h.reshape(-1)
    assert (h >= 0).all()
    # The extreme voxels define min/max and therefore sit on the edges of
    # the histogram, i.e. on knots of the even-order splines, where the
    # Hessian jumps and finite differences average both sides.
    m = mov.reshape(-1)
    keep = (m > m.min()) & (m < m.max())
    assert relerr(h[keep], ref[keep]) < 10 * TOL, relerr(h[keep], ref[keep])


@pytest.mark.parametrize('order', [1, 3])
def test_mi_hess_slicewise(order):
    """Slicewise: each slice is an independent loss."""
    mov, fix, mn, mx, _, h, kw = run_mi([5, 5, 2], 2, 'studholme', True,
                                        order=order)
    for s in range(len(mov)):
        _, _, h1 = mi(mov[s], fix[s], minmax=(mn[s], mx[s]), **kw)
        assert torch.allclose(h[s], h1, rtol=1e-10, atol=0)


@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('fwhm', [0, 2])
def test_jointhist_backward_hess(order, fwhm):
    """JointHistCount.backward_hess == derivative of JointHistCount.backward
    (soft weights, smoothing). For order 1, `B''` is a sum of Diracs and
    the result is its average over one bin: a one-bin finite difference."""
    g = torch.Generator().manual_seed(0)
    n = 200
    x = torch.rand([1, n, 2], dtype=torch.double, generator=g)
    w = torch.rand([1, n], dtype=torch.double, generator=g)
    gg = torch.randn([1, BINS, BINS], dtype=torch.double, generator=g)
    # keep the extreme samples (which define min/max, and sit on knots of
    # the order-2 spline) out of the comparison
    mn, mx = x.min(-2).values, x.max(-2).values
    keep = ((x > mn[:, None]) & (x < mx[:, None])).all(-1)[0]
    hist = JointHistCount(bins=BINS, order=order, fwhm=fwhm)
    out = hist.backward_hess(gg, x, w=w, min=mn, max=mx)[0]
    for k in range(2):
        # half a bin, for order 1
        eps = 0.5 * (mx - mn)[0, k] / BINS if order == 1 else 1e-6
        d = torch.zeros(2, dtype=torch.double)
        d[k] = eps
        ref = (hist.backward(gg, x + d, w=w, min=mn, max=mx)[0, :, k]
               - hist.backward(gg, x - d, w=w, min=mn, max=mx)[0, :, k])
        ref = ref / (2 * eps)
        if order == 1:
            # the one-bin window must not cross the histogram edges
            u = (x[0, :, k] - mn[0, k]) / (mx - mn)[0, k] * BINS - 0.5
            keep = keep & (u > 0.5) & (u < BINS - 1.5)
        assert relerr(out[keep, k], ref[keep]) < 1e-8, \
            relerr(out[keep, k], ref[keep])


@pytest.mark.parametrize('fwhm', [0, 2])
def test_jointhist_backward2_exact(fwhm):
    """JointHistCount.backward2 (bin-diagonal + marginal terms, soft
    weights, samples on the histogram edges) == diag(J^T M J)."""
    g = torch.Generator().manual_seed(0)
    n = 60
    x = torch.rand([1, n, 2], dtype=torch.double, generator=g)
    x[0, :5, 0] = 0             # on the min edge
    x[0, 5:10, 0] = 1           # on the max edge
    x[0, 10:15, 1] = 0
    w = torch.rand([1, n], dtype=torch.double, generator=g)
    hd = torch.rand([1, BINS, BINS], dtype=torch.double, generator=g) + 0.5
    hx = torch.randn([1, BINS], dtype=torch.double, generator=g)
    hy = torch.randn([1, BINS], dtype=torch.double, generator=g)
    hist = JointHistCount(bins=BINS, order=1, fwhm=fwhm)
    mn, mx = x.min(-2).values, x.max(-2).values
    jac = torch.autograd.functional.jacobian(
        lambda x: hist.forward(x, w=w, min=mn, max=mx), x)
    jac = jac.reshape(BINS, BINS, n, 2)
    eye = torch.eye(BINS, dtype=torch.double)
    one = torch.ones([BINS] * 2, dtype=torch.double)
    M = (torch.diag(hd.reshape(-1))
         + torch.einsum('ac,a,bd->abcd', eye, hx[0], one).reshape([BINS**2] * 2)
         + torch.einsum('bd,b,ac->abcd', eye, hy[0], one).reshape([BINS**2] * 2))
    jac = jac.reshape(BINS**2, n, 2)
    ref = torch.einsum('ink,ij,jnk->nk', jac, M, jac)
    out = hist.backward2(hd, x, w=w, min=mn, max=mx, hx=hx, hy=hy)[0]
    assert relerr(out, ref) < 1e-10, relerr(out, ref)
    # gradient: adjoint smoothing
    gg = torch.randn([1, BINS, BINS], dtype=torch.double, generator=g)
    ref = torch.einsum('ab,abnk->nk', gg[0], jac.reshape(BINS, BINS, n, 2))
    out = hist.backward(gg, x, w=w, min=mn, max=mx)[0]
    assert relerr(out, ref) < 1e-10, relerr(out, ref)
