"""Gradient and Gauss-Newton Hessian of the (normalized) mutual information
loss (`nitorch.tools.registration.losses.mi`) against autograd.

The returned Hessian is the per-voxel diagonal of the Gauss-Newton form
`J^T M J`, where `J` is the Jacobian of the smoothed joint histogram with
respect to the moving voxels and `M` is minus the histogram-space Hessian
of the loss (without the rank-2 terms of the normalisation). For plain MI
this is exactly minus the per-voxel diagonal of the true Hessian (which,
for spline order 1, has no second-derivative term almost everywhere).
"""
import torch
import pytest
from nitorch.core._hist import JointHistCount
from nitorch.tools.registration.losses.mi import mi

BINS = 8


def make_data(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix = torch.rand([1, *shape], dtype=torch.double, generator=g)
    mov = 0.7 * fix + 0.3 * torch.rand([1, *shape], dtype=torch.double,
                                       generator=g)
    return mov, fix


def loss_from_p(p, norm):
    p = p.clamp(1e-8)
    px, py = p.sum(-1, keepdim=True), p.sum(-2, keepdim=True)
    hxy = -(p * p.log()).sum()
    hx = -(px * px.log()).sum()
    hy = -(py * py.log()).sum()
    if norm == 'studholme':
        return 2 - (hx + hy) / hxy
    return hxy - hx - hy


def gn_matrix(p, norm):
    """M = diag(cxy/pxy) + [a == c] cx/px + [b == d] cy/py (bins^2, bins^2)"""
    p = p.clamp(1e-8)
    px, py = p.sum(-1), p.sum(-2)
    hxy = -(p * p.log()).sum()
    hx = -(px * px.log()).sum()
    hy = -(py * py.log()).sum()
    cxy, scl = 1., 1.
    if norm == 'studholme':
        cxy, scl = (hx + hy) / hxy, hxy
    eye, one = torch.eye(BINS, dtype=p.dtype), torch.ones([BINS] * 2, dtype=p.dtype)
    M = torch.diag((cxy / p).reshape(-1))
    M = M - torch.einsum('ac,a,bd->abcd', eye, 1 / px, one).reshape([BINS**2] * 2)
    M = M - torch.einsum('bd,b,ac->abcd', eye, 1 / py, one).reshape([BINS**2] * 2)
    return M / scl


def per_slice_reference(mov, fix, mn, mx, fwhm, norm):
    """Autograd gradient, exact Hessian diagonal (via J^T F J) and
    diag(J^T M J) for one image (or one slice)."""
    n = mov.numel()
    hist = JointHistCount(bins=BINS, order=1, fwhm=fwhm)
    fixf = fix.reshape(-1)

    def p_of(m):
        return hist.forward(torch.stack([m, fixf], -1)[None],
                            min=mn, max=mx)[0] / n

    m0 = mov.reshape(-1).clone()
    J = torch.autograd.functional.jacobian(p_of, m0).reshape(BINS**2, n)
    p0 = p_of(m0)
    F = torch.autograd.functional.hessian(
        lambda q: loss_from_p(q.reshape(BINS, BINS), norm), p0.reshape(-1))
    m = m0.clone().requires_grad_()
    grad, = torch.autograd.grad(loss_from_p(p_of(m), norm), m)
    exact = torch.einsum('in,ij,jn->n', J, F, J)
    gn = torch.einsum('in,ij,jn->n', J, gn_matrix(p0, norm), J)
    return grad, exact, gn


def references(mov, fix, mn, mx, fwhm, norm, slicewise):
    if not slicewise:
        return per_slice_reference(mov, fix, mn[0], mx[0], fwhm, norm)
    out = [per_slice_reference(mov[s], fix[s], mn[s], mx[s], fwhm, norm)
           for s in range(len(mov))]
    return tuple(torch.cat(o) for o in zip(*out))


def run_mi(shape, fwhm, norm, slicewise, seed=0):
    mov, fix = make_data(shape, seed)
    dim = len(shape)
    if slicewise:  # batch of 2D slices along the last axis
        mov, fix = mov.movedim(-1, 0), fix.movedim(-1, 0)
        dim = dim - 1
    kw = dict(dim=dim, bins=BINS, order=1, fwhm=fwhm, norm=norm)
    _, mn, mx = mi(mov, fix, grad=False, hess=False, minmax=True, **kw)
    _, g, h = mi(mov, fix, minmax=(mn, mx), **kw)
    mn, mx = mn.reshape(-1, 2), mx.reshape(-1, 2)
    ref = references(mov, fix, mn, mx, fwhm, norm, slicewise)
    return g.reshape(-1), h.reshape(-1), ref


# Tolerance of the loss-level checks: `mi` clamps empty bins of the counts
# (not of the probabilities, as done here), which moves the results by
# ~1e-6. Before the fix, the errors were 3e-2 to 2 (gradient: 0.1).
TOL = 1e-4


def relerr(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


cases = [([8, 8, 8], False), ([8, 8, 3], True)]


@pytest.mark.parametrize('norm', ['studholme', None])
@pytest.mark.parametrize('shape,slicewise', cases)
def test_mi_grad_smoothed(norm, shape, slicewise):
    """The backward smoothing must be the adjoint of the forward one
    (it is not self-adjoint near the histogram edges)."""
    g, _, (grad, _, _) = run_mi(shape, 2, norm, slicewise)
    assert relerr(g, grad) < TOL, relerr(g, grad)


@pytest.mark.parametrize('fwhm', [0, 2])
@pytest.mark.parametrize('shape,slicewise', cases)
def test_mi_hess_is_exact_diag(fwhm, shape, slicewise):
    """Plain MI: the GN Hessian is exactly minus the true per-voxel
    Hessian diagonal (which is <= 0, -MI being concave in the histogram)."""
    _, h, (_, exact, _) = run_mi(shape, fwhm, None, slicewise)
    assert (exact <= 1e-12 * exact.abs().max()).all()
    assert relerr(h, -exact) < TOL, relerr(h, -exact)


@pytest.mark.parametrize('fwhm', [0, 2])
@pytest.mark.parametrize('shape,slicewise', cases)
def test_nmi_hess_is_gn_diag(fwhm, shape, slicewise):
    """Studholme NMI: the GN Hessian is the per-voxel diagonal of J^T M J
    and is non-negative."""
    _, h, (_, _, gn) = run_mi(shape, fwhm, 'studholme', slicewise)
    assert relerr(h, gn) < TOL, relerr(h, gn)
    assert (h >= 0).all()


@pytest.mark.parametrize('fwhm', [0, 1, 2])
@pytest.mark.parametrize('norm', ['studholme', 'arithmetic', 'geometric', None])
def test_mi_hess_nonnegative(norm, fwhm):
    """With a shared background (a bin where pxy ~ px ~ py), taking the
    bin-diagonal of the marginal terms made the Hessian negative for
    ~40% of the voxels (all but studholme, at fwhm 0 and 1)."""
    mov, fix = make_data([10, 10, 10], seed=3)
    mov[..., :4] = 0
    fix[..., :4] = 0
    _, _, h = mi(mov, fix, dim=3, bins=16, fwhm=fwhm, norm=norm)
    assert (h >= 0).all(), h.min()


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
