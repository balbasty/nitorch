"""`exp_backward` (SVF): finite-difference Jacobian of the squared field,
and accuracy against autograd (#158). The closed-form small-matrix helpers
it uses are tested in `test_linalg_small.py`."""
import torch
import pytest
from nitorch import spatial
from nitorch.spatial import _svf

dt = torch.double


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('bound', ['dft', 'dct2'])
@pytest.mark.parametrize('batch', [(), (2,)])
def test_disp_jacobian(dim, bound, batch):
    g = torch.Generator().manual_seed(0)
    v = torch.randn([*batch, *[7] * dim, dim], dtype=dt, generator=g)
    ref = spatial.grid_jacobian(v, bound=bound, type='disp')
    assert torch.equal(_svf._disp_jacobian(v, bound), ref)
    out = torch.full_like(ref, float('nan'))
    jac = _svf._disp_jacobian(v, bound, out=out)
    assert torch.equal(jac, ref)
    if bound == 'dft':
        assert jac is out                       # written into the buffer


def _smooth_field(dim, n, fwhm, amp, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn([dim, *[n] * dim], dtype=dt, generator=g)
    return spatial.smooth(v, fwhm=fwhm, dim=dim).movedim(0, -1) * amp


@pytest.mark.parametrize('dim,n,fwhm,amp,tol', [
    (2, 32, 8, 2, 0.08),
    (3, 16, 6, 1.5, 0.08),
])
def test_exp_backward_matches_autograd(dim, n, fwhm, amp, tol):
    """The DARTEL-style backward pass approximates the gradient of the
    discrete scaling-and-squaring (a few % on smooth fields)."""
    v = _smooth_field(dim, n, fwhm, amp)
    g = torch.Generator().manual_seed(1)
    G = torch.randn([*[n] * dim, dim], dtype=dt, generator=g)
    vv = v.clone().requires_grad_()
    phi = spatial.exp_forward(vv, steps=6, displacement=True)
    truth, = torch.autograd.grad((phi * G).sum(), vv)
    est = spatial.exp_backward(v, G, rotate_grad=True, steps=6)
    err = ((est - truth).norm() / truth.norm()).item()
    assert err < tol, err


@pytest.mark.parametrize('dim', [2, 3])
def test_exp_backward_hessian(dim):
    v = _smooth_field(dim, 12, 4, 1.)
    g = torch.Generator().manual_seed(2)
    G = torch.randn([*[12] * dim, dim], dtype=dt, generator=g)
    H = torch.rand([*[12] * dim, dim * (dim + 1) // 2], dtype=dt, generator=g)
    g1 = spatial.exp_backward(v, G, steps=4)
    g2, h2 = spatial.exp_backward(v, G, H, steps=4)
    assert torch.equal(g1, g2)
    assert h2.shape == H.shape and torch.isfinite(h2).all()


@pytest.mark.parametrize('dim', [2, 3])
def test_exp_backward_hessian_stays_psd_where_field_folds(dim):
    """The push forward weights by the volume change |det J|: with a
    folding field, a signed determinant made the Hessian indefinite (and
    the Gauss-Newton solve diverge)."""
    n = 32 if dim == 2 else 16
    v = _smooth_field(dim, n, fwhm=2, amp=16, seed=1)
    # the field folds already halfway through the squaring (the last
    # step's Jacobian is that of exp(v/2))
    half = spatial.exp_forward(v / 2, steps=7, displacement=True)
    assert (spatial.grid_jacdet(half, type='disp') <= 0).any()
    g = torch.Generator().manual_seed(2)
    grad = torch.randn(v.shape, dtype=dt, generator=g)
    nh = dim * (dim + 1) // 2
    hess = torch.zeros([*v.shape[:-1], nh], dtype=dt)
    hess[..., :dim] = 1                          # positive definite input
    _, h = spatial.exp_backward(v, grad, hess, steps=8)
    full = torch.zeros([*v.shape[:-1], dim, dim], dtype=dt)
    c = dim
    for i in range(dim):
        full[..., i, i] = h[..., i]
        for j in range(i + 1, dim):
            full[..., i, j] = full[..., j, i] = h[..., c]
            c += 1
    eig = torch.linalg.eigvalsh(full)
    assert eig.min() >= -1e-10 * eig.max(), eig.min()
