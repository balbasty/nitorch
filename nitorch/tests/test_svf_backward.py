"""`exp_backward` (SVF): small-matrix helpers, finite-difference Jacobian
of the squared field, and accuracy against autograd (#158)."""
import torch
import pytest
from nitorch import spatial
from nitorch.spatial import _svf

dt = torch.double


def _mats(dim, n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn([n, dim, dim], dtype=dt, generator=g)
    m = m + 3 * torch.eye(dim, dtype=dt)
    v = torch.randn([n, dim], dtype=dt, generator=g)
    return m, v


@pytest.mark.parametrize('dim', [1, 2, 3, 4])
def test_small_matrix_ops(dim):
    m, v = _mats(dim)
    assert torch.allclose(_svf._small_det(m), m.det(), rtol=1e-12)
    ref = torch.linalg.solve(m, v)
    assert torch.allclose(_svf._small_solve(m, v), ref, rtol=1e-10, atol=1e-12)
    mt = m.transpose(-1, -2)                    # non-contiguous views
    ref = torch.linalg.solve(mt, v)
    assert torch.allclose(_svf._small_solve(mt, v), ref, rtol=1e-10, atol=1e-12)
    ref = (m @ v[..., None])[..., 0]
    assert torch.allclose(_svf._small_matvec(m, v), ref, rtol=1e-12)
    assert torch.allclose(_svf._small_matvec(mt, v), (mt @ v[..., None])[..., 0],
                          rtol=1e-12)


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
