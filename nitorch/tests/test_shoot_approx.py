"""`shoot_approx`: forward matches `shoot` and the approximate backward
uses the right arguments/saved tensor for every flag combination (#171)."""
import torch
import pytest
from nitorch import spatial

dt = torch.double


def _smooth(x, dim):
    return spatial.smooth(x.movedim(-1, 0), fwhm=3, dim=dim).movedim(0, -1)


def _fd_grad(fn, v, eps=1e-4):
    """Central finite-difference gradient of scalar `fn` at `v`."""
    g = torch.zeros_like(v).flatten()
    vf = v.flatten()
    for i in range(vf.numel()):
        vp, vm = vf.clone(), vf.clone()
        vp[i] += eps
        vm[i] -= eps
        g[i] = (fn(vp.reshape(v.shape)) - fn(vm.reshape(v.shape))) / (2*eps)
    return g.reshape(v.shape)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('return_inverse', [False, True])
@pytest.mark.parametrize('displacement', [False, True])
def test_shoot_approx(dim, return_inverse, displacement):
    n = 6 if dim == 2 else 4
    g = torch.Generator().manual_seed(0)
    v = _smooth(torch.randn([*[n]*dim, dim], dtype=dt, generator=g), dim)[None]
    v = v * (0.5 / v.abs().max())
    w = torch.randn([2, *v.shape], dtype=dt, generator=g)
    opt = dict(return_inverse=return_inverse, displacement=displacement)

    def loss(out):
        if return_inverse:
            return (out[0] * w[0]).sum() + (out[1] * w[1]).sum()
        return (out * w[0]).sum()

    # forward is exactly `shoot`
    vv = v.clone().requires_grad_()
    out = spatial.shoot_approx(vv, **opt)
    ref = spatial.shoot(v, **opt)
    if return_inverse:
        assert isinstance(out, tuple) and len(out) == 2
        for o, r in zip(out, ref):
            assert torch.allclose(o, r)
    else:
        assert torch.is_tensor(out)
        assert torch.allclose(out, ref)

    # approximate backward points the same way as finite differences
    grad, = torch.autograd.grad(loss(out), vv)
    assert grad.shape == v.shape
    fd = _fd_grad(lambda x: loss(spatial.shoot(x, **opt)), v)
    cos = (grad * fd).sum() / (grad.norm() * fd.norm())
    assert cos > 0.8, cos.item()
