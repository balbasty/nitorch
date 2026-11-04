"""Backpropagation through `spatial.shoot` (issue #171).

With the TorchScript backend, `grid_pull` returns a view created inside
a custom autograd Function, which `shoot` used to modify in place, so
any backward pass through `shoot` failed.
"""
import itertools
import pytest
import torch
from nitorch.spatial import shoot

FLAGS = list(itertools.product([False, True], repeat=3))


def _ids(flags):
    ri, disp, fast = flags
    return f"inv={int(ri)}-disp={int(disp)}-fast={int(fast)}"


def _loss(out, return_inverse):
    outs = out if return_inverse else (out,)
    # pseudo-random weights, so that the loss depends on every output
    loss = 0
    for k, o in enumerate(outs):
        w = torch.arange(o.numel(), dtype=o.dtype).reshape(o.shape)
        loss = loss + (o * (w + k).sin()).sum() + o.square().sum()
    return loss


@pytest.mark.parametrize("shape", [(6, 5, 2), (5, 4, 6, 3), (2, 6, 5, 2)],
                         ids=["2d", "3d", "2d-batch"])
@pytest.mark.parametrize("flags", FLAGS, ids=_ids)
def test_shoot_backward_runs(shape, flags):
    return_inverse, displacement, fast = flags
    torch.manual_seed(0)
    vel = torch.randn(shape, dtype=torch.double).mul_(0.5)
    vel.requires_grad_()
    out = shoot(vel, steps=3, return_inverse=return_inverse,
                displacement=displacement, fast=fast)
    _loss(out, return_inverse).backward()
    assert vel.grad is not None
    assert vel.grad.shape == vel.shape
    assert torch.isfinite(vel.grad).all()
    assert vel.grad.abs().sum() > 0

    # values with and without autograd are bitwise identical
    with torch.no_grad():
        ref = shoot(vel.detach(), steps=3, return_inverse=return_inverse,
                    displacement=displacement, fast=fast)
    out = out if return_inverse else (out,)
    ref = ref if return_inverse else (ref,)
    for o, r in zip(out, ref):
        assert torch.equal(o.detach(), r)


@pytest.mark.parametrize("flags", [(True, False, True)],
                         ids=_ids)
def test_shoot_gradcheck(flags):
    return_inverse, displacement, fast = flags
    torch.manual_seed(1)
    vel = torch.randn([6, 5, 2], dtype=torch.double).mul_(0.3)
    vel.requires_grad_()

    def fn(v):
        return shoot(v, steps=2, return_inverse=return_inverse,
                     displacement=displacement, fast=fast)

    assert torch.autograd.gradcheck(fn, (vel,), eps=1e-6, atol=1e-5)
