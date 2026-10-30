import torch

from nitorch import nn, spatial
from nitorch.cli.registration.autoreg import struct


def energy(fn, v):
    return fn(v).mul(v).sum(-1).mean()


def test_grid_loss_linear_elastic_factors():
    torch.manual_seed(0)
    v = torch.randn(1, 8, 8, 8, 3, dtype=torch.double)
    zero = dict(absolute=0, membrane=0, bending=0)

    div = energy(spatial.lame_div, v)
    shear = energy(spatial.lame_shear, v)
    loss = nn.GridLoss(**zero, lame=(0.05, 0.2))(v)
    assert torch.allclose(loss, 0.05 * div + 0.2 * shear)

    # the linear-elastic terms must not depend on the membrane factor
    loss_mem = nn.GridLoss(**dict(zero, membrane=1e-3), lame=(0.05, 0.2))(v)
    mem = energy(spatial.membrane_grid, v)
    assert torch.allclose(loss_mem, 1e-3 * mem + 0.05 * div + 0.2 * shear)


def test_autoreg_linear_elastic_loss():
    torch.manual_seed(0)
    v = torch.randn(1, 8, 8, 8, 3, dtype=torch.double)
    div = energy(spatial.lame_div, v)
    shear = energy(spatial.lame_shear, v)
    loss = struct.LinearElasticLoss(factor=(0.05, 0.2)).call(v)
    assert torch.allclose(loss, 0.05 * div + 0.2 * shear)
