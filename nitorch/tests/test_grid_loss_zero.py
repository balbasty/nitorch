import torch

from nitorch import nn


def test_grid_loss_all_penalties_disabled():
    zero = dict(absolute=0, membrane=0, bending=0, lame=0)
    # dense displacement field
    v = torch.randn(1, 8, 8, 8, 3, dtype=torch.double)
    loss = nn.GridLoss(**zero)(v)
    assert torch.is_tensor(loss) and loss.item() == 0
    # vector of affine parameters (as used by autoreg's linear transforms)
    q = torch.randn(12, dtype=torch.double)
    loss = nn.GridLoss(**zero)(q)
    assert torch.is_tensor(loss) and loss.item() == 0
