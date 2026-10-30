import pytest
import torch

from nitorch.spatial import mind, rmind


@pytest.mark.parametrize('fn', [mind, rmind])
def test_mind_2d_channels_are_independent(fn):
    # the variance normalization of each channel should only depend on
    # that channel (i.e., average over spatial dimensions only)
    torch.manual_seed(0)
    x = torch.rand(2, 16, 16, dtype=torch.double)
    x[1] *= 100
    both = fn(x, dim=2)
    first = fn(x[:1], dim=2)
    assert torch.allclose(both[:1], first)


@pytest.mark.parametrize('fn', [mind, rmind])
def test_mind_2d_without_channel(fn):
    x = torch.rand(16, 16, dtype=torch.double)
    assert fn(x, dim=2).shape[:2] == (16, 16)


def test_rmind_radius_too_small():
    x = torch.rand(1, 8, 8, 8, dtype=torch.double)
    with pytest.raises(ValueError, match='two rings'):
        rmind(x, radius=1, dim=3)
    assert rmind(x, radius=1.5, dim=3).shape[-1] == 1
