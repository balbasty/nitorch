"""`spatial.pool`/`spatial.conv`: every zero-padding alias takes torch's
native (implicit padding) path and gives the same result as explicit
padding."""
import pytest
import torch

from nitorch import spatial
from nitorch.core import utils

ALIASES = ['zero', 'zeros', 'constant', 'Zeros']


def _count_pads(monkeypatch):
    calls = []
    orig = utils.pad

    def pad(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(utils, 'pad', pad)
    return calls


@pytest.mark.parametrize('reduction', ['mean', 'max'])
@pytest.mark.parametrize('bound', ALIASES)
def test_pool_zero_aliases_native(monkeypatch, bound, reduction):
    torch.manual_seed(0)
    x = torch.randn(2, 7, 9)
    # reference: explicit-pad path (dilation forces our implementation)
    ref = spatial.pool(2, x, kernel_size=3, stride=2, padding=1,
                       bound='zero', reduction=reduction)
    calls = _count_pads(monkeypatch)
    out = spatial.pool(2, x, kernel_size=3, stride=2, padding=1,
                       bound=bound, reduction=reduction)
    assert not calls, 'zero-padding alias took the explicit-pad path'
    assert torch.allclose(out, ref)


@pytest.mark.parametrize('reduction', ['mean', 'max'])
def test_pool_native_matches_explicit(monkeypatch, reduction):
    torch.manual_seed(0)
    x = torch.randn(2, 7, 9)
    native = spatial.pool(2, x, kernel_size=3, stride=2, padding=1,
                          bound='constant', reduction=reduction)
    # force the explicit path by padding by hand with the same fill value
    fill = -float('inf') if reduction == 'max' else 0
    xp = utils.pad(x, [1, 1], 'constant', value=fill, side='both')
    explicit = spatial.pool(2, xp, kernel_size=3, stride=2, padding=0,
                            bound='zero', reduction=reduction)
    assert torch.allclose(native, explicit)


def test_pool_nonzero_bound_still_pads(monkeypatch):
    x = torch.randn(7, 9)
    calls = _count_pads(monkeypatch)
    spatial.pool(2, x, kernel_size=3, padding=1, bound='dct2')
    assert calls


@pytest.mark.parametrize('bound', ALIASES)
def test_conv_zero_aliases_native(monkeypatch, bound):
    torch.manual_seed(0)
    x = torch.randn(7, 9)
    k = torch.randn(3, 3)
    ref = spatial.conv(2, x, k, padding='same', bound='zero')
    calls = _count_pads(monkeypatch)
    out = spatial.conv(2, x, k, padding='same', bound=bound)
    assert not calls
    assert torch.allclose(out, ref)
