"""Intensity rescaling of the slices shown in the live registration plot
(`register -v 3+`)."""
import torch
from nitorch.tools.registration.pairwise_impl import _rescale2d


def test_empty_slice():
    # a slice that is entirely background (e.g., the central slice of a
    # masked image) used to crash `quantile` on an empty tensor
    x = torch.zeros(5, 6)
    y = _rescale2d(x)
    assert y.shape == x.shape
    assert (y == 0).all()


def test_empty_slice_int():
    y = _rescale2d(torch.zeros(5, 6, dtype=torch.int16))
    assert y.dtype.is_floating_point
    assert (y == 0).all()


def test_background_and_nan_excluded():
    x = torch.rand(32, 32) + 1
    x[:4] = 0
    x[5, 5] = float('nan')
    y = _rescale2d(x)
    assert y.shape == x.shape
    # background maps below the foreground range
    assert (y[:4] == 0).all()
    fg = y[8:]
    assert fg.min() == 0 and fg.max() == 1


def test_batch():
    x = torch.rand(3, 16, 16) + 1
    x[1, 2, 3] = 0
    x[2] *= 10
    y = _rescale2d(x)
    assert y.shape == x.shape
    for y1 in y:
        assert y1.min() == 0 and y1.max() == 1
