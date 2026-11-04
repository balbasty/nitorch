"""Regression tests for `nitorch.nn.modules.spatial.Resize` (issue #171)."""
import pytest
import torch
from nitorch import spatial
from nitorch.nn.modules.spatial import Resize

ANCHORS = ['c', 'e', 'f', 'l']
INSHAPES = [(8, 8, 8), (8, 8), (7, 9)]


def _ref(x, **kw):
    return spatial.resize(x, interpolation='linear', bound='dct2',
                          extrapolate=True, prefilter=True, **kw)


@pytest.mark.parametrize('inshape', INSHAPES)
@pytest.mark.parametrize('anchor', ANCHORS)
@pytest.mark.parametrize('factor', [2, 0.5])
def test_factor_only_matches_spatial_resize(inshape, anchor, factor):
    x = torch.randn(1, 1, *inshape)
    y = Resize(factor=factor, anchor=anchor)(x)
    ref = _ref(x, factor=factor, anchor=anchor)
    assert y.shape == ref.shape
    assert torch.allclose(y, ref)
    assert Resize(factor=factor, anchor=anchor).shape(x) == tuple(ref.shape)


@pytest.mark.parametrize('inshape', INSHAPES)
@pytest.mark.parametrize('anchor', ANCHORS)
@pytest.mark.parametrize('factor', [None, 2])
def test_explicit_shape(inshape, anchor, factor):
    x = torch.randn(1, 1, *inshape)
    shape = [5] * len(inshape)
    y = Resize(factor=factor, shape=shape, anchor=anchor)(x)
    ref = _ref(x, factor=factor, shape=shape, anchor=anchor)
    assert tuple(y.shape) == (1, 1, *shape)
    assert torch.allclose(y, ref)


def test_forward_output_shape_override():
    x = torch.randn(1, 1, 8, 8)
    y = Resize(factor=2)(x, output_shape=[3, 3])
    assert tuple(y.shape) == (1, 1, 3, 3)


def test_str():
    assert str(Resize(shape=[4, 4])) == 'ResizeTo([4, 4], anchor=c)'
    assert str(Resize(factor=2)) == 'Upsample(2, anchor=c)'
