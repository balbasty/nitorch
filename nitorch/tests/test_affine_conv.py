"""Numeric ground truth for `spatial.affine_conv` (incl. dilation).

A single bright voxel is placed at a known input index; the signal is
convolved/pooled; the location of the peak in the output is mapped
through the output affine and must land on the same world position as
the input peak.
"""
import itertools
import pytest
import torch
import torch.nn.functional as F
from nitorch import spatial


def _affine(ndim):
    # non-trivial voxel size, rotation-free but shifted/scaled
    aff = torch.eye(ndim + 1, dtype=torch.double)
    for d in range(ndim):
        aff[d, d] = 0.7 + 0.3 * d
        aff[d, -1] = -5.0 + 2.0 * d
    return aff


def _world(aff, idx):
    idx = torch.as_tensor(idx, dtype=torch.double)
    return aff[:-1, :-1] @ idx + aff[:-1, -1]


def _peak(x):
    return list(map(int, torch.nonzero(x == x.max())[0]))


# (K, S, D, (Plo, Phi))
CASES = [
    (3, 1, 1, (1, 1)),
    (3, 2, 1, (0, 0)),
    (3, 1, 2, (2, 2)),
    (3, 2, 2, (0, 0)),
    (5, 1, 3, (1, 4)),
    (3, 3, 4, (2, 0)),
    (1, 2, 1, (0, 1)),
    (5, 2, 2, (3, 1)),
]


def _delta_input(L, K, S, D, Plo):
    # choose an input index that is the centre of output window j=2
    x = 2 * S - Plo + D * (K - 1) // 2
    assert 0 <= x < L
    return x


@pytest.mark.parametrize('K,S,D,P', CASES)
def test_affine_conv_peak_1d(K, S, D, P):
    """F.conv1d with a centre-tap kernel + explicit asymmetric padding."""
    L = 32
    Plo, Phi = P
    x = _delta_input(L, K, S, D, Plo)
    inp = torch.zeros(1, 1, L, dtype=torch.double)
    inp[..., x] = 1
    w = torch.zeros(1, 1, K, dtype=torch.double)
    w[..., K // 2] = 1      # centre tap (K odd)
    out = F.conv1d(F.pad(inp, (Plo, Phi)), w, stride=S, dilation=D)
    aff = _affine(1)
    oaff, oshape = spatial.affine_conv(aff, [L], K, S, [P], dilation=D)
    assert tuple(out.shape[-1:]) == tuple(oshape)
    j = _peak(out[0, 0])
    assert torch.allclose(_world(oaff, j), _world(aff, [x]))


@pytest.mark.parametrize('K,S,D,P', CASES)
@pytest.mark.parametrize('reduction', ['mean', 'median'])
def test_affine_conv_pool_3d(K, S, D, P, reduction):
    """spatial.pool(affine=...) (the code path that calls affine_conv).

    Pooling a single bright voxel is ambiguous (it spreads over several
    windows), so the ground truth here is a coordinate ramp: the mean
    (or median) of input indices over a window is the index of the
    window's centre, which the output affine must map to the same world
    position as the output voxel. Only windows that lie fully inside
    the input are checked (padding values are not coordinates).
    """
    L = 20
    Plo, Phi = P
    aff = _affine(3)
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(L, dtype=torch.double)] * 3, indexing='ij'))
    padding = [P] * 3 if Plo != Phi else [Plo] * 3
    out, oaff = spatial.pool(3, grid, K, stride=S, dilation=D,
                             padding=padding, reduction=reduction,
                             affine=aff)
    # indices of output voxels whose window is fully inside the input
    j = torch.arange(out.shape[-1], dtype=torch.double)
    first = j * S - Plo
    ok = (first >= 0) & (first + D * (K - 1) <= L - 1)
    assert ok.any()
    jj = j[ok]
    for a, b, c in itertools.product(jj[:3], jj[:3], jj[:3]):
        ijk = [int(a), int(b), int(c)]
        centre = out[(slice(None), *ijk)]        # input index of centre
        assert torch.allclose(_world(oaff, ijk), _world(aff, centre))


@pytest.mark.parametrize('K,S,D,P', [c for c in CASES if c[3][0] == c[3][1]])
def test_affine_conv_peak_transposed_1d(K, S, D, P):
    """F.conv_transpose1d with a centre-tap kernel."""
    L = 12
    P = P[0]
    j = 3
    inp = torch.zeros(1, 1, L, dtype=torch.double)
    inp[..., j] = 1
    w = torch.zeros(1, 1, K, dtype=torch.double)
    w[..., K // 2] = 1
    out = F.conv_transpose1d(inp, w, stride=S, padding=P, dilation=D)
    aff = _affine(1)
    oaff, oshape = spatial.affine_conv(aff, [L], K, S, P, dilation=D,
                                       transposed=True)
    assert tuple(out.shape[-1:]) == tuple(oshape)
    i = _peak(out[0, 0])
    assert torch.allclose(_world(oaff, [i]), _world(aff, [j]))


def test_spatial_lattice_conv_keywords():
    """SpatialLattice.conv used to pass `transposed` as `dilation`."""
    from nitorch.spatial._layout import SpatialLattice
    lat = SpatialLattice([10, 10], torch.eye(3, dtype=torch.double))
    out = lat.conv(3, 1)
    ref_aff, ref_shape = spatial.affine_conv(lat.affine, lat.shape, 3, 1)
    assert tuple(out.shape) == (8, 8) == tuple(ref_shape)
    assert torch.allclose(out.affine, ref_aff)
    assert tuple(lat.conv(3).shape) == (8, 8)
