"""Tests for `optimal_svf` (used by `nitorch vopt`)."""
import torch
import pytest
from nitorch.spatial._svf_optimal import optimal_svf, svf_labels

SHAPE = (4, 4, 4)


def _subjects(seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn([3, *SHAPE, 3], generator=g, dtype=torch.double)
    return z - z.mean(0)  # zero-centred "template-to-subject" fields


def _one_direction(z):
    # pairs (i, j) -> v_ij = z_i - z_j, given in one direction only
    return {(0, 1): z[0] - z[1], (1, 2): z[1] - z[2], (0, 2): z[0] - z[2]}


def _both_directions(svf):
    out = dict(svf)
    out.update({(b, a): -v for (a, b), v in svf.items()})
    return out


def _ones(svf):
    return {k: torch.ones([*SHAPE, 1], dtype=torch.double) for k in svf}


@pytest.mark.parametrize("weighted", [False, True])
def test_one_direction_matches_both(weighted):
    svf = _one_direction(_subjects())
    both = _both_directions(svf)
    w1 = _ones(svf) if weighted else None
    w2 = _ones(both) if weighted else None
    z1 = optimal_svf(svf, w1, dtype=torch.double)
    z2 = optimal_svf(both, w2, dtype=torch.double)
    assert z1.shape == (3, *SHAPE, 3)
    assert torch.allclose(z1.double(), z2.double(), atol=1e-5)


@pytest.mark.parametrize("weighted", [False, True])
def test_consistent_pairs_are_reproduced(weighted):
    z = _subjects(1)
    svf = _one_direction(z)
    # consistent system: v_01 + v_12 == v_02
    assert torch.allclose(svf[(0, 1)] + svf[(1, 2)], svf[(0, 2)])
    w = _ones(svf) if weighted else None
    opt = optimal_svf(svf, w, dtype=torch.double).double()
    assert torch.allclose(opt, z, atol=1e-3)
    for (i, j), v in svf.items():
        assert torch.allclose(opt[i] - opt[j], v, atol=1e-3)


def test_mixed_weighted_unweighted_pairs():
    svf = _one_direction(_subjects(2))
    full = _ones(svf)
    partial = {k: v for k, v in full.items() if k != (0, 2)}
    z_full = optimal_svf(svf, full, dtype=torch.double)
    z_part = optimal_svf(svf, partial, dtype=torch.double)
    # missing weight is filled with the mean weight (== 1 here)
    assert torch.allclose(z_part, z_full, atol=1e-6)


def test_string_labels_deterministic_order():
    z = _subjects(3)
    names = ['b', 'c', 'a']  # z[0] is 'b', z[1] is 'c', z[2] is 'a'
    svf = {(names[i], names[j]): v
           for (i, j), v in _one_direction(z).items()}
    assert svf_labels(svf) == ['a', 'b', 'c']
    opt = optimal_svf(svf, dtype=torch.double).double()
    for n, name in enumerate(['a', 'b', 'c']):
        assert torch.allclose(opt[n], z[names.index(name)], atol=1e-3)


def test_svf_labels_order():
    assert svf_labels({(2, 0): 0, (1, 0): 0}) == [0, 1, 2]
    # non-comparable labels: order of first appearance
    assert svf_labels({('x', 1): 0, (0, 'y'): 0}) == ['x', 1, 0, 'y']


def test_no_stray_prints(capsys):
    optimal_svf(_one_direction(_subjects()), dtype=torch.double)
    assert capsys.readouterr().out == ''
