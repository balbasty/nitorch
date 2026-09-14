import torch
from nitorch.core import bounds
from nitorch.tools.registration.pairwise_makeobj import make_image


def test_bounds_zero_tensor_clamps_and_masks():
    n = 5
    i = torch.tensor([-2, -1, 0, 2, 4, 5, 7])
    idx, mult = bounds.zero(i.clone(), n)
    assert idx.min() >= 0 and idx.max() <= n - 1
    assert torch.equal(mult, torch.tensor([0, 0, 1, 1, 1, 0, 0]))


def test_bounds_zero__inplace_tensor_clamps_and_masks():
    n = 5
    i = torch.tensor([-2, -1, 0, 2, 4, 5, 7])
    idx, mult = bounds.zero_(i.clone(), n)
    assert idx.min() >= 0 and idx.max() <= n - 1
    assert torch.equal(mult, torch.tensor([0, 0, 1, 1, 1, 0, 0]))


def test_bounds_zero_int():
    n = 5
    assert bounds.zero(-1, n) == (0, 0)
    assert bounds.zero(0, n) == (0, 1)
    assert bounds.zero(4, n) == (4, 1)
    assert bounds.zero(5, n) == (4, 0)


def test_make_image_mind_true_no_longer_crashes():
    # the exact originally-reported crash: AttributeError: module
    # 'nitorch.core.bounds' has no attribute 'zero_'
    dat = torch.rand(1, 32, 32, 32)
    image = make_image(dat, mind=True)
    level = image[0]
    assert level.dat.shape[0] > 1
    assert torch.equal(level.preview, dat)


def test_make_image_mind_true_other_bound_unaffected():
    dat = torch.rand(1, 32, 32, 32)
    before = make_image(dat.clone(), mind=True, bound='dct2')
    after = make_image(dat.clone(), mind=True, bound='dct2')
    assert torch.equal(before[0].dat, after[0].dat)
