import torch

from nitorch.tools.registration.objects import Image


def test_center_of_mass_applies_affine_and_is_intensity_invariant():
    # get_center_of_mass() previously computed the centroid in voxel space
    # and returned it directly, without ever applying the image's affine
    # (despite its own docstring: "Compute the RAS coordinate of the
    # center of mass") -- and divided by voxel/mask count instead of total
    # intensity mass, so the result also scaled with the image's arbitrary
    # intensity units instead of being a true weighted average. Found while
    # debugging a `nitorch register -i com` initialization that produced a
    # translation in the hundreds of thousands of mm.
    shape = (10, 10, 10)
    dat = torch.zeros(1, *shape)
    dat[0, 7, 2, 5] = 1.0

    affine = torch.eye(4)
    affine[:3, :3] = torch.diag(torch.tensor([2., 1.5, 1.]))
    affine[:3, 3] = torch.tensor([10., -5., 3.])

    img = Image(dat, affine=affine)
    com = img.get_center_of_mass(masked=False)

    voxel = torch.tensor([7., 2., 5.])
    expected = affine[:3, :3] @ voxel + affine[:3, 3]
    assert torch.allclose(com, expected, atol=1e-4)

    # a center of MASS (weighted average) must be invariant to uniformly
    # rescaling every intensity -- the old voxel/mask-count denominator was
    # not scale invariant
    img_scaled = Image(dat * 1000, affine=affine)
    com_scaled = img_scaled.get_center_of_mass(masked=False)
    assert torch.allclose(com_scaled, com, atol=1e-4)


def test_center_of_mass_respects_mask():
    shape = (10, 10, 10)
    dat = torch.zeros(1, *shape)
    dat[0, 1, 1, 1] = 1.0
    dat[0, 8, 8, 8] = 1.0
    mask = torch.zeros(1, *shape, dtype=torch.bool)
    mask[0, 1, 1, 1] = True

    img = Image(dat, affine=torch.eye(4), mask=mask)
    com = img.get_center_of_mass(masked=True)
    assert torch.allclose(com, torch.tensor([1., 1., 1.]), atol=1e-4)
