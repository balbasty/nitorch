import torch

from nitorch.spatial import compute_fov


def test_compute_fov_single_vs_batched():
    mat = torch.eye(4, dtype=torch.double)
    aff = torch.eye(4, dtype=torch.double)
    aff[:3, -1] = torch.as_tensor([2., -3., 5.])
    shape = torch.as_tensor([10, 20, 30])

    mn_batch, mx_batch = compute_fov(mat, aff[None], shape[None])
    mn_single, mx_single = compute_fov(mat, aff, shape)
    assert torch.allclose(mn_single, mn_batch)
    assert torch.allclose(mx_single, mx_batch)
    # corners (1, 1, 1) and (10, 20, 30), shifted by the translation
    assert torch.allclose(mn_batch, torch.as_tensor([3., -2., 6.]).double())
    assert torch.allclose(mx_batch, torch.as_tensor([12., 17., 35.]).double())


def test_compute_fov_shared_shape():
    mat = torch.eye(4, dtype=torch.double)
    affs = torch.eye(4, dtype=torch.double).repeat(2, 1, 1)
    affs[1, :3, -1] = 10
    shape = torch.as_tensor([4, 4, 4])
    mn, mx = compute_fov(mat, affs, shape)
    assert torch.allclose(mn, torch.full([3], 1.).double())
    assert torch.allclose(mx, torch.full([3], 14.).double())


def test_fov_max_single_space_is_identity():
    """The bounding box of a single space, in that space, is itself:
    it used to be shifted by one voxel (and could gain a voxel through
    round-off errors)."""
    from nitorch.spatial import fov_max, mean_space
    mat = torch.eye(4, dtype=torch.double)
    shape = (10, 12, 1)
    out, oshape = fov_max(mat, mat[None], [shape])
    assert oshape == shape
    assert torch.allclose(out, mat)

    aff = torch.as_tensor([[0.8, 0, 0, -30.], [0, 1.2, 0, 10.],
                           [0, 0, 2., 5.], [0, 0, 0, 1]], dtype=torch.double)
    shape = (20, 16, 8)
    out, oshape = mean_space(aff[None], [shape])
    assert oshape == shape
    assert torch.allclose(out, aff, atol=1e-3)  # affine_mean is approximate
