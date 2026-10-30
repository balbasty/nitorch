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
