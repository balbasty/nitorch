import numpy as np
import pytest
import torch
import scipy.linalg
from scipy.spatial.transform import Rotation

from nitorch.core.linalg import sqrtm


def _affines(n=20, seed=0):
    rng = np.random.default_rng(seed)
    mats = []
    i = -1
    while len(mats) < n:
        i += 1
        A = np.eye(4)
        if i % 4 == 0:
            # large rotation close to 180 degrees
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            # (with isotropic scaling, else eigenvalues may become
            #  real negative and no real square root exists)
            R = Rotation.from_rotvec(axis * np.deg2rad(179)).as_matrix()
            S = np.eye(3) * rng.uniform(0.5, 2)
            Sh = np.eye(3)
        else:
            R = Rotation.random(random_state=rng).as_matrix()
            S = np.diag(rng.uniform(0.5, 2, 3))
            Sh = np.eye(3)
            Sh[np.triu_indices(3, 1)] = rng.uniform(-0.3, 0.3, 3)
        A[:3, :3] = R @ S @ Sh
        A[:3, 3] = rng.normal(scale=20, size=3)
        ev = np.linalg.eigvals(A)
        if np.any((np.abs(ev.imag) < 1e-6) & (ev.real < 0)):
            # anisotropy + large rotation can create real negative
            # eigenvalues: no real square root, skip.
            continue
        mats.append(A)
    return torch.as_tensor(np.stack(mats), dtype=torch.float64)


def test_sqrtm_square_and_scipy():
    A = _affines()
    S = sqrtm(A)
    assert S.dtype == A.dtype and S.shape == A.shape
    res = (S @ S - A).abs().amax(dim=(-1, -2)) / A.abs().amax(dim=(-1, -2))
    assert res.max() < 1e-12
    ref = np.stack([scipy.linalg.sqrtm(a).real for a in A.numpy()])
    assert np.abs(S.numpy() - ref).max() < 1e-8
    # homogeneous last row preserved exactly
    assert (S[:, -1, :] == torch.tensor([0., 0., 0., 1.])).all()


def test_sqrtm_float32_and_batch_shape():
    A = _affines(6).reshape(2, 3, 4, 4).float()
    S = sqrtm(A)
    assert S.dtype == torch.float32 and S.shape == (2, 3, 4, 4)
    assert torch.allclose(S @ S, A, atol=1e-4)


def test_sqrtm_reflection_raises():
    A = torch.eye(4, dtype=torch.float64)
    A[0, 0] = -1
    with pytest.raises(ValueError):
        sqrtm(A)
