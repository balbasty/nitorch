"""Tests for `io.savef` dispatch across formats (part of #171)."""
import numpy as np
import pytest
import torch

from nitorch import io


def _affine():
    aff = torch.eye(4, dtype=torch.float64)
    aff[:3, :3] = torch.tensor([[0.9, 0.1, 0.0],
                                [-0.1, 0.9, 0.0],
                                [0.0, 0.0, 1.1]], dtype=torch.float64)
    aff[:3, 3] = torch.tensor([1.5, -2.0, 3.25], dtype=torch.float64)
    return aff


@pytest.mark.parametrize('savef', [io.savef, io.transforms.savef])
def test_savef_affine_lta_roundtrip(tmp_path, savef):
    aff = _affine()
    fname = str(tmp_path / 'xfm.lta')
    savef(aff, fname)
    for loadf in (io.transforms.loadf, io.loadf):
        out = torch.as_tensor(loadf(fname), dtype=torch.float64)
        out = out.reshape(4, 4)
        assert torch.allclose(out, aff, atol=1e-5)


def test_savef_volume_to_lta_raises_clear_error(tmp_path):
    vol = torch.rand(5, 6, 7)
    fname = str(tmp_path / 'vol.lta')
    with pytest.raises(ValueError, match='4x4'):
        io.savef(vol, fname)


def test_savef_nifti_still_works(tmp_path):
    vol = torch.rand(5, 6, 7)
    fname = str(tmp_path / 'vol.nii.gz')
    io.savef(vol, fname)
    out = io.loadf(fname)
    assert tuple(out.shape) == (5, 6, 7)
    assert np.allclose(np.asarray(out), vol.numpy(), atol=1e-5)
