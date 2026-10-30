import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from nitorch.core.linalg import logm, expm


def _random_affines(n=4, seed=0):
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n):
        A = np.eye(4)
        A[:3, :3] = (Rotation.random(random_state=rng).as_matrix()
                     @ np.diag(rng.uniform(0.8, 1.2, 3)))
        A[:3, 3] = rng.normal(size=3)
        mats.append(A)
    return torch.as_tensor(np.stack(mats), dtype=torch.float64)


def _random_rotations(n=4, seed=1):
    R = Rotation.random(n, random_state=seed).as_matrix()
    return torch.as_tensor(R, dtype=torch.float64)


@pytest.mark.parametrize('kind', ['rotation', 'affine'])
def test_expm_logm_roundtrip_batched(kind):
    A = _random_rotations() if kind == 'rotation' else _random_affines()
    L = logm(A)
    assert L.shape == A.shape and not L.is_complex()
    assert torch.allclose(expm(L), A, atol=1e-6)


def test_logm_backward():
    A = _random_affines(1)[0].requires_grad_()
    logm(A).sum().backward()
    assert torch.isfinite(A.grad).all()


def test_register_nonlin_svf_runs(tmp_path):
    nib = pytest.importorskip('nibabel')
    rng = np.random.default_rng(0)
    x = np.zeros((16, 16, 16), dtype=np.float32)
    x[4:12, 4:12, 4:12] = 1
    y = np.roll(x, 1, axis=0) + 0.01 * rng.random(x.shape).astype(np.float32)
    fix, mov = str(tmp_path / 'fix.nii.gz'), str(tmp_path / 'mov.nii.gz')
    nib.save(nib.Nifti1Image(x, np.eye(4)), fix)
    nib.save(nib.Nifti1Image(y, np.eye(4)), mov)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=root)
    args = ['register', '@loss', 'mse', '@@fix', fix, '@@mov', mov,
            '@affine', 'rigid', '@nonlin', 'svf',
            '--cpu', '-v', '0', '-o', str(tmp_path)]
    code = ('import sys; from nitorch.cli import cli; '
            'sys.exit(cli(sys.argv[1:]))')
    res = subprocess.run([sys.executable, '-c', code] + args,
                         cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
