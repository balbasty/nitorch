import math
import os
import time
import numpy as np
import nibabel as nib
import torch
from nitorch.tools.registration.losses.mse import (
    MSE, weighted_precision, PRIOR_FRACTION, PRIOR_DOF)
from nitorch.tools.registration.optim.wolfe import StrongWolfe
from nitorch.tools.registration.optim.base import Optim
from nitorch.cli.registration.register.cli import _cli
from nitorch import io


def _rand(shape=(1, 16, 16, 16), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g, dtype=torch.float64)


def test_precision_closed_form():
    fix, mov = _rand(), _rand(seed=1)
    w = (_rand(seed=2) > 0.3).double()
    lam = weighted_precision(mov, fix, weights=w, dim=3)
    n = w.sum()
    s = ((mov - fix).square() * w).sum()
    mean = (fix * w).sum() / n
    s0 = PRIOR_FRACTION * ((fix - mean).square() * w).sum() / n
    expected = (PRIOR_DOF + n) / (PRIOR_DOF * s0 + s)
    assert torch.allclose(lam, expected.reshape(1), rtol=1e-12)
    # weighted-average form
    assert torch.allclose(1 / lam, ((PRIOR_DOF * s0 + n * (s / n))
                                    / (PRIOR_DOF + n)).reshape(1))
    # explicit s0^2, per channel
    lam2 = weighted_precision(mov.expand(2, 16, 16, 16), fix, dim=3,
                              prior_variance=[0.1, 0.2])
    s2 = (mov - fix).square().sum()
    exp2 = (PRIOR_DOF + 16**3) / (PRIOR_DOF * torch.tensor([0.1, 0.2],
                                  dtype=torch.double) + s2)
    assert torch.allclose(lam2, exp2)


def test_precision_prior_vanishes_with_n():
    rel = []
    for n in (8, 16, 32, 64):
        fix, mov = _rand((1, n, n, n)), _rand((1, n, n, n), seed=1)
        lam = weighted_precision(mov, fix, dim=3)
        ml = 1 / (mov - fix).square().mean()
        rel.append(abs(lam / ml - 1).item())
    assert all(a > b for a, b in zip(rel, rel[1:]))
    assert rel[-1] < 1e-4


def test_precision_identical_images():
    # one voxel's worth of data: lam = (nu0 + 1) / (nu0 s0^2) ~= 1/s0^2
    fix = _rand((1, 2, 2, 2))
    w = torch.zeros_like(fix); w[..., 0, 0, :] = 0.5
    lam = weighted_precision(fix, fix, weights=w, dim=3, prior_variance=0.01)
    assert torch.allclose(lam, torch.tensor([(PRIOR_DOF + 1) / PRIOR_DOF / 0.01],
                                            dtype=lam.dtype))
    # in general, finite and (nu0 + N) / (nu0 s0^2)
    lam = weighted_precision(fix, fix, dim=3)
    mean_ = fix.mean()
    s0 = PRIOR_FRACTION * (fix - mean_).square().mean()
    assert torch.allclose(lam, ((PRIOR_DOF + 8) / (PRIOR_DOF * s0)).reshape(1))
    assert torch.isfinite(weighted_precision(fix * 0, fix * 0, dim=3)).all()


def test_mse_estimated_lam_finite():
    fix = _rand((1, 8, 8, 8)).float()
    for mov, mask in [(fix.clone(), None),
                      (fix + 1e-7, (fix > 0.9).float())]:
        cube = torch.zeros_like(fix); cube[..., 2:6, 2:6, 2:6] = 1
        for f, m in [(fix, mov), (cube, cube.clone())]:
            out = MSE().loss_grad_hess(m, f, mask=mask if f is fix else cube)
            for o in out:
                assert torch.isfinite(torch.as_tensor(o)).all()


def test_wolfe_backtracks_on_nan():
    # f(x) = (x-1)^2 along delta=+1, NaN for steps > 1.5
    def closure(a, grad=False):
        a = float(a)
        f = torch.tensor(float('nan') if a > 1.5 else (a - 1) ** 2)
        g = torch.tensor([float('nan') if a > 1.5 else 2 * (a - 1)])
        return (f, g) if grad else f
    f0, g0 = closure(0, grad=True)
    a, f, g = StrongWolfe(lr=8).step(0, closure, torch.ones(1),
                                     loss=f0, grad=g0)
    assert math.isfinite(f) and f < f0 and 0 < a <= 1.5


def test_plain_step_rejects_nonfinite():
    class Bad(Optim):
        def search_direction(self, *d):
            return torch.full([2], float('inf'))
    p = torch.zeros(2)
    p2, loss = Bad().step(p.clone(), lambda x: x.square().sum())
    assert torch.equal(p2, p) and math.isfinite(loss)


def _cube(path, noise=False):
    dat = np.zeros([16] * 3, dtype=np.float32)
    dat[4:12, 4:12, 4:12] = 1
    nib.save(nib.Nifti1Image(dat, np.eye(4)), path)
    return path


def _run(tmp_path, extra):
    fix = _cube(str(tmp_path / 'fix.nii'))
    mov = _cube(str(tmp_path / 'mov.nii'))
    out = str(tmp_path / 'aff.lta')
    t = time.time()
    _cli(['-o', str(tmp_path), '@loss', 'mse', '@@fix', fix, '@@mov', mov,
          '@affine', 'translation', '-o', out, *extra])
    assert time.time() - t < 60
    aff = io.transforms.map(out).fdata()
    assert torch.isfinite(torch.as_tensor(aff)).all()
    assert torch.allclose(torch.as_tensor(aff, dtype=torch.double)[..., :3, -1],
                          torch.zeros(3, dtype=torch.double), atol=1e-3)


def test_register_identical_cubes(tmp_path):
    _run(tmp_path, ['@pyramid', '--levels', '1'])


def test_register_identical_cubes_pyramid(tmp_path):
    _run(tmp_path, ['@pyramid', '-l', '0:2'])
