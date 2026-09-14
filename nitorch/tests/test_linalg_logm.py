import math

import torch

from nitorch.core._linalg_logm import logm
from nitorch.core.linalg import meanm


def test_logm_matches_known_value():
    # logm(exp(x) * I) == x * I -- avoids depending on scipy's exact
    # internal algorithm while still exercising the real scipy.linalg.logm
    # call path (nitorch.core._linalg_logm._scipy_logm). Found broken with
    # scipy>=1.18, which dropped logm()'s `disp` kwarg and stopped
    # returning a (result, errest) tuple: nitorch's `MeanSpace` (used by
    # any `nitorch register @nonlin` stage) called
    # `scipy.linalg.logm(x, disp=False)[0]` unconditionally and crashed
    # with `TypeError: logm() got an unexpected keyword argument 'disp'`.
    mat = torch.eye(3, dtype=torch.double)[None] * math.e
    out = logm(mat)
    expected = torch.eye(3, dtype=torch.double)[None]
    assert torch.allclose(out, expected, atol=1e-6)


def test_meanm_of_identical_matrices_is_itself():
    # meanm() calls logm() internally (via affine_mean) -- this is the
    # exact call path used by nitorch.tools.registration.objects.MeanSpace,
    # which every `@nonlin` registration stage constructs.
    mat = torch.eye(4, dtype=torch.double) * 2
    mat[-1, -1] = 1
    mats = torch.stack([mat, mat])
    out = meanm(mats)
    assert torch.allclose(out, mat, atol=1e-5)
