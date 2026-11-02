"""Reductions that used to accumulate in double now sum in the native
dtype and cast the reduced result. Check that float32 results stay
close to the float64 ones."""
import torch
from nitorch.core import math
from nitorch.tools.registration.losses.utils_gmm2 import sumspatial


def test_softmax_lse_float32_matches_double():
    g = torch.Generator().manual_seed(0)
    x = torch.randn([4, 64, 64, 64], generator=g, dtype=torch.double) * 3
    w = torch.rand([64, 64, 64], generator=g, dtype=torch.double)
    _, ref = math.softmax_lse(x, dim=0, lse=True, weights=w)
    _, lse = math.softmax_lse(x.float(), dim=0, lse=True, weights=w.float())
    assert lse.dtype == torch.double
    assert abs(lse.item() - ref.item()) <= 1e-6 * abs(ref.item())


def test_sumspatial_float32_matches_double():
    g = torch.Generator().manual_seed(0)
    x = torch.rand([2, 96, 96, 96, 3], generator=g, dtype=torch.double)
    ref = sumspatial(x, 3)
    out = sumspatial(x.float(), 3)
    assert out.dtype == torch.float32
    assert torch.allclose(out.double(), ref, rtol=1e-6, atol=0)
