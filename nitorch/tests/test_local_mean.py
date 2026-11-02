"""Local means (LCC, LGMMH): shared mask normaliser (#158).

The normaliser of a masked local mean (conv(mask)) can be computed once
(`local_mean_norm`) and shared by all calls that use the same mask.
"""
import torch
import pytest
from nitorch.tools.registration.losses import utils_local as U
from nitorch.tools.registration.losses.cc import lcc

dt = torch.double


def _data(dim, n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand([2] + [n] * dim, dtype=dt, generator=g)
    y = torch.rand([2] + [n] * dim, dtype=dt, generator=g)
    mask = (torch.rand([n] * dim, dtype=dt, generator=g) > 0.3).to(dt)
    return x, y, mask


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('kernel_fwhm', [[6.], [6., 0.]])
def test_backward_is_adjoint(dim, kernel_fwhm):
    # <fwd(x), y> == <x, bwd(y)>, also with an unsmoothed axis
    x, y, mask = _data(dim)
    kernel_fwhm = kernel_fwhm[:dim]
    fwd = U.local_mean(x, kernel_fwhm, dim=dim, mask=mask)
    bwd = U.local_mean(y, kernel_fwhm, dim=dim, mask=mask, backward=True,
                       shape=list(x.shape[1:]))
    assert torch.allclose((fwd * y).sum(), (x * bwd).sum(), rtol=1e-12)


@pytest.mark.parametrize('mode,fwhm,stride', [
    ('g', [6.], [1]),
    ('g', [1.], [1]),
    ('g', [6.], [2]),       # strided
    ('s', [4.], [1]),       # square patches
    ('s', [4.], [2]),
])
@pytest.mark.parametrize('backward', [False, True])
def test_shared_norm_is_exact(mode, fwhm, stride, backward):
    x, _, mask = _data(3)
    opt = dict(dim=3, mode=mode, mask=mask)
    norm = U.local_mean_norm(mask, fwhm, stride, mode=mode, dim=3)
    if backward:
        y = U.local_mean(x, fwhm, stride, **opt)
        bopt = dict(opt, backward=True, shape=list(x.shape[1:]))
        a = U.local_mean(y, fwhm, stride, **bopt)
        b = U.local_mean(y, fwhm, stride, norm=norm, **bopt)
    else:
        a = U.local_mean(x, fwhm, stride, **opt)
        b = U.local_mean(x, fwhm, stride, norm=norm, **opt)
    assert torch.equal(a, b)


def test_local_mean_is_scriptable():
    # `utils_gmm2` calls `local_mean` from TorchScript
    scripted = torch.jit.script(U.local_mean)
    x, _, mask = _data(3)
    for fwhm in ([6.], [1.]):
        a = scripted(x.float(), fwhm, mask=mask.float())
        b = U.local_mean(x.float(), fwhm, mask=mask.float())
        assert torch.equal(a, b)


def test_exact_zeros_far_from_mask():
    # Far from the mask, conv(mask) is exactly 0 (then clamped): the
    # local means must not see rounding noise there (they divide by it)
    mask = torch.zeros([1, 64, 64, 64])
    mask[..., :8, :8, :8] = 1
    norm = U.local_mean_norm(mask, [8.], dim=3)
    assert norm.dtype == torch.float32
    assert (norm[..., 40:, 40:, 40:] == 1e-5).all()


@pytest.mark.parametrize('patch', [1., 6.])
def test_lcc_no_mask_is_all_ones_mask(patch):
    # `lcc` used to replace `mask=None` by ones (which disabled the cached
    # conv(ones)); it now keeps None: same value, gradient and Hessian
    x, y, _ = _data(3, seed=1)
    x, y = x[:1], y[:1]
    out0 = lcc(x, y, dim=3, patch=patch)
    out1 = lcc(x, y, dim=3, patch=patch, mask=torch.ones_like(x[0]))
    for a, b in zip(out0, out1):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)
