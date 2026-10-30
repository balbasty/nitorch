"""`nitorch.core.utils.quantile(..., mask=)` must only use in-mask values."""
import torch
import pytest
from nitorch.core import utils

qs = [0., 0.005, 0.25, 0.5, 0.77, 1.]


def make(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, dtype=torch.double, generator=g)
    return x


def reference(x, q, mask):
    """torch.quantile on in-mask values, per batch row (last 2 dims)."""
    x = x.reshape([-1, x.shape[-2] * x.shape[-1]])
    mask = mask.expand(*x.shape[:-1], -1) if mask.dim() == 1 else \
        mask.reshape(x.shape)
    q = torch.as_tensor(q, dtype=x.dtype)
    return torch.stack([torch.quantile(x1[m1 > 0], q)
                        for x1, m1 in zip(x, mask)])


@pytest.mark.parametrize('where', ['min', 'max', 'both', 'random'])
def test_quantile_mask_excludes_extremes(where):
    x = make([3, 6, 7])
    flat = x.reshape([3, -1])
    mask = torch.ones_like(flat)
    if where in ('min', 'both'):
        mask.scatter_(-1, flat.argsort(-1)[:, :5], 0)
    if where in ('max', 'both'):
        mask.scatter_(-1, flat.argsort(-1)[:, -5:], 0)
    if where == 'random':
        g = torch.Generator().manual_seed(1)
        mask = (torch.rand(mask.shape, generator=g) > 0.4).double()
    # put outliers in the masked-out voxels
    flat = flat.clone()
    flat[mask == 0] = 1e3 * (torch.arange((mask == 0).sum()) % 2 * 2 - 1).double()
    x = flat.reshape(x.shape)
    mask = mask.reshape(x.shape)

    out = utils.quantile(x, qs, dim=[-2, -1], mask=mask)
    ref = reference(x, qs, mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=1e-12), (out, ref)


def test_quantile_mask_keepdim_and_dims():
    x = make([2, 5, 4, 3])
    mask = (make([2, 5, 4, 3], seed=3) > 0.3).double()
    out = utils.quantile(x, qs, dim=[1, 2], keepdim=True, mask=mask)
    assert out.shape == (2, 1, 1, 3, len(qs))
    xm = x.movedim(-1, 1)            # (2, 3, 5, 4)
    mm = mask.movedim(-1, 1)
    ref = reference(xm, qs, mm).reshape(2, 3, len(qs))
    assert torch.allclose(out[:, 0, 0], ref, atol=1e-12)


def test_quantile_mask_all_ones_matches_unmasked():
    x = make([4, 9])
    out = utils.quantile(x, qs, dim=-1, mask=torch.ones_like(x))
    ref = utils.quantile(x, qs, dim=-1)
    assert torch.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize('where', ['min', 'max', 'both', 'random'])
def test_quantile_mask_bins_ignores_outliers(where):
    """Histogram path: masked-out outliers must not affect the bin range."""
    bins = 256
    x = make([3, 40, 50])  # enough samples for a histogram
    flat = x.reshape([3, -1])
    mask = torch.ones_like(flat)
    if where in ('min', 'both'):
        mask.scatter_(-1, flat.argsort(-1)[:, :100], 0)
    if where in ('max', 'both'):
        mask.scatter_(-1, flat.argsort(-1)[:, -100:], 0)
    if where == 'random':
        g = torch.Generator().manual_seed(1)
        mask = (torch.rand(mask.shape, generator=g) > 0.4).double()
    clean = flat.clone()
    flat = flat.clone()
    flat[mask == 0] = 1e3 * (torch.arange((mask == 0).sum()) % 2 * 2 - 1).double()

    out = utils.quantile(flat, qs, dim=-1, mask=mask, bins=bins)
    assert torch.isfinite(out).all()
    # same as the bins= result with masked-out voxels set to in-mask values
    # (which only changes values with zero weight) -- i.e. outliers ignored
    clean[mask == 0] = clean[mask > 0].min()  # within the in-mask range
    ref_bins = utils.quantile(clean, qs, dim=-1, mask=mask, bins=bins)
    assert torch.allclose(out, ref_bins, atol=1e-10), (out, ref_bins)
    # close to the exact quantiles on in-mask values (within a bin width)
    ref = reference(flat.reshape(x.shape), qs, mask.reshape(x.shape))
    width = torch.stack([(r[m > 0].max() - r[m > 0].min()) / bins
                         for r, m in zip(flat, mask)])
    assert ((out - ref).abs() <= 1.5 * width[:, None] + 1e-12).all(), \
        (out, ref, width)
