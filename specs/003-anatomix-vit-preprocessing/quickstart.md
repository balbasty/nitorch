# Quickstart: Validating Anatomix 3D ViT Preprocessing

See `contracts/anatomix-vit-api.md` for exact API shapes and `data-model.md`
for entity details.

## Prerequisites

- nitorch with this feature implemented (`anatomix_vit=` on `make_image()`,
  `--anatomix-vit` on `nitorch register`).
- Either network access (for `auto_download=True` / bare `--anatomix-vit`,
  fetching `anatomix-dev-vit` on first use) or a local `.pth` checkpoint for
  that variant.

## Scenario 1 — Enable ViT features through the existing interface (US1)

```python
import torch
from nitorch.tools.registration.pairwise_makeobj import make_image

dat = torch.rand(1, 64, 64, 64)  # single-channel volume
image = make_image(dat, anatomix_vit=True)
level = image[0]
print(level.dat.shape)   # (output_nc, 64, 64, 64) -- same spatial shape as input
```

**Expected outcome**: completes without requiring the caller to pad/reshape
`dat` to 128³ themselves (FR-002, SC-001).

## Scenario 2 — Combine ViT with MIND and/or U-Net anatomix (US2)

```python
image = make_image(dat, mind=True, anatomix=True, anatomix_vit=True)
level = image[0]
# channel count = mind_channels + unet_output_nc + vit_output_nc
```

**Expected outcome**: `level.dat`'s channel count is exactly the sum of each
individually-enabled option's own channel count, in mind → anatomix →
anatomix_vit order (FR-004, SC-003) — verify by comparing against each
option run alone, the same way the existing MIND+anatomix joint test does
(`nitorch/tests/test_anatomix_image.py::test_make_image_mind_and_anatomix_concatenate_channels`).

## Scenario 3 — Frictionless weight resolution (US3)

```python
# no local weights_path supplied
image = make_image(dat, anatomix_vit=True)   # downloads + caches anatomix-dev-vit on first use
image2 = make_image(dat, anatomix_vit=True)  # reuses the cached checkpoint, no re-download
```

```python
from nitorch._models.anatomix import AnatomixWeightsError

try:
    make_image(dat, anatomix_vit={'auto_download': False})
except AnatomixWeightsError as e:
    print(e)  # clear, actionable message -- not a generic crash
```

**Expected outcome**: matches the existing anatomix U-Net's own
download/cache/error behavior exactly (FR-005, FR-006, SC-004).

## Scenario 4 — Arbitrary input shapes (pad vs. tile)

```python
small = torch.rand(1, 32, 40, 28)     # every axis < 128 -> pad path
large = torch.rand(1, 200, 192, 176)  # some axis > 128 -> sliding-window path

for vol in (small, large):
    image = make_image(vol, anatomix_vit=True)
    assert image[0].dat.shape[1:] == vol.shape[1:]
```

**Expected outcome**: both the pad path (small input) and the tiling path
(large input) return features covering the full input at its original
shape, with no visible seams from tile blending in the overlap regions
(spot-check numerically: neighboring-tile overlap region values should vary
smoothly, not show a sharp discontinuity at the tile boundary).

## Scenario 5 — Multi-channel input raises a clear error (FR-007)

```python
multi = torch.rand(2, 64, 64, 64)  # 2 channels
try:
    make_image(multi, anatomix_vit=True)
except ValueError as e:
    print(e)  # "anatomix_vit expects a single-channel image, got 2 channels"
```

## Scenario 6 — CLI end-to-end (US1, regression FR-003)

```
nitorch register --gpu \
  @loss lcc \
  @@fix fixed.nii.gz --anatomix-vit \
  @@mov moving.nii.gz --anatomix-vit \
  @affine affine -o out.lta
```

**Expected outcome**: registration completes and writes `out.lta`, exactly
as the equivalent `--anatomix`-based command already does.

## Scenario 7 — Existing behavior is unaffected (regression, FR-003)

Run the existing test suite (`nitorch/tests/test_anatomix_image.py` and
related registration tests) and confirm `mind=`/`anatomix=`-only behavior,
and any registration run that does not pass `anatomix_vit=`, is unchanged.
