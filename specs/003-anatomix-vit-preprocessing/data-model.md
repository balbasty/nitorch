# Data Model: Anatomix 3D ViT Preprocessing

This feature has no persisted data entities in the usual sense — it adds a
new *preprocessing transform* and its supporting configuration/weight
-resolution objects, extending the pattern already established for MIND and
the anatomix U-Net.

## Entities

### `AnatomixViT` (nn.Module)

The vendored 3D Vision Transformer architecture itself.

| Field | Type | Notes |
|---|---|---|
| `input_size` | `int` (fixed) | 128 (voxels, per axis) — the ViT's native working resolution, per `research.md` §2. |
| `patch_size`, `embed_dim`, `depth`, `num_heads` | `int` | Exact values reverse-engineered from the downloaded checkpoint's `state_dict` tensor shapes during implementation (`research.md` §4) — not user-configurable defaults the way the U-Net's `num_downs`/`ngf` are, since a mismatch against the pretrained checkpoint would break weight loading. |
| `output_nc` | `int` | Number of output feature channels; also reverse-engineered from the checkpoint. |

Mirrors `AnatomixUNet` (`nitorch/_models/anatomix/unet.py`) in role: a frozen,
pretrained `nn.Module` mapping `(1, 1, *spatial)` volumes to
`(1, output_nc, *spatial)` feature maps — but only for `spatial ==
(128, 128, 128)` exactly (see `SlidingWindowRunner` below for arbitrary
input shapes).

### `SlidingWindowRunner`

New supporting component (no upstream equivalent needed for the U-Net, which
tolerates arbitrary shapes via simple pad/crop). Wraps `AnatomixViT` to
present the same "any single-channel 3D volume in, same-spatial-shape
features out" contract already relied on by `make_image()`.

| Field | Type | Notes |
|---|---|---|
| `window` | `int` | Fixed at 128 (matches `AnatomixViT.input_size`). |
| `overlap` | `float` | Fraction of `window` two adjacent tiles overlap by, for blending at tile borders. Implementation default; not user-facing per FR-002 (no caller shape pre-processing required). |

Behavior (`research.md` §2):
- If every spatial axis of the input is ≤ 128: pad up to exactly 128³
  (replicate padding, matching the U-Net auto-pad precedent from PR #90),
  run `AnatomixViT` once, crop the output back to the original shape.
- Otherwise: tile the volume into overlapping 128³ windows covering the full
  extent, run `AnatomixViT` per tile, and reassemble via overlap blending
  (e.g. linear/cosine feathering at tile borders) into a full-size feature
  volume.

### `AnatomixViTFeatureExtractor`

The public wrapper class, directly analogous to the existing
`AnatomixFeatureExtractor` (`nitorch/_models/anatomix/__init__.py`).

| Field | Type | Notes |
|---|---|---|
| `weights_path` | `str`, optional | Local checkpoint path; takes precedence over `auto_download`. |
| `auto_download` | `bool` | Opt into fetching `anatomix-dev-vit` via the existing `resolve_weights_path(variant='anatomix-dev-vit')` (`research.md` §1) — no new download mechanism. |
| `cache_dir` | `str`, optional | Same cache directory convention as the U-Net extractor. |
| `normalize` | `bool`, default `True` | Applies the per-voxel feature normalization from `research.md` §3 (unit-norm or zero-mean/unit-std across channels) before returning features. |

Call contract: `extractor(volume)` where `volume` is `(1, 1, *spatial)`
(or reshaped from `(*spatial)`/`(1, *spatial)` the same way the existing
extractor accepts it) → returns `(1, output_nc, *spatial)`, spatial shape
unchanged from the input (via `SlidingWindowRunner`).

### `Preprocessing Option Set` (conceptual, spec-level entity)

Not a class — the set of independently-toggleable feature-extraction options
`make_image()` accepts: `mind`, `anatomix` (U-Net), `anatomix_vit` (new).
Any non-empty subset may be enabled together; their outputs are concatenated
along the channel dimension in a fixed, deterministic order (mind, then
U-Net anatomix, then ViT anatomix) so combined-mode output is reproducible.

## Relationships

```text
make_image()
 ├─ mind=...           -> spatial.rmind(...)                    -> feats[0]
 ├─ anatomix=...        -> AnatomixFeatureExtractor(...)         -> feats[1]
 └─ anatomix_vit=...    -> AnatomixViTFeatureExtractor(...)      -> feats[2]
                                │
                                ├─ resolve_weights_path(variant='anatomix-dev-vit')
                                ├─ AnatomixViT (frozen nn.Module)
                                └─ SlidingWindowRunner (pad or tile+blend)

level.dat = concat(feats, dim=0)   # only the enabled subset
```

No state transitions or persistence: every entity above is constructed fresh
per `make_image()` call (the frozen model/weights are cached process-wide the
same way `AnatomixFeatureExtractor` already lazily builds-and-reuses its
model on first use).
