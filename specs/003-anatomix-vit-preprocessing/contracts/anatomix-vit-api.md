# Contract: Anatomix 3D ViT Preprocessing

Like MIND and the existing anatomix U-Net, this feature is reached through
the same registration entry points nitorch already exposes — no new command
or top-level API surface. This is a contract on the *additive* option, not a
new interface.

## 1. Python API (`make_image()`)

```python
from nitorch.tools.registration.pairwise_makeobj import make_image

image = make_image(dat, anatomix_vit=True)                       # auto-download anatomix-dev-vit
image = make_image(dat, anatomix_vit='/path/to/checkpoint.pth')  # local weights
image = make_image(dat, anatomix_vit={'weights_path': '...'})    # dict form, mirrors anatomix=
image = make_image(dat, mind=True, anatomix=True, anatomix_vit=True)  # all three, concatenated
```

- `anatomix_vit` follows the exact same `None`/`False` (disabled) /
  `True` (enabled, auto-download) / `str` (local path shorthand) / `dict`
  (full control) normalization already used for `anatomix=`
  (`_normalize_anatomix_config`), generalized to a shared helper so both
  options behave identically (FR-001, FR-005).
- Enabling `anatomix_vit` together with `mind` and/or `anatomix` MUST
  concatenate all enabled feature sets along the channel dimension in a
  fixed order: mind, then anatomix (U-Net), then anatomix_vit (FR-004).
- `anatomix_vit=None` (the default) MUST leave `make_image()`'s behavior
  byte-for-byte identical to before this feature existed (FR-003) — this is
  a regression contract, not just a new-feature one.

## 2. CLI (`nitorch register`)

```
nitorch register ... @@fix scan.nii.gz --anatomix-vit ...
nitorch register ... @@mov scan.nii.gz --anatomix-vit /path/to/checkpoint.pth ...
```

- `--anatomix-vit [PATH]` is a new per-file option under `@@fix`/`@@mov`,
  positioned alongside the existing `--mind`/`--anatomix` options in
  `nitorch/cli/registration/register/parser.py`'s `file` group (FR-001).
- Bare `--anatomix-vit` (no value) opts into automatic download, matching
  `--anatomix`'s existing bare-flag convention exactly.
- `nitorch register -h 3` (full help) MUST document how `--anatomix-vit`
  differs from `--anatomix` (U-Net vs. ViT, both usable together) (FR-008).

## 3. Weight resolution

```python
from nitorch._models.anatomix.weights import resolve_weights_path

path = resolve_weights_path(auto_download=True, variant='anatomix-dev-vit')
```

- No new download/cache mechanism: `anatomix_vit=True` (or CLI bare
  `--anatomix-vit`) resolves via the *existing*
  `resolve_weights_path(variant='anatomix-dev-vit')`, sharing the same
  `HF_REPO_ID`, cache directory, and `AnatomixWeightsError` error type as
  the U-Net extractor (`research.md` §1, FR-005, FR-006).
- A `weights_path` that exists but fails to load (corrupt file, or a
  checkpoint that doesn't match `AnatomixViT`'s reverse-engineered
  architecture) MUST raise `AnatomixWeightsError` with an actionable
  message, mirroring `load_state_dict_into`'s existing U-Net behavior
  (FR-006).

## 4. Input shape handling

```python
from nitorch._models.anatomix import AnatomixViTFeatureExtractor

extractor = AnatomixViTFeatureExtractor(auto_download=True)
features = extractor(volume)  # volume: (1, 1, *any_spatial_shape)
# features: (1, output_nc, *any_spatial_shape) -- same spatial shape as input
```

- The extractor MUST accept a single-channel 3D volume of *any* spatial
  shape and return features of the *same* spatial shape — the ViT's fixed
  128³ working resolution MUST be an internal implementation detail (pad
  when every axis ≤ 128, tile+blend otherwise), never a caller-visible
  constraint (FR-002, `research.md` §2, `data-model.md` `SlidingWindowRunner`).
- A multi-channel input (`volume.shape[1] != 1` after normalization to
  `(1, C, *spatial)`) MUST raise a clear `ValueError`, mirroring the
  existing U-Net extractor's own single-channel check (FR-007).

## 5. Test contract

Each contract clause above MUST have a corresponding automated test
(Constitution Principle III):

- `anatomix_vit=None`/absent reproduces byte-identical `make_image()`
  output to before this feature (regression guard for FR-003).
- `anatomix_vit=True` with a real (or fixture) checkpoint produces
  multi-channel features of the expected shape from a single-channel input.
- `anatomix_vit` combined with `mind=True` and/or `anatomix=True` produces
  the exact channel-wise concatenation of the individually-enabled outputs,
  in the fixed mind → anatomix → anatomix_vit order (FR-004).
- An input smaller than 128³ in every axis and an input larger than 128³ in
  at least one axis both produce full, same-shape-as-input feature output
  (pad path and tile path both exercised).
- A multi-channel input raises a clear error (FR-007).
- An unresolvable checkpoint (`auto_download=False`, no `weights_path`)
  raises `AnatomixWeightsError`, not a generic exception (FR-006).
- `nitorch register --anatomix-vit ...` end-to-end completes a registration
  run and produces a transform, exactly as `--anatomix` already does.
