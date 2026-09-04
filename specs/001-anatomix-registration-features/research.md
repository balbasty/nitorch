# Phase 0 Research: Anatomix Feature-Based Registration

## 1. Integration point: mirror the existing `mind=` feature-transform, not a new loss

- **Decision**: Anatomix is a one-time **image preprocessing / feature-transform**
  step, not a new similarity loss. It plugs into
  `nitorch/tools/registration/pairwise_makeobj.py::make_image()` as a new `anatomix=`
  parameter, exactly mirroring the existing `mind=` parameter on the same function.
  After the image pyramid is built, for each level: `level.preview = level.dat` (keep
  the original intensities for display), then `level.dat = extract_features(level.dat,
  ...)` (swap in the 16-channel anatomix feature map). No new `OptimizationLoss`
  subclass is added, and `pairwise_makeobj.make_loss()` is **not modified at all** —
  whichever existing loss the user already selects (`'lcc'`, `'cc'`, `'mse'`, `'mi'`,
  ...) runs unchanged on the feature-valued `dat`, exactly as it already does for MIND
  features today.
- **Rationale**: `mind=` already solves the identical structural problem — swap the
  space similarity is computed in, without touching the loss machinery — for a
  different modality-agnostic feature descriptor (Heinrich et al.'s MIND). Reusing this
  exact pattern (rather than the previously-considered `'anatomix'` loss-string design)
  means: (a) no autograd needs to flow through the frozen network during optimization,
  because extraction happens once, before the optimizer runs, and the optimizer only
  ever sees the resulting feature tensor as ordinary (multi-channel) data — consistent
  with nitorch's gradient-less-autograd optimizer design (`optim/base.py`: "these are
  generic optimizers that do not require autograd"); (b) `make_loss()` and every
  existing loss class are untouched, satisfying FR-008 by construction rather than by
  a compatibility shim; (c) it follows Constitution Principle I (follow existing
  conventions) more directly than inventing a new dispatch axis.
- **Alternatives considered**: A new `'anatomix'` loss key in `make_loss()`, wrapping a
  `base_loss` and requiring autograd through the network on every optimizer iteration
  (originally proposed; rejected — conflates "what space to compare in" with "what
  similarity function to use," and fights the optimizer module's no-autograd design). A
  brand-new orthogonal `features=` parameter independent of both `make_image` and
  `make_loss` (rejected — `mind=` already established this exact idiom on `make_image`,
  so a second one would be redundant).

## 2. Scope: only the feature-extraction/model-loading portion of anatomix

- **Decision**: Adopt only anatomix's pretrained model + weight loading. Its own
  reference registration script
  (`anatomix/registration/run_convex_adam_with_network_feats.py` in the upstream repo)
  additionally bundles a specific ConvexAdam-based instance-optimization pipeline
  (`grid_sp`, `lambda_weight`, `selected_niter`, its own MIND-search-width `disp_hw`,
  masking, intensity clipping, label warping) — none of that is adopted.
- **Rationale**: nitorch already has its own optimizer stack (Gauss-Newton, L-BFGS,
  CG, ...), masking (`make_image(mask=...)`), intensity handling (`--rescale`), and
  label-map warping (`--label`). Reimplementing anatomix's own optimizer would
  duplicate existing, tested nitorch functionality and contradict FR-008's requirement
  that the rest of the registration workflow stay unchanged.
- **Alternatives considered**: Port the full ConvexAdam pipeline as an alternative
  registration backend (rejected — far larger scope than the spec calls for, and
  duplicates existing nitorch optimizer functionality).

## 3. Weight distribution strategy

- **Decision**: By default, require a user-supplied local path to a `.pth` checkpoint.
  Provide an explicit opt-in convenience path that downloads the official weights from
  the anatomix HuggingFace Hub distribution and caches them locally, only when the user
  asks for it.
- **Rationale**: Matches the `/speckit-clarify` decision to default to no implicit
  network access (safe for HPC/offline environments such as this repository's own
  cluster deployment). Anatomix's code and weights are MIT-licensed, so redistribution
  or referencing the official distribution carries no license conflict with nitorch's
  own MIT license.
- **Alternatives considered**: Always auto-download on first use (rejected — violates
  the clarified default and risks silent network calls on offline compute nodes);
  vendor the weight file directly inside the nitorch source tree (rejected — a
  multi-megabyte binary checkpoint does not belong in a source/PyPI package).

## 4. Model architecture implementation

- **Decision**: Vendor a minimal, dependency-light, **parameterized** 3D U-Net inside
  nitorch (`nitorch/_models/anatomix/unet.py`), rather than depending on the external
  `anatomix` GitHub package or reusing nitorch's own generic `UNet2`
  (`nitorch/nn/modules/cnn.py`). The upstream reference script exposes 6 architecture
  knobs that must match whatever checkpoint is loaded — `num_downs` (default 4), `ngf`
  (default 16), `output_nc` (default 16), `norm` (default `'batch'`), `interp` (default
  `'nearest'`), `pooling` (default `'Max'`) — so the vendored U-Net accepts all six,
  defaulted to the published `anatomix` checkpoint's values, so the common case needs
  no overrides.
- **Rationale**: The upstream `anatomix` package pulls in MONAI, TorchIO, and other
  training-time dependencies unnecessary for pure frozen-weight inference, and is not a
  stable PyPI dependency (git-installed). Reusing nitorch's own `UNet2` is attractive
  for code reuse, but its internal layer naming is not guaranteed to match the
  state-dict keys baked into the published checkpoint. Exposing the 6 architecture
  knobs (rather than hard-coding them) is necessary because users may load a different
  checkpoint variant (e.g. the 94M-parameter `anatomix-dev`) with different
  hyperparameters.
- **Alternatives considered**: Depend on the `anatomix` GitHub package directly
  (rejected — unstable install path, heavy unrelated dependencies); reimplement using
  `UNet2` (rejected — state-dict compatibility is unverified); hard-code a single fixed
  architecture with no overrides (rejected — breaks for any checkpoint other than the
  one default variant).

## 5. `anatomix=` parameter shape

- **Decision**: Mirror `mind=`'s own shorthand-vs-explicit pattern (`mind=True` →
  shorthand `[1, 2]`; `mind=[fwhm, radius]` → explicit), extended with a string and a
  dict form to cover the larger config surface:
  ```python
  anatomix=None                          # disabled (default)
  anatomix="/path/to/anatomix.pth"       # shorthand for {"weights_path": "..."}
  anatomix=True                          # shorthand for {"auto_download": True}
  anatomix={"weights_path": "...",       # full form, only needed for a non-default
            "num_downs": 6, "ngf": 32,   # checkpoint/architecture
            "output_nc": 16, "norm": "batch",
            "interp": "nearest", "pooling": "max"}
  ```
- **Rationale**: A single string or bool (as in the originally-proposed `'anatomix'`
  loss key, or a bare `anatomix=path` design) cannot express the architecture overrides
  needed for a non-default checkpoint (Decision 4), but a dict-only design would make
  the common case (`anatomix="/path.pth"`) more verbose than necessary. The
  string/bool/dict hybrid keeps the common case a one-liner while allowing full control
  when needed, and follows a pattern (`True` → shorthand, explicit form → full control)
  already established by `mind=` in the very same function.
- **Alternatives considered**: Flat kwargs (`anatomix=False, anatomix_weights=None,
  anatomix_num_downs=4, ...` — 8 new top-level parameters on an already-large function
  signature; rejected as excessive for a config surface that's rarely touched beyond
  the weights path); dict-only, no string/bool shorthand (rejected — makes the common
  case needlessly verbose compared to `mind`'s own shorthand precedent).

## 6. Per-pyramid-level extraction

- **Decision**: Run feature extraction once per pyramid level (the network is applied
  independently to each already-downsampled resolution level's `dat`), exactly
  mirroring how `mind` is computed inside the existing `for level in image:` loop in
  `make_image()`.
- **Rationale**: Keeps the change minimal and structurally identical to the existing,
  working `mind` code path (Constitution Principle I). Anatomix was trained with
  synthetic multi-scale appearance augmentation, so running it directly at each
  pyramid resolution (rather than extracting once at full resolution and
  downsampling the resulting feature maps) is expected to be reasonable quality-wise.
- **Alternatives considered**: Extract once at full resolution, then downsample the
  feature maps through the pyramid (deferred — a valid future optimization if per-level
  network inference proves too slow, but not needed for a first implementation, and it
  would require new pyramid-downsampling code for feature (not intensity) tensors that
  doesn't currently exist).

## 7. Standalone feature extraction (User Story 2)

- **Decision**: Expose a plain function,
  `nitorch._models.anatomix.extract_features(volume, weights_path=None,
  auto_download=False, **architecture_overrides) -> Tensor`, usable independently of
  the registration workflow. `make_image()`'s `anatomix=` wiring calls this same
  function internally for each pyramid level.
- **Rationale**: Satisfies FR-007/US2 without a full registration run, and avoids
  duplicating weight-loading/error-handling logic (FR-006) between the `make_image`
  wiring and standalone use.

## 8. Device handling

- **Decision**: No new device-selection mechanism. The feature extractor module is
  moved to whichever device the input volume tensor already resides on (mirroring the
  existing `.cuda()`/`.cuda_()` pattern on registration objects in
  `nitorch/tools/registration/objects.py`).
- **Rationale**: Matches the spec's Assumptions section and existing codebase
  conventions (Constitution Principle I).

## 9. Dependency footprint

- **Decision**: No new entries in `install_requires` (`setup.cfg`). The vendored U-Net
  (Decision 4) only needs `torch`, already required. Weight download (opt-in) uses a
  plain HTTP request, matching the existing optional `wget`/`appdirs` pattern already
  used for the `data` extra in `setup.cfg`.
- **Rationale**: Keeps the feature's dependency footprint consistent with nitorch's
  existing minimal-core / optional-extras convention, and avoids forcing MONAI/TorchIO
  on all nitorch users for a single opt-in feature.

## 10. CLI exposure

- **Decision**: `--mind` is already exposed in the `nitorch register` CLI
  (`nitorch/cli/registration/register/parser.py`, forwarded to `make_image()` via a
  keyword-forwarding tuple in `nitorch/cli/registration/register/cli.py:296`). Mirror
  this for v1 with a single `--anatomix [PATH]` flag (bare flag = auto-download,
  matching `--mind [FWHM=1 [RADIUS=0]]`'s optional-value shape) added to both files.
  Architecture-override options (`num_downs`, `ngf`, ...) are Python-API-only for v1 —
  not exposed as CLI flags, since they are expected to be rarely needed (non-default
  checkpoints only).
- **Rationale**: Keeps CLI parity with `mind` for the common case at low cost (two
  small, well-precedented edits), while not bloating the CLI parser with 6 rarely-used
  architecture flags.
- **Alternatives considered**: No CLI exposure, Python-API-only (rejected — `mind`, the
  feature this mirrors, is itself CLI-exposed, so parity is the more consistent
  choice); full architecture-override CLI flags now (deferred — not needed until a
  concrete non-default-checkpoint use case arises).
