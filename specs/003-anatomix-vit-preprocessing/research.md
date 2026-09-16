# Research: Anatomix 3D ViT Preprocessing

## §1. Correct checkpoint identity

**Decision**: The feature targets the checkpoint variant named `anatomix-dev-vit`
(not the bare name `anatomix-dev` used in the initial clarification answer,
which is a *different*, 94M-parameter experimental **U-Net** variant — not
the ViT).

**Rationale**: The upstream `neel-dey/anatomix` project (GitHub) documents
three distinct pretrained variants:
- `anatomix` — 6M params, the original ICLR 2025 U-Net (already vendored in
  `nitorch/_models/anatomix/unet.py` as the default `anatomix=` option).
- `anatomix-dev` — 94M params, an experimental *U-Net* (out of scope here;
  it's an alternative to the existing U-Net, not a ViT).
- `anatomix-dev-vit` — 26M params, the experimental **3D ViT** — this is
  what this feature is actually about.

All three are loaded upstream via a single convenience function,
`load_from_hf(<variant-name>)`, which strongly suggests they live in the
*same* HuggingFace repository under different checkpoint filenames — exactly
matching nitorch's own existing `resolve_weights_path(variant=...,
repo_id=...)` design (`nitorch/_models/anatomix/weights.py`), which already
parameterizes the checkpoint filename as `f'{variant}.pth'` against a fixed
`repo_id`. No new weight-resolution mechanism is needed: the existing
function can be reused unmodified, called with `variant='anatomix-dev-vit'`.

**Alternatives considered**: Depending on the upstream `anatomix` PyPI
package's own `load_from_hf` helper directly. Rejected — nitorch's
established pattern (Principle: Code Quality; existing `unet.py` vendoring)
is a dependency-light, vendored reimplementation using only `torch`, with no
new hard dependency. Reusing the existing `resolve_weights_path` achieves the
same download/cache/local-override behavior without adding a package
dependency.

## §2. Input-size constraint: fixed 128³, no known auto-pad/crop precedent

**Decision**: `anatomix-dev-vit` requires exactly 128×128×128 input volumes
(patch-embedding architectures generally cannot process arbitrary shapes the
way a fully-convolutional U-Net can). The upstream project describes it as
"amenable to sliding window processing" for larger volumes, but does not ship
a ready-made sliding-window implementation.

**Rationale/impact**: This is a materially different constraint from the
existing U-Net's "divisible by 16" requirement (which balbasty's upstream fix
already resolves transparently via pad-then-crop, see PR #90,
`AnatomixFeatureExtractor.__call__`). A 128³ *fixed* size cannot be satisfied
by simple pad-to-multiple; an input smaller than 128³ needs padding, and an
input larger than 128³ needs either padding+crop-back (only valid up to
128³) or genuine sliding-window tiling with overlap-blending to cover the
full volume without a fixed-size ceiling. Given `nitorch register`'s pyramid
levels can produce volumes of many different sizes (as encountered
extensively during the hackathon26 registration testing this session, e.g.
padded shapes from 64³ up to 320×192×256), a pad-only strategy would silently
fail (or truncate) for any level exceeding 128 along some axis.

**Alternatives considered**:
- *Pad-only (mirror the U-Net's approach)*: simplest, but only correct for
  inputs ≤ 128³ in every axis; silently wrong (crops away real content) for
  larger pyramid levels. Rejected as the sole strategy, but still the right
  behavior for the ≤ 128³ case (avoids needless tiling overhead).
- *Sliding-window tiling with overlap-add/blend*: correct for arbitrary
  sizes, standard practice for patch-based 3D models (e.g. MONAI's
  `sliding_window_inference`). Adds real implementation complexity (window
  stride/overlap choice, blending at tile borders) but is the only
  approach that doesn't silently misbehave on larger volumes.
- **Chosen**: pad-up-to-128³ when every axis is ≤ 128 (cheap, exact); fall
  back to sliding-window tiling with a blended overlap when any axis exceeds
  128. This mirrors the existing U-Net extractor's external contract (accepts
  any single-channel 3D volume, returns same-shape-spatial features) so
  `make_image()`'s integration code does not need to special-case the ViT.

## §3. Output feature normalization

**Decision**: Apply per-voxel feature normalization (unit-norm or zero-mean
/unit-std across the output channel dimension) to `anatomix-dev-vit`'s raw
output before it is used as a registration feature map.

**Rationale**: The upstream project's own documentation explicitly warns
that the experimental dev models' raw features are not well-scaled
out-of-the-box the way the published `anatomix` checkpoint's are, and
recommends this normalization step for downstream use (including
registration-style applications). Skipping it risks poorly-conditioned
similarity losses (e.g. LCC/NCC on raw unnormalized activations), consistent
with numerically-sensitive-pipeline concerns already called out in the
project constitution (Principle I).

**Alternatives considered**: Leaving normalization as a caller responsibility
(matching the existing U-Net extractor, whose published checkpoint doesn't
need it). Rejected: it would silently produce a working-but-poor-quality
integration by default, contradicting FR-002/SC-001's "same integration
ergonomics as the existing options" expectation — the ViT extractor should
be usable via the same bare-flag opt-in as `mind=`/`anatomix=` without the
caller needing to know about this normalization quirk.

## §4. Exact architecture parameters are unknown pre-implementation

**Decision**: Defer resolving the ViT's exact architecture hyperparameters
(patch size, embedding dimension, depth, number of attention heads) to the
implementation phase, by reverse-engineering them from the downloaded
checkpoint's `state_dict` tensor shapes — the same technique already used to
reconstruct the U-Net's flat-`nn.Sequential`-to-named-module mapping
(`nitorch/_models/anatomix/weights.py::_flat_index_to_name_map`).

**Rationale**: Neither the upstream repository's documentation nor its model
card expose these parameters in prose; they are only implicit in the
checkpoint file itself. This is an implementation-time task (downloading the
real checkpoint and inspecting it), not something resolvable through
research alone, and does not block planning: the *contract* (single-channel
3D in, multi-channel modality-invariant 3D features out, fixed 128³ working
resolution) is fully known regardless of the internal parameter values.

**Alternatives considered**: Blocking this feature on upstream publishing
detailed architecture docs. Rejected — reverse-engineering from the
checkpoint is a proven, already-precedented technique in this exact codebase.
