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

**Decision**: Apply per-voxel feature normalization to `anatomix-dev-vit`'s
raw output before it is used as a registration feature map. **Superseded by
§4**: the exact, verified operation is channel-wise spatial demeaning
(`out_norm="demean"` in the real architecture: `x - x.mean(dim=(2,3,4),
keepdim=True)`), not a generic "unit-norm or zero-mean/unit-std" choice as
originally guessed here before the real source was found.

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

## §4. Real architecture: PrimusV2 (hybrid CNN-tokenizer + EVA transformer + CNN decoder), not a vanilla ViT [T001, RESOLVED]

**Finding**: `anatomix-dev-vit` is far more elaborate than a vanilla 3D ViT.
Downloading the real checkpoint (105MB, 27.1M params) and inspecting its
`state_dict` (411 keys), then locating and reading the upstream
`neel-dey/anatomix` GitHub repo's actual source
(`anatomix/model/vit3d/architectures.py`, `load_from_hf.py`) reveals the
exact recipe, verified by a **strict, zero-mismatch `load_state_dict`**:

```python
from dynamic_network_architectures.architectures.primus import PrimusV2

model = PrimusV2(
    input_channels=1, num_classes=32, embed_dim=396, eva_depth=12,
    eva_numheads=6, patch_embed_size=(8, 8, 8), input_shape=(128, 128, 128),
    num_register_tokens=8, init_values=0.1, scale_attn_inner=True,
)
# anatomix's own small addition (not in upstream PrimusV2): per-head
# LayerNorm on queries/keys in every EVA block (12 blocks x 2 norms x
# {weight, bias} = 48 extra params, confirmed exactly against the checkpoint)
for block in model.eva.blocks:
    head_dim = block.attn.q_proj.out_features // block.attn.num_heads  # 66
    block.attn.q_norm = torch.nn.LayerNorm(head_dim)
    block.attn.k_norm = torch.nn.LayerNorm(head_dim)

model.load_state_dict(checkpoint_state_dict, strict=True)  # verified: succeeds exactly
```

Architecture, ground-truth-verified: a 4-stage residual CNN encoder
(InstanceNorm, `in_eps=1e-2`) downsamples 128³ -> 16³ while expanding
channels 1 -> 32 -> 32 -> 64 -> 128, a 1x1x1 conv projects to
`embed_dim=396`, 8 register tokens are prepended (sequence length
16³+8=4104, matching `eva.pos_embed`'s shape exactly), 12 EVA-style
transformer blocks (6 heads, head_dim=66, SwiGLU MLP hidden=1056,
LayerScale `init_values=0.1`, per-head QK-LayerNorm) process the sequence,
and a 3-stage transpose-conv decoder upsamples back to 128³ with 32 output
channels. Output normalization is `out_norm="demean"`: subtract each output
channel's own spatial mean (`x - x.mean(dim=(2,3,4), keepdim=True)`), a
*stateless* operation (no learned parameters, doesn't affect
`load_state_dict`) — this supersedes and makes concrete §3's earlier,
vaguer "unit-norm or zero-mean/unit-std" normalization guess: it is
specifically per-channel spatial demeaning, nothing more.

**Decision**: Depend on the third-party `dynamic_network_architectures`
PyPI package (real, published, pip-installable — part of the
nnU-Net/MIC-DKFZ ecosystem) for the verified-correct `PrimusV2`
architecture, rather than hand-reimplementing it. Vendor only anatomix's
own small addition (the ~10-line QK-LayerNorm attachment loop and the
stateless demean output norm) directly in `nitorch/_models/anatomix/vit.py`,
mirroring how thin anatomix's *own* wrapper around upstream `PrimusV2`
already is.

**Rationale**: This revises plan.md's Technical Context assumption of "no
new hard dependency," which was reasonable *before* the real architecture
was known (a vanilla ViT would have been small enough to vendor, like the
U-Net) but does not hold now that the real model is confirmed to depend on
published third-party research infrastructure. Hand-reimplementing
`PrimusV2` (the CNN tokenizer, EVA attention/SwiGLU/LayerScale details, and
CNN decoder) to bit-exact behavioral fidelity would be substantial,
slower, and carries a materially different risk profile than the
U-Net's own vendoring: a subtle mistake in a from-scratch reimplementation
would not necessarily crash (shapes could still match) — it could silently
produce numerically-wrong features, which is a much harder class of bug to
catch than the shape/key-mismatch errors a wrong vendored U-Net would
produce. Depending on the actual, verified-correct upstream package
eliminates that entire risk class for a mechanical cost (one new
dependency). This decision was made explicitly with the user rather than
assumed silently, given it revises a plan-level architectural assumption.

**Dependency footprint**: `pip install dynamic_network_architectures`
(tested: v0.4.4) pulls in `torchvision`, `timm`, `huggingface_hub`, `einops`,
and their own transitive dependencies (14 packages total in a clean
install) — non-trivial, but consistent with the scale of what it provides
(a real research-grade hybrid CNN/ViT architecture library), and installed
under a new optional extra (mirroring the `zarr`/`dask` optional-extra
precedent from the nifti-zarr feature), not a hard requirement for nitorch
users who don't use `anatomix_vit=`.

**Alternatives considered**:
- *Hand-reimplement `PrimusV2` from scratch* (the plan's original
  assumption): rejected per the risk/effort rationale above, now that the
  real complexity is known.
- *Depend on the `anatomix` PyPI package directly* (rather than just
  `dynamic_network_architectures`): rejected — `anatomix` itself pulls in
  unrelated registration/segmentation/pretraining code nitorch doesn't need;
  depending on the narrower `dynamic_network_architectures` package plus
  vendoring anatomix's own ~10-line QK-norm/demean addition keeps the
  dependency surface minimal while still being 100% verified-correct.
- *Reverse-engineer purely from checkpoint tensor shapes with no source
  code* (the original plan for this section): superseded once the actual
  upstream source was located and read — verifying against real source
  code plus a strict `load_state_dict` check is strictly more reliable than
  shape-only inference (which cannot recover operations like InstanceNorm
  vs. GroupNorm, SwiGLU vs. plain MLP, or the exact residual wiring).
