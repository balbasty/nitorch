# Feature Specification: Anatomix 3D ViT Preprocessing

**Feature Branch**: `003-anatomix-vit-preprocessing`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "create a new feature for adding anatomix dev branch 3d vit as a preprocessing step alongside the two other types"

## Clarifications

### Session 2026-09-16

- Q: Does a publicly downloadable checkpoint for the anatomix 3D ViT already exist, or is only a local weights file realistic for v1? → A: A public checkpoint exists — hosted as `anatomix-dev` on HuggingFace (there is also an "anatomix+brains" alternative checkpoint for the existing U-Net, noted for context but out of scope for this feature).
- Q: Should the ViT feature extractor be combinable with the existing U-Net anatomix extractor in the same run, or mutually exclusive since both play the same role? → A: Combinable — follows the same precedent as MIND+anatomix, concatenating feature channels from any combination of enabled extractors.

**Planning research correction** (`research.md` §1): the precise checkpoint
name is `anatomix-dev-vit` — `anatomix-dev` alone names a *different*,
94M-parameter experimental U-Net variant, not the ViT. All references below
use the corrected name.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Register using ViT-based anatomix features (Priority: P1)

A researcher running cross-modal image registration wants to use anatomix's 3D
Vision Transformer (ViT) feature extractor — from anatomix's development
branch — as an alternative modality-invariant feature-extraction
preprocessing step, so they can evaluate its registration quality against the
existing U-Net-based anatomix extractor without leaving nitorch's existing
registration workflow.

**Why this priority**: This is the core capability requested. Without it,
there is no feature at all.

**Independent Test**: Can be fully tested by running `nitorch register` (or
the equivalent Python `make_image()` call) with the new ViT option enabled on
a real single-channel volume, and confirming registration completes and
produces a transform, exactly as it does today with the existing `anatomix=`
option.

**Acceptance Scenarios**:

1. **Given** a single-channel 3D image and a pretrained ViT checkpoint,
   **When** a user enables the anatomix ViT preprocessing option, **Then**
   the image is transformed into a modality-invariant feature representation
   before the registration similarity measure is evaluated, and the
   raw-intensity registration path is unaffected when the option is not
   enabled.
2. **Given** a user has already been using the existing U-Net-based anatomix
   option successfully, **When** they enable the ViT option instead, **Then**
   registration completes using ViT features, with no other required
   changes to their existing setup.

---

### User Story 2 - Combine ViT features with MIND and/or U-Net anatomix (Priority: P2)

A researcher wants to combine the ViT-based features with the existing MIND
and/or U-Net-based anatomix features in the same registration run, so they
can evaluate whether combining modality-invariant feature spaces improves
alignment beyond any single method alone.

**Why this priority**: Builds directly on the existing joint MIND+anatomix
concatenation capability. Valuable, but secondary to having the ViT option
work on its own.

**Independent Test**: Can be tested by enabling MIND, U-Net anatomix, and ViT
anatomix together on the same pair of images and confirming the resulting
per-level feature representation is the channel-wise concatenation of all
enabled feature sets, with registration completing normally.

**Acceptance Scenarios**:

1. **Given** both U-Net anatomix and ViT anatomix are enabled together,
   **When** preprocessing runs, **Then** the feature channels from both are
   concatenated into a single multi-channel representation, mirroring how
   MIND and anatomix are already concatenated today.
2. **Given** MIND, U-Net anatomix, and ViT anatomix are all enabled together,
   **When** preprocessing runs, **Then** all three feature sets are
   concatenated without error.

---

### User Story 3 - Frictionless access to ViT weights (Priority: P3)

A researcher without local ViT weights wants nitorch to fetch the pretrained
`anatomix-dev-vit` checkpoint automatically from its public hosting location, so
they can start using the feature without manually locating and downloading
model weights.

**Why this priority**: Improves ease of adoption, but registration with a
manually supplied local checkpoint already satisfies the core need — this is
a convenience layer on top of User Story 1.

**Independent Test**: Can be tested by enabling the ViT option with no local
weights path supplied and confirming the system downloads the public
`anatomix-dev-vit` checkpoint automatically, caches it, and reuses the cached
copy on subsequent runs — mirroring the existing anatomix U-Net
weight-resolution behavior exactly.

**Acceptance Scenarios**:

1. **Given** no local ViT checkpoint path is supplied and automatic download
   is requested, **When** preprocessing runs for the first time, **Then**
   the public `anatomix-dev-vit` checkpoint is fetched once and cached for later
   reuse, exactly as the existing anatomix U-Net checkpoint is.
2. **Given** no local checkpoint path is supplied, automatic download is not
   requested, and none is cached, **When** preprocessing runs, **Then** a
   clear, actionable error is raised instead of a silent failure or generic
   crash.

---

### Edge Cases

- What happens when the ViT variant is enabled on a multi-channel input image
  (the existing U-Net variant requires single-channel input)?
- How does the system handle an input volume smaller than the ViT's fixed
  working resolution (pad) versus larger than it (tile with sliding
  -window, then reassemble)? (Resolved: `research.md` §2 — pad when every
  axis is ≤ the fixed size, tile with blended overlap otherwise.)
- What happens when a user enables the ViT variant with a checkpoint file
  that does not match the expected architecture (corrupt or wrong file)?
- What happens when the ViT and U-Net variants are both requested but their
  output feature channel counts differ (should concatenation still
  proceed)? (Resolved: yes — channel-wise concatenation is agnostic to how
  many channels each source contributes; no validation is needed, and this
  is exercised incidentally by the combinability tests since the U-Net's
  and ViT's `output_nc` are not expected to match.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to enable the anatomix 3D ViT
  feature-extraction preprocessing step through the same registration entry
  points (CLI and Python API) that already expose MIND and U-Net anatomix
  preprocessing.
- **FR-002**: The system MUST transform single-channel input image data into
  a modality-invariant feature representation using the ViT model before the
  registration similarity measure is evaluated, at every pyramid level,
  mirroring how U-Net anatomix features are computed today. Because the ViT
  architecture requires a fixed input size, the system MUST transparently
  pad and/or tile (sliding-window) arbitrarily-shaped input volumes to
  produce features covering the full input, without requiring the caller to
  pre-shape their data.
- **FR-003**: The system MUST leave existing MIND-only and U-Net-anatomix
  -only registration behavior completely unchanged when the ViT option is
  not enabled.
- **FR-004**: Users MUST be able to enable the ViT preprocessing step
  together with MIND and/or the existing U-Net anatomix preprocessing step
  in the same registration run, with all enabled feature sets concatenated
  into a single multi-channel representation.
- **FR-005**: The system MUST support supplying a local pretrained ViT
  checkpoint path, and MUST support automatically fetching the default
  pretrained checkpoint (hosted as `anatomix-dev-vit`) when no local path is
  supplied, consistent with the existing anatomix U-Net weight-resolution
  behavior.
- **FR-006**: The system MUST raise a clear, actionable error (not a generic
  crash) when ViT preprocessing is requested but no usable checkpoint can be
  resolved.
- **FR-007**: The system MUST raise a clear, actionable error when the ViT
  preprocessing step is applied to a multi-channel input image (single
  -channel input is required), or to an input whose spatial shape cannot be
  handled even after padding/tiling (e.g., a degenerate zero-size
  dimension).
- **FR-008**: The system MUST document, in user-facing help/reference text,
  how the ViT preprocessing option differs from the existing U-Net anatomix
  option, so users can make an informed choice between them.

### Key Entities

- **Anatomix ViT Feature Extractor**: A pretrained 3D Vision Transformer
  model that maps a single-channel image volume to a modality-invariant
  multi-channel feature representation; sourced from the anatomix project's
  development branch, analogous in role to the existing vendored anatomix
  U-Net.
- **Preprocessing Option Set**: The collection of feature-extraction
  preprocessing steps (MIND, U-Net anatomix, ViT anatomix) that a
  registration run may enable independently or in combination.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can enable ViT-based preprocessing and complete a
  registration run using only documentation and existing familiarity with
  the current MIND/anatomix options, without needing to consult source code.
- **SC-002**: Registration runs that do not enable ViT preprocessing produce
  identical results before and after this feature is added (zero
  regression).
- **SC-003**: A user can combine ViT preprocessing with the existing MIND
  and/or U-Net anatomix preprocessing in a single run without any additional
  configuration beyond enabling each option.
- **SC-004**: When no usable ViT checkpoint is available, 100% of affected
  registration attempts fail with a specific, actionable error message
  rather than an unrelated or generic exception.

## Assumptions

- The pretrained 3D ViT checkpoint (`anatomix-dev-vit`) is publicly hosted on
  HuggingFace, the same distribution mechanism already used for the
  existing anatomix U-Net checkpoint; a local-path override remains
  available for users who supply their own file. The separately-mentioned
  "anatomix+brains" checkpoint is an alternative weight set for the
  *existing U-Net* extractor and is out of scope for this feature.
- The ViT option is combinable (channel-concatenated) with MIND and/or the
  existing U-Net anatomix option, per Clarifications above — not mutually
  exclusive with them.
- The ViT preprocessing step operates on the same single-channel-input
  assumption as the existing anatomix U-Net extractor.
- This feature reuses the same integration points already used by
  MIND/anatomix (the `nitorch register` CLI and the underlying Python
  image-preprocessing API); no new registration entry points are
  introduced.
- Performance/runtime characteristics of the ViT model (e.g., relative to
  the U-Net) are out of scope for this specification and will be evaluated
  during implementation.
