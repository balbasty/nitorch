# Implementation Plan: Anatomix 3D ViT Preprocessing

**Branch**: `003-anatomix-vit-preprocessing` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-anatomix-vit-preprocessing/spec.md`

## Summary

Add anatomix's experimental 3D Vision Transformer (`anatomix-dev-vit`, 26M
params) as a third, independently-toggleable modality-invariant
feature-extraction preprocessing step, alongside the existing MIND and
U-Net-based anatomix options — combinable with either or both (channel
-concatenated), reachable through the exact same `make_image()`/`nitorch
register` entry points, and reusing the existing weight-resolution
machinery (`resolve_weights_path`) unmodified via a new `variant`. The one
genuinely new piece of engineering is input-shape handling: the ViT requires
a fixed 128³ input, so a `SlidingWindowRunner` transparently pads (small
inputs) or tiles-with-blended-overlap (large inputs) so callers never have
to pre-shape their data — mirroring the external contract the U-Net
extractor already presents (any shape in, same shape out).

## Technical Context

**Language/Version**: Python 3.x (matches existing `nitorch` codebase; no version change)

**Primary Dependencies**: `torch` (existing), plus a **new optional
dependency**, `dynamic_network_architectures` (PyPI, real published
nnU-Net-ecosystem package), added as a new optional extra — revised during
implementation (T001, see `research.md` §4) once the real `anatomix-dev-vit`
architecture (`PrimusV2`: CNN tokenizer + EVA transformer + CNN decoder) was
found to be a genuine hybrid research architecture, not a small vanilla ViT
reasonably vendored dependency-free like the U-Net. Checkpoint download
still reuses `urllib` (stdlib), already used by `weights.py`, unchanged.

**Storage**: N/A (stateless preprocessing transform); downloaded checkpoint cached to the same user cache directory the U-Net extractor already uses (`_default_cache_dir()`).

**Testing**: `pytest`, matching the existing `nitorch/tests/test_anatomix_image.py` / `nitorch/tests/test_anatomix_extraction.py` conventions (synthetic small checkpoints as fixtures, no real network access required for unit tests).

**Target Platform**: Same as the rest of nitorch — Linux/HPC and general Python environments, CPU or CUDA GPU.

**Project Type**: Library (Python package) with a CLI entry point — extends existing `nitorch.tools.registration` / `nitorch.cli.registration.register` modules; no new project type.

**Performance Goals**: Out of scope per spec Assumptions (ViT vs. U-Net runtime/memory comparison deferred to evaluation, not a shipping requirement). Sliding-window tiling for large inputs must remain a single-pass, non-recursive algorithm (no unbounded runtime blowup), but no specific throughput target is required.

**Constraints**: The ViT's fixed 128³ input size (`research.md` §2) is the core technical constraint driving `SlidingWindowRunner`'s design. Exact architecture hyperparameters (patch size, embed dim, depth, heads) are unknown until the real `anatomix-dev-vit` checkpoint is downloaded and its `state_dict` inspected during implementation (`research.md` §4) — `AnatomixViT`'s constructor signature should therefore avoid hard-coding guessed values as defaults; they should be derived from the checkpoint or made explicit constructor arguments filled in once known.

**Scale/Scope**: Single new preprocessing option, additive to two existing ones; touches `nitorch/_models/anatomix/` (new files), `nitorch/tools/registration/pairwise_makeobj.py` (extend existing `mind`/`anatomix` handling), and `nitorch/cli/registration/register/parser.py` + `cli.py` (new `--anatomix-vit` option mirroring `--anatomix`).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Code Quality**: PASS. Follows the existing convention set by the U-Net anatomix integration (docstrings on public classes/functions; no speculative generalization beyond what MIND+anatomix already established for combinability). One deviation, justified below: the new optional dependency on `dynamic_network_architectures`, since the real architecture is not reasonably vendored dependency-free — see Complexity Tracking.
- **II. Atomic & Regular Commits**: PASS (process constraint, not a design gate) — implementation will be delivered as separate atomic commits per logical unit (new optional extra + dependency, vendored `AnatomixViT` wrapper module, `SlidingWindowRunner`, `make_image()` integration, CLI option, tests), matching how the original anatomix U-Net feature (spec 001) was delivered.
- **III. Testing Discipline**: PASS. `contracts/anatomix-vit-api.md` §5 and `quickstart.md` enumerate the required automated tests (regression guard for unchanged existing behavior, shape-handling pad/tile paths, combinability, error paths) up front, before implementation.

*Post-Phase-1 re-check (post-T001 architecture discovery)*: PASS, with one
tracked, user-approved exception (see Complexity Tracking at the end of this
document). `data-model.md` and the contract still hold at the API-contract
level (`AnatomixViTFeatureExtractor`'s external shape contract is
unchanged); the *internal* implementation of `AnatomixViT` now wraps
`dynamic_network_architectures.PrimusV2` plus anatomix's own small
QK-norm/demean addition (`research.md` §4, verified via a strict,
zero-mismatch `load_state_dict` against the real checkpoint) rather than a
from-scratch reimplementation. `SlidingWindowRunner` remains the one
genuinely new component, unaffected by this change.

## Project Structure

### Documentation (this feature)

```text
specs/003-anatomix-vit-preprocessing/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   └── anatomix-vit-api.md
└── tasks.md              # Phase 2 output (/speckit-tasks - not created here)
```

### Source Code (repository root)

```text
setup.cfg   # MODIFIED: add a new `anatomix-vit` optional extra
            #   (dynamic_network_architectures), mirroring the existing
            #   `zarr`/`dask` optional-extra pattern from the nifti-zarr
            #   feature; not part of the default install

nitorch/_models/anatomix/
├── __init__.py         # MODIFIED: add AnatomixViTFeatureExtractor,
│                       #   VIT_ARCHITECTURE_DEFAULTS, and generalize
│                       #   _normalize_anatomix_config into a shared
│                       #   helper reused by both anatomix= and
│                       #   anatomix_vit=
├── unet.py             # unchanged (existing AnatomixUNet)
├── vit.py              # NEW: AnatomixViT (nn.Module), architecture
│                       #   params filled in from the real checkpoint
│                       #   during implementation
├── sliding_window.py   # NEW: SlidingWindowRunner (pad-or-tile+blend)
└── weights.py          # MODIFIED (minimal): confirm/extend
                        #   resolve_weights_path's `variant` handles
                        #   'anatomix-dev-vit' (likely already works
                        #   unmodified, per research.md §1); ViT-specific
                        #   state_dict loading/remapping added alongside
                        #   the existing U-Net-specific
                        #   `_remap_flat_sequential_state_dict`

nitorch/tools/registration/
└── pairwise_makeobj.py  # MODIFIED: add `anatomix_vit=` parameter to
                         #   make_image(), extend the existing
                         #   mind/anatomix concatenation block to a third
                         #   optional feature source

nitorch/cli/registration/register/
├── parser.py   # MODIFIED: add `--anatomix-vit` file-group option
│               #   alongside existing `--mind`/`--anatomix`; update help
│               #   text (FR-008)
└── cli.py      # MODIFIED: thread `anatomix_vit` through to make_image()
                #   the same way `anatomix` already is

nitorch/tests/
├── test_anatomix_vit_extraction.py  # NEW: AnatomixViTFeatureExtractor
│                                    #   standalone tests (shapes, pad vs.
│                                    #   tile paths, error paths)
└── test_anatomix_image.py           # MODIFIED: add ViT + combinability
                                     #   tests alongside the existing
                                     #   mind+anatomix ones
```

**Structure Decision**: Single-project library structure (matches the rest
of nitorch — no new top-level project/package). The new ViT model and its
sliding-window wrapper are vendored as siblings to the existing U-Net inside
`nitorch/_models/anatomix/`, exactly mirroring how `unet.py` sits alongside
`weights.py` today; the public extractor classes both live in
`anatomix/__init__.py` so `from nitorch._models.anatomix import
AnatomixFeatureExtractor, AnatomixViTFeatureExtractor` reads symmetrically.
Registration-pipeline integration touches the same two files (
`pairwise_makeobj.py`, the CLI parser/cli) that the original MIND/anatomix
work already touched, extending rather than replacing their existing
patterns.

## Complexity Tracking

> One deviation, discovered during T001 (not assumed up front) and
> explicitly discussed with and approved by the user before proceeding.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|---------------------------------------|
| New optional dependency: `dynamic_network_architectures` (Principle I's established "no new hard dependency" precedent from the U-Net) | The real `anatomix-dev-vit` architecture (`PrimusV2`: 4-stage residual CNN tokenizer + 12-block EVA transformer with SwiGLU/LayerScale/QK-norm + 3-stage transpose-conv decoder) is a substantial published research architecture, not a small model reasonably vendored dependency-free the way the 6M-param U-Net was. Verified via `research.md` §4: a strict `load_state_dict(strict=True)` against the real checkpoint succeeds with zero key/shape mismatches once `PrimusV2` is constructed with anatomix's documented kwargs. | Hand-reimplementing `PrimusV2` bit-exactly (CNN tokenizer InstanceNorm epsilons, EVA attention/SwiGLU/LayerScale internals, transpose-conv decoder) was considered and rejected: substantially more implementation effort than depending on the verified-correct upstream package, and — critically — a subtle mistake in a from-scratch reimplementation would not necessarily crash (shapes could still match) but could silently produce numerically-wrong features, a harder class of bug to catch than a dependency's already-tested behavior. |
