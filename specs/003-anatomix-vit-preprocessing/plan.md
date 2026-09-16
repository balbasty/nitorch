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

**Primary Dependencies**: PyTorch only (`torch`), matching the existing anatomix U-Net's "no new hard dependency" precedent (`nitorch/_models/anatomix/__init__.py` docstring). Checkpoint download reuses `urllib` (stdlib), already used by `weights.py`.

**Storage**: N/A (stateless preprocessing transform); downloaded checkpoint cached to the same user cache directory the U-Net extractor already uses (`_default_cache_dir()`).

**Testing**: `pytest`, matching the existing `nitorch/tests/test_anatomix_image.py` / `nitorch/tests/test_anatomix_extraction.py` conventions (synthetic small checkpoints as fixtures, no real network access required for unit tests).

**Target Platform**: Same as the rest of nitorch — Linux/HPC and general Python environments, CPU or CUDA GPU.

**Project Type**: Library (Python package) with a CLI entry point — extends existing `nitorch.tools.registration` / `nitorch.cli.registration.register` modules; no new project type.

**Performance Goals**: Out of scope per spec Assumptions (ViT vs. U-Net runtime/memory comparison deferred to evaluation, not a shipping requirement). Sliding-window tiling for large inputs must remain a single-pass, non-recursive algorithm (no unbounded runtime blowup), but no specific throughput target is required.

**Constraints**: The ViT's fixed 128³ input size (`research.md` §2) is the core technical constraint driving `SlidingWindowRunner`'s design. Exact architecture hyperparameters (patch size, embed dim, depth, heads) are unknown until the real `anatomix-dev-vit` checkpoint is downloaded and its `state_dict` inspected during implementation (`research.md` §4) — `AnatomixViT`'s constructor signature should therefore avoid hard-coding guessed values as defaults; they should be derived from the checkpoint or made explicit constructor arguments filled in once known.

**Scale/Scope**: Single new preprocessing option, additive to two existing ones; touches `nitorch/_models/anatomix/` (new files), `nitorch/tools/registration/pairwise_makeobj.py` (extend existing `mind`/`anatomix` handling), and `nitorch/cli/registration/register/parser.py` + `cli.py` (new `--anatomix-vit` option mirroring `--anatomix`).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Code Quality**: PASS. Follows the exact existing convention set by the U-Net anatomix integration (vendored, dependency-light, PyTorch-only implementation; docstrings on public classes/functions; no speculative generalization beyond what MIND+anatomix already established for combinability).
- **II. Atomic & Regular Commits**: PASS (process constraint, not a design gate) — implementation will be delivered as separate atomic commits per logical unit (vendored `AnatomixViT` module, `SlidingWindowRunner`, `make_image()` integration, CLI option, tests), matching how the original anatomix U-Net feature (spec 001) was delivered.
- **III. Testing Discipline**: PASS. `contracts/anatomix-vit-api.md` §5 and `quickstart.md` enumerate the required automated tests (regression guard for unchanged existing behavior, shape-handling pad/tile paths, combinability, error paths) up front, before implementation.

No constitution violations requiring justification — no entries needed in Complexity Tracking.

*Post-Phase-1 re-check*: PASS, unchanged. `data-model.md` and the contract confirm the design reuses existing infrastructure (`resolve_weights_path`, the `mind`/`anatomix` concatenation pattern) rather than introducing parallel/competing mechanisms, and introduces exactly one genuinely new component (`SlidingWindowRunner`), scoped to the one genuinely new problem (fixed input size) this feature has that the U-Net didn't.

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

*No violations — table intentionally omitted.*
