# Discussion: Handling Multiscale NIfTI-Zarr Stores

**Status**: Decided — Option A, with Integration Point 1 for the
registration-pyramid use case (see spec.md's Assumptions, FR-007/FR-008,
and User Story 3).
**Feature**: [spec.md](./spec.md)
**Context**: A nifti-zarr store may contain an OME-Zarr-style multiscale
pyramid (multiple resolution levels of the same volume). The spec's original
draft assumed only the finest level would be exposed and everything else
would be out of scope; that assumption was rejected and this decision is now
deferred to a separate discussion rather than resolved through the usual
`/speckit-clarify` question loop.

This document lays out the options considered so far, for that discussion.

## Why this matters

Whichever option is chosen changes:
- What "loading a nifti-zarr store" actually returns (a single array vs. a
  collection of arrays)
- Whether the existing, format-agnostic loading interface stays unchanged
  for the common case, or gains new arguments/behavior
- What the acceptance tests for this feature need to check
- Whether this feature can reuse an existing nitorch concept
  (`ImagePyramid`, used in registration) or needs a new one

## Options

### Option A — Finest level by default, explicit opt-in to others

Loading a nifti-zarr store returns the finest (full) resolution level
through nitorch's existing, single-array loading interface — unchanged from
how every other format already behaves. A store with additional coarser
levels makes them reachable only if the user explicitly asks for a specific
level (e.g., by index/argument).

- **Pros**: Zero behavior change to the existing loading contract for the
  common case; smallest, most contained implementation; matches how every
  other currently-supported format works (one call → one array).
- **Cons**: A user who wants to work across multiple levels at once (e.g.
  compare or select among them) has to load the same path multiple times
  with different arguments, rather than getting them all up front.

### Option B — Expose all levels as a sequence (mirrors `ImagePyramid`)

Loading a nifti-zarr store returns (or otherwise makes available) all of its
resolution levels at once, as an ordered collection — deliberately similar
to nitorch's existing `ImagePyramid` concept already used in the
registration tools (`nitorch/tools/registration/objects.py`), which
represents "the same volume at several resolutions" today by downsampling a
single image on the fly.

- **Pros**: Familiar shape for anyone already using nitorch's registration
  pyramid; a natural fit if this feature's stores are ever used directly as
  registration inputs (skipping nitorch's own on-the-fly downsampling,
  since the levels already exist on disk); every level is available
  immediately without extra calls.
- **Cons**: Changes what "loading" returns compared to every other format
  (a collection instead of a single array) — the standard loading interface
  would need new handling for this case, or a new, format-specific entry
  point; more surface area to implement and test up front.

## Registration-pyramid integration: making `--levels` reuse the store's native pyramid

A concrete motivating use case for exposing multiple levels at all: `nitorch
register`'s `-l/--levels` option (`@pyramid --levels`) currently always
builds its pyramid by downsampling the finest loaded resolution itself
(`ImagePyramid`'s Gaussian/average/median/stride methods, driven by
`pairwise_pyramid.pyramid_levels()`). For a nifti-zarr input, the desired
behavior is for `--levels` to fetch the store's *already-built* resolution
levels instead of recomputing them.

Two integration points were identified for this:

1. **At the `ImagePyramid`/`make_image` layer.** Give a `MappedArray`
   backend an optional "native levels" capability. When `ImagePyramid` is
   built from a source that has one, it wraps each requested native level
   directly as an `Image` instead of calling its own downsampling — so
   `--levels` transparently gets the zarr's own pyramid, and anything else
   that builds an `ImagePyramid` (not just the CLI) benefits too.
2. **At the `pyramid_levels()`/CLI level-selection layer.** When the loss's
   input is a nifti-zarr, `-l/--levels N:M` indexes directly into the
   store's native levels instead of `pyramid_levels()` synthesizing
   evenly-spaced targets from the base voxel size. `ImagePyramid` stays
   untouched, but this logic would need to be reimplemented for any other
   future format with native pyramids.

Either way, there's a shared reconciliation problem: nitorch's `--levels`
currently assumes it controls a uniform, factor-of-2-ish pyramid it built
itself, while a zarr store's native levels can have an arbitrary count and
downsampling factor. Both integration points need a rule for what happens
when more levels are requested than the store natively has — most likely,
falling back to nitorch downsampling further from the coarsest native
level.

**How this bears on Options A vs. B**: `ImagePyramid` builds itself
level-by-level in a loop, for whatever subset `--levels` asks for (e.g.
`1:3` skips level 0). What that loop needs from the backend is "fetch level
`i`" on demand — which is Option A's primitive (a parameterized single-level
fetch), just exposed one layer down at the reader/backend level rather than
only at the top-level `load()` call. Swapping "downsample this level" for
"fetch native level `i`" inside that existing loop is then a small, local
change. Option B's shape (return all levels as one eager collection)
doesn't fit as cleanly here: it would either materialize levels nobody
asked for, or need per-item laziness added back into the collection — at
which point it is effectively Option A's primitive wearing a collection
wrapper. In other words: integration point 1 *implements Option B's stated
goal* (every level reachable) but is *cheaper to build on Option A's
mechanism* (fetch-by-index) than on Option B's (eager collection).

## Recommendation (not yet decided)

Option A was proposed as the default recommendation going into this
discussion: it keeps the existing loading contract completely unchanged for
the common case (so the already-drafted User Story 1 / SC-001 need no
rework) while still making every level reachable for users who need one.
Option B is the strongest alternative if there's an appetite for a tighter
integration with nitorch's existing pyramid concept in registration — that
tradeoff (bigger surface area now vs. a nicer registration-side integration
later) is the crux of the discussion. The registration-pyramid analysis
above sharpens this further: even the registration-side integration that
motivates Option B turns out to be cheaper to build on Option A's
fetch-by-index primitive, which weakens Option B's main advantage.

## Next step

Once a decision is made, update `spec.md`'s Assumptions section (currently
marked as deferred, pointing here) to state the chosen behavior concretely,
and re-run the spec quality checklist before proceeding to `/speckit-plan`.
