# Feature Specification: Fix mind=True Crash

**Feature Branch**: `002-fix-mind-bug`

**Created**: 2026-09-04

**Status**: Draft

**Input**: User description: "fix the mind=True bug"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use MIND-based feature registration without crashing (Priority: P1)

As a user of nitorch's registration tooling, I want to build an image pyramid with
MIND (Modality Independent Neighbourhood Descriptor) features enabled, so that I can
register images using this existing, modality-agnostic feature option instead of raw
intensities.

**Why this priority**: This is a previously-available capability that currently
crashes for any use of its default configuration, making it entirely unusable. There
is no lower-priority scope to this fix — it either works or it doesn't.

**Independent Test**: Can be fully tested by building an image pyramid with MIND
features enabled on a representative 3D volume, using default settings, and confirming
it completes successfully and returns the expected feature maps rather than raising an
error.

**Acceptance Scenarios**:

1. **Given** a 3D image volume, **When** a user builds an image pyramid with MIND
   features enabled using default settings, **Then** the operation completes
   successfully and returns MIND feature maps in place of the original intensities.
2. **Given** a 3D image volume and an explicitly-specified boundary condition that
   already works correctly today (e.g., one other than the default), **When** a user
   builds an image pyramid with MIND features enabled, **Then** the operation continues
   to complete successfully and produce the same output as before this fix.

---

### Edge Cases

- What happens when MIND features are requested together with the default boundary
  condition versus a non-default one that already works — does the fix apply to both
  consistently?
- What happens for other, currently-untested code paths that rely on the same shared
  boundary-condition-handling mechanism that MIND features use — are they fixed
  consistently as well, or does the fix apply narrowly only to the MIND code path?
- What happens when MIND features are requested with alternate spellings/aliases of the
  same default boundary condition (e.g., a "zeros" or "constant" alias) — do they all
  behave consistently after the fix?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow building an image pyramid with MIND features enabled,
  using default settings, without raising an error.
- **FR-002**: System MUST correctly support the default boundary condition wherever the
  shared mechanism MIND features rely on to handle out-of-bounds neighborhood sampling
  is used, consistent with how the other boundary conditions it already supports behave.
- **FR-003**: The fix MUST NOT change behavior for any boundary condition or call site
  that already produces correct output today.
- **FR-004**: This bug fix MUST be accompanied by an automated test that fails before
  the fix is applied and passes after, per the project's testing discipline.

### Key Entities

- **MIND Feature Transform**: The existing, previously-available option to compute
  Modality Independent Neighbourhood Descriptor features when building an image
  pyramid, currently broken for its default configuration.
- **Boundary Condition**: The out-of-bounds handling rule (e.g., zero-padding,
  replication) used when sampling neighboring voxels; the default one is currently
  broken for the shared mechanism the MIND feature transform depends on.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Building an image pyramid with MIND features enabled, using default
  settings, completes successfully and returns feature maps 100% of the time (no
  crash), for representative 3D volumes.
- **SC-002**: Every boundary condition that produces correct output today continues to
  produce identical output after the fix (zero behavioral regressions).
- **SC-003**: The project's full existing automated test suite continues to pass at
  100% after the fix is applied, with no new failures introduced.

## Assumptions

- The fix addresses the shared, underlying mechanism responsible for the crash (rather
  than working around it narrowly only for the MIND feature transform specifically),
  since that mechanism is not exclusive to MIND features and a narrow workaround would
  leave the same crash available to any other current or future caller relying on the
  default boundary condition through that shared mechanism.
- This crash is unrelated to and predates any other recent feature work in this
  repository; it reproduces on a clean checkout with no other changes applied.
- No change to the public signature or documented behavior of the MIND feature option
  is required — only its currently-broken default-configuration behavior is being
  corrected.
