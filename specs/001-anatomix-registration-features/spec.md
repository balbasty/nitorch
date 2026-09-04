# Feature Specification: Anatomix Feature-Based Registration

**Feature Branch**: `001-anatomix-registration-features`

**Created**: 2026-09-04

**Status**: Draft

**Input**: User description: "Add the anatomix feature extraction to nitorch register to help for the registration https://www.neeldey.com/anatomix/"

## Clarifications

### Session 2026-09-04

- Q: Should the system ever download the anatomix pretrained weights over the network automatically, or must weights always come from a path the user provides explicitly? → A: Require an explicit user-supplied local weights path by default; auto-download only if the user opts in.
- Q: What metric should measure "measurably better alignment" in SC-002? → A: Reuse one of nitorch's existing registration quality metrics (already implemented as loss/evaluation functions, e.g., Dice label-overlap or local cross-correlation), rather than defining new bespoke evaluation tooling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Register images across modalities using anatomix features (Priority: P1)

As a researcher aligning two 3D biomedical volumes acquired with different imaging
modalities or contrasts (for example, CT to MRI, or two differently-weighted MRI
sequences), I want to register them by comparing extracted, modality-agnostic anatomix
features instead of raw intensities, so that registration succeeds in cases where
intensity-based similarity metrics fail to find a good alignment.

**Why this priority**: This is the core value of the feature — enabling reliable
registration in exactly the cross-modality/cross-contrast scenario where existing
intensity-based criteria are least reliable. Without this, the feature delivers no value.

**Independent Test**: Can be fully tested by registering a known multimodal image pair
(with an expert-verified or otherwise known-good alignment) using anatomix-based
feature comparison, and confirming the registration converges to an accurate alignment
of corresponding anatomical structures.

**Acceptance Scenarios**:

1. **Given** two 3D volumes of different modalities depicting the same anatomy, **When**
   the user runs registration with anatomix-based feature comparison enabled, **Then**
   the tool produces a spatial transform that aligns corresponding anatomical structures
   in both volumes.
2. **Given** a registration configuration that previously compared images using raw
   intensities, **When** the user enables anatomix-based feature comparison without
   changing anything else, **Then** the rest of the registration workflow (transformation
   model, alignment criterion, optimizer, output handling) continues to work unchanged.

---

### User Story 2 - Extract anatomix features from a single volume (Priority: P2)

As a user, I want to extract anatomix's modality-agnostic feature representation for a
single 3D volume independent of running a full registration, so that I can inspect the
features or reuse them in other analyses.

**Why this priority**: A useful secondary capability explicitly implied by "feature
extraction," but the core registration benefit (User Story 1) can be delivered and
demonstrated without it.

**Independent Test**: Can be fully tested by requesting feature extraction on a single
volume and confirming a feature representation is produced, without needing a second
image or a full registration run.

**Acceptance Scenarios**:

1. **Given** a single 3D biomedical volume, **When** the user requests anatomix feature
   extraction on it, **Then** the system returns the corresponding feature
   representation for that volume.

---

### User Story 3 - Clear failure handling when prerequisites are missing (Priority: P3)

As a user who has not supplied a local path to the pretrained anatomix weights (and has
not opted into automatic download, or has no network access to complete it), I want to
receive a clear, actionable message when I attempt to use anatomix-based registration or
extraction, so that I understand what is missing and how to resolve it.

**Why this priority**: Important for usability and robustness, but not required for the
core value in User Story 1 to be demonstrated in the common case where weights are
available.

**Independent Test**: Can be fully tested by attempting anatomix-based registration or
extraction in an environment lacking the pretrained weights, and confirming a clear,
descriptive error is raised rather than an unhandled crash or silent failure.

**Acceptance Scenarios**:

1. **Given** the user has not supplied a local weights path and has not opted into
   automatic download, **When** the user attempts feature-based registration or
   extraction, **Then** the system reports a clear error explaining that a weights path
   is required (or how to opt into automatic download) instead of crashing or failing
   silently.
2. **Given** the user has opted into automatic download but weights cannot be retrieved
   (e.g., no network access), **When** the user attempts feature-based registration or
   extraction, **Then** the system reports a clear error explaining the retrieval failure
   and how to resolve it, instead of crashing or failing silently.

---

### Edge Cases

- What happens when the two input images have a much smaller or larger field of view, or
  a very different resolution, than what anatomix was trained on?
- How does the system behave in an offline environment where pretrained weights cannot
  be downloaded and no local copy is supplied?
- What happens when a 2D image is provided instead of a 3D volume?
- How does the system behave when the extracted feature maps are too large for available
  memory (e.g., very large or high-resolution volumes)?
- What happens if a user selects anatomix-based alignment together with a transformation
  or workflow designed for label maps (e.g., a Dice-based segmentation-overlap
  criterion) rather than image alignment?
- What happens when the two input volumes have incompatible shapes/orientations that
  the rest of the registration workflow would otherwise accept for intensity-based
  criteria?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a way to extract an anatomix modality-agnostic feature
  representation from a 3D biomedical image volume.
- **FR-002**: System MUST allow registration to compare images via their extracted
  anatomix features instead of raw intensities, as a selectable option in the existing
  pairwise registration workflow, independent of which alignment criterion (e.g.,
  correlation- or intensity-based) is used to score that comparison.
- **FR-003**: System MUST support registering pairs of images that come from different
  imaging modalities or contrasts.
- **FR-004**: System MUST, by default, require the user to supply a local path to the
  pretrained anatomix weights before feature extraction can run. System MUST additionally
  support an explicit opt-in mode in which weights are retrieved automatically from the
  official anatomix distribution and cached locally, for users who choose it.
- **FR-005**: System MUST produce, from an anatomix-based registration run, a spatial
  transform usable by nitorch's existing registration output and reporting mechanisms
  in the same way as transforms produced by existing alignment criteria.
- **FR-006**: System MUST report a clear, actionable error identifying the missing
  prerequisite and a remedy when the required pretrained weights cannot be obtained or
  loaded, rather than crashing or failing silently.
- **FR-007**: System MUST allow a user to run anatomix feature extraction on a single
  volume independent of performing a full registration.
- **FR-008**: System MUST leave existing registration behavior unchanged for users who
  do not enable anatomix-based feature comparison.

### Key Entities

- **Anatomix Feature Extractor**: The pretrained component that maps a 3D image volume
  to a modality-agnostic feature representation; used in place of, or as an alternative
  to, raw image intensities when scoring alignment quality.
- **Feature-Space Comparison**: A selectable mode in the pairwise registration workflow
  in which images are compared via their extracted anatomix features instead of raw
  intensities; whichever alignment criterion the user has chosen (e.g., correlation- or
  intensity-based) then scores that comparison exactly as it would score raw
  intensities.
- **Pretrained Weights**: The trained parameters of the anatomix model, obtained from
  the official anatomix distribution, required before any anatomix feature extraction
  can run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can register two images from different modalities by enabling
  anatomix-based feature comparison, without writing any custom code beyond the normal
  registration configuration.
- **SC-002**: For a representative multimodal image pair where intensity-based
  registration fails to converge or misaligns corresponding structures, anatomix-based
  registration produces a measurably better score on one of nitorch's existing
  registration quality metrics (e.g., Dice label-overlap or local cross-correlation)
  than intensity-based registration achieves on the same pair.
- **SC-003**: A user can obtain the extracted feature representation for a single volume
  in one operation, without performing a full two-image registration.
- **SC-004**: A user who has already obtained the pretrained anatomix weights can enable
  anatomix-based registration by changing a single configuration choice plus supplying
  the weights location; a user who instead opts into automatic retrieval needs no manual
  weight-file management.
- **SC-005**: 100% of attempts to use anatomix-based registration or extraction without
  the required pretrained weights produce a descriptive error identifying the cause and
  remedy, rather than an unhandled crash.

## Assumptions

- Anatomix features are used frozen (no fine-tuning) for the registration and
  extraction use cases described here; fine-tuning anatomix on new data is out of scope
  for this feature.
- Anatomix-based feature comparison is offered as an additional, independently
  selectable option — orthogonal to the choice of alignment criterion — rather than
  replacing raw-intensity comparison by default, so current default registration
  behavior is preserved (FR-008).
- Pretrained anatomix weights are obtained from the official anatomix distribution. By
  default the system requires a user-supplied local weights location (safe for offline
  or network-restricted environments such as HPC clusters); automatic download and
  local caching is available only as an explicit opt-in.
- Input images are 3D biomedical volumes consistent with what nitorch's registration
  tooling already supports; 2D-only workflows are out of scope for this feature.
- Standalone feature extraction (User Story 2) operates on one volume at a time.
- Feature extraction runs on whichever compute device (CPU or GPU) the rest of the
  registration pipeline is already configured to use, with no new device-selection
  mechanism introduced.
