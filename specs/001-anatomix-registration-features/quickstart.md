# Quickstart: Validating Anatomix Feature-Based Registration

Manual/scripted validation scenarios proving the feature works end-to-end. See
`contracts/anatomix-api.md` for the exact API shapes and `data-model.md` for entity
details. Anatomix is a feature-transform option (mirroring the existing `mind=`
parameter) — no new loss is introduced.

## Prerequisites

- nitorch installed (no new hard dependency — see `research.md` §9).
- An anatomix pretrained checkpoint (`.pth`) obtained from the official anatomix
  distribution and saved locally, e.g. `~/weights/anatomix.pth`. (Automatic download is
  available but off by default — see below.)
- Two 3D volumes for a registration test, ideally from different modalities/contrasts
  (satisfies US1); a single volume suffices for the extraction-only check (US2).

## Scenario 1 — Standalone feature extraction (US2)

```python
import torch
from nitorch._models.anatomix import extract_features

volume = torch.rand(1, 1, 64, 64, 64)  # or a loaded real volume
features = extract_features(volume, weights_path="~/weights/anatomix.pth")

assert features.shape == (1, 16, 64, 64, 64)
assert features.device == volume.device
```

**Expected outcome**: a 16-channel feature tensor is returned without performing any
registration (FR-007, SC-003).

## Scenario 2 — Cross-modality registration using anatomix features (US1)

```python
from nitorch.tools.registration.pairwise_makeobj import make_image

fixed = make_image(fixed_dat, anatomix="~/weights/anatomix.pth")
moving = make_image(moving_dat, anatomix="~/weights/anatomix.pth")
# `fixed`/`moving` now carry 16-channel anatomix features in place of intensities;
# hand them to the existing pairwise registration entry point exactly as before,
# selecting any existing loss (e.g. loss='lcc') — see
# nitorch/tools/registration/pairwise_run.py
```

**Expected outcome**: registration converges and produces a spatial transform through
the existing output/reporting path (FR-005), unchanged from how any other loss's
transform is produced — because, from the loss's point of view, nothing changed: it is
still just comparing two `(C, *spatial)` tensors.

**Validation against SC-002**: on a representative multimodal pair, compute an existing
nitorch registration-quality metric (e.g. `losses.Dice` on available label maps, or
`losses.LCC` on intensities as an independent post-hoc check) for both the
anatomix-based result and an intensity-based baseline (`anatomix=None`, `loss='lcc'`
directly on intensities), and confirm the anatomix-based result scores better.

## Scenario 3 — Switching only the feature transform leaves the rest of the workflow unchanged (US1, AS2)

Take an existing working registration configuration (any transformation model,
optimizer, loss, output settings), change only `make_image`'s `anatomix=` argument
(disabled → a valid weights path), and confirm the run completes and produces output
through the same code path — no other configuration changes required. This mirrors how
toggling `mind=` alone already works today.

## Scenario 4 — Missing weights produce a clear error (US3)

```python
from nitorch.tools.registration.pairwise_makeobj import make_image

try:
    make_image(dat, anatomix=True)  # auto_download=True but e.g. offline / unreachable
except Exception as e:
    print(e)  # MUST name the missing prerequisite and how to resolve it
```

**Expected outcome**: a descriptive error is raised — not a bare crash or silent
failure (FR-006, SC-005). Repeat with `anatomix="/nonexistent/path.pth"` to cover the
other failure branch documented in User Story 3's acceptance scenarios.

## Scenario 5 — Existing behavior is unaffected (regression check, FR-008)

Run the existing test suite (or a quick manual check) with `anatomix=None` (the
default) and confirm `make_image()`'s output and every existing loss's behavior (e.g.
`'mse'`, `'lcc'`, and `mind=`-based registration) is identical to before this feature
was added — `make_loss()` and every `OptimizationLoss` subclass are untouched by this
feature.
