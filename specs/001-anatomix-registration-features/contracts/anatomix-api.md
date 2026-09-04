# Contract: Anatomix Feature API

nitorch is a library, so its "interface contract" is the public Python API surface
(also reachable through the existing `nitorch` CLI). This document fixes the shape of
that surface so implementation and tests agree on it.

Anatomix is a **feature-transform / preprocessing option**, mirroring the existing
`mind=` parameter — it is not a new loss. `pairwise_makeobj.make_loss()` and every
existing `OptimizationLoss` subclass are unmodified by this feature (`research.md` §1).

## 1. Image-construction contract (`make_image`)

`nitorch/tools/registration/pairwise_makeobj.py::make_image` gains a new parameter,
alongside the existing `mind`, `discretize`, `soft`, ...:

```python
make_image(
    dat, mask=None, affine=None,
    pyramid=0, pyramid_method='gaussian',
    discretize=False, soft=False, mind=None,
    anatomix=None,     # NEW: None | str | True | dict — see data-model.md
    bound='zero', extrapolate=False, **kwargs,
) -> ImagePyramid
```

- `anatomix=None` (default): behavior identical to today — `level.dat` stays raw
  intensities.
- `anatomix=<str>`: shorthand for `{"weights_path": <str>}`.
- `anatomix=True`: shorthand for `{"auto_download": True}`.
- `anatomix=<dict>`: full `AnatomixFeatureExtractor` config (see `data-model.md`).
- When set (any non-`None` form), for every pyramid level: `level.preview = level.dat`
  (original intensities preserved for display, mirroring `mind`'s own `preview`
  handling), then `level.dat` is replaced by the extracted feature map
  (`(1, output_nc, *spatial)`, `output_nc=16` by default).
- **Precondition**: exactly one weight source must resolve — either `weights_path`
  (direct or via the string shorthand) points to a readable checkpoint, or
  `auto_download=True` and retrieval succeeds.
- **Error contract**: if neither weight source is usable, or the checkpoint fails to
  load (missing file, network failure, corrupt/incompatible checkpoint), `make_image`
  raises a `RuntimeError` (or a dedicated `AnatomixWeightsError` subclass) whose
  message names the missing prerequisite and how to resolve it (FR-006, SC-005). It
  MUST NOT raise a bare/unlabeled exception (e.g. an unqualified
  `KeyError`/`AttributeError` from a failed download or `torch.load`).
- Every other existing `make_image` parameter and its behavior is unchanged
  (regression contract for FR-008).

## 2. Standalone feature extraction contract (User Story 2)

```python
from nitorch._models.anatomix import extract_features

extract_features(
    volume: torch.Tensor,          # shape (1, 1, *spatial) or (*spatial,)
    weights_path: str | None = None,
    auto_download: bool = False,
    num_downs: int = 4,
    ngf: int = 16,
    output_nc: int = 16,
    norm: str = 'batch',
    interp: str = 'nearest',
    pooling: str = 'max',
) -> torch.Tensor                   # shape (1, output_nc, *spatial)
```

- Same weight-resolution precondition and error contract as §1.
- Pure function with respect to its inputs: no full registration workflow objects
  required (satisfies US2 / FR-007). Internally reused by `make_image()`'s `anatomix=`
  wiring for each pyramid level.
- Output feature map is on the same device as the input `volume`.

## 3. CLI contract (v1)

`nitorch register`, per-image options block (`@@fix`/`@@mov`), mirroring the existing
`--mind [FWHM=1 [RADIUS=0]]`:

```
--anatomix [PATH]     Compute anatomix features (path to local weights;
                       bare flag opts into automatic download)
```

- Wired in `nitorch/cli/registration/register/parser.py` (flag definition) and
  `nitorch/cli/registration/register/cli.py:296` (added to the existing
  keyword-forwarding tuple `('soft', 'bound', 'extrapolate', 'mind')` →
  `('soft', 'bound', 'extrapolate', 'mind', 'anatomix')`).
- Architecture-override parameters (`num_downs`, `ngf`, `output_nc`, `norm`, `interp`,
  `pooling`) are **not** exposed as CLI flags in v1 — Python-API-only
  (`research.md` §10).

## 4. Backward-compatibility contract

- No existing `make_image` or `make_loss` parameter, default, or return type changes.
- `pairwise_makeobj.make_loss()` source is untouched by this feature.
- `nitorch`'s `install_requires` in `setup.cfg` is unchanged; the feature adds no new
  hard dependency (`research.md` §9).

## 5. Test contract

Each contract clause above MUST have a corresponding automated test (Constitution
Principle III):

- `make_image(dat, anatomix=<valid local checkpoint path>)` produces an `ImagePyramid`
  whose levels' `dat` has `output_nc` channels and whose `preview` holds the original
  intensities.
- `make_image(dat, anatomix=True)` in an environment with no reachable weights raises
  the documented, descriptive error (not a bare crash).
- `make_image(dat, anatomix=None)` (default) is byte-for-byte identical to calling
  `make_image` without the parameter at all (regression guard for FR-008).
- `extract_features(...)` on a synthetic volume returns a tensor of the documented
  shape and device.
- Selecting any existing loss (e.g. `'mse'`, `'lcc'`) on `anatomix`-transformed images
  runs without modification to `make_loss()` or the loss classes (regression guard,
  confirms no coupling was introduced).
- `--anatomix [PATH]` CLI flag is parsed and forwarded to `make_image()` correctly.
