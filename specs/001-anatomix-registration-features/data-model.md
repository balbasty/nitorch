# Phase 1 Data Model: Anatomix Feature-Based Registration

Derived from the spec's Key Entities section, made concrete against nitorch's existing
image-pyramid architecture (`nitorch/tools/registration/pairwise_makeobj.py::make_image`,
`nitorch/tools/registration/objects.py::ImagePyramid`) — the same architecture the
existing `mind=` feature transform already uses (see `research.md` §1).

## AnatomixFeatureExtractor

The pretrained component that maps a 3D image volume to a modality-agnostic feature
representation (spec: "Anatomix Feature Extractor"). Lives in
`nitorch/_models/anatomix/`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `weights_path` | `str \| None` | `None` | Local filesystem path to a `.pth` checkpoint. Required unless `auto_download=True`. |
| `auto_download` | `bool` | `False` | Opt-in flag to fetch weights from the official anatomix HuggingFace Hub distribution and cache them locally when `weights_path` is not supplied. |
| `num_downs` | `int` | `4` | Architecture: number of U-Net downsampling levels. Must match the loaded checkpoint. |
| `ngf` | `int` | `16` | Architecture: base channel multiplier. |
| `output_nc` | `int` | `16` | Architecture: number of output feature channels. |
| `norm` | `{'batch', 'instance', 'none'}` | `'batch'` | Architecture: normalization layer type. |
| `interp` | `{'nearest', 'trilinear'}` | `'nearest'` | Architecture: decoder upsampling mode. |
| `pooling` | `{'max', 'avg'}` | `'max'` | Architecture: pooling type. |
| `model` | vendored 3D U-Net (`nitorch/_models/anatomix/unet.py`) | — | Constructed from the fields above, loaded from `weights_path`/download, and frozen (`requires_grad_(False)` on all parameters). |
| `device` | `torch.device` | — | Not separately configured; the module is moved to match the input tensor's device on each call. |

**Validation rules**:
- Constructing/using an extractor with `weights_path is None` and `auto_download is
  False` MUST raise a descriptive error before any computation is attempted (FR-006).
- A download or checkpoint-load failure MUST raise a descriptive error identifying the
  cause (network failure, missing file, corrupt/incompatible checkpoint) and a remedy
  (FR-006, SC-005).

**Lifecycle**: Stateless from the caller's perspective beyond the loaded, frozen
weights — one extractor instance can be reused across multiple `extract_features`
calls, and across multiple pyramid levels within a single `make_image()` call.

## `anatomix=` configuration (as passed to `make_image()`)

Not a persistent entity — the parameter value itself, mirroring `mind=`'s
shorthand-vs-explicit shape (see `research.md` §5):

| Form | Meaning |
|---|---|
| `None` (default) | Feature transform disabled; `level.dat` stays as raw intensities. |
| `str` | Shorthand for `{"weights_path": str}`; all architecture fields take their defaults. |
| `True` | Shorthand for `{"auto_download": True}`; all architecture fields take their defaults. |
| `dict` | Full form — any subset of `AnatomixFeatureExtractor`'s fields above. |

## ImagePyramid / level (existing entity, now optionally carrying feature data)

No new type — `nitorch/tools/registration/objects.py::ImagePyramid` and its per-level
`dat`/`preview` fields already exist (this is exactly how `mind` works today). When
`anatomix=` is set, `make_image()`'s existing `for level in image:` loop (the same one
`mind` uses) additionally does:

```python
level.preview = level.dat                                  # original intensities, kept for display
level.dat = extract_features(level.dat, **extractor_config)  # (1, output_nc, *spatial)
```

for each pyramid level, using one shared `AnatomixFeatureExtractor` instance across all
levels of a given image. Downstream, `Similarity(loss, moving, fixed)` and whichever
existing loss the user selected (`make_loss('lcc')`, `'cc'`, etc.) then operate on the
now-feature-valued `dat` exactly as they already do for raw intensities or MIND
features — no change to `Similarity`, `make_loss()`, or any `OptimizationLoss`
subclass.

## Pretrained Weights (external artifact)

Not a code entity — the `.pth` checkpoint file itself.

| Attribute | Value |
|---|---|
| Source | Official anatomix distribution (HuggingFace Hub, or a local copy the user supplies) |
| License | MIT (compatible with nitorch's MIT license — see `research.md` §3) |
| Format | PyTorch `state_dict`, loadable via `torch.load` into the vendored U-Net (`nitorch/_models/anatomix/unet.py`), given matching architecture parameters |
| Required by | `AnatomixFeatureExtractor` (both the `make_image(anatomix=...)` wiring and standalone `extract_features`) |
