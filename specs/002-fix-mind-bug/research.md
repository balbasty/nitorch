# Phase 0 Research: Fix mind=True Crash

## 1. Exact root cause

- **Finding**: `make_image(dat, mind=True)` defaults to `bound='zero'`
  (`make_image`'s own default parameter). This is forwarded through
  `objects.Image.bound` to `spatial.rmind(level.dat, ..., bound=level.bound)`
  (`nitorch/tools/registration/pairwise_makeobj.py`), which forwards it to
  `nitorch.core.utils.roll(x, ..., bound=bound)`
  (`nitorch/spatial/_finite_differences.py`). `roll` normalizes the bound
  name via `bounds.to_nitorch` (which correctly maps `'zero'`, `'zeros'`,
  `'constant'`, and `BoundType.zero` all to the canonical string `'zero'`),
  then does `getattr(bounds, 'zero' + '_')` — i.e. `getattr(bounds,
  'zero_')`. **`nitorch.core.bounds` has no `zero_` function.** It defines
  `<name>`/`<name>_` index-remapping function pairs for 6 of its 7
  documented canonical boundary types (`replicate`, `dct1`, `dct2`, `dst1`,
  `dst2`, `dft`) but not for `zero`, even though the module's own docstring
  table explicitly lists `zero` as a first-class canonical boundary
  (`"0  0 | a b c d |  0  0"`, i.e. literal zero-padding) alongside the
  other six.
- **Verified independent of any other change**: reproduced via `git stash`
  against a clean `master` checkout with no other modifications applied —
  this is a pre-existing bug, unrelated to any other feature work.
- **Not a bug in `dirichlet`**: `to_nitorch` maps `'dirichlet'` (and
  `'antireflect'`) to `'dst2'`, not to `'zero'` (`bounds.py`:
  `antireflect = dirichlet = dst2`). `dirichlet` here names the
  antireflective DST-II boundary, a different, already-working boundary
  type — not an alias for literal zero-padding. This is a naming quirk in
  the existing codebase, not something this fix needs to touch.
- **Not a bug in the compiled/TorchScript grid backend**: `nitorch._C`'s
  `grid_pull`/`grid_push`/`affine_grid` (used extensively elsewhere,
  including throughout the anatomix feature's own tests with
  `bound='zero'`) already handle zero-boundary correctly. The gap is
  isolated to this one pure-Python index-remapping utility
  (`nitorch.core.bounds`) used by `nitorch.core.utils.roll`.

## 2. Blast radius

- **Decision**: Fix at the root — add `zero`/`zero_` functions to
  `nitorch/core/bounds.py`, matching the existing `<name>`/`<name>_`
  function-pair contract used by `replicate`/`dct1`/`dct2`/`dst1`/`dst2`.
- **Rationale**: `grep` across the codebase shows exactly two call sites of
  `nitorch.core.utils.roll` — both inside
  `nitorch/spatial/_finite_differences.py` (`mind()` and `rmind()`, the two
  MIND-descriptor implementations). Both currently expose the crash only
  when `bound='zero'` is passed through to them (their own defaults are
  `bound='dct2'`, which already works — it is specifically `make_image`'s
  own default of `bound='zero'`, forwarded through unmodified, that
  triggers the crash for the common `make_image(dat, mind=True)` case). No
  other caller of `utils.roll` exists today, so a root-cause fix has a
  small, fully enumerable blast radius: both existing call sites, and any
  future caller of `utils.roll` or `mind`/`rmind` with a zero-family bound.
- **Alternatives considered**: Patch around the crash locally in
  `rmind`/`mind` (e.g. special-casing `bound='zero'` to substitute a
  different boundary before calling `utils.roll`) — rejected, since it
  would leave `utils.roll` itself broken for its documented `'zero'` bound
  for any other/future caller, contradicting the spec's Assumptions
  section and Constitution Principle I (fix the actual defect, not its
  symptom at one call site).

## 3. Fix contract

- **Decision**: Add `zero(i, n)` and `zero_(i, n)` to `bounds.py`, matching
  the exact `(index, multiplier) -> (remapped_index, sign_or_mask)`
  contract already used by every other bound function in the file (see
  `replicate`/`replicate_` for the closest analog — replicate always
  returns multiplier `1` and clamps the index into `[0, n-1]`). For `zero`:
  clamp the index into `[0, n-1]` for safe gathering (as `replicate` does —
  the sampled value is irrelevant once multiplied by 0), and return a
  multiplier of `1` where the *original, unclamped* index was in
  `[0, n-1]` and `0` everywhere else. `nitorch.core.utils.roll` already
  multiplies the gathered value by this returned multiplier
  (`out = inp.flatten()[grid]; out *= mult`), so a `0` multiplier correctly
  realizes "out-of-bounds samples are treated as literal zero."
- **Rationale**: This is exactly the existing contract every other bound
  function already implements; no change to `utils.roll`, `rmind`, `mind`,
  `make_image`, or any caller is needed — the fix is additive-only within
  `bounds.py`.

## 3b. Second latent bug, found during implementation

- **Finding**: Implementing `zero`/`zero_` alone was not sufficient — it exposed a
  second, pre-existing latent defect in `nitorch.core.utils.roll` (and the structurally
  identical `_pad_bound`) that had never been triggered before. `roll` initializes
  `mult = [1] * inp.dim()` (a plain Python int, for every dimension of the *whole*
  tensor, e.g. including a leading channel dimension), then only overwrites `mult[d]`
  for dimensions in `dims` (the spatial ones actually being resampled). Every existing
  bound function that returns a *tensor* multiplier only does so for `dst1`/`dst2`
  (antisymmetric sign flips); `dct1`/`dct2`/`replicate`/`dft` always return the bare
  Python int `1` regardless of input, so `mult` ends up either all-scalar (skips the
  `meshgrid_ij(*mult)` branch entirely) or (for dst1/dst2, apparently never exercised
  through this exact path in existing tests) already broken the same way. `zero`/`zero_`
  is the first bound to return a genuine, always-real per-index tensor multiplier for
  ordinary in-bounds/out-of-bounds data, which reliably triggers
  `any(map(torch.is_tensor, mult))` and then `meshgrid_ij(*mult)` — which fails because
  the untouched channel dimension's `mult[0]` is still the bare int `1`, and
  `torch.meshgrid` requires every argument to be a tensor.
- **Decision**: Fix `roll` and `_pad_bound` (the identical pattern, in the same file) by
  wrapping any remaining scalar entries in `mult` as a 1-element tensor immediately
  before the `meshgrid_ij(*mult)` call — broadcasts correctly against the real
  per-index multipliers from other dimensions, with no change to any dimension that
  already produces a tensor multiplier.
- **Rationale**: This is the same shared-mechanism scope already established in §2 —
  both functions are in `nitorch/core/utils.py`, both have the exact same defect shape,
  and leaving one fixed and one not would be an inconsistent half-fix for what is
  observably the same underlying bug.
- **Test coverage**: `test_make_image_mind_true_no_longer_crashes` in
  `nitorch/tests/test_bounds_zero.py` exercises this exact path (a real image, so
  `zero_`'s multiplier is a genuine, non-degenerate 0/1 tensor) and is the test that
  caught this during implementation (T004, first attempt).

## 4. Test strategy

- **Decision**: Add a regression test that calls
  `nitorch.tools.registration.pairwise_makeobj.make_image(dat, mind=True)`
  (the exact reported failure) and confirms it completes and returns MIND
  feature maps, plus a focused unit test on the new `bounds.zero`/`zero_`
  functions directly (confirming in-bounds indices pass through unchanged
  with multiplier 1, and out-of-bounds indices are clamped with multiplier
  0), per the spec's FR-004 (a test that fails before the fix and passes
  after).
- **Rationale**: Matches Constitution Principle III; the two-level test
  (unit-level on the new bound functions, integration-level on the
  originally-reported `mind=True` crash) gives both precise coverage of the
  fix itself and an end-to-end guard against the exact regression that was
  reported.
