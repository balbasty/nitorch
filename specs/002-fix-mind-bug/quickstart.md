# Quickstart: Validating the mind=True Fix

## Prerequisites

- nitorch installed (no new dependency introduced by this fix).

## Scenario 1 — The originally-reported crash is fixed (US1, SC-001)

```python
import torch
from nitorch.tools.registration.pairwise_makeobj import make_image

dat = torch.rand(1, 32, 32, 32)
image = make_image(dat, mind=True)  # previously raised AttributeError

level = image[0]
assert level.dat.shape[0] > 1          # MIND feature channels
assert torch.equal(level.preview, dat)  # original intensities preserved
```

**Expected outcome**: completes without error and returns MIND feature maps,
matching how `anatomix=` already behaves (see
`specs/001-anatomix-registration-features/`).

## Scenario 2 — Other boundary conditions are unaffected (SC-002)

```python
from nitorch.tools.registration.pairwise_makeobj import make_image

dat = torch.rand(1, 32, 32, 32)
before = make_image(dat, mind=True, bound='dct2')  # already worked before the fix
after = make_image(dat, mind=True, bound='dct2')    # must still work identically
assert torch.equal(before[0].dat, after[0].dat)
```

**Expected outcome**: identical output before and after the fix for any
boundary condition that already worked (`dct1`, `dct2`, `dst1`, `dst2`,
`replicate`, `dft`).

## Scenario 3 — New `bounds.zero`/`zero_` functions behave correctly

```python
import torch
from nitorch.core import bounds

n = 5
i = torch.tensor([-2, -1, 0, 2, 4, 5, 7])
idx, mult = bounds.zero_(i.clone(), n)
assert idx.min() >= 0 and idx.max() <= n - 1  # safely clamped for gathering
assert torch.equal(mult, torch.tensor([0, 0, 1, 1, 1, 0, 0]))  # 0 exactly where out-of-bounds
```

## Scenario 4 — Full regression suite passes

Run the project's existing test suite and confirm zero new failures (SC-003).
