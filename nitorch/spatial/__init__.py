# -*- coding: utf-8 -*-
"""Tools related to spatial sampling (displacement, warps, etc.).

Spatial ordering conventions in nitorch
---------------------------------------

NiTorch uses consistent ordering conventions throughout its API:
. We use abbreviations ``(B[atch], C[hannel], X, Y, Z)``
  to name dimensions.
. Tensors that represent series (resp. images or volumes) should always
  be ordered as ``(B, C, X, [Y, [Z]])``.
. We use ``x``, ``y``, ``z`` to denote axis/coordinates along the
  ``X``, ``Y``, ``Z`` dimensions.
. Displacement or deformation fields are ordered as ``(B, X, [Y, [Z],] D)``.
  There, the ``D``[irection] dimension contains displacements or
  deformations along the ``x``, ``y``, ``z`` axes. ``D`` must be equal
  to the Tensor's spatial dimension (1, 2, or 3); i.e.,
  ``(B, X, 1)`` or ``(B, X, Y, 2)`` or ``(B, X, Y, Z, 3)``
. Similarly, Jacobian fields are stored as ``(B, X, [Y, [Z],] D, D)``.
  The second to last dimension corresponds to x, y, z components and the
  last dimension corresponds to derivatives along the x, y, z axes.
  I.e., ``jac[..., i, j]`` = d{u_i}/d{x_j}
. This means that we usually do not store deformations per *imaging*
  channel (they are assumed to lie in the same space).
. Arguments that relate to spatial dimensions are ordered as ``(x, y, z)``.

These conventions are *not* consistent with those used in PyTorch
(conv, grid_sampler, etc.), but we find them more intuitive.
Furthermore, they are consistent with nibabel, where columns of the
orientation matrices have the same order as dimensions in the
corresponding ND-array.

"""

# Conventions for developers:
# . Spatial transformations (voxel to coordinates) are called ``grid``
# . Relative transformations (voxel to shift) are called ``displacement``
# . Voxel size is abbreviated ``vs``
# . Interpolation order: ``interpolation``
#  > Prefer str representation for the default values; e.g., ``'linear'``
# . Boundary type: ``bound``
# . In each submodule, define ``__all__`` so that only public symbols are
#   exported here

# TODO:
#     . What about time series?
#     . Should we always have a batch dimension for affine matrices?
#     . Should the default storage be compact or square for affine matrices?


from ._affine import *
from ._affine_optimal import *
from ._conv import *
from ._spconv import *
from ._finite_differences import *
from ._grid import *
from ._grid_inv import *
from ._morpho import *
from ._mrfield import *
from ._regularisers import *
from ._solvers import *
from ._svf import *
from ._svf_optimal import *
from ._svf1d import *
from ._shoot import *
from ._distances import *
