"""
WORK IN PROGRESS
- the idea is too have an object that represents an "oriented lattice"
  and can easily be sliced/transformed/etc, which would be easier to
  use than the affine_* functions
"""

from ._affine import (
    affine_default,
    affine_sub,
    affine_conv,
    affine_pad,
    affine_permute,
    affine_transpose,
    affine_resize,
)
import torch
import math as pymath


class SpatialLattice:
    """Cartesian grid associated with an affine orientation matrix."""

    def __init__(self, shape, affine=None):
        if affine is None:
            affine = affine_default(shape)
        self.shape = shape
        self.affine = affine

    def __getitem__(self, item):
        affine, shape = affine_sub(self.affine, self.shape, item)
        return SpatialLattice(shape, affine)

    def dim(self):
        return len(self.shape)

    def numel(self):
        return pymath.prod(self.shape)

    def conv(self, kernel, stride=None, padding=0, output_padding=0,
             transposed=False):
        if torch.is_tensor(kernel):
            kernel = kernel.shape[-self.dim():]
        affine, shape = affine_conv(
            self.affine, self.shape, kernel, stride, padding,
            output_padding, transposed)
        return SpatialLattice(shape, affine)

    def pad(self, padsize, side=None):
        affine, shape = affine_pad(self.affine, self.shape, padsize, side)
        return SpatialLattice(shape, affine)

    def permute(self, perm):
        affine, shape = affine_permute(self.affine, perm, shape=self.shape)
        return SpatialLattice(shape, affine)

    def transpose(self, dim0, dim1):
        affine, shape = affine_transpose(self.affine, dim0, dim1, shape=self.shape)
        return SpatialLattice(shape, affine)

    def resize(self, factor, anchor='c'):
        affine, shape = affine_transpose(self.affine, self.shape, factor, anchor)
        return SpatialLattice(shape, affine)
