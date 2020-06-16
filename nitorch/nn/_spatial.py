# -*- coding: utf-8 -*-
"""Spatial transformation layers."""

import torch
from torch import nn as tnn
from nitorch import spatial


class GridPull(tnn.Module):
    """Pull/Sample an image according to a deformation.

    This module is parameter-free.
    """

    def __init__(self, interpolation='linear', bound='dct', extrapolate=True):
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, input, grid):
        return spatial.grid_pull(input, grid, self.interpolation,
                                 self.bound, self.extrapolate)


class GridPush(tnn.Module):
    """Push/Splat an image according to a deformation.

    This module is parameter-free.
    """

    def __init__(self, interpolation='linear', bound='dct', extrapolate=True):
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, input, grid):
        return spatial.grid_push(input, grid, self.interpolation,
                                 self.bound, self.extrapolate)


class GridPushCount(tnn.Module):
    """Push/Splat an image **and** ones according to a deformation.

    Both an input image and an image of ones of the same shape are pushed.
    The results are concatenated along the channel dimension.

    This module is parameter-free.
    """

    def __init__(self, interpolation='linear', bound='dct',
                 extrapolate=True):
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, input, grid):
        push = spatial.grid_push(input, grid, self.interpolation,
                                 self.bound, self.extrapolate)
        count = spatial.grid_count(grid, self.interpolation,
                                   self.bound, self.extrapolate)
        return torch.cat((push, count), dim=1)
