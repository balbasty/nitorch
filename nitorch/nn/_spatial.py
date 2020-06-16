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

    def forward(self, x, grid):
        return spatial.grid_pull(x, grid, self.interpolation,
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

    def forward(self, x, grid):
        return spatial.grid_push(x, grid, self.interpolation,
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

    def forward(self, x, grid):
        push = spatial.grid_push(x, grid, self.interpolation,
                                 self.bound, self.extrapolate)
        count = spatial.grid_count(grid, self.interpolation,
                                   self.bound, self.extrapolate)
        return torch.cat((push, count), dim=1)


class GridExp(tnn.Module):
    """Exponentiate an inifinitesimal deformation field (velocity)."""

    def __init__(self, fwd=True, inverse=False, steps=None,
                 interpolation='linear',
                 bound='dft', displacement=False, energy=None, vs=None,
                 greens=None, try_inplace=True):
        super().__init__()

        self.fwd = fwd
        self.inverse = inverse
        self.steps = steps
        self.interpolation = interpolation
        self.bound = bound
        self.displacement = displacement
        self.energy = energy
        self.vs = vs
        self.greens = greens
        self.try_inplace = try_inplace

    def forward(self, x):
        inplace = False  # TODO
        output = []
        if self.fwd:
            y = spatial.exp(x, False, self.steps, self.interpolation,
                            self.bound, self.displacement, self.energy,
                            self.vs, self.greens, inplace)
            output.append(y)
        if self.inverse:
            iy = spatial.exp(x, True, self.steps, self.interpolation,
                             self.bound, self.displacement, self.energy,
                             self.vs, self.greens, inplace)
            output.append(iy)
        return tuple(output)
