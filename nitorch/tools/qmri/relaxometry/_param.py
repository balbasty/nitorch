from ..param import ParameterMap
import torch
import copy


class ParameterMaps:
    pd: ParameterMap = None
    r1: ParameterMap = None
    r2s: ParameterMap = None
    shape: tuple = None
    affine: torch.tensor = None

    def __len__(self):
        return 3 + hasattr(self, 'mt')

    def __iter__(self):
        maps = [self.pd, self.r1, self.r2s]
        if hasattr(self, 'mt'):
            maps.append(self.mt)
        for map in maps:
            yield map

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
