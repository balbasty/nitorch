from .. import io as qio
import torch
import copy


class ParameterMap(qio.Volume3D):
    min = None
    max = None

    def __new__(cls, input=None, fill=None, dtype=None, device=None, **kwargs):
        if isinstance(input, (list, tuple)):
            volume = torch.zeros(input, dtype=dtype, device=device)
            if fill is not None:
                volume[...] = fill
            return cls.__new__(cls, volume, **kwargs)
        return super().__new__(cls, input, **kwargs)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


class ParameterMaps:
    intercepts: list = None
    decay: ParameterMap = None
    shape: tuple = None
    affine: torch.tensor = None

    def __len__(self):
        return len(self.intercepts) + 1

    def __iter__(self):
        maps = self.intercepts + [self.decay]
        for map in maps:
            yield map

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
