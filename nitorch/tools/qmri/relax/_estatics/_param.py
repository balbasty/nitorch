from nitorch.tools.qmri.param import ParameterMap
import torch
import copy


class ESTATICSParameterMaps:
    """ESTATICS-specific parameter maps

    Attributes
    ----------
    intercepts: list[ParameterMap]  GRE-volumes extrapolated to TE=0
    decay: ParameterMap             Apparent transverse relaxation rate (R2*)
    shape: tuple[int]               (nb_prm, *spatial) Shape
    affine: (4, 4) tensor           Affine orientation matrix
    volume : tensor                 Stacked maps

    Methods
    -------
    Iteration over `[*intercepts, decay]` is implemented:
    ```python
    >> maps = ESTATICSParameterMaps(2, [128, 128, 128])
    >> for prm in maps:
    >>    do_something(prm)
    ```
    """

    intercepts: list = None
    decay: ParameterMap = None
    volume: torch.Tensor = None

    def __init__(self, nb_contrasts, shape, dtype=None, device=None, affine=None):
        """

        Parameters
        ----------
        nb_contrasts : int
        shape : sequence[int]
        dtype : torch.dtype, optional
        device : torch.device, optional
        affine : (D+1, D+1) tensor, optional
        """
        full_shape = (nb_contrasts+1, *shape)
        volume = torch.zeros(full_shape, dtype=dtype, device=device)
        volume[-1].fill_(1)
        self.intercepts = [ParameterMap(volume[c], affine=affine)
                           for c in range(nb_contrasts)]
        self.decay = ParameterMap(volume[-1], affine=affine, min=0)
        self.volume = volume

    def __len__(self):
        return len(self.intercepts) + 1

    @property
    def shape(self):
        return (len(self), *self.decay.shape)

    @property
    def affine(self):
        return self.decay.affine

    @affine.setter
    def affine(self, value):
        self.decay.affine = value
        for inter in self.intercepts:
            inter.affine = value

    def __iter__(self):
        maps = self.intercepts + [self.decay]
        for map in maps:
            yield map

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
