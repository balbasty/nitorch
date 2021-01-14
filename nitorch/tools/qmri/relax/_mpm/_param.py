from nitorch.tools.qmri.param import ParameterMap
import copy


class GREEQParameterMaps:
    """GREEQ-specific parameter maps

    Attributes
    ----------
    pd: ParameterMap                Proton density -- up to an arbitrary factor
    r1: ParameterMap                Longitudinal relaxation rate
    r2s: ParameterMap               Apparent transverse relaxation rate (R2*)
    mt: ParameterMap, optional      Magnetisation-transfer saturation
    shape: tuple[int]               (nb_prm, *spatial) Shape
    affine: (4, 4) tensor           Affine orientation matrix

    Methods
    -------
    Iteration over `[pd, r1, r2s, mt]` is implemented:
    ```python
    >> maps = GREEQParameterMaps()
    >> for prm in maps:
    >>    do_something(prm)
    ```
    """
    pd: ParameterMap = None
    r1: ParameterMap = None
    r2s: ParameterMap = None

    def __len__(self):
        return 3 + hasattr(self, 'mt')

    @property
    def shape(self):
        return (len(self), *self.pd.shape)

    @property
    def affine(self):
        return self.pd.affine

    @affine.setter
    def affine(self, value):
        self.pd.affine = value
        self.r1.affine = value
        self.r2s.affine = value
        if hasattr(self, 'mt'):
            self.mt.affine = value

    def __iter__(self):
        yield self.pd
        yield self.r1
        yield self.r2s
        if hasattr(self, 'mt'):
            yield self.mt
            
    def __getitem__(self, index):
        maps = [self.pd, self.r1, self.r2s]
        if hasattr(self, 'mt'):
            maps.append(self.mt)
        return maps[index]

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
