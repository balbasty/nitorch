from nitorch.tools.qmri.param import ParameterMap, MultiParameterMaps
import copy


class GREEQParameterMaps(MultiParameterMaps):
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

    def __init__(self, *args, **kwargs):
        self._greeq_volume = None
        self._greeq_affine = None
        super().__init__(*args, **kwargs)
        if len(self) not in (3, 4):
            raise ValueError('Exected 3 or 4 maps')
        self.pd = ParameterMap(self.volume[0], affine=self.affine)
        self.r1 = ParameterMap(self.volume[1], affine=self.affine)
        self.r2s = ParameterMap(self.volume[2], affine=self.affine)
        if len(self) == 4:
            self.mt = ParameterMap(self.volume[3], affine=self.affine)

    def drop_mt(self):
        if not hasattr(self, 'mt'):
            return self
        return GREEQParameterMaps(self.volume[:-1], affine=self.affine)

    @property
    def volume(self):
        return self._greeq_volume

    @volume.setter
    def volume(self, value):
        self._greeq_volume = value
        if hasattr(self, 'pd'):
            self.pd.volume = self.volume[0]
        if hasattr(self, 'r1'):
            self.r1.volume = self.volume[1]
        if hasattr(self, 'r2s'):
            self.r2s.volume = self.volume[2]
        if hasattr(self, 'mt'):
            self.mt.volume = self.volume[3]

    @property
    def affine(self):
        return self._greeq_affine

    @affine.setter
    def affine(self, value):
        self._greeq_affine = value
        if hasattr(self, 'pd'):
            self.pd.affine = self.affine
        if hasattr(self, 'r1'):
            self.r1.affine = self.affine
        if hasattr(self, 'r2s'):
            self.r2s.affine = self.affine
        if hasattr(self, 'mt'):
            self.mt.affine = self.affine

    def __iter__(self):
        if hasattr(self, 'pd'):
            yield self.pd
        if hasattr(self, 'r1'):
            yield self.r1
        if hasattr(self, 'r2s'):
            yield self.r2s
        if hasattr(self, 'mt'):
            yield self.mt
            
    def __getitem__(self, index):
        maps = [self.pd, self.r1, self.r2s]
        if hasattr(self, 'mt'):
            maps.append(self.mt)
        return maps[index]