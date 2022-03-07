from nitorch.tools.qmri.param import ParameterMap, MultiParameterMaps


class ESTATICSParameterMaps(MultiParameterMaps):
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
    def __init__(self, *args, **kwargs):
        self._estatics_volume = []
        self._estatics_affine = None
        # self.intercepts = []
        # self.decay = None
        super().__init__(*args, **kwargs)
        # self.intercepts = [ParameterMap(self.volume[i], affine=self.affine)
        #                    for i in range(len(self)-1)]
        # self.decay = ParameterMap(self.volume[-1], affine=self.affine)

    @property
    def intercepts(self):
        if self.volume is None:
            return []
        return [ParameterMap(self.volume[i], affine=self.affine)
                for i in range(len(self)-1)]

    @property
    def decay(self):
        if self.volume is None:
            return None
        return ParameterMap(self.volume[-1], affine=self.affine)

    @property
    def volume(self):
        return self._estatics_volume

    @volume.setter
    def volume(self, value):
        self._estatics_volume = value
        for map, val in zip(self, value):
            if map:
                map.volume = val

    @property
    def affine(self):
        return self._estatics_affine

    @affine.setter
    def affine(self, value):
        self._estatics_affine = value
        for map, val in zip(self, value):
            if map:
                map.affine = val

    def __iter__(self):
        for map in self.intercepts:
            yield map
        yield self.decay

    def __getitem__(self, index):
        maps = [*self.intercepts, self.decay]
        return maps[index]
