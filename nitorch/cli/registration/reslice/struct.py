from copy import copy
from nitorch.core.struct import Structure


class FileWithInfo(Structure):
    fname: str = None               # Full path
    shape: tuple = None             # Spatial shape
    affine = None                   # Orientation matrix
    dir: str = None                 # Directory
    base: str = None                # Base name (without extension)
    ext: str = None                 # Extension
    channels: int = None            # Number of channels
    float: bool = True              # Is raw dtype floating point
    type: str = None


class Transform(Structure):
    file: str = None
    inv: bool = False
    square: bool = False


class Linear(Transform):
    pass


class Displacement(Transform):
    order: int = 1
    unit: str = 'vox'


class Velocity(Transform):
    json: str = None
    pass


class Reslicer(Structure):
    files: list = []
    transformations: list = []
    target: str = None
    voxel_size: list = None
    chunk: list = None
    output: str = '{dir}{sep}{base}.resliced{ext}'
    interpolation: int = 1
    bound: str = 'dct2'
    extrapolate: bool = False
    dtype: str = None
    device: str = 'cpu'
    prefilter: bool = True
    channels: list = None
    log: bool = False
    logit: bool = False
    clip: bool = False
