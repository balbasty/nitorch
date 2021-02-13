from copy import copy


class Base:
    """Base class that implements initialization/copy of parameters"""
    def __init__(self, **kwargs):
        # make a copy of all class attributes to avoid cross-talk
        for k in dir(self):
            if k.startswith('_'):
                continue
            v = getattr(self, k)
            if callable(v):
                continue
            setattr(self, k, copy(v))
        # update user-provided attributes
        for k, v in kwargs.items():
            setattr(self, k, copy(v))

    def _ordered_keys(self):
        """Get all class attributes, including inherited ones, in order.
        Private members and methods are skipped.
        """
        unrolled = [type(self)]
        while unrolled[0].__base__ is not object:
            unrolled = [unrolled[0].__base__, *unrolled]
        keys = []
        for klass in unrolled:
            for key in klass.__dict__.keys():
                if key.startswith('_'):
                    continue
                if key in keys:
                    continue
                val = getattr(self, key)
                if callable(val):
                    continue
                keys.append(key)
        return keys

    def _lines(self):
        """Build all lines of the representation of this object.
        Returns a list of str.
        """
        lines = []
        for k in self._ordered_keys():
            v = getattr(self, k)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], Base):
                lines.append(f'{k} = [')
                pad = '  '
                for vv in v:
                    l = [pad + ll for ll in vv._lines()]
                    lines.extend(l)
                    lines[-1] += ','
                lines.append(']')
            elif isinstance(v, Base):
                ll = v._lines()
                lines.append(f'{k} = {ll[0]}')
                lines.extend(ll[1:])
            else:
                lines.append(f'{k} = {v}')

        superpad = '  '
        lines = [superpad + line for line in lines]
        lines = [f'{type(self).__name__}('] + lines + [')']
        return lines

    def __repr__(self):
        return '\n'.join(self._lines())

    def __str__(self):
        return repr(self)


class FileWithInfo(Base):
    fname: str = None               # Full path
    shape: tuple = None             # Spatial shape
    affine = None                   # Orientation matrix
    dir: str = None                 # Directory
    base: str = None                # Base name (without extension)
    ext: str = None                 # Extension
    channels: int = None            # Number of channels
    float: bool = True              # Is raw dtype floating point


class Transform(Base):
    file: str = None
    inv: bool = False


class Linear(Transform):
    pass


class Displacement(Transform):
    order: int = 1


class Velocity(Transform):
    pass


class Reslicer(Base):
    files: list = []
    transformations: list = []
    target: str = None
    output: str = '{dir}/{base}.resliced{ext}'
    interpolation: int = 1
    bound: str = 'dct2'
    extrapolate: bool = False
    device: str = 'cpu'


