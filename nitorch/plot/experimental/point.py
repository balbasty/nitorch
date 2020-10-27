from ...core.pyutils import make_list


class Point(list):

    axis_names = ()
    _fixed_length = False

    def __init__(self, *args, dtype=None, dim=None, **kwargs):
        """

        Parameters
        ----------
        args : list or scalar(s)
            Point value, by position.
        dtype : type, optional
            Data type for the scalar coordinates.
        dim : int, optional
            Dimension of the point.
        <axis_name> : scalar, optional
            Point value, by axis name.
        """
        dim = max(len(self.axis_names), dim or 1)

        if len(args) == 1:
            seq = make_list(args[0], n=dim, crop=False, default=0)
        elif len(args) > 1:
            seq = make_list(args, n=dim, crop=False, default=0)
        else:
            seq = [0] * dim

        for d, axis_name in enumerate(self.axis_names):
            if axis_name in kwargs.keys():
                seq[d] = kwargs[axis_name]

        if dtype is not None:
            seq = [dtype(elem) for elem in seq]
        super().__init__(seq)

    def dim(self):
        return len(self)

    def __str__(self):
        return self.__class__.__name__ + str(tuple(self))

    def __repr__(self):
        return self.__class__.__name__ + tuple(self).__repr__()

    def _fn1_mute(self, fn):
        if isinstance(fn, (tuple, list)):
            fn, alt = fn
        else:
            alt = fn
        for d in range(len(self)):
            if hasattr(self[d], fn):
                f = getattr(self[d], fn)
            else:
                f = lambda:  getattr(self[d], alt)
            self[d] = f()
        return self

    def _fn1_copy(self, fn):
        new = self.__class__(self)
        for d in range(len(new)):
            f = getattr(new[d], fn)
            new[d] = f()
        return new

    def _fn2_mute(self, other, fn):
        if isinstance(fn, (tuple, list)):
            fn, alt = fn
        else:
            alt = fn
        try:
            other = list(other)
            for d, elem in enumerate(other):
                if d < len(self):
                    if hasattr(self[d], fn):
                        f = getattr(self[d], fn)
                    else:
                        f = lambda x:  getattr(self[d], alt)(x)
                    self[d] = f(elem)
                else:
                    f = getattr(0, alt)
                    self.append(f(elem))
        except TypeError:
            # other is a scalar value
            for d in range(len(self)):
                f = getattr(self[d], fn)
                self[d] = f(other)
        return self

    def _fn2_copy(self, other, fn):
        new = self.__class__(self)
        try:
            other = list(other)
            for d, elem in enumerate(other):
                if d < len(self):
                    f = getattr(new[d], fn)
                    new[d] = f(elem)
                else:
                    f = getattr(0, fn)
                    new.append(f(elem))
        except TypeError:
            # other is a scalar value
            for d in range(len(self)):
                f = getattr(new[d], fn)
                new[d] = f(other)
        return new

    def append(self, *args, **kwargs):
        if self._fixed_length:
            raise NotImplementedError('Fixed-length points cannot be '
                                      'appended.')
        else:
            return super().append(*args, **kwargs)

    def to(self, dtype, inplace=False):
        if inplace:
            return self.to_(dtype)
        new = self.__class__(self)
        for d in range(len(self)):
            new[d] = dtype(new[d])
        return new

    def to_(self, dtype):
        for d in range(len(self)):
            self[d] = dtype(self[d])
        return self

# overload operators
_fn1_copy = ['__abs__', '__ceil__', '__float__', '__floor__', '__int__',
             '__invert__', '__neg__', '__pos__', '__trunc__']
_fn1_mute = []
_fn2_copy = ['__add__', '__and__', '__divmod__', '__floordiv__',
             '__lshift__', '__mod__', '__mul__', '__or__', '__pow__',
             '__radd__', '__rand__', '__rdivmod__', '__rfloordiv__',
             '__rlshift__', '__rmod__', '__rmul__', '__ror__',
             '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
             '__rtruediv__', '__rxor__', '__sub__', '__truediv__',
             '__xor__']
_fn2_mute = [('__iadd__', '__add__'),
             ('__iand__', '__and__'),
             ('__idivmod__', '__divmod__'),
             ('__ifloordiv__', '__floordiv__'),
             ('__ilshift__', '__lshift__'),
             ('__imod__', '__mod__'),
             ('__imul__', '__mul__'),
             ('__ior__', '__or__'),
             ('__ipow__', '__pow__'),
             ('__irshift__', '__rshift__'),
             ('__isub__', '__sub__'),
             ('__itruediv__', '__truediv__'),
             ('__ixor__', '__xor__')]

for fn in _fn1_copy:
    setattr(Point, fn, lambda *args, fn=fn, **kwargs: Point._fn1_copy(*args, **kwargs, fn=fn))
for fn in _fn1_mute:
    setattr(Point, fn[0], lambda *args, fn=fn, **kwargs: Point._fn1_mute(*args, **kwargs, fn=fn))
for fn in _fn2_copy:
    setattr(Point, fn, lambda *args, fn=fn, **kwargs: Point._fn2_copy(*args, **kwargs, fn=fn))
for fn in _fn2_mute:
    setattr(Point, fn[0], lambda *args, fn=fn, **kwargs: Point._fn2_mute(*args, **kwargs, fn=fn))


# ---------------------------
# -   Specialized classes   -
# ---------------------------

class Point2d(Point):

    axis_names = ('x', 'y')
    _fixed_length = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dim=2)

    def x(self):
        return self[0]

    def y(self):
        return self[1]


class Point3d(Point):

    axis_names = ('x', 'y', 'z')
    _fixed_length = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dim=3)

    def x(self):
        return self[0]

    def y(self):
        return self[1]

    def z(self):
        return self[2]
