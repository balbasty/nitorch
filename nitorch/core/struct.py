"""This file implements structure classes, which are quite related to
the `dataclasses` library in python >= 3.7 (although it deviates from
it in several aspects).
We use a base class rather than a decorator to allow for nested structures.
"""
from copy import copy


class MISSING:
    """Tag an argument as MISSING (= not user-provided)"""
    pass


class STRONGTYPE:
    """Special validator that enforces the type of a value based on
    decorators."""
    pass


def _donothing(x):
    return x


class Field:
    """This object can be used to describe slightly more complex
    default values."""

    def __init__(self, default=MISSING, default_factory=MISSING,
                 init=True, repr=True, compare=True, convert=None):
        """

        Parameters
        ----------
        default : object, optional
            A default value.
            Cannot be used with `default_factory`
        default_factory : callable, optional
            A zero-argument factory for the default value.
            Cannot be used with `default`
        init : bool, default=True
            Can be specified as a __init__ argument
        repr : bool, default=True
            Is not hidden in the representation of the structure
        compare : bool, default=True
            Is involved in the comparison between structures.
        """
        if (default is not MISSING) and (default_factory is not MISSING):
            raise ValueError('Cannot use both `default` and `default_factory`')
        self.default = default
        self.default_factory = default_factory
        self.init = init
        self.repr = repr
        self.compare = compare
        self.convert = convert or _donothing


class ValidatedField(Field):
    """A field with a validator function."""

    def __init__(self, validator=STRONGTYPE, *args, **kwargs):
        """

        Parameters
        ----------
        validator : callable, optional
            A single-argument validator function.
            By default, enforced annotated type.
        default : object, optional
            A default value.
            Cannot be used with `default_factory`
        default_factory : callable, optional
            A zero-argument factory for the default value.
            Cannot be used with `default`
        init : bool, default=True
            Can be specified as a `__init__` argument
        repr : bool, default=True
            Is not hidden in the representation of the structure
        compare : bool, default=True
            Is involved in the comparison between structures.
        """
        super().__init__(*args, **kwargs)
        self.validator = validator


class Structure:
    """A class that mimics a C-like structure

    A structure class should inherit from Structure. Its fields
    are annotated class attributes. The values of all attributes can be
    set at initialization. If an attribute does not have a default value
    at the class level, it MUST be set at initialization.
    ```{python}
    >> class Option(Structure):
    >>   read_all: bool = True
    >>   verbose: int = 2
    ```

    At initialization, a shallow copy of the default class-level value
    is made and assigned to the instance. This avoids that multiple
    instances point to the same underlying object. This differs from
    the behaviour in `dataclasses`.

    Some flexibility can be obtained by using the `Field` object to
    specify default values:
    ```{python}
    >> class Option(Structure):
    >>   read_all: bool = True
    >>   verbose: int = 2
    >>   hidden_param: bool = Field(default=False, repr=False)
    ```

    """
    _fields = {}
    DefaultField = Field

    def __init__(self, as_dict=None, **kwargs):
        """
        """
        as_dict = as_dict or {}
        as_dict.update(kwargs)
        kwargs = as_dict
        # 0) Save all "Fields"
        self._fields = {}
        for k in self._all_annotations().keys():
            if hasattr(self, k):
                if isinstance(getattr(self, k), Field):
                    self._fields[k] = _donothing(getattr(self, k))
                else:
                    self._fields[k] = self.DefaultField(default=getattr(self, k))
            else:
                self._fields[k] = self.DefaultField()
        # 1) Set attributes that are user-defined
        for k, v in kwargs.items():
            if k in self._fields and not self._fields[k].init:
                raise TypeError(f'Attribute {k} cannot be set at init')
            setattr(self, k, copy(v))
        # 3) Use class-level default value
        for k, val in self._fields.items():
            if k in kwargs:
                continue
            if isinstance(val, Field):
                if val.default is not MISSING:
                    setattr(self, k, copy(val.default))
                elif val.default_factory is not MISSING:
                    setattr(self, k, val.default_factory())
        # 3) Check that all fields have been set
        for k in self._all_annotations().keys():
            if not hasattr(self, k):
                raise TypeError(f'Missing required argument {k}')

    def __setattr__(self, key, value):
        annotations = self._all_annotations()
        if key in self._fields:
            field = self._fields[key]
            if isinstance(field, ValidatedField):
                if field.validator is STRONGTYPE:
                    validator = lambda x: isinstance(x, annotations[key])
                else:
                    validator = field.validator
                if not validator(value):
                    raise ValueError(f'Value {value} failed validation '
                                     f'for attribute {key}')
            value = field.convert(value)
        return super().__setattr__(key, value)

    def update(self, other=None, **kwargs):
        """Update attributes by copying from on another dict_like object

        Parameters
        ----------
        other : Structure or dict_like, optional
            If `other` is an instance of a subclass of `type(self)`,
            its attributes and validators are copied into `self`.
            Else, if `other` has `a `keys` method:
                `for k in other: self[k] = other[k]`
            Else
                `for k, v in other, self[k] = v`.
        kwargs : dict
            `for k in kwargs: self[k] = kwargs[k]`

        Returns
        -------
        self

        """
        if isinstance(other, type(self)):
            for key, value in other.items():
                if key in self.keys():
                    self._fields[key] = other._fields[key]
                    setattr(self, key, getattr(other, key))
        else:
            if hasattr(other, 'keys'):
                for key in other.keys():
                    if key not in self.keys():
                        raise KeyError(key)
                    setattr(self, key, other[key])
            elif hasattr(other, '__iter__'):
                for key, value in other:
                    if key not in self.keys():
                        raise KeyError(key)
                    setattr(self, key, value)
            else:
                raise TypeError('Other does not seem to be dict-like')
        if kwargs:
            self.update(kwargs)
        return self

    def keys(self):
        """

        Returns
        -------
        list[str]
            All existing keys, in the same order as they were defined.

        """
        return self._fields.keys()

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        """Return all (key, value) pairs in the option dict.

        Returns
        -------
        iterator of `(key, value)`

        """
        for key in self.keys():
            yield key, self[key]

    def values(self):
        """Return all values in the option dict.

        Returns
        -------
        iterator of `value`

        """
        for key in self.keys():
            yield self[key]

    def _all_annotations(self):
        """Get all annotations from all base classes"""
        unrolled = [type(self)]
        while unrolled[0].__base__ is not object:
            unrolled = [unrolled[0].__base__, *unrolled]
        annotations = {}
        for klass in unrolled:
            if hasattr(klass, '__annotations__'):
                annotations.update(klass.__annotations__)
        return annotations

    def _lines(self):
        """Build all lines of the representation of this object.
        Returns a list of str.
        """
        lines = []
        for k in self._all_annotations().keys():
            if not self._fields[k].repr:
                continue
            v = getattr(self, k)
            if (isinstance(v, (list, tuple)) and v and
                    isinstance(v[0], Structure)):
                lines.append(f'{k} = [')
                pad = '  '
                for vv in v:
                    l = [pad + ll for ll in vv._lines()]
                    lines.extend(l)
                    lines[-1] += ','
                lines.append(']')
            elif isinstance(v, Structure):
                ll = v._lines()
                lines.append(f'{k} = {ll[0]}')
                lines.extend(ll[1:])
            else:
                lines.append(f'{k} = {v}')

        superpad = '  '
        lines = [superpad + line + ',' for line in lines]
        lines = [f'{type(self).__name__}('] + lines + [')']
        return lines

    def __repr__(self):
        return '\n'.join(self._lines())

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        for key, val in self.items():
            if not key in other.keys():
                return False
            if val != other[key]:
                return False
        for key in other.keys():
            if key not in self.keys():
                return False
        return True

    def __ne__(self, other):
        return not (self == other)


class TypedStructure(Structure):
    """A strongly typed structure"""
    DefaultField = ValidatedField
