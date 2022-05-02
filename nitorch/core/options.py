"""Utility to define """
import copy


class Validated:
    """"An object used to associate a validator function to an option value.

    Examples
    --------
    class MyOption(Option):
        free_param: int = 1
        validated_param: float = Validated(0.5, lambda x: x > 0)
    """

    def __init__(self, default, validator):
        """

        Parameters
        ----------
        default : object
        validator : callable -> bool

        """
        self.default = default
        self.validator = validator


class Option:
    """A base class for a structure of options.

    The attributes of an option object can be Python objects *or*
    other nested option objects. Attributes should be defined at the class
    level and given default value. When an instance of an Option class
    is created, it deep-copies all the class attributes. This ensures
    that nested option objects are not shared between option instances.

    All public class attributes (that do not start with '_') are
    considered options and are deep-copied on instantiation, except from
    a few protected methods (`keys`, `copy`, `items`).

    Validators can be specified when the option class is defined by
    warping the default value in a `Validated` call.

    Examples
    --------
    ```python
    >> class SubOption(Option):
    >>     param1: int = 1
    >>     param2: str = 'value'
    >>
    >> class MainOption(Option):
    >>     nested_param: SubOption = SubOption()
    >>     other_param: float = 0.5
    >>
    >> main_opt = MainOption()
    >> other_opt = MainOption()
    >>
    >> main_opt.nested_param.param1 = 2
    >> print(main_opt.nested_param.param1)
    >> print(other_opt.nested_param.param1)
    ```

    ```python
    >> class ValidatedOption(Option):
    >>     free_param: int = 1
    >>     validated_param: float = Validated(0.5, lambda x: x > 0)
    >>
    >> opt = ValidatedOption()
    >>
    >> main_opt.nested_param.param1 = 2
    >> print(main_opt.nested_param.param1)
    >> print(other_opt.nested_param.param1)
    ```
    """
    # def __new__(cls, *args, **kwargs):
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args : sequence
            ordered as in `self.keys()`
        kwargs : dict of `option_name: value`
        """
        # Initial deep-copy from template object
        # obj = super().__new__(cls)
        obj = self
        cls = type(self)
        obj._validators = dict()

        for attr in obj.keys():
            value = getattr(cls, attr)
            if isinstance(value, Validated):
                validator = value.validator
                value = value.default
            else:
                validator = None
            setattr(obj, attr, copy.deepcopy(value))
            if validator is not None:
                obj._validators[attr] = validator

        # Copy fields from list
        if len(args) > len(obj.keys()):
            raise ValueError('Too many values for this object')
        for key, val in zip(obj.keys(), args):
            if isinstance(val, Validated):
                raise ValueError('Validated objects can only be (static) '
                                 'class attributes and cannot be set '
                                 '(dynamically) at instantiation time.')
            setattr(obj, key, val)

        # Copy fields from dictionary
        # This time, we fail if non-existing keys are provided.
        for key, val in kwargs.items():
            if key not in obj.keys():
                raise KeyError(key)
            setattr(obj, key, val)

    _validators = None
    __protected_fields__ = ('copy', 'keys', 'items', 'values', 'update')

    def __setattr__(self, key, value):
        if key.startswith('_') or key in self.__protected_fields__:
            return super().__setattr__(key, value)
        if key not in self.keys():
            raise KeyError(f'Key "{key}" does not exist in structure {type(self)}')
        if key in self._validators and self._validators[key] is not None:
            validator = self._validators[key]
            if not validator(value):
                raise ValueError(f'Value {value} failed validation')
        if isinstance(getattr(self, key), Option):
            if getattr(self, key) is getattr(type(self), key):
                # special case -> we are inside __init__ and want to
                # instantiate a new Option object to avoid writing into
                # the class instance
                super().__setattr__(key, value)
            else:
                # re-use the existing object (TODO: is this a good idea?)
                getattr(self, key).update(value)
            return self
        return super().__setattr__(key, value)

    def update(self, other=None, **kwargs):
        """Update attributes by copying from on another dict_like object

        Parameters
        ----------
        other : Option or dict_like, optional
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
                    self._validators[key] = other._validators.get(key, None)
                    setattr(self, key, getattr(other, key))
        else:
            if hasattr(other, 'keys'):
                for key in other.keys():
                    if key not in self.keys():
                        raise KeyError(key)
                    setattr(self, key, other[key])
            else:
                for key, value in other:
                    if key not in self.keys():
                        raise KeyError(key)
                    setattr(self, key, value)
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
        keys = type(self).__dict__.keys()
        return [key for key in keys
                if not key.startswith('_')
                and key not in self.__protected_fields__]

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

    def __eq__(self, other):
        other = type(self)(other)
        if type(self) != type(other):
            return False
        for attr in self.keys():
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def copy(self):
        """Recursive (deep) copy of the object."""
        return copy.deepcopy(self)

    def __str__(self):
        return self._str()

    def _str(self, level=0):
        as_str = ''
        key_length = str(max(len(key) for key in self.keys()))
        pad = ' ' * level
        for key, val in self.items():
            vals = ''
            if isinstance(val, Option):
                vals = val._str(level + int(key_length) + 3) + '\n'
                val = type(val).__name__
            as_str += pad + ('{:' + key_length + 's} : {}\n').format(key, val)
            as_str += vals
        return as_str[:-1]

    def __repr__(self):
        return dict(self).__repr__()


class TypedOption(Option):
    """An Option object that strongly enforces its type hints"""

    def __setattr__(self, key, value):
        if key in type(self).__annotations__:
            valtype = type(self).__annotations__[key]
            if not isinstance(value, valtype):
                raise TypeError(f'Value {value} failed validation: '
                                f'expected type {valtype.__name__} but '
                                f'got type {type(value).__name__}')
        return super().__setattr__(key, value)
