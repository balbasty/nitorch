import copy


class OptionBase:
    """A base class for a structure of options.

    The attributes of an option object can be Python objects *or*
    other nested option objects. Attributes should be defined at the class
    level and given default value. When an instance of an Option class
    is created, it deep-copies all the class attributes. This ensures
    that nested option objects are not shared between option instances.

    All public class attributes (that do not start with '_') are
    considered options and are deep-copied on instantiation, except from
    a few protected methods (`keys`, `copy`, `items`).

    Examples
    --------
    ```python
    >> class SubOption(OptionBase):
    >>     param1: int = 1
    >>     param2: str = 'value'
    >>
    >> class MainOption(OptionBase):
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
    """
    def __new__(cls, *args, **kwargs):
        """

        Parameters
        ----------
        args : sequence of values, ordered as in `self.keys()`
        kwargs : dictionary of `option_name: value`
        """
        # Initial deep-copy from template object
        obj = super().__new__(cls)
        for attr in obj.keys():
            setattr(obj, attr, copy.deepcopy(getattr(obj, attr)))

        # Copy fields from list
        if len(args) > len(obj.keys()):
            raise ValueError('Too many values for this object')
        for key, val in zip(obj.keys(), args):
            setattr(obj, key, val)

        # Copy fields from dictionary
        # This time, we fail if non-existing keys are provided.
        for key, val in kwargs.items():
            if key not in obj.keys():
                raise KeyError(key)
            setattr(obj, key, val)

        return obj

    __protected_fields__ = ('copy', 'keys', 'items', 'values')

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
            if isinstance(val, OptionBase):
                vals = val._str(level + int(key_length) + 3) + '\n'
                val = type(val).__name__
            as_str += pad + ('{:' + key_length + 's} : {}\n').format(key, val)
            as_str += vals
        return as_str[:-1]

    def __repr__(self):
        return dict(self).__repr__()
