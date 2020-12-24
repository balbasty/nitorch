import copy


class OptionBase:
    def __init__(self, *args, **kwargs):
        for attr in self.keys():
            setattr(self, attr, copy.deepcopy(getattr(self, attr)))
        if len(args) == 1 and isinstance(args[0], OptionBase):
            for key, val in args[0].iteritems():
                if key in self.keys():
                    setattr(self, key, val)
        elif len(args) != 0:
            raise TypeError(f'Expected at most one argument, got {len(args)}')

        for key, val in kwargs.items():
            if key not in self.keys():
                raise KeyError(key)
            setattr(self, key, val)

    def keys(self):
        return [key for key in dir(self)
                if not key.startswith('_')
                and key not in ('keys', 'copy', 'items')]

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __eq__(self, other):
        other = type(self)(other)
        if type(self) != type(other):
            return False
        for attr in self.keys():
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def copy(self):
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
