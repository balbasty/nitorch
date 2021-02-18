"""Command-line utilities."""
from copy import copy


class ParseError(RuntimeError):
    """A specialized error for command-line parsers"""
    pass


def istag(arg):
    """Return true if the argument starts with a dash ('-') and is not a number

    Parameters
    ----------
    arg : str

    Returns
    -------
    bool

    """
    return arg.startswith('-') and len(arg) > 1 and arg[1] not in '0123456789'


def isvalue(arg):
    """Return true if the argument does not starts with a dash ('-')

    Parameters
    ----------
    arg : str

    Returns
    -------
    bool

    """
    return not istag(arg)


def next_istag(args):
    """Return true if the next argument starts with a dash ('-')

    Parameters
    ----------
    args : list of str

    Returns
    -------
    bool

    """
    return args and istag(args[0])


def next_isvalue(args):
    """Return true if the next argument does not starts with a dash ('-')

    Parameters
    ----------
    args : list of str

    Returns
    -------
    bool

    """
    return args and isvalue(args[0])


def check_next_isvalue(args, group=''):
    """Raise ParseError if the next argument starts with a dash ('-')

    Parameters
    ----------
    args : list of str
    group : str, default=''
        A name for the current group of values

    Returns
    -------
    bool

    """
    if not next_isvalue(args):
        raise ParseError(f'Expected a value for tag {group} but found nothing.')


class ParsedStructure:
    """Base class that implements initialization/copy of parameters"""
    def __init__(self, **kwargs):
        """
        """
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
            if (isinstance(v, (list, tuple)) and v and
                    isinstance(v[0], ParsedStructure)):
                lines.append(f'{k} = [')
                pad = '  '
                for vv in v:
                    l = [pad + ll for ll in vv._lines()]
                    lines.extend(l)
                    lines[-1] += ','
                lines.append(']')
            elif isinstance(v, ParsedStructure):
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
