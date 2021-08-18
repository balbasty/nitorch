"""Command-line utilities."""
from copy import copy
import collections
import pprint


class ParseError(RuntimeError):
    """A specialized error for command-line parsers"""
    pass


def istag(arg, symbol='-'):
    """Return true if the argument starts with a dash ('-') and is not a number

    Parameters
    ----------
    arg : str

    Returns
    -------
    bool

    """
    return arg.startswith(symbol) and len(arg) > 1 and arg[1] not in '0123456789'


def isvalue(arg, symbols='-'):
    """Return true if the argument does not starts with a dash ('-')

    Parameters
    ----------
    arg : str

    Returns
    -------
    bool

    """
    return not any(istag(arg, s) for s in symbols)


def next_istag(args, symbol=''):
    """Return true if the next argument starts with a dash ('-')

    Parameters
    ----------
    args : list of str

    Returns
    -------
    bool

    """
    return args and istag(args[0], symbol)


def next_isvalue(args, symbols='-'):
    """Return true if the next argument does not starts with a dash ('-')

    Parameters
    ----------
    args : list of str

    Returns
    -------
    bool

    """
    return args and isvalue(args[0], symbols)


def check_next_isvalue(args, group='', symbols='-'):
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
    if not next_isvalue(args, symbols):
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


class Actions:
    """A list of classmethods that can be used as actions"""
    @classmethod
    def store_value(cls, value):
        return (lambda: value)

    @classmethod
    def store_true(cls):
        return True

    @classmethod
    def store_false(cls):
        return False


class Validations:

    @classmethod
    def choice(cls, choices):
        def validate(value):
            return value in choices
        return validate


class Parsed(ParsedStructure):
    def merge(self, *others):
        for other in others:
            for key, value in other.__dict__.items():
                setattr(self, key, value)
        return self

    @classmethod
    def _recursive_to_dict(cls, item):
        if isinstance(item, dict):
            return {key: cls._recursive_to_dict(val)
                    for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(cls._recursive_to_dict(val) for val in item)
        if isinstance(item, Parsed):
            return {key: cls._recursive_to_dict(val)
                    for key, val in item.__dict__.items()}
        return item

    def todict(self):
        return self._recursive_to_dict(self)

    def __repr__(self):
        return repr(self.todict())

    def __str__(self):
        return pprint.pformat(self.todict(), indent=2)


class CommandParser:

    def __init__(self, name=None, help=None, title=None, add_help=True):
        self.name = name
        self.help = help
        self.title = title
        self.positionals = PositionalList()
        self.options = OptionList()
        self.groups = GroupList()
        if add_help:
            hopt = Option('help', ('-h', '--help'), nargs=0,
                          help='Show this help', default=False)
            self.options.append(hopt)

    def parse(self, args):
        nargs = len(args)
        alltags = list(self.options.tags) + list(self.groups.tags)
        obj = self.positionals.parse(args, stops=alltags)
        obj.merge(self.options.parse(args, stops=self.groups.tags))
        if obj.help:
            return obj
        obj.merge(self.groups.parse(args))
        if args:
            raise ParseError(f"Don't know what to do with '{args[0]}' "
                             f"at position {nargs-len(args)}")
        return obj

    def add_option(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Option):
            self.options.append(args[0])
        else:
            self.options.append(Option(*args, **kwargs))

    def add_positional(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Positional):
            self.positionals.append(args[0])
        else:
            self.positionals.append(Positional(*args, **kwargs))

    def add_group(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Group):
            self.groups.append(args[0])
        else:
            self.groups.append(Group(*args, **kwargs))


def _n_to_minmax(n):
    if not isinstance(n, str):
        n = str(n)
    if '+' in n:
        # substitute '1*' for '+' (equivalent)
        n = n.replace('+', '*')
        if n.startswith('*'):
            n = '1' + n
    if n.startswith('*'):
        # implicit minimum: 0
        n = '0' + n
    if n.startswith('?'):
        # implicit maximum: 1
        n = '1' + n
    if '?' in n and '*' not in n:
        # implicit minimum: 0
        n = '0*' + n
    if '?' not in n and '*' not in n:
        # minimum == maximum == value
        n = n + '*' + n + '?'
    mn, mx = n.split('*')
    if not mx:
        return int(mn), float('inf')
    mx = mx.split('?')[0]
    return int(mn), int(mx)


class Option:

    def __init__(self, name, tags=None, n='?', nargs='*', help=None,
                 default=None, validation=None, action=None, convert=None):
        """

        Parameters
        ----------
        name : str
            Name of the option in the results tree.
        tags : str or sequence[str], default='--{name}'
            Tags opening this option
        n : int or str, default='?'
            Number of times this options can appear.
                '{mx=1}?'      : up to `mx` times
                '{mn=0}*'      : at least `mn` times
                '{mn=0}*{mx}?' : `mn` to `mx` times
                '{n}'          : exactly `n` times
        nargs : int or str, default='*'
            Number of arguments to this option.
                '{mx=1}?'      : up to `mx` arguments
                '{mn=0}*'      : at least `mn` arguments
                '{mn=0}*{mx}?' : `mn` to `mx` arguments
                '{n}'          : exactly `n` arguments
        help : str, optional
            Help string
        default : optional
            A value to store in the results tree if the option is not
            specified.
        validation : callable(obj) -> bool, optional
            A function to call on each argument to check that it belongs.
        action : callable(), optional
            A function to call when the option is used without arguments.
            Should return a value to store in the results tree.
        convert : callable(obj), optional
            A function to call on each argument to convert it from
            str to a target type.
        """
        super().__init__()
        self.name = name
        self.tags = tags or ('--' + name,)
        if not isinstance(self.tags, (list, tuple)):
            self.tags = [self.tags]
        self.n = n
        self.nargs = nargs
        self.help = help
        self.default = default
        self.validation = validation or (lambda x: True)
        self.action = action or (lambda: True)
        self.convert = convert or str

    def __repr__(self):
        if len(self.tags) == 1:
            pattern = f'{self.tags[0]}'
        else:
            pattern = f'{tuple(self.tags)}'
        mn, mx = _n_to_minmax(self.nargs)
        if not (mn == mx == 0) and not (mn == mx == 1):
            pattern += f' {self.nargs}'
        if isinstance(self.convert, type):
            pattern += f' <{self.convert.__name__}>'
        if self.default is not None:
            pattern += f' [{self.default}]'

        s = f'{type(self).__name__}[{self.n}]({self.name}): {pattern}'
        return s

    __str__ = __repr__

    def parse(self, args, stops=tuple()):
        """

        Parameters
        ----------
        args : list[str]
        stops : sequence[str]

        Returns
        -------
        results : Parsed

        """
        mn, mx = _n_to_minmax(self.nargs)
        values = []
        for nargs_count in range(mn):
            if args:
                next_arg = args[0]
                if not next_arg in stops:
                    try:
                        next_arg = self.convert(next_arg)
                        if self.validation(next_arg):
                            args.pop(0)
                            values.append(next_arg)
                            continue
                    except Exception:
                        pass
            raise ParseError(f'Not enough arguments for option '
                             f'{self.name}: expected at least {mn} '
                             f'but got {nargs_count}.')
        if mx == float('inf'):
            while args:
                next_arg = args[0]
                if next_arg in stops:
                    break
                try:
                    next_arg = self.convert(next_arg)
                except Exception:
                    break
                if self.validation(next_arg):
                    args.pop(0)
                    values.append(next_arg)
                    continue
                break
        else:
            for nargs_count in range(mn, mx):
                if args:
                    next_arg = args[0]
                    if not next_arg in stops:
                        try:
                            next_arg = self.convert(next_arg)
                            if self.validation(next_arg):
                                args.pop(0)
                                values.append(next_arg)
                                continue
                        except Exception:
                            pass
                break
        if not values:
            values = self.action()
        elif mx == 1:
            values = values[0]
        return values


class Positional(Option):
    def __init__(self, name, nargs='*', help=None,
                 default=None, validation=None, convert=None):
        super().__init__(name, tuple(), '?', nargs, help, default,
                         validation, Actions.store_value(default), convert)


class Group:

    def __init__(self, name, tags=None, n=1, help=None, make_default=True):
        super().__init__()
        self.name = name
        self.tags = tags or ('@' + name,)
        if not isinstance(self.tags, (list, tuple)):
            self.tags = [self.tags]
        self.n = n
        self.help = help
        self.options = OptionList()
        self.groups = GroupList()
        self.positionals = PositionalList()
        self.make_default = make_default

    def parse(self, args, stops=tuple()):
        stops = list(stops)
        obj = self.positionals.parse(args, stops=stops + self.options.tags + self.groups.tags)
        obj2 = self.options.parse(args, stops=stops + self.groups.tags)
        obj3 = self.groups.parse(args)
        obj = obj.merge(obj2, obj3)
        return obj

    def add_option(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Option):
            self.options.append(args[0])
        else:
            self.options.append(Option(*args, **kwargs))

    def add_positional(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Positional):
            self.positionals.append(args[0])
        else:
            self.positionals.append(Positional(*args, **kwargs))

    def add_group(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Group):
            self.groups.append(args[0])
        else:
            self.groups.append(Group(*args, **kwargs))

    def copy_from(self, other):
        self.positionals.extend(other.positionals)
        self.options.extend(other.options)
        self.groups.extend(other.groups)
        return self

    @property
    def default(self):
        if self.make_default:
            results = Parsed()
            for option in self.positionals:
                setattr(results, option.name, option.default)
            for option in self.options:
                setattr(results, option.name, option.default)
            for option in self.groups:
                setattr(results, option.name, option.default)
            return results
        return None


class NamedGroup(Group):

    def __init__(self, name, choices, tags=None, n=1, help=None, default=None,
                 make_default=True):
        super().__init__(name, tags=tags, n=n, help=help, make_default=make_default)
        if not isinstance(choices, Positional):
            choices = Positional('name',
                                 nargs='?' if default is not None else 1,
                                 default=default,
                                 validation=Validations.choice(choices))
        self.switch = choices
        self.suboptions = dict()

    def add_suboption(self, condition, *args, **kwargs):
        if condition not in self.suboptions:
            self.suboptions[condition] = OptionList()
        if len(args) == 1 and not kwargs and isinstance(args[0], Option):
            self.suboptions[condition].append(args[0])
        else:
            self.suboptions[condition].append(Option(*args, **kwargs))

    def parse(self, args, stops=tuple()):
        all_stops = list(stops) + list(self.options.tags)
        for option in self.suboptions.values():
            all_stops += option.tags
        all_stops += self.groups.tags
        results = Parsed()
        results.name = self.switch.parse(args, all_stops)
        results.merge(self.positionals.parse(args, all_stops))
        options = self.options + self.suboptions.get(results.name, [])
        results.merge(options.parse(args, all_stops))
        results.merge(self.groups.parse(args, all_stops))
        return results

    @property
    def default(self):
        if self.make_default:
            results = Parsed()
            results.name = self.switch.default
            for option in self.positionals:
                setattr(results, option.name, option.default)
            for option in self.options:
                setattr(results, option.name, option.default)
            for option in self.groups:
                setattr(results, option.name, option.default)
            for option in self.suboptions.get(results.name, []):
                setattr(results, option.name, option.default)
            return results
        else:
            return None


class TypedList(collections.UserList):
    subtype = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, item):
        if item is not None and not isinstance(item, self.subtype):
            raise TypeError(f'Item should be a {self.subtype}')
        super().__setitem__(key, item)

    def append(self, item):
        if item is not None and not isinstance(item, self.subtype):
            raise TypeError(f'Item should be a {self.subtype}')
        super().append(item)

    def extend(self, other):
        # may lose performance but only way to check type if other is a
        # consumable
        for item in other:
            self.append(item)


class ListWithTags(TypedList):

    def iter_tags(self):
        for option in self:
            for tag in option.tags:
                yield tag

    @property
    def tags(self):
        return list(self.iter_tags())

    def parse(self, args, stops=tuple()):
        """

        Parameters
        ----------
        args : list[str]
        stops : sequence[str]

        Returns
        -------
        results : Parsed

        """
        subtags = list(self.iter_tags())
        substops = list(stops) + subtags
        results = Parsed()
        next_arg = args[0] if args else None
        while next_arg in subtags:
            for option in self:
                mn, mx = _n_to_minmax(option.n)
                if next_arg in option.tags:
                    if len(getattr(results, option.name, [])) >= mx:
                        raise ParseError(f'Too many {next_arg}. Expected '
                                         f'{mn} to {mx} but got at least '
                                         f'{mx + 1}')
                    args.pop(0)
                    value = option.parse(args, stops=substops)
                    if mx == 1:
                        setattr(results, option.name, value)
                    else:
                        if not hasattr(results, option.name):
                            setattr(results, option.name, [])
                        getattr(results, option.name).append(value)
                    break
            next_arg = args[0] if args else None

        # set default values / check minimum values
        for option in self:
            mn, mx = _n_to_minmax(option.n)
            if mx > 1 and len(getattr(results, option.name, [])) < mn:
                cnt = len(getattr(results, option.name, []))
                raise ParseError(f'Not enough {option.tags[0]}. Expected '
                                 f'at least {mn} but got {cnt}.')
            elif mn == mx == 1 and not hasattr(results, option.name):
                raise ParseError(f'No {option.tags[0]}. Expected 1 but got 0.')
            if not hasattr(results, option.name):
                default = option.default
                if mx > 1:
                    default = [default]
                setattr(results, option.name, default)

        return results


class OptionList(ListWithTags):
    subtype = Option


class PositionalList(TypedList):
    subtype = Positional

    def parse(self, args, stops=tuple()):
        results = Parsed()
        for option in self:
            value = option.parse(args, stops)
            setattr(results, option.name, value)
        return results


class GroupList(ListWithTags):
    subtype = Group

