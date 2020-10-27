"""Unit descriptors.

The mechanisms implemented here do not aim at performance but at
easing conversions between units.
"""
# We use a 'constant singleton' pattern


# registered units and converters
units = set()
converters = dict()


# -------------------------
# -        helpers        -
# -------------------------


class Abstract:
    """Inherit this two specify that a class is abstract and cannot be
    used directly"""
    pass


class Singleton(Abstract):
    """Abstract base class for singleton instances."""
    _instance = None

    def __new__(cls):
        cls.__str__ = lambda self: self.__class__.__name__
        if type(cls._instance) is not cls:
            cls._instance = super().__new__(cls)
        return cls._instance


# -------------------------
# -         units         -
# -------------------------


class Unit(Singleton, Abstract):
    """Abstract base class for units."""

    names: list = property(lambda self: [self.__class__.__name__.lower()])
    symbols: list = property(lambda self: [])
    scale: float = 1
    offset: float = 0
    relative_to = None

    def to(self, other, raise_error=False):
        """Return a converter from unit `self` to unit `other`

        Parameters
        ----------
        other : Unit
            Another unit
        raise_error : bool, default=False
            If True and no converter found, raise an error.

        Returns
        -------
        converter : callable
            A converter function

        """
        converter = None
        if (self, other) in converters.keys():
            converter = converters[(self, other)]
        if converter is None and raise_error:
            raise TypeError('No converter registered for types {} and {}'
                            .format(self, other))
        return converter


class UnitPrefix(Singleton, Abstract):
    """Abstract base class for unit prefixes."""
    names: list = property(lambda self: [self.__class__.__name__.lower()])
    symbols: list = property(lambda self: [])
    scale: float


def _prefixed(prefix, unit):
    """Create a new prefixed unit.

    Parameters
    ----------
    prefix : type or UnitPrefix
        Prefix class or singleton.
    unit : type or Unit
        Unit class or singleton.

    Returns
    -------
    prefixed_unit : type or Unit
        New unit class (if unit is a class) or singleton (if unit is a
        singleton).

    """
    isclass = hasattr(unit, '__base__')
    if not hasattr(unit, '__base__'):
        unit = unit.__class__
    if not hasattr(prefix, '__base__'):
        prefix = prefix.__class__

    klassname = prefix.__name__ + unit.__name__
    names = [prefixname + unitname
             for prefixname in prefix.names
             for unitname in unit.names]
    symbols = [prefixsymbol + unitsymbol
               for prefixsymbol in prefix.symbols
               for unitsymbol in unit.symbols]
    klass = type(klassname, (unit.__base__,), {
        'names': names,
        'symbols': symbols,
        'scale': prefix.scale,
        'relative_to': unit,
    })

    return klass if isclass else klass()


class DecimalPrefix(UnitPrefix, Abstract):
    logscale: float
    scale = property(lambda self: 10 ** self.logscale)


# decimal prefix types
Yocto = type("Yocto", (DecimalPrefix,), {'symbols': ['y'], 'logscale': -24})
Zepto = type("Zepto", (DecimalPrefix,), {'symbols': ['z'], 'logscale': -21})
Atto = type("Atto", (DecimalPrefix,), {'symbols': ['a'], 'logscale': -18})
Femto = type("Femto", (DecimalPrefix,), {'symbols': ['f'], 'logscale': -15})
Pico = type("Pico", (DecimalPrefix,), {'symbols': ['p'], 'logscale': -12})
Nano = type("Nano", (DecimalPrefix,), {'symbols': ['n'], 'logscale': -9})
Micro = type("Micro", (DecimalPrefix,), {'symbols': ['u', '\u00B5', '\u03BC'], 'logscale': -6})
Milli = type("Milli", (DecimalPrefix,), {'symbols': ['m'], 'logscale': -3})
Centi = type("Centi", (DecimalPrefix,), {'symbols': ['c'], 'logscale': -2})
Deci = type("Deci", (DecimalPrefix,), {'symbols': ['d'], 'logscale': -1})
Deca = type("Deca", (DecimalPrefix,), {'symbols': ['da', 'D'], 'logscale': 1})
Hecto = type("Hecto", (DecimalPrefix,), {'symbols': ['h', 'H'], 'logscale': 2})
Kilo = type("Kilo", (DecimalPrefix,), {'symbols': ['k', 'K'], 'logscale': 3})
Mega = type("Mega", (DecimalPrefix,), {'symbols': ['M'], 'logscale': 6})
Giga = type("Giga", (DecimalPrefix,), {'symbols': ['G'], 'logscale': 9})
Tera = type("Tera", (DecimalPrefix,), {'symbols': ['T'], 'logscale': 12})
Peta = type("Peta", (DecimalPrefix,), {'symbols': ['P'], 'logscale': 15})
Exa = type("Exa", (DecimalPrefix,), {'symbols': ['E'], 'logscale': 18})
Zetta = type("Zetta", (DecimalPrefix,), {'symbols': ['Z'], 'logscale': 21})
Yotta = type("Yotta", (DecimalPrefix,), {'symbols': ['Y'], 'logscale': 24})

decimal_prefixes = [Yocto, Zepto, Atto, Femto, Pico, Nano, Micro, Milli,
                    Centi, Deci, Deca, Hecto, Kilo, Mega, Giga, Tera,
                    Peta, Exa, Zetta, Yotta]


class Length(Unit, Abstract):
    pass


class PercentLength(Unit, Abstract):
    pass


class NumericLength(Length, Abstract):
    pass


class Pixel(NumericLength):
    pass


class MetricLength(Length, Abstract):
    pass


# build unit classes
# TODO: there should be a dynamic (but probably ugly) way to do this
Meter = Metre = type('Meter', (MetricLength,), {'names': ['metre', 'meter'], 'symbols': 'm'})
YoctoMeter = YoctoMetre = _prefixed(Yocto, Meter)
ZeptoMeter = ZeptoMetre = _prefixed(Zepto, Meter)
AttoMeter = AttoMetre = _prefixed(Atto, Meter)
FemtoMeter = FemtoMetre = _prefixed(Femto, Meter)
PicoMeter = PicoMetre = _prefixed(Pico, Meter)
NanoMeter = NanoMetre = _prefixed(Nano, Meter)
MicroMeter = MicroMetre = Micron = _prefixed(Micro, Meter)
MilliMeter = MilliMetre = _prefixed(Milli, Meter)
CentiMeter = CentiMetre = _prefixed(Centi, Meter)
DeciMeter = DeciMetre = _prefixed(Deci, Meter)
DecaMeter = DecaMetre = _prefixed(Deca, Meter)
HectoMeter = HectoMetre = _prefixed(Hecto, Meter)
KiloMeter = KiloMetre = _prefixed(Kilo, Meter)
MegaMeter = MegaMetre = _prefixed(Mega, Meter)
GigaMeter = GigaMetre = _prefixed(Giga, Meter)
TeraMeter = TeraMetre = _prefixed(Tera, Meter)
PetaMeter = PetaMetre = _prefixed(Peta, Meter)
ExaMeter = ExaMetre = _prefixed(Exa, Meter)
ZettaMeter = ZettaMetre = _prefixed(Zetta, Meter)
YottaMeter = YottaMetre = _prefixed(Yotta, Meter)

# build instances
m = meter = metre = Meter()
ym = yoctometer = yoctometre = YoctoMeter()
zm = zeptometer = zeptometre = ZeptoMeter()
am = attometer = attometre = AttoMeter()
fm = femtometer = femtometre = FemtoMeter()
pm = picometer = pciometre = PicoMeter()
nm = nanometer = nanometre = NanoMeter()
um = micron = micrometer = micrometre = MicroMeter()
mm = millimeter = millimetre = MilliMeter()
dm = decimeter = decimetre = DeciMeter()
dam = Dm = decameter = decametre = DecaMeter()
hm = Hm = hectometer = hectometre = HectoMeter()
km = Km = kilometer = kilometre = KiloMeter()
Mm = megameter = megametre = MegaMeter()
Gm = gigameter = gigametre = GigaMeter()
Tm = terameter = terametre = TeraMeter()
Pm = petameter = petametre = PetaMeter()
Em = exameter = exametre = ExaMeter()
Zm = zettameter = zettametre = ZettaMeter()
Ym = yottameter = yottametre = YottaMeter()


class ImperialLength(Length, Abstract):
    """Super class for Imperial length unit.

    References
    ----------
    .. https://en.wikipedia.org/wiki/Imperial_units#Length
    """
    pass

# build classes
Yard = type('Yard', (ImperialLength,), {'symbols': ['yd'], 'scale': 0.9144, 'relative_to': Meter})
Foot = type('Foot', (ImperialLength,), {'names': ['foot', 'feet'], 'symbols': ['ft', "'"], 'scale': 1/3, 'relative_to': Yard})
Inch = type('Inch', (ImperialLength,), {'names': ['inch', 'inches'], 'symbols': ['in', '"'], 'scale': 1/12, 'relative_to': Foot})
Thou = type('Thou', (ImperialLength,), {'names': ['thou', 'thousandth', 'mil'], 'symbols': ['th'], 'scale': 1E-3, 'relative_to': Inch})
Chain = type('Chain', (ImperialLength,), {'symbols': ['ch'], 'scale': 22, 'relative_to': Yard})
Furlong = type('Furlong', (ImperialLength,), {'symbols': ['fur'], 'scale': 10, 'relative_to': Chain})
Mile = type('Mile', (ImperialLength,), {'symbols': ['mi'], 'scale': 8, 'relative_to': Furlong})
League = type('League', (ImperialLength,), {'symbols': ['lea'], 'scale': 3, 'relative_to': Mile})

# build instances
yd = yard = Yard()
ft = foot = Foot()
inch = Inch()
th = thou = thousandth = Thou()
ch = chain = Chain()
fur = furlong = Furlong()
mi = mile = Mile()
lea = league = League()

# -------------------------
# -    value with unit    -
# -------------------------


class ValueWithUnit:
    """Base class for scalar types associated to a unit."""
    _unit: Unit
    unit = property(lambda self: self._unit)

    def to(self, *args, **kwargs):
        """Convert from unit `self` to unit `other`

        Parameters
        ----------
        unit : Unit, optional
            Another unit
        type : type, optional
            Another scalar type

        Returns
        -------
        value : ValueWithUnit
            A converted value

        """
        return to(self, *args, **kwargs)


# store known classes with units
_class_with_unit = {}

# list of methods that should be wrapped so that:
# 1) a conversion is performed a priori so that both inputs have the same unit
# 2) a ValueWithUnit is built and returned
_scalar_fn1 = ['__abs__', '__ceil__', '__float__', '__floor__', '__int__',
               '__invert__', '__neg__', '__pos__', '__trunc__']
_scalar_fn2 = ['__add__', '__and__', '__divmod__', '__floordiv__',
               '__lshift__', '__mod__', '__mul__', '__or__', '__pow__',
               '__radd__', '__rand__', '__rdivmod__', '__rfloordiv__',
               '__rlshift__', '__rmod__', '__rmul__', '__ror__',
               '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
               '__rtruediv__', '__rxor__', '__sub__', '__truediv__',
               '__xor__']
_scalar_fn2_bool = ['__eq__', '__ge__', '__gt__', '__le__', '__lt__', '__ne__']


def to(x, *args, **kwargs):
    """Convert a value to a unit."""

    # read arguments
    scalar_type = None
    unit = None
    while len(args) > 0:
        if isinstance(args[0], Unit):
            if unit is None:
                unit = args[0]
            else:
                raise TypeError('More than one unit provided.')
        else:
            if scalar_type is None:
                scalar_type = args[0]
            else:
                raise TypeError('More than one scalar type provided.')
        args = args[1:]
    if 'type' in kwargs.keys():
        if scalar_type is None:
            scalar_type = kwargs['type']
        else:
            raise TypeError('More than one scalar type provided.')
    if 'unit' in kwargs.keys():
        if unit is None:
            unit = kwargs['unit']
        else:
            raise TypeError('More than one unit provided.')

    # if no target unit is specified:
    # * either x has a unit and we store it and continue with the algo
    # * or x is a pure scalar and we return a pure scalar
    #   (potentially casted if a target scalar_type is specified)
    #   (if no cast necessary, we avoid a copy and return the input)
    if unit is None:
        if isinstance(x, ValueWithUnit):
            unit = x.unit
        elif scalar_type is None or isinstance(x, scalar_type):
            return x
        else:
            return scalar_type(x)

    if not isinstance(unit, Unit):
        raise TypeError('`unit` should be a Unit. Got {}.'.format(type(unit)))

    # at the point, the target unit is defined.
    # if the input has a unit, we need a conversion.
    if isinstance(x, ValueWithUnit):
        if (scalar_type is None or isinstance(x, scalar_type)) \
                and x.unit == unit:
            # no need to do anything: we can return the input
            return x
        # else, we get the appropriate converter
        converter = converters.get((x.unit, unit), None)
        if converter is None:
            raise TypeError('No converter registered for types {} and {}'
                            .format(x.unit, unit))
        x = converter(x)

    # now that we know the converted type of x, we can choose a
    # default value for scalar_type.
    # we cannot do this earlier, because an integer type can end up
    # being a float after conversion.
    if scalar_type is None:
        scalar_type = type(x)

    # helper functions
    # scalar methods can be:
    # * arg1:       scalar_type              -> scalar_type
    # * arg2:       scalar_type, scalar_type -> scalar_type
    # * arg2_bool:  scalar_type, scalar_type -> bool

    def arg1(fn):
        return lambda x: to(getattr(scalar_type, fn)(x), x.unit)

    def arg2(fn):
        return lambda x, y: to(getattr(scalar_type, fn)(x, to(y, x.unit)), x.unit)

    def arg2_bool(fn):
        return lambda x, y: getattr(scalar_type, fn)(x, to(y, x.unit))

    if scalar_type not in _class_with_unit.keys():
        # build class
        klass_name = scalar_type.__name__ + 'WithUnit'
        attr = {
            '_unit': None,
            '__str__': lambda x: scalar_type.__str__(x) + ' ' +  x.unit.symbols[0],
            '__repr__': lambda x: scalar_type.__repr__(x) + ' ' +  x.unit.symbols[0],
        }
        for fn in _scalar_fn1:
            attr[fn] = arg1(fn)         # function with one argument
        for fn in _scalar_fn2:
            attr[fn] = arg2(fn)         # function with two argument
        for fn in _scalar_fn2_bool:
            attr[fn] = arg2_bool(fn)    # function -> bool with two argument
        klass = type(klass_name, (scalar_type, ValueWithUnit), attr)
        _class_with_unit[scalar_type] = klass
    else:
        # get class
        klass = _class_with_unit[scalar_type]

    # build object and set unit
    obj = klass(x)
    obj._unit = unit
    return obj


# -------------------------
# -   unit registration   -
# -------------------------


def register_units(unit_set=None, out=None):
    """Add units to the set of registered units.

    Parameters
    ----------
    unit_set : sequence, default=`locals().items()`
        Sequence (of `Unit` instances)
        Only non-`Abstract` `Unit` instances are registered.
    out  : set, default=`units`
        Output set. Units in `unit_set` will be appended.

    Returns
    -------
    out : set
        Output set.

    """
    if unit_set is None:
        unit_set = globals().values()
    # filter out non-unit objects
    unit_set = [var for var in unit_set
                if isinstance(var, Unit) and
                Abstract not in var.__class__.__bases__]
    if out is None:
        out = units
    out.update(unit_set)
    return out


def _upflow(target, root_to_node=None, root=None):
    """Navigate through the tree(s) upflow.

    Unit store the relationship to one parent in the form
    of an affine transform: parent = a * child + b
    When taken all together, units form a collection of non-overlapping
    trees. However, only the child-to-parent relationship is stored;
    we therefore need to build the trees by browsing connections upflow.
    To save time, we navigate one branch at a time and stop when
    we reach a branch that is already known (we can then compose
    the two affine transforms:
        unknown_unit -(aff1)-> known_unit -(aff2)-> root

    Parameters
    ----------
    target : Unit
        A node in the tree(s).
    root_to_node : dict, default=dict()
        A nested dictionary {root: {node: (scale, offset)}}
        that stores root to node affine transforms
        (root = scale * node + offset)
    root : Unit, optional
        A hint of who the root is.

    Returns
    -------
    root_to_node : dict
        A nested dictionary {root: {node: (scale, offset)}}
        In the end, it will should have all directed pairs (root, node)
        in the trees.

    """
    if root_to_node is None:
        root_to_node = dict()

    # check if target is known
    # (we may have found it during a previous call)
    for _root, nodes in root_to_node.items():
        if target in nodes.keys():
            return root_to_node, _root

    # move cursor
    cursor = target.relative_to

    # if root provided, it's an internal sign that we know that cursor
    # is already in the graph
    if root is not None:
        scale_cursor, offset_cursor = root_to_node[root][cursor]
        offset = target.offset + offset_cursor
        scale = target.scale * scale_cursor
        root_to_node[root] = root_to_node.get(root, {})
        root_to_node[root][target] = (scale, offset)
        return root_to_node, root

    # check if we are the root
    if cursor is None:
        return root_to_node, target

    # check if we are just before the root
    if cursor.relative_to is None:
        root_to_node[cursor] = root_to_node.get(cursor, {})
        root_to_node[cursor][target] = (target.scale, target.offset)
        return root_to_node, cursor

    # check if cursor is known
    for root, nodes in root_to_node.items():
        if cursor in nodes.keys():
            scale_cursor, offset_cursor = root_to_node[root][cursor]
            offset = target.offset + offset_cursor
            scale = target.scale * scale_cursor
            root_to_node[root] = root_to_node.get(root, {})
            root_to_node[root][target] = (scale, offset)
            return root_to_node, root

    # else
    # 1) start a new flow from the cursor.
    _, root = _upflow(cursor, root_to_node)
    # 2) because of step 1, we already know the path from source to root
    #    so we can just compose our transforms.
    #    we call the same function again for simplicity.
    _upflow(target, root_to_node, root)

    return root_to_node, root


def _browse_units(unit_set, out=None):
    """Build all possible converters between pairs of units."""

    if out is None:
        out = dict()

    # 1) build trees
    root_to_node = dict()
    for unit in unit_set:
        _upflow(unit, root_to_node)

    def make_affine(scale, offset):
        def affine(x): return scale * x + offset
        return affine

    # 2) we have all (node -> root) transforms in the dict.
    #    we just need to build all pairs (node1 <-> node2)
    #    by composition
    for root, subdict in root_to_node.items():
        out[(root, root)] = lambda x: x
        nodes = list(subdict.keys())
        for i in range(len(nodes)):
            node1 = nodes[i]
            scale1, offset1 = subdict[node1]
            out[(node1, node1)] = lambda x: x
            out[(node1, root)] = make_affine(scale1, offset1)
            out[(root, node1)] = make_affine(1/scale1, -offset1/scale1)
            for j in range(i+1, len(nodes)):
                node2 = nodes[j]
                scale2, offset2 = subdict[node2]
                scale21 = scale1 / scale2
                offset21 = offset1 - scale21 * offset2
                out[(node1, node2)] = make_affine(scale21, offset21)
                out[(node2, node1)] = make_affine(1/scale21, -offset21/scale21)

    return out


def register_converters(unit_set=None, converter_dict=None, out=None):
    """Add converters to the dictionary of registered converters.

    Parameters
    ----------
    unit_set : sequence, optional
        Set of known units. Their directed graph of relations will be
        browsed and corresponding converter functions created.
    converter_dict : dict, optional
        Dictionary of known converters
    out : dict, default=`converters`
        Output dict. Known and computed converters will be appended.

    Returns
    -------

    """

    if out is None:
        out = converters

    # 1. build converters by browsing unit graph
    if unit_set is not None:
        unit_set = [unit for unit in unit_set
                    if isinstance(unit, Unit)
                    and Abstract not in unit.__class__.__bases__]
        _browse_units(unit_set, out=out)

    # 2. register known converters
    # (if conflict, they have priority over computed converters)
    if converter_dict is not None:
        for source, subdict in converter_dict.items():
            for target, conv_fn in subdict.items():
                out[(source, target)] = conv_fn
                if (target, source) not in out.keys():
                    out[(target, source) ] = None

    return out


# register all units in local scope
register_units()

# register all converters build from unit list
register_converters(units)
