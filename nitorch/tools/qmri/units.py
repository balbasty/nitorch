from typing import Tuple, List, Union
import math


# ----------------------------------------------------------------------
#       UNITS
# ----------------------------------------------------------------------

class Unit:
    factor: float = 1  # multiplicative factor wrt standard unit
    symbol: str = ''

    def __init__(self, value):
        if isinstance(value, Unit):
            value = value.to(type(self))
        self.value = value

    @classmethod
    def map(cls, f, x):
        if isinstance(x, (list, tuple)):
            return type(x)(map(f, x))
        elif isinstance(x, dict):
            return {k: f(v) for k, v in x.items()}
        else:
            return f(x)

    @classmethod
    def make_to(cls, unit: type):
        return lambda x: unit.factor * x / cls.factor

    def to(self, unit: type):
        if unit is type(self):
            return self
        if issubclass(unit, self.__base__):
            return unit(self.map(self.make_to(unit), self.value))
        raise TypeError('Cannot convert {type(self).__name__} '
                        'to {unit.__name__}')

    def repr(self):
        return f'{self.value} {self.symbol}'

    def __repr__(self):
        return self.repr()

    def __str__(self):
        return self.repr()


class Angle(Unit):
    pass


class Degree(Angle):
    symbol = '°'
    factor = 180 / math.pi


class Radian(Angle):
    symbol = 'rad'
    pass


class Time(Unit):
    pass


class Second(Time):
    symbol = 's'
    pass


class MicroSecond(Time):
    symbol = 'μs'
    factor = 1e-6


class NanoSecond(Time):
    symbol = 'ns'
    factor = 1e-9


class MilliSecond(Time):
    symbol = 'ms'
    factor = 1e-3


class Minute(Time):
    symbol = 'min'
    factor = 60


class Hour(Time):
    symbol = 'h'
    factor = 60 * Minute.factor


class Day(Time):
    symbol = 'd'
    factor = 24 * Hour.factor


class Distance(Unit):
    pass


class Metre(Distance):
    symbol = 'm'
    pass


class KiloMetre(Distance):
    symbol = 'Km'
    factor = 1e3


class HectoMetre(Distance):
    symbol = 'Hm'
    factor = 1e2


class DecaMetre(Distance):
    symbol = 'Dm'
    factor = 1e1


class DeciMetre(Distance):
    symbol = 'cm'
    factor = 1e-1


class CentiMetre(Distance):
    symbol = 'cm'
    factor = 1e-2


class MilliMetre(Distance):
    symbol = 'mm'
    factor = 1e-3


class MicroMetre(Distance):
    symbol = 'μm'
    factor = 1e-6


class NanoMetre(Distance):
    symbol = 'nm'
    factor = 1e-9


class Unary(Unit):
    base: Unit = None


class PowerBase(Unary):
    exponent: float = 1

    @classmethod
    def make_to(cls, unit: type):
        base_to = cls.base.make_to(unit.base)
        return lambda x: base_to(x ** (1/cls.exponent)) ** cls.exponent


class ProdUnitBase(Unit):
    base: List[Unit] = []

    @classmethod
    def make_to(cls, unit: type):
        def convert(x):
            for base1, base2 in (cls.base, unit.base):
                base_to = base1.make_to(base2)
                x = base_to(x).value
            return x
        return convert


_kls_families = {Angle: Radian, Distance: Metre, Time: Second}
_kls_power = dict()
_kls_prod = dict()


def pow(unit, exponent):
    """Take the power of a unit"""

    if isinstance(unit, PowerBase):
        return pow(unit.base, exponent * unit.exponent)

    if isinstance(unit, ProdUnitBase):
        return prod([pow(sub, exponent) for sub in unit.base])

    if exponent == 1:
        return unit

    if (unit, exponent) not in _kls_power:
        _exponent = exponent
        class Power(PowerBase):
            base = unit
            exponent: float = _exponent
        Power.symbol = f'{unit.symbol} ** {_exponent}'
        _kls_power[(unit, exponent)] = Power

    return _kls_power[(unit, exponent)]


def prod(units):
    units0, units = units, []
    for unit in units0:
        if isinstance(unit, ProdUnitBase):
            units += unit.base
        else:
            units.append(unit)
    del units0

    units0, units = units, {}
    for family in _kls_families:
        units[family] = []
    for unit in units0:
        for family in _kls_families:
            if isinstance(unit, family) or \
                (isinstance(unit, PowerBase) and
                 isinstance(unit.base, family)):
                units[family].append(unit)
    units0, units = units, []
    for family, same_units in units0.items():
        standard = _kls_families[family]
        units.append(merge(standard, same_units))

    if len(units) == 1:
        return units[0]

    units = tuple(units)
    if units not in _kls_prod:
        class Prod(ProdUnitBase):
            base = units
        Prod.symbol = '·'.join([unit.symbol for unit in units])
        _kls_prod[units] = Prod

    return _kls_prod[units]


def merge(standard, units):
    exponent = 1
    for unit in units:
        if isinstance(unit, PowerBase):
            exponent *= unit.exponent
    return pow(standard, exponent)