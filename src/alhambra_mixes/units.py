from __future__ import annotations

import decimal
from decimal import Decimal
from typing import Sequence, TypeVar, Union, overload
from typing_extensions import TypeAlias
import pint
from pint import Quantity
from pint.facets.plain import PlainQuantity

# This needs to be here to make Decimal NaNs behave the way that NaNs
# *everywhere else in the standard library* behave.
decimal.setcontext(decimal.ExtendedContext)

__all__ = [
    "ureg",
    "uL",
    "nM",
    "uM",
    "Q_",
    "DNAN",
    "ZERO_VOL",
    "NAN_VOL",
    "NAN_CONC",
    "Decimal",
    "Quantity",
    "DecimalQuantity",
]

ureg = pint.UnitRegistry(non_int_type=Decimal)
ureg.default_format = "~P"

uL = ureg.uL
# µL = ureg.uL
uM = ureg.uM
nM = ureg.nM

DecimalQuantity: TypeAlias = "Quantity[Decimal]"


def Q_(
    qty: int | str | Decimal | float, unit: str | pint.Unit | None = None
) -> DecimalQuantity:
    "Convenient constructor for units, eg, :code:`Q_(5.0, 'nM')`.  Ensures that the quantity is a Decimal."
    if unit is not None:
        return ureg.Quantity(Decimal(qty), unit)
    else:
        return ureg.Quantity(qty)


class VolumeError(ValueError):
    pass


DNAN = Decimal("nan")
ZERO_VOL = Q_("0.0", "µL")
NAN_VOL = Q_("nan", "µL")

NAN_CONC = Q_("nan", "µM")

T = TypeVar("T", bound=Union[float, Decimal])


@overload
def _ratio(
    top: Sequence[PlainQuantity[T]], bottom: Sequence[Quantity[Union[float, Decimal]]]
) -> Sequence[T]:
    ...


@overload
def _ratio(
    top: PlainQuantity[T], bottom: Sequence[PlainQuantity[T]]
) -> Sequence[Union[float, Decimal]]:
    ...


@overload
def _ratio(
    top: Sequence[PlainQuantity[T]], bottom: PlainQuantity[T]
) -> Sequence[Union[float, Decimal]]:
    ...


@overload
def _ratio(top: PlainQuantity[T], bottom: PlainQuantity[T]) -> Union[float, Decimal]:
    ...


def _ratio(
    top: PlainQuantity[T] | Sequence[PlainQuantity[T]],
    bottom: PlainQuantity[T] | Sequence[PlainQuantity[T]],
) -> Union[float, Decimal] | Sequence[Union[float, Decimal]]:
    if isinstance(top, Sequence) and isinstance(bottom, Sequence):
        return [(x / y).m_as("") for x, y in zip(top, bottom)]
    elif isinstance(top, Sequence):
        return [(x / bottom).m_as("") for x in top]
    elif isinstance(bottom, Sequence):
        return [(top / y).m_as("") for y in bottom]
    return (top / bottom).m_as("")


def _parse_conc_optional(v: str | Quantity | None) -> DecimalQuantity:
    """Parses a string or Quantity as a concentration; if None, returns a NaN
    concentration."""
    if isinstance(v, str):
        q = ureg.Quantity(v)
        if not q.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        return q
    elif isinstance(v, Quantity):
        if not v.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    elif v is None:
        return Q_(DNAN, nM)
    raise ValueError


def _parse_conc_required(v: str | Quantity) -> DecimalQuantity:
    """Parses a string or Quantity as a concentration, requiring that
    it result in a value."""
    if isinstance(v, str):
        q = ureg.Quantity(v)
        if not q.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        return q
    elif isinstance(v, Quantity):
        if not v.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be molarity).")


def _parse_vol_optional(v: str | Quantity) -> DecimalQuantity:
    """Parses a string or quantity as a volume, returning a NaN volume
    if the value is None.
    """
    # if isinstance(v, (float, int)):  # FIXME: was in quantitate.py, but potentially unsafe
    #    v = f"{v} µL"
    if isinstance(v, str):
        q = ureg.Quantity(v)
        if not q.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        return q
    elif isinstance(v, Quantity):
        if not v.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    elif v is None:
        return Q_(DNAN, uL)
    raise ValueError


def _parse_vol_optional_none_zero(v: str | Quantity) -> DecimalQuantity:
    """Parses a string or quantity as a volume, returning a NaN volume
    if the value is None.
    """
    # if isinstance(v, (float, int)):  # FIXME: was in quantitate.py, but potentially unsafe
    #    v = f"{v} µL"
    if isinstance(v, str):
        q = ureg.Quantity(v)
        if not q.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        return q
    elif isinstance(v, Quantity):
        if not v.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    elif v is None:
        return ZERO_VOL
    raise ValueError


def _parse_vol_required(v: str | Quantity) -> DecimalQuantity:
    """Parses a string or quantity as a volume, requiring that it result in a
    value.
    """
    # if isinstance(v, (float, int)):
    #    v = f"{v} µL"
    if isinstance(v, str):
        q = ureg.Quantity(v)
        if not q.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        return q
    elif isinstance(v, Quantity):
        if not v.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be volume).")


def normalize(quantity: Quantity) -> Quantity:
    """
    Normalize `quantity` so that it is "compact" (uses units within the correct "3 orders of magnitude":
    https://pint.readthedocs.io/en/0.18/tutorial.html#simplifying-units)
    and eliminate trailing zeros.

    :param quantity:
        a pint Quantity[Decimal]
    :return:
        `quantity` normalized to be compact and without trailing zeros.
    """
    quantity = quantity.to_compact()
    mag_int = quantity.magnitude.to_integral()
    if mag_int == quantity.magnitude:
        # can be represented exactly as integer, so return that;
        # quantity.magnitude.normalize() would use scientific notation in this case, which we don't want
        quantity = Q_(mag_int, quantity.units)
    else:
        # is not exact integer, so normalize will return normal float literal such as 10.2
        # and not scientific notation like it would for an integer
        mag_norm = quantity.magnitude.normalize()
        quantity = Q_(mag_norm, quantity.units)
    return quantity
