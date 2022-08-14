from __future__ import annotations

import decimal
from decimal import Decimal
from typing import Sequence, TypeVar, overload

import pint
from pint import Quantity

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
    "Decimal",
    "Quantity",
]

ureg = pint.UnitRegistry(non_int_type=Decimal)
ureg.default_format = "~P"

uL = ureg.uL
# µL = ureg.uL
uM = ureg.uM
nM = ureg.nM


def Q_(qty: int | str | Decimal | float, unit: str | pint.Unit) -> pint.Quantity:
    "Convenient constructor for units, eg, :code:`Q_(5.0, 'nM')`.  Ensures that the quantity is a Decimal."
    return ureg.Quantity(Decimal(qty), unit)


class VolumeError(ValueError):
    pass


DNAN = Decimal("nan")
ZERO_VOL = Q_("0.0", "µL")
NAN_VOL = Q_("nan", "µL")

T = TypeVar("T")


@overload
def _ratio(
    top: Sequence[pint.Quantity[T]], bottom: Sequence[pint.Quantity[T]]
) -> Sequence[T]:
    ...


@overload
def _ratio(top: pint.Quantity[T], bottom: Sequence[pint.Quantity[T]]) -> Sequence[T]:
    ...


@overload
def _ratio(top: Sequence[pint.Quantity[T]], bottom: pint.Quantity[T]) -> Sequence[T]:
    ...


@overload
def _ratio(top: pint.Quantity[T], bottom: pint.Quantity[T]) -> T:
    ...


def _ratio(
    top: pint.Quantity[T] | Sequence[pint.Quantity[T]],
    bottom: pint.Quantity[T] | Sequence[pint.Quantity[T]],
) -> T | Sequence[T]:
    if isinstance(top, Sequence) and isinstance(bottom, Sequence):
        return [(x / y).m_as("") for x, y in zip(top, bottom)]
    elif isinstance(top, Sequence):
        return [(x / bottom).m_as("") for x in top]
    elif isinstance(bottom, Sequence):
        return [(top / y).m_as("") for y in bottom]
    return (top / bottom).m_as("")


def _parse_conc_optional(v: str | pint.Quantity | None) -> pint.Quantity:
    """Parses a string or Quantity as a concentration; if None, returns a NaN
    concentration."""
    if isinstance(v, str):
        q = ureg(v)
        if not q.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        return q
    elif isinstance(v, pint.Quantity):
        if not v.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    elif v is None:
        return Q_(DNAN, nM)
    raise ValueError


def _parse_conc_required(v: str | pint.Quantity) -> pint.Quantity:
    """Parses a string or Quantity as a concentration, requiring that
    it result in a value."""
    if isinstance(v, str):
        q = ureg(v)
        if not q.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        return q
    elif isinstance(v, pint.Quantity):
        if not v.check(nM):
            raise ValueError(f"{v} is not a valid quantity here (should be molarity).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be molarity).")


def _parse_vol_optional(v: str | pint.Quantity) -> pint.Quantity:
    """Parses a string or quantity as a volume, returning a NaN volume
    if the value is None.
    """
    # if isinstance(v, (float, int)):  # FIXME: was in quantitate.py, but potentially unsafe
    #    v = f"{v} µL"
    if isinstance(v, str):
        q = ureg(v)
        if not q.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        return q
    elif isinstance(v, pint.Quantity):
        if not v.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    elif v is None:
        return Q_(DNAN, uL)
    raise ValueError


def _parse_vol_required(v: str | pint.Quantity) -> pint.Quantity:
    """Parses a string or quantity as a volume, requiring that it result in a
    value.
    """
    # if isinstance(v, (float, int)):
    #    v = f"{v} µL"
    if isinstance(v, str):
        q = ureg(v)
        if not q.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        return q
    elif isinstance(v, pint.Quantity):
        if not v.check(uL):
            raise ValueError(f"{v} is not a valid quantity here (should be volume).")
        v = Q_(v.m, v.u)
        return v.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be volume).")
