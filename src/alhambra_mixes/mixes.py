"""
A module for handling mixes.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from decimal import Decimal

import decimal
import json

# This needs to be here to make Decimal NaNs behave the way that NaNs
# *everywhere else in the standard library* behave.
decimal.setcontext(decimal.ExtendedContext)

import enum

import logging
import math
from os import PathLike
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    overload,
    Callable,
    Tuple,
)
from typing_extensions import (
    TypeAlias,
)
from pathlib import Path

import numpy as np
import pandas as pd
import pint
from tabulate import tabulate, TableFormat, Line

if TYPE_CHECKING:
    from pandas.core.indexing import _LocIndexer

import attrs

import warnings

from pint import Quantity

warnings.filterwarnings(
    "ignore",
    "The unit of the quantity is " "stripped when downcasting to ndarray",
    pint.UnitStrippedWarning,
)

warnings.filterwarnings(
    "ignore",
    "pint-pandas does not support magnitudes of class <class 'int'>",
    RuntimeWarning,
)

__all__ = (
    "uL",
    "uM",
    "nM",
    "Q_",
    "Component",
    "Strand",
    "FixedVolume",
    "FixedConcentration",
    "MultiFixedVolume",
    "MultiFixedConcentration",
    "Mix",
    "AbstractComponent",
    "AbstractAction",
    "WellPos",
    "MixLine",
    "Reference",
    "load_reference",
    "_format_title",
    "ureg",
    "DNAN",
    #    "D",
    "VolumeError",
    "save_mixes",
    "load_mixes",
)

log = logging.getLogger("alhambra")

ureg = pint.UnitRegistry(non_int_type=Decimal)
ureg.default_format = "~P"

uL = ureg.uL
uM = ureg.uM
nM = ureg.nM


def Q_(qty: int | str | Decimal | float, unit: str | pint.Unit) -> pint.Quantity:
    "Convenient constructor for units, eg, :code:`Q_(5.0, 'nM')`.  Ensures that the quantity is a Decimal."
    return ureg.Quantity(Decimal(qty), unit)


DNAN = Decimal("nan")
ZERO_VOL = Q_("0.0", "µL")
NAN_VOL = Q_("nan", "µL")

ROW_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWX"

MIXHEAD_EA = (
    "Component",
    "[Src]",
    "[Dest]",
    "#",
    "Ea Tx Vol",
    "Tot Tx Vol",
    "Location",
    "Note",
)
MIXHEAD_NO_EA = ("Component", "[Src]", "[Dest]", "Tx Vol", "Location", "Note")


class VolumeError(ValueError):
    pass


T = TypeVar("T")


def _maybesequence(object_or_sequence: Sequence[T] | T) -> list[T]:
    if isinstance(object_or_sequence, Sequence):
        return list(object_or_sequence)
    return [object_or_sequence]


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


@attrs.define(init=False, frozen=True, order=True, hash=True)
class WellPos:
    """A Well reference, allowing movement in various directions and bounds checking.

    This uses 1-indexed row and col, in order to match usual practice.  It can take either
    a standard well reference as a string, or two integers for the row and column.
    """

    row: int = attrs.field()
    col: int = attrs.field()
    platesize: Literal[96, 384] = 96

    @row.validator
    def _validate_row(self, v: int) -> None:
        rmax = 8 if self.platesize == 96 else 16
        if (v <= 0) or (v > rmax):
            raise ValueError(
                f"Row {ROW_ALPHABET[v - 1]} ({v}) out of bounds for plate size {self.platesize}"
            )

    @col.validator
    def _validate_col(self, v: int) -> None:
        cmax = 12 if self.platesize == 96 else 24
        if (v <= 0) or (v > cmax):
            raise ValueError(
                f"Column {v} out of bounds for plate size {self.platesize}"
            )

    @overload
    def __init__(
        self, ref_or_row: int, col: int, /, *, platesize: Literal[96, 384] = 96
    ) -> None:  # pragma: no cover
        ...

    @overload
    def __init__(
        self, ref_or_row: str, col: None = None, /, *, platesize: Literal[96, 384] = 96
    ) -> None:  # pragma: no cover
        ...

    def __init__(
        self,
        ref_or_row: str | int,
        col: int | None = None,
        /,
        *,
        platesize: Literal[96, 384] = 96,
    ) -> None:
        if isinstance(ref_or_row, str) and (col is None):
            row: int = ROW_ALPHABET.index(ref_or_row[0]) + 1
            col = int(ref_or_row[1:])
        elif isinstance(ref_or_row, WellPos) and (col is None):
            row = ref_or_row.row
            col = ref_or_row.col
            platesize = ref_or_row.platesize
        elif isinstance(ref_or_row, int) and isinstance(col, int):
            row = ref_or_row
            col = col
        else:
            raise TypeError

        if platesize not in (96, 384):
            raise ValueError(f"Plate size {platesize} not supported.")
        object.__setattr__(self, "platesize", platesize)

        self._validate_col(cast(int, col))
        self._validate_row(row)

        object.__setattr__(self, "row", row)
        object.__setattr__(self, "col", col)

    def __str__(self) -> str:
        return f"{ROW_ALPHABET[self.row - 1]}{self.col}"

    def __repr__(self) -> str:
        return f'WellPos("{self}")'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WellPos):
            return (other.row == self.row) and (other.col == self.col)
        elif isinstance(other, str):
            return self == WellPos(other, platesize=self.platesize)

        return False

    def key_byrow(self) -> tuple[int, int]:
        "Get a tuple (row, col) key that can be used for ordering by row."
        try:
            return (self.row, self.col)
        except AttributeError:
            return (-1, -1)

    def key_bycol(self) -> tuple[int, int]:
        "Get a tuple (col, row) key that can be used for ordering by column."
        try:
            return (self.col, self.row)
        except AttributeError:
            return (-1, -1)

    def next_byrow(self) -> WellPos:
        "Get the next well, moving right along rows, then down."
        CMAX = 12 if self.platesize == 96 else 24
        return WellPos(
            self.row + (self.col + 1) // (CMAX + 1),
            (self.col) % CMAX + 1,
            platesize=self.platesize,
        )

    def next_bycol(self) -> WellPos:
        "Get the next well, moving down along columns, and then to the right."
        RMAX = 8 if self.platesize == 96 else 16
        return WellPos(
            (self.row) % RMAX + 1,
            self.col + (self.row + 1) // (RMAX + 1),
            platesize=self.platesize,
        )

    def is_last(self) -> bool:
        """
        :return:
            whether WellPos is the last well on this type of plate
        """
        rows = _96WELL_PLATE_ROWS if self.platesize == 96 else _384WELL_PLATE_ROWS
        cols = _96WELL_PLATE_COLS if self.platesize == 96 else _384WELL_PLATE_COLS
        return self.row == len(rows) and self.col == len(cols)

    def advance(self, order: Literal["row", "col"] = "col") -> WellPos:
        """
        Advances to the "next" well position. Default is column-major order, i.e.,
        A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, ...
        To switch to row-major order, select `order` as `'row'`, i.e.,
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, B1, B2, ...

        :return:
            new WellPos representing the next well position
        """
        rows = _96WELL_PLATE_ROWS if self.platesize == 96 else _384WELL_PLATE_ROWS
        cols = _96WELL_PLATE_COLS if self.platesize == 96 else _384WELL_PLATE_COLS
        next_row = self.row
        next_col = self.col
        if order == "col":
            next_row += 1
            if next_row == len(rows) + 1:
                next_row = 1
                next_col += 1
                if next_col == len(cols) + 1:
                    raise ValueError("cannot advance WellPos; already on last well")
        else:
            next_col += 1
            if next_col == len(cols) + 1:
                next_col = 1
                next_row += 1
                if next_row == len(rows) + 1:
                    raise ValueError("cannot advance WellPos; already on last well")

        return WellPos(next_row, next_col, platesize=self.platesize)


def emphasize(text: str, tablefmt: str | TableFormat, strong: bool = False) -> str:
    """
    Emphasizes `text` according to `tablefmt`, e.g., for Markdown (e.g., `tablefmt` = `'pipe'`),
    surrounds with pair of *'s; if `strong` is True, with double *'s. For `tablefmt` = `'html'`,
    uses ``<emph>`` or ``<strong>``.

    :param text:
        text to emphasize
    :param tablefmt:
        format in which to add emphasis markup
    :return:
        emphasized version of `text`
    """
    # formats a title for a table produced using tabulate,
    # in the formats tabulate understands
    if tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:  # type: ignore
        if strong:
            emph_text = f"<strong>{text}</strong>"
        else:
            emph_text = f"<em>{text}</em>"
    elif tablefmt in ["latex", "latex_raw", "latex_booktabs", "latex_longtable"]:
        if strong:
            emph_text = r"\textbf{" + text + r"}"
        else:
            emph_text = r"\emph{" + text + r"}"
    else:  # use the emphasis for tablefmt == "pipe" (Markdown)
        star = "**" if strong else "*"
        emph_text = f"{star}{text}{star}"
    return emph_text


@attrs.define(eq=True)
class MixLine:
    """Class for handling a line of a (processed) mix recipe.

    Each line should represent a single step, or series of similar steps (same volume per substep)
    in the mixing process.

    Parameters
    ----------

    names
        A list of component names.  For a single step, use [name].

    source_conc
        The source concentration; may not be provided (will be left blank), or be a descriptive string.

    dest_conc
        The destination/target concentration; may not be provided (will be left blank), or be a descriptive string.

    total_tx_vol
        The total volume added to the mix by the step.  If zero, the amount will still be included in tables.
        If None, the amount will be blank.  If provided, and the line is not fake, the value must be correct
        and interpretable for calculations involving the mix.

    number
        The number of components added / subste

    each_tx_vol
        The volume per component / substep.  May be omitted, or a descriptive string.

    plate
        The plate name for the mix, a descriptive string for location / source type (eg, "tube") or None (omitted).
        A single MixLine, at present, should not involve multiple plates.

    wells
        A list of wells for the components in a plate.  If the components are not in a plate, this must be an
        empty list.  This *does not* parse strings; wells must be provided as WellPos instances.

    note
        A note to add for the line

    fake
        Denotes that the line is not a real step, eg, for a summary/total information line.  The line
        will be distinguished in some way in tables (eg, italics) and will not be included in calculations.
    """

    names: list[str] = attrs.field(factory=list)
    source_conc: Quantity[Decimal] | str | None = None
    dest_conc: Quantity[Decimal] | str | None = None
    total_tx_vol: Quantity[Decimal] = NAN_VOL
    number: int = 1
    each_tx_vol: Quantity[Decimal] = NAN_VOL  # | str | None = None
    plate: str = ""
    wells: list[WellPos] = attrs.field(factory=list)
    note: str | None = None
    fake: bool = False

    def __attrs_post_init__(self):
        if (
            math.isnan(self.each_tx_vol.m)
            and not math.isnan(self.total_tx_vol.m)
            and self.number == 1
        ):
            self.each_tx_vol = self.total_tx_vol

    @wells.validator
    def _check_wells(self, _: str, v: Any) -> None:
        if (not isinstance(v, list)) or any(not isinstance(x, WellPos) for x in v):
            raise TypeError(f"MixLine.wells of {v} is not a list of WellPos.")

    @names.validator
    def _check_names(self, _: str, v: Any) -> None:
        if (not isinstance(v, list)) or any(not isinstance(x, str) for x in v):
            raise TypeError(f"MixLine.names of {v} is not a list of strings.")

    def location(
        self, tablefmt: str | TableFormat = "pipe", split: bool = True
    ) -> tuple[str | list[str], list[int]]:
        "A formatted string (according to `tablefmt`) for the location of the component/components."
        if len(self.wells) == 0:
            return f"{self.plate}", []
        elif len(self.wells) == 1:
            return f"{self.plate}: {self.wells[0]}", []

        byrow = mixgaps(
            sorted(list(self.wells), key=WellPos.key_byrow),
            by="row",
        )
        bycol = mixgaps(
            sorted(list(self.wells), key=WellPos.key_bycol),
            by="col",
        )

        sortnext = WellPos.next_bycol if bycol <= byrow else WellPos.next_byrow

        splits = []
        wells_formatted = []
        next_well_iter = iter(self.wells)
        prevpos = next(next_well_iter)
        formatted_prevpos = emphasize(f"{self.plate}: {prevpos}", tablefmt, strong=True)
        wells_formatted.append(formatted_prevpos)
        for i, well in enumerate(next_well_iter):
            if (sortnext(prevpos) != well) or (
                (prevpos.col != well.col) and (prevpos.row != well.row)
            ):
                formatted_well = emphasize(f"{well}", tablefmt, strong=True)
                wells_formatted.append(formatted_well)
                if split:
                    splits.append(i)
            else:
                wells_formatted.append(f"{well}")
            prevpos = well

        return wells_formatted, splits

    def toline(
        self, incea: bool, tablefmt: str | TableFormat = "pipe"
    ) -> Sequence[str]:
        locations, splits = self.location(tablefmt=tablefmt)
        if incea:
            return [
                _formatter(
                    self.names, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.source_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.dest_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.number, italic=self.fake, tablefmt=tablefmt)
                if self.number != 1
                else "",
                _formatter(self.each_tx_vol, italic=self.fake, tablefmt=tablefmt)
                if not math.isnan(self.each_tx_vol.m)
                else "",
                _formatter(self.total_tx_vol, italic=self.fake, tablefmt=tablefmt),
                _formatter(
                    locations, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.note, italic=self.fake),
            ]
        else:
            return [
                _formatter(
                    self.names, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.source_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.dest_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.total_tx_vol, italic=self.fake, tablefmt=tablefmt),
                _formatter(
                    locations, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.note, italic=self.fake, tablefmt=tablefmt),
            ]


def _format_error_span(out, tablefmt):
    if tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:
        return f"<span style='color:red'>{out}</span>"
    else:
        return f"**{out}**"


_NL = {
    "pipe": "\n",
    "html": "<br/>",
    "unsafehtml": "<br/>",
}


def _formatter(
    x: int | float | str | list[str] | Quantity[Decimal] | None,
    italic: bool = False,
    tablefmt: str | TableFormat = "pipe",
    splits: list = [],
) -> str:
    if isinstance(x, (int, str)):
        out = str(x)
    elif x is None:
        out = ""
    elif isinstance(x, float):
        out = f"{x:,.2f}"
        if math.isnan(x):
            out = _format_error_span(out, tablefmt)
    elif isinstance(x, Quantity):
        out = f"{x:,.2f~#P}"
        if math.isnan(x.m):
            out = _format_error_span(out, tablefmt)
        if x.m < 0:
            out = _format_error_span(out, tablefmt)
    elif isinstance(x, (list, np.ndarray, pd.Series)):
        out = ", ".join(
            (_NL[tablefmt] if i - 1 in splits else "") + _formatter(y)
            for i, y in enumerate(x)
        )
    else:
        raise TypeError
    if not out:
        return ""
    if italic:
        return emphasize(out, tablefmt=tablefmt, strong=False)
    return out


class AbstractComponent(ABC):
    """Abstract class for a component in a mix.  Custom components that don't inherit from
    a concrete class should inherit from this class and implement the methods here.
    """

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        "Name of the component."
        ...

    @property
    def location(self) -> tuple[str, WellPos | None]:
        return ("", None)

    @property
    def plate(self) -> str:
        return ""

    @property
    def well(self) -> WellPos | None:
        return None

    @property
    def _well_list(self) -> list[WellPos]:
        if self.well is not None:
            return [self.well]
        return []

    @property
    @abstractmethod
    def concentration(self) -> Quantity[Decimal]:  # pragma: no cover
        "(Source) concentration of the component as a pint Quantity.  NaN if undefined."
        ...

    @abstractmethod
    def all_components(self) -> pd.DataFrame:  # pragma: no cover
        "A dataframe of all components."
        ...

    @abstractmethod
    def with_reference(self: T, reference: Reference) -> T:  # pragma: no cover
        ...

    def printed_name(self, tablefmt: str | TableFormat) -> str:
        return self.name


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


def _parse_wellpos_optional(v: str | WellPos | None) -> WellPos | None:
    """Parse a string (eg, "C7"), WellPos, or None as potentially a
    well position, returning either a WellPos or None."""
    if isinstance(v, str):
        return WellPos(v)
    elif isinstance(v, WellPos):
        return v
    elif v is None:
        return None
    try:
        if v.isnan():  # type: ignore
            return None
    except:
        pass
    try:
        if np.isnan(v):  # type: ignore
            return None
    except:
        pass
    raise ValueError(f"Can't interpret {v} as well position or None.")


def _none_as_empty_string(v: str | None) -> str:
    return "" if v is None else v


@attrs.define()
class Component(AbstractComponent):
    """A single named component, potentially with a concentration and location.

    Location is stored as a `plate` and `well` property. `plate` is

    """

    name: str
    concentration: Quantity[Decimal] = attrs.field(
        converter=_parse_conc_optional, default=None, on_setattr=attrs.setters.convert
    )
    # FIXME: this is not a great way to do this: should make code not give None
    # Fortuitously, mypy doesn't support this converter, so problems should give type errors.
    plate: str = attrs.field(
        default="",
        kw_only=True,
        converter=_none_as_empty_string,
        on_setattr=attrs.setters.convert,
    )
    well: WellPos | None = attrs.field(
        converter=_parse_wellpos_optional,
        default=None,
        kw_only=True,
        on_setattr=attrs.setters.convert,
    )

    def __eq__(self, other: Any) -> bool:
        if not other.__class__ == Component:
            return False
        if self.name != other.name:
            return False
        if isinstance(self.concentration, Quantity) and isinstance(
            other.concentration, Quantity
        ):
            if math.isnan(self.concentration.m) and math.isnan(other.concentration.m):
                return True
            return self.concentration == other.concentration
        elif hasattr(self, "concentration") and hasattr(other, "concentration"):
            return bool(self.concentration == other.concentration)
        return False

    @property
    def location(self) -> tuple[str, WellPos | None]:
        return (self.plate, self.well)

    def all_components(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "concentration_nM": [self.concentration.to(nM).magnitude],
                "component": [self],
            },
            index=pd.Index([self.name], name="name"),
        )
        return df

    def with_reference(self: Component, reference: Reference) -> Component:
        if reference.df.index.name == "Name":
            ref_by_name = reference.df
        else:
            ref_by_name = reference.df.set_index("Name")
        try:
            ref_comps = ref_by_name.loc[
                [self.name], :
            ]  # using this format to force a dataframe result
        except KeyError:
            return self

        mismatches = []
        matches = []
        for _, ref_comp in ref_comps.iterrows():
            ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], nM)
            if not math.isnan(self.concentration.m) and not (
                ref_conc == self.concentration
            ):
                mismatches.append(("Concentration (nM)", ref_comp))
                continue

            ref_plate = ref_comp["Plate"]
            if self.plate and ref_plate != self.plate:
                mismatches.append(("Plate", ref_comp))
                continue

            ref_well = _parse_wellpos_optional(ref_comp["Well"])
            if self.well and self.well != ref_well:
                mismatches.append(("Well", ref_well))
                continue

            matches.append(ref_comp)

        if len(matches) > 1:
            log.warning(
                "Component %s has more than one location: %s.  Choosing first.",
                self.name,
                [(x["Plate"], x["Well"]) for x in matches],
            )
        elif (len(matches) == 0) and len(mismatches) > 0:
            raise ValueError(
                "Component has only mismatched references: %s", self, mismatches
            )

        match = matches[0]
        ref_conc = ureg.Quantity(match["Concentration (nM)"], nM)
        ref_plate = match["Plate"]
        ref_well = _parse_wellpos_optional(match["Well"])

        return attrs.evolve(
            self, name=self.name, concentration=ref_conc, plate=ref_plate, well=ref_well
        )


@attrs.define()
class Strand(Component):
    """A single named strand, potentially with a concentration, location and sequence."""

    sequence: str | None = None

    def with_reference(self: Strand, reference: Reference) -> Strand:
        if reference.df.index.name == "Name":
            ref_by_name = reference.df
        else:
            ref_by_name = reference.df.set_index("Name")
        try:
            ref_comps = ref_by_name.loc[
                [self.name], :
            ]  # using this format to force a dataframe result
        except KeyError:
            return self

        mismatches = []
        matches = []
        for _, ref_comp in ref_comps.iterrows():
            ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], nM)
            if not math.isnan(self.concentration.m) and not (
                ref_conc == self.concentration
            ):
                mismatches.append(("Concentration (nM)", ref_comp))
                continue

            ref_plate = ref_comp["Plate"]
            if self.plate and ref_plate != self.plate:
                mismatches.append(("Plate", ref_comp))
                continue

            ref_well = _parse_wellpos_optional(ref_comp["Well"])
            if self.well and self.well != ref_well:
                mismatches.append(("Well", ref_well))
                continue

            if isinstance(self.sequence, str) and isinstance(ref_comp["Sequence"], str):
                y = ref_comp["Sequence"]
                self.sequence = self.sequence.replace(" ", "").replace("-", "")
                y = y.replace(" ", "").replace("-", "")
                if self.sequence != y:
                    mismatches.append(("Sequence", ref_comp["Sequence"]))
                    continue

            matches.append(ref_comp)

        del ref_comp  # Ensure we never use this again

        if len(matches) > 1:
            log.warning(
                "Strand %s has more than one location: %s.  Choosing first.",
                self.name,
                [(x["Plate"], x["Well"]) for x in matches],
            )
        elif (len(matches) == 0) and len(mismatches) > 0:
            raise ValueError(
                "Strand has only mismatched references: %s", self, mismatches
            )

        m = matches[0]
        ref_conc = ureg.Quantity(m["Concentration (nM)"], nM)
        ref_plate = m["Plate"]
        ref_well = _parse_wellpos_optional(m["Well"])
        ss, ms = self.sequence, m["Sequence"]
        if (ss is None) and (ms is None):
            seq = None
        elif isinstance(ss, str) and ((ms is None) or (ms == "")):
            seq = ss
        elif isinstance(ms, str) and ((ss is None) or isinstance(ss, str)):
            seq = ms
        else:
            raise RuntimeError("should be unreachable")

        return attrs.evolve(
            self,
            name=self.name,
            concentration=ref_conc,
            plate=ref_plate,
            well=ref_well,
            sequence=seq,
        )


class AbstractAction(ABC):
    """
    Abstract class defining an action in a mix recipe.
    """

    @property
    def name(self) -> str:  # pragma: no cover
        ...

    @abstractmethod
    def tx_volume(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> Quantity[Decimal]:  # pragma: no cover
        """The total volume transferred by the action to the sample.  May depend on the total mix volume.

        Parameters
        ----------

        mix_vol
            The mix volume.  Does not accept strings.
        """
        ...

    @abstractmethod
    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
    ) -> Sequence[MixLine]:  # pragma: no cover
        ...

    @abstractmethod
    def all_components(
        self, mix_vol: Quantity[Decimal]
    ) -> pd.DataFrame:  # pragma: no cover
        """A dataframe containing all base components added by the action.

        Parameters
        ----------

        mix_vol
            The mix volume.  Does not accept strings.
        """
        ...

    @abstractmethod
    def with_reference(self: T, reference: Reference) -> T:  # pragma: no cover
        """Returns a copy of the action updated from a reference dataframe."""
        ...

    def dest_concentration(self, mix_vol: pint.Quantity) -> pint.Quantity:
        """The destination concentration added to the mix by the action.

        Raises
        ------

        ValueError
            There is no good definition for a single destination concentration
            (the action may add multiple components).
        """
        raise ValueError("Single destination concentration not defined.")

    def dest_concentrations(
        self, mix_vol: pint.Quantity
    ) -> Sequence[Quantity[Decimal]]:
        raise ValueError

    @property
    @abstractmethod
    def components(self) -> list[AbstractComponent]:
        ...

    @abstractmethod
    def each_volumes(self, total_volume: Quantity[Decimal]) -> list[Quantity[Decimal]]:
        ...


def findloc(locations: pd.DataFrame | None, name: str) -> str | None:
    loc = findloc_tuples(locations, name)

    if loc is None:
        return None

    _, plate, well = loc
    if well:
        return f"{plate}: {well}"
    else:
        return f"{plate}"


def findloc_tuples(
    locations: pd.DataFrame | None, name: str
) -> tuple[str, str, WellPos | str] | None:
    if locations is None:
        return None
    locs = locations.loc[locations["Name"] == name]

    if len(locs) > 1:
        log.warning(f"Found multiple locations for {name}, using first.")
    elif len(locs) == 0:
        return None

    loc = locs.iloc[0]

    try:
        well = WellPos(loc["Well"])
    except Exception:
        well = loc["Well"]

    return (loc["Name"], loc["Plate"], well)


def mixgaps(wl: Iterable[WellPos], by: Literal["row", "col"]) -> int:
    score = 0

    wli = iter(wl)

    getnextpos = WellPos.next_bycol if by == "col" else WellPos.next_byrow
    prevpos = next(wli)

    for pos in wli:
        if not (getnextpos(prevpos) == pos):
            score += 1
        prevpos = pos
    return score


def _empty_components() -> pd.DataFrame:
    cps = pd.DataFrame(
        index=pd.Index([], name="name"),
    )
    cps["concentration_nM"] = pd.Series([], dtype=object)
    cps["component"] = pd.Series([], dtype=object)
    return cps


@attrs.define()
class FixedVolume(AbstractAction):
    """An action adding one or multiple components, with a set destination volume (potentially keeping equal concentration).

    FixedVolume adds a selection of components, with a specified transfer volume.  Depending on the setting of
    `equal_conc`, it may require that the destination concentrations all be equal, may not care, and just transfer
    a fixed volume of each strand, or may treat the fixed transfer volume as the volume as the minimum or maximum
    volume to transfer, adjusting volumes of each strand to make this work and have them at equal destination
    concentrations.

    Parameters
    ----------

    components
        A list of :ref:`Components`.

    fixed_volume
        A fixed volume for the action.  Input can be a string (eg, "5 µL") or a pint Quantity.  The interpretation
        of this depends on equal_conc.

    set_name
        The name of the mix.  If not set, name is based on components.

    compact_display
        If True (default), the action tries to display compactly in mix recipes.  If False, it displays
        each component as a separate line.

    equal_conc
        If `False`, the action transfers the same `fixed_volume` volume of each component, regardless of
        concentration.  If `True`, the action still transfers the same volume of each component, but will
        raise a `ValueError` if this will not result in every component having the same destination concentration
        (ie, if they have different source concentrations).  If `"min_volume"`, the action will transfer *at least*
        `fixed_volume` of each component, but will transfer more for components with lower source concentration,
        so that the destination concentrations are all equal (but not fixed to a specific value).  If `"max_volume"`,
        the action instead transfers *at most* `fixed_volume` of each component, tranferring less for higher
        source concentration components.  If ('max_fill', buffer_name), the fixed volume is the maximum, while for
        every component that is added at a lower volume, a corresponding volume of buffer is added to bring the total
        volume of the two up to the fixed volume.

    Examples
    --------

    >>> from alhambra.mixes import *
    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ... ]

    >>> print(Mix([FixedVolume(components, "5 uL")], name="example"))
    Table: Mix: example, Conc: 66.67 nM, Total Vol: 15.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   |   # | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|----:|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 66.67 nM  |   3 | 5.00 µl     | 15.00 µl     |       |        |

    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ...     Component("c4", "100 nM")
    ... ]

    >>> print(Mix([FixedVolume(components, "5 uL", equal_conc="min_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 25.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 5.00 µl     | 15.00 µl     |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 10.00 µl    | 10.00 µl     |       |        |

    >>> print(Mix([FixedVolume(components, "5 uL", equal_conc="max_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 12.50 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 2.50 µl     | 7.50 µl      |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 5.00 µl     | 5.00 µl      |       |        |

    """

    components: list[AbstractComponent] = attrs.field(
        converter=_maybesequence, on_setattr=attrs.setters.convert
    )
    fixed_volume: Quantity[Decimal] = attrs.field(
        converter=_parse_vol_required, on_setattr=attrs.setters.convert
    )
    set_name: str | None = None
    compact_display: bool = True
    equal_conc: bool | Literal["max_volume", "min_volume"] | tuple[
        Literal["max_fill"], str
    ] = True

    def with_reference(self, reference: Reference) -> FixedVolume:
        return FixedVolume(
            [c.with_reference(reference) for c in self.components],  # type: ignore
            self.fixed_volume,
            self.set_name,
            self.compact_display,
            self.equal_conc,
        )

    @property
    def source_concentrations(self) -> Sequence[Quantity[Decimal]]:
        concs = [c.concentration.to(nM) for c in self.components]
        if any(x != concs[0] for x in concs) and not self.equal_conc:
            raise ValueError("Not all components have equal concentration.")
        return concs

    def all_components(self, mix_vol: Quantity[Decimal]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp, dc, sc in zip(
            self.components,
            self.dest_concentrations(mix_vol),
            self.source_concentrations,
        ):
            comps = comp.all_components()
            comps.concentration_nM *= _ratio(dc, sc)

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=Decimal("0.0"))
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentrations(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        return [
            x * y
            for x, y in zip(
                self.source_concentrations, _ratio(self.each_volumes(mix_vol), mix_vol)
            )
        ]

    def each_volumes(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        # match self.equal_conc:
        if self.equal_conc == "min_volume":
            sc = self.source_concentrations
            scmax = max(sc)
            return [self.fixed_volume * x for x in _ratio(scmax, sc)]
        elif (self.equal_conc == "max_volume") | (
            isinstance(self.equal_conc, Sequence) and self.equal_conc[0] == "max_fill"
        ):
            sc = self.source_concentrations
            scmin = min(sc)
            return [self.fixed_volume * x for x in _ratio(scmin, sc)]
        elif self.equal_conc is True:
            sc = self.source_concentrations
            if any(x != sc[0] for x in sc):
                raise ValueError("Concentrations")
            return [self.fixed_volume.to(uL)] * len(self.components)
        elif self.equal_conc is False:
            return [self.fixed_volume.to(uL)] * len(self.components)

        raise ValueError(f"equal_conc={repr(self.equal_conc)} not understood")

    def tx_volume(self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)) -> Quantity[Decimal]:
        if isinstance(self.equal_conc, Sequence) and (self.equal_conc[0] == "max_fill"):
            return self.fixed_volume * len(self.components)
        return sum(self.each_volumes(mix_vol), ureg("0.0 uL"))

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
        locations: pd.DataFrame | None = None,
    ) -> list[MixLine]:
        if not self.compact_display:
            ml = [
                MixLine(
                    [comp.printed_name(tablefmt=tablefmt)],
                    comp.concentration,
                    dc,
                    ev,
                    plate=comp.plate,
                    wells=comp._well_list,
                )
                for dc, ev, comp in zip(
                    self.dest_concentrations(mix_vol),
                    self.each_volumes(mix_vol),
                    self.components,
                )
            ]
        else:
            ml = list(self._compactstrs(tablefmt=tablefmt, mix_vol=mix_vol))

        if isinstance(self.equal_conc, Sequence) and (self.equal_conc[0] == "max_fill"):
            fv = self.fixed_volume * len(self.components) - sum(self.each_volumes())
            if not fv == Q_(Decimal("0.0"), uL):
                ml.append(MixLine([self.equal_conc[1]], None, None, fv))

        return ml

    @property
    def number(self) -> int:
        return len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return ", ".join(c.name for c in self.components)
        else:
            return self.set_name

    def _compactstrs(
        self, tablefmt: str | TableFormat, mix_vol: pint.Quantity
    ) -> list[MixLine]:
        # locs = [(c.name,) + c.location for c in self.components]
        # names = [c.name for c in self.components]

        # if any(x is None for x in locs):
        #     raise ValueError(
        #         [name for name, loc in zip(names, locs) if loc is None]
        #     )

        locdf = pd.DataFrame(
            {
                "names": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_concs": self.source_concentrations,
                "dest_concs": self.dest_concentrations(mix_vol),
                "ea_vols": self.each_volumes(mix_vol),
                "plate": [c.plate for c in self.components],
                "well": [c.well for c in self.components],
            }
        )

        locdf.fillna({"plate": ""}, inplace=True)

        locdf.sort_values(
            by=["plate", "ea_vols", "well"], ascending=[True, False, True]
        )

        names: list[list[str]] = []
        source_concs: list[Quantity[Decimal]] = []
        dest_concs: list[Quantity[Decimal]] = []
        numbers: list[int] = []
        ea_vols: list[Quantity[Decimal]] = []
        tot_vols: list[Quantity[Decimal]] = []
        plates: list[str] = []
        wells_list: list[list[WellPos]] = []

        for plate, plate_comps in locdf.groupby("plate"):  # type: str, pd.DataFrame
            for vol, plate_vol_comps in plate_comps.groupby(
                "ea_vols"
            ):  # type: Quantity[Decimal], pd.DataFrame
                if pd.isna(plate_vol_comps["well"].iloc[0]):
                    if not pd.isna(plate_vol_comps["well"]).all():
                        raise ValueError
                    names.append(list(plate_vol_comps["names"]))
                    ea_vols.append((vol))
                    tot_vols.append((vol * len(plate_vol_comps)))
                    numbers.append((len(plate_vol_comps)))
                    source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                    dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                    plates.append(plate)
                    wells_list.append([])
                    continue
                byrow = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_byrow),
                    by="row",
                )
                bycol = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_bycol),
                    by="col",
                )

                sortkey = WellPos.key_bycol if bycol <= byrow else WellPos.key_byrow

                plate_vol_comps["sortkey"] = [
                    sortkey(c) for c in plate_vol_comps["well"]
                ]

                plate_vol_comps.sort_values(by="sortkey", inplace=True)

                names.append(list(plate_vol_comps["names"]))
                ea_vols.append((vol))
                numbers.append((len(plate_vol_comps)))
                tot_vols.append((vol * len(plate_vol_comps)))
                source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                plates.append(plate)
                wells_list.append(list(plate_vol_comps["well"]))

        return [
            MixLine(
                name,
                source_conc=source_conc,
                dest_conc=dest_conc,
                number=number,
                each_tx_vol=each_tx_vol,
                total_tx_vol=total_tx_vol,
                plate=plate,
                wells=wells,
            )
            for name, source_conc, dest_conc, number, each_tx_vol, total_tx_vol, plate, wells in zip(
                names,
                source_concs,
                dest_concs,
                numbers,
                ea_vols,
                tot_vols,
                plates,
                wells_list,
            )
        ]


@attrs.define()
class FixedConcentration(AbstractAction):
    """An action adding one or multiple components, with a set destination concentration per component (adjusting volumes).

    FixedConcentration adds a selection of components, with a specified destination concentration.

    Parameters
    ----------

    components
        A list of :ref:`Components`.

    fixed_concentration
        A fixed concentration for the action.  Input can be a string (eg, "50 nM") or a pint Quantity.

    set_name
        The name of the mix.  If not set, name is based on components.

    compact_display
        If True (default), the action tries to display compactly in mix recipes.  If False, it displays
        each component as a separate line.

    min_volume
        Specifies a minimum volume that must be transferred per component.  Currently, this is for
        validation only: it will cause a VolumeError to be raised if a volume is too low.

    Raises
    ------

    VolumeError
        One of the volumes to transfer is less than the specified min_volume.

    Examples
    --------

    >>> from alhambra.mixes import *
    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ...     Component("c4", "100 nM")
    ... ]

    >>> print(Mix([FixedConcentration(components, "20 nM")], name="example", fixed_total_volume="25 uL"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 25.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 20.00 nM  | 3   | 2.50 µl     | 7.50 µl      |       |        |
    | c4         | 100.00 nM | 20.00 nM  |     | 5.00 µl     | 5.00 µl      |       |        |
    | Buffer     |           |           |     |             | 12.50 µl     |       |        |
    | *Total:*   |           | 40.00 nM  |     |             | 25.00 µl     |       |        |
    """

    components: list[AbstractComponent] = attrs.field(
        converter=_maybesequence, on_setattr=attrs.setters.convert
    )
    fixed_concentration: Quantity[Decimal] = attrs.field(
        converter=_parse_conc_required, on_setattr=attrs.setters.convert
    )
    set_name: str | None = None
    compact_display: bool = True
    min_volume: Quantity[Decimal] = attrs.field(
        converter=_parse_vol_optional,
        default=Q_(DNAN, uL),
        on_setattr=attrs.setters.convert,
    )

    def with_reference(self, reference: Reference) -> FixedConcentration:
        return attrs.evolve(
            self, components=[c.with_reference(reference) for c in self.components]
        )

    @property
    def source_concentrations(self) -> list[Quantity[Decimal]]:
        concs = [c.concentration for c in self.components]
        return concs

    def all_components(self, mix_vol: Quantity[Decimal]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp, dc, sc in zip(
            self.components,
            self.dest_concentrations(mix_vol),
            self.source_concentrations,
        ):
            comps = comp.all_components()
            comps.concentration_nM *= _ratio(dc, sc)

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=Decimal("0.0"))
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentrations(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        return [self.fixed_concentration] * len(self.components)

    def each_volumes(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        ea_vols = [
            mix_vol * r
            for r in _ratio(self.fixed_concentration, self.source_concentrations)
        ]
        if not math.isnan(self.min_volume.m):
            below_min = []
            for comp, vol in zip(self.components, ea_vols):
                if vol < self.min_volume:
                    below_min.append((comp.name, vol))
            if below_min:
                raise VolumeError(
                    "Volume of some components is below minimum: "
                    + ", ".join(f"{n} at {v}" for n, v in below_min)
                    + ".",
                    below_min,
                )
        return ea_vols

    def tx_volume(self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)) -> Quantity[Decimal]:
        return sum(self.each_volumes(mix_vol), Q_(Decimal("0"), "uL"))

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
        locations: pd.DataFrame | None = None,
    ) -> list[MixLine]:
        if not self.compact_display:
            ml = [
                MixLine(
                    [comp.printed_name(tablefmt=tablefmt)],
                    comp.concentration,
                    dc,
                    ev,
                    plate=comp.plate,
                    wells=comp._well_list,
                )
                for dc, ev, comp in zip(
                    self.dest_concentrations(mix_vol),
                    self.each_volumes(mix_vol),
                    self.components,
                )
            ]
        else:
            ml = list(self._compactstrs(tablefmt=tablefmt, mix_vol=mix_vol))

        return ml

    @property
    def number(self) -> int:
        return len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return ", ".join(c.name for c in self.components)
        else:
            return self.set_name

    def _compactstrs(
        self, tablefmt: str | TableFormat, mix_vol: pint.Quantity
    ) -> Sequence[MixLine]:
        # locs = [(c.name,) + c.location for c in self.components]
        # names = [c.name for c in self.components]

        # if any(x is None for x in locs):
        #     raise ValueError(
        #         [name for name, loc in zip(names, locs) if loc is None]
        #     )

        locdf = pd.DataFrame(
            {
                "names": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_concs": self.source_concentrations,
                "dest_concs": self.dest_concentrations(mix_vol),
                "ea_vols": self.each_volumes(mix_vol),
                "plate": [c.plate for c in self.components],
                "well": [c.well for c in self.components],
            }
        )

        locdf.fillna({"plate": ""}, inplace=True)

        locdf.sort_values(
            by=["plate", "ea_vols", "well"], ascending=[True, False, True]
        )

        names: list[list[str]] = []
        source_concs: list[Quantity[Decimal]] = []
        dest_concs: list[Quantity[Decimal]] = []
        numbers: list[int] = []
        ea_vols: list[Quantity[Decimal]] = []
        tot_vols: list[Quantity[Decimal]] = []
        plates: list[str] = []
        wells_list: list[list[WellPos]] = []

        for plate, plate_comps in locdf.groupby("plate"):  # type: str, pd.DataFrame
            for vol, plate_vol_comps in plate_comps.groupby(
                "ea_vols"
            ):  # type: Quantity[Decimal], pd.DataFrame
                if pd.isna(plate_vol_comps["well"].iloc[0]):
                    if not pd.isna(plate_vol_comps["well"]).all():
                        raise ValueError
                    names.append(list(plate_vol_comps["names"]))
                    ea_vols.append((vol))
                    tot_vols.append((vol * len(plate_vol_comps)))
                    numbers.append((len(plate_vol_comps)))
                    source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                    dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                    plates.append(plate)
                    wells_list.append([])
                    continue

                byrow = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_byrow),
                    by="row",
                )
                bycol = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_bycol),
                    by="col",
                )

                sortkey = WellPos.key_bycol if bycol <= byrow else WellPos.key_byrow

                plate_vol_comps["sortkey"] = [
                    sortkey(c) for c in plate_vol_comps["well"]
                ]

                plate_vol_comps.sort_values(by="sortkey", inplace=True)

                names.append(list(plate_vol_comps["names"]))
                ea_vols.append((vol))
                numbers.append((len(plate_vol_comps)))
                tot_vols.append((vol * len(plate_vol_comps)))
                source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                plates.append(plate)
                wells_list.append(list(plate_vol_comps["well"]))

        return [
            MixLine(
                name,
                source_conc=source_conc,
                dest_conc=dest_conc,
                number=number,
                each_tx_vol=each_tx_vol,
                total_tx_vol=total_tx_vol,
                plate=p,
                wells=wells,
            )
            for name, source_conc, dest_conc, number, each_tx_vol, total_tx_vol, p, wells in zip(
                names,
                source_concs,
                dest_concs,
                numbers,
                ea_vols,
                tot_vols,
                plates,
                wells_list,
            )
        ]


MultiFixedConcentration = FixedConcentration
MultiFixedVolume = FixedVolume

_96WELL_PLATE_ROWS: list[str] = ["A", "B", "C", "D", "E", "F", "G", "H"]
_96WELL_PLATE_COLS: list[int] = list(range(1, 13))

_384WELL_PLATE_ROWS: list[str] = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]
_384WELL_PLATE_COLS: list[int] = list(range(1, 25))


@enum.unique
class PlateType(enum.Enum):
    """Represents two different types of plates in which DNA sequences can be ordered."""

    wells96 = 96
    """96-well plate."""

    wells384 = 384
    """384-well plate."""

    def rows(self) -> list[str]:
        """
        :return:
            list of all rows in this plate (as letters 'A', 'B', ...)
        """
        return _96WELL_PLATE_ROWS if self is PlateType.wells96 else _384WELL_PLATE_ROWS

    def cols(self) -> list[int]:
        """
        :return:
            list of all columns in this plate (as integers 1, 2, ...)
        """
        return _96WELL_PLATE_COLS if self is PlateType.wells96 else _384WELL_PLATE_COLS

    def num_wells_per_plate(self) -> int:
        """
        :return:
            number of wells in this plate type
        """
        if self is PlateType.wells96:
            return 96
        elif self is PlateType.wells384:
            return 384
        else:
            raise AssertionError("unreachable")

    def min_wells_per_plate(self) -> int:
        """
        :return:
            minimum number of wells in this plate type to avoid extra charge by IDT
        """
        if self is PlateType.wells96:
            return 24
        elif self is PlateType.wells384:
            return 96
        else:
            raise AssertionError("unreachable")


@attrs.define()
class Mix(AbstractComponent):
    """Class denoting a Mix, a collection of source components mixed to
    some volume or concentration.
    """

    actions: Sequence[AbstractAction] = attrs.field(
        converter=_maybesequence, on_setattr=attrs.setters.convert
    )
    name: str
    test_tube_name: str | None = attrs.field(kw_only=True, default=None)
    "A short name, eg, for labelling a test tube."
    fixed_total_volume: Quantity[Decimal] = attrs.field(
        converter=_parse_vol_optional,
        default=Q_(DNAN, uL),
        kw_only=True,
        on_setattr=attrs.setters.convert,
    )
    fixed_concentration: str | Quantity[Decimal] | None = attrs.field(
        default=None, kw_only=True, on_setattr=attrs.setters.convert
    )
    buffer_name: str = "Buffer"
    reference: Reference | None = None
    min_volume: Quantity[Decimal] = attrs.field(
        converter=_parse_vol_optional,
        default=Q_(Decimal(0.5), uL),
        kw_only=True,
        on_setattr=attrs.setters.convert,
    )

    def __attrs_post_init__(self) -> None:
        if self.reference is not None:
            self.actions = [
                action.with_reference(self.reference) for action in self.actions
            ]
        if self.actions is None:
            raise ValueError(
                f"Mix.actions must contain at least one action, but it was not specified"
            )
        elif len(self.actions) == 0:
            raise ValueError(
                f"Mix.actions must contain at least one action, but it is empty"
            )

    def printed_name(self, tablefmt: str | TableFormat) -> str:
        return self.name + (
            ""
            if self.test_tube_name is None
            else f" ({emphasize(self.test_tube_name, tablefmt=tablefmt, strong=False)})"
        )

    @property
    def concentration(self) -> Quantity[Decimal]:
        """
        Effective concentration of the mix.  Calculated in order:

        1. If the mix has a fixed concentration, then that concentration.
        2. If `fixed_concentration` is a string, then the final concentration of
           the component with that name.
        3. If `fixed_concentration` is none, then the final concentration of the first
           mix component.
        """
        if isinstance(self.fixed_concentration, pint.Quantity):
            return self.fixed_concentration
        elif isinstance(self.fixed_concentration, str):
            ac = self.all_components()
            return ureg.Quantity(
                Decimal(ac.loc[self.fixed_concentration, "concentration_nM"]), ureg.nM
            )
        elif self.fixed_concentration is None:
            return self.actions[0].dest_concentrations(self.total_volume)[0]
        else:
            raise NotImplemented

    @property
    def total_volume(self) -> Quantity[Decimal]:
        """
        Total volume of the mix.  If the mix has a fixed total volume, then that,
        otherwise, the sum of the transfer volumes of each component.
        """
        if self.fixed_total_volume is not None and not (
            math.isnan(self.fixed_total_volume.m)
        ):
            return self.fixed_total_volume
        else:
            return sum(
                [
                    c.tx_volume(self.fixed_total_volume or Q_(DNAN, ureg.uL))
                    for c in self.actions
                ],
                Q_(Decimal(0.0), ureg.uL),
            )

    @property
    def buffer_volume(self) -> Quantity[Decimal]:
        """
        The volume of buffer to be added to the mix, in addition to the components.
        """
        mvol = sum(c.tx_volume(self.total_volume) for c in self.actions)
        return self.total_volume - mvol

    def table(
        self,
        tablefmt: TableFormat | str = "pipe",
        raise_failed_validation: bool = False,
        buffer_name: str = "Buffer",
        stralign="default",
        missingval="",
        showindex="default",
        disable_numparse=False,
        colalign=None,
    ) -> str:
        """Generate a table describing the mix.

        Parameters
        ----------

        tablefmt
            The output format for the table.

        validate
            Ensure volumes make sense.

        buffer_name
            Name of the buffer to use. (Default="Buffer")
        """
        mixlines = list(self.mixlines(buffer_name=buffer_name, tablefmt=tablefmt))

        validation_errors = self.validate(mixlines=mixlines)

        # If we're validating and generating an error, we need the tablefmt to be
        # a text one, so we'll call ourselves again:
        if validation_errors and raise_failed_validation:
            raise VolumeError(self.table("pipe"))

        mixlines.append(
            MixLine(
                ["Total:"],
                None,
                self.concentration,
                self.total_volume,
                fake=True,
                number=sum(m.number for m in mixlines),
            )
        )

        include_numbers = any(ml.number != 1 for ml in mixlines)

        if validation_errors:
            errline = _format_errors(validation_errors, tablefmt) + "\n"
        else:
            errline = ""

        return errline + tabulate(
            [ml.toline(include_numbers, tablefmt=tablefmt) for ml in mixlines],
            MIXHEAD_EA if include_numbers else MIXHEAD_NO_EA,
            tablefmt=tablefmt,
            stralign=stralign,
            missingval=missingval,
            showindex=showindex,
            disable_numparse=disable_numparse,
            colalign=colalign,
        )

    def mixlines(
        self, tablefmt: str | TableFormat, buffer_name: str = "Buffer"
    ) -> Sequence[MixLine]:
        mixlines: list[MixLine] = []

        for action in self.actions:
            mixlines += action._mixlines(tablefmt=tablefmt, mix_vol=self.total_volume)

        if self.has_fixed_total_volume():
            mixlines.append(MixLine([buffer_name], None, None, self.buffer_volume))
        return mixlines

    def has_fixed_concentration_action(self) -> bool:
        return any(isinstance(action, FixedConcentration) for action in self.actions)

    def has_fixed_total_volume(self) -> bool:
        return not math.isnan(self.fixed_total_volume.m)

    def validate(
        self,
        tablefmt: str | TableFormat | None = None,
        mixlines: Sequence[MixLine] | None = None,
        raise_errors: bool = False,
    ) -> list[VolumeError]:
        if mixlines is None:
            if tablefmt is None:
                raise ValueError("If mixlines is None, tablefmt must be specified.")
            mixlines = self.mixlines(tablefmt=tablefmt)
        ntx = [
            (m.names, m.total_tx_vol) for m in mixlines if m.total_tx_vol is not None
        ]

        error_list: list[VolumeError] = []

        # special case check for FixedConcentration action(s) used
        # without corresponding Mix.fixed_total_volume
        if not self.has_fixed_total_volume() and self.has_fixed_concentration_action():
            error_list.append(
                VolumeError(
                    "If a FixedConcentration action is used, "
                    "then Mix.fixed_total_volume must be specified."
                )
            )

        nan_vols = [", ".join(n) for n, x in ntx if math.isnan(x.m)]
        if nan_vols:
            error_list.append(
                VolumeError(
                    "Some volumes aren't defined (mix probably isn't fully specified): "
                    + "; ".join(x or "" for x in nan_vols)
                    + "."
                )
            )

        tot_vol = self.total_volume
        high_vols = [(n, x) for n, x in ntx if x > tot_vol]
        if high_vols:
            error_list.append(
                VolumeError(
                    "Some items have higher transfer volume than total mix volume of "
                    f"{tot_vol} "
                    "(target concentration probably too high for source): "
                    + "; ".join(f"{', '.join(n)} at {x}" for n, x in high_vols)
                    + "."
                )
            )

        # ensure we pipette at least self.min_volume from each source

        for mixline in mixlines:
            if (
                not math.isnan(mixline.each_tx_vol.m)
                and mixline.each_tx_vol != ZERO_VOL
                and mixline.each_tx_vol < self.min_volume
            ):
                if mixline.names == [self.buffer_name]:
                    # This is the line for the buffer
                    # TODO: tell them what is the maximum source concentration they can have
                    msg = (
                        f'Negative buffer volume of mix "{self.name}"; '
                        f"this is typically caused by requesting too large a target concentration in a "
                        f"FixedConcentration action,"
                        f"since the source concentrations are too low. "
                        f"Try lowering the target concentration."
                    )
                else:
                    # FIXME: why do these need :f?
                    msg = (
                        f"Some items have lower transfer volume than {self.min_volume}\n"
                        f'This is in creating mix "{self.name}", '
                        f"attempting to pipette {mixline.each_tx_vol} of these components:\n"
                        f"{mixline.names}"
                    )
                error_list.append(VolumeError(msg))

        # We'll check the last tx_vol first, because it is usually buffer.
        if ntx[-1][1] < ZERO_VOL:
            error_list.append(
                VolumeError(
                    f"Last mix component ({ntx[-1][0]}) has volume {ntx[-1][1]} < 0 µL. "
                    "Component target concentrations probably too high."
                )
            )

        neg_vols = [(n, x) for n, x in ntx if x < ZERO_VOL]
        if neg_vols:
            error_list.append(
                VolumeError(
                    "Some volumes are negative: "
                    + "; ".join(f"{', '.join(n)} at {x}" for n, x in neg_vols)
                    + "."
                )
            )

        # check for sufficient volume in intermediate mixes
        # XXX: this assumes 1-1 correspondence between mixlines and actions (true in current implementation)
        for action in self.actions:
            for component, volume in zip(
                action.components, action.each_volumes(self.total_volume)
            ):
                if isinstance(component, Mix):
                    if component.fixed_total_volume < volume:
                        error_list.append(
                            VolumeError(
                                f'intermediate Mix "{component.name}" needs {volume} to create '
                                f'Mix "{self.name}", but Mix "{component.name}" contains only '
                                f"{component.fixed_total_volume}."
                            )
                        )
            # for each_vol, component in zip(mixline.each_tx_vol, action.all_components()):

        return error_list

    def all_components(self) -> pd.DataFrame:
        """
        Return a Series of all component names, and their concentrations (as pint nM).
        """
        cps = _empty_components()

        for action in self.actions:
            mcomp = action.all_components(self.total_volume)
            cps, _ = cps.align(mcomp)
            cps.loc[:, "concentration_nM"].fillna(Decimal("0.0"), inplace=True)
            cps.loc[mcomp.index, "concentration_nM"] += mcomp.concentration_nM
            cps.loc[mcomp.index, "component"] = mcomp.component
        return cps

    def _repr_markdown_(self) -> str:
        return f"Table: {self.infoline()}\n" + self.table(tablefmt="pipe")

    def _repr_html_(self) -> str:
        return f"<p>Table: {self.infoline()}</p>\n" + self.table(tablefmt="unsafehtml")

    def infoline(self) -> str:
        elems = [
            f"Mix: {self.name}",
            f"Conc: {self.concentration:,.2f~#P}",
            f"Total Vol: {self.total_volume:,.2f~#P}",
            # f"Component Count: {len(self.all_components())}",
        ]
        if self.test_tube_name:
            elems.append(f"Test tube name: {self.test_tube_name}")
        return ", ".join(elems)

    def __repr__(self) -> str:
        return f'Mix("{self.name}", {len(self.actions)} actions)'

    def __str__(self) -> str:
        return f"Table: {self.infoline()}\n\n" + self.table()

    def with_reference(self: Mix, reference: Reference) -> Mix:
        new = attrs.evolve(
            self, actions=[action.with_reference(reference) for action in self.actions]
        )
        new.reference = reference
        return new

    @property
    def location(self) -> tuple[str, WellPos | None]:
        return ("", None)

    def vol_to_tube_names(
        self,
        tablefmt: str | TableFormat = "pipe",
        validate: bool = True,
    ) -> dict[Quantity[Decimal], list[str]]:
        """
        :return:
             dict mapping a volume `vol` to a list of names of strands in this mix that should be pipetted
             with volume `vol`
        """
        mixlines = list(self.mixlines(tablefmt=tablefmt))

        if validate:
            try:
                self.validate(tablefmt=tablefmt, mixlines=mixlines)
            except ValueError as e:
                e.args = e.args + (
                    self.vol_to_tube_names(tablefmt=tablefmt, validate=False),
                )
                raise e

        result: dict[Quantity[Decimal], list[str]] = {}
        for mixline in mixlines:
            if len(mixline.names) == 0 or (
                len(mixline.names) == 1 and mixline.names[0].lower() == "buffer"
            ):
                continue
            if mixline.plate.lower() != "tube":
                continue
            assert mixline.each_tx_vol not in result
            result[mixline.each_tx_vol] = mixline.names

        return result

    def _tube_map_from_mixline(self, mixline: MixLine) -> str:
        joined_names = "\n".join(mixline.names)
        return f"## tubes, {mixline.each_tx_vol} each\n{joined_names}"

    def tubes_markdown(self, tablefmt: str | TableFormat = "pipe") -> str:
        """
        :param tablefmt:
            table format (see :meth:`PlateMap.to_table` for description)
        :return:
            a Markdown (or other format according to `tablefmt`)
            string indicating which strands in test tubes to pipette, grouped by the volume
            of each
        """
        entries = []
        for vol, names in self.vol_to_tube_names(tablefmt=tablefmt).items():
            joined_names = "\n".join(names)
            entry = f"## tubes, {vol} each\n{joined_names}"
            entries.append(entry)
        return "\n".join(entries)

    def display_instructions(
        self,
        plate_type: PlateType = PlateType.wells96,
        raise_failed_validation: bool = False,
        combine_plate_actions: bool = True,
        well_marker: None | str | Callable[[str], str] = None,
        title_level: Literal[1, 2, 3, 4, 5, 6] = 3,
        warn_unsupported_title_format: bool = True,
        buffer_name: str = "Buffer",
        tablefmt: str | TableFormat = "unsafehtml",
        include_plate_maps: bool = True,
    ) -> None:
        """
        Displays in a Jupyter notebook the result of calling :meth:`Mix.instructions()`.

        :param plate_type:
            96-well or 384-well plate; default is 96-well.
        :param raise_failed_validation:
            If validation fails (volumes don't make sense), raise an exception.
        :param combine_plate_actions:
            If True, then if multiple actions in the Mix take the same volume from the same plate,
            they will be combined into a single :class:`PlateMap`.
        :param well_marker:
            By default the strand's name is put in the relevant plate entry. If `well_marker` is specified
            and is a string, then that string is put into every well with a strand in the plate map instead.
            This is useful for printing plate maps that just put,
            for instance, an `'X'` in the well to pipette (e.g., specify ``well_marker='X'``),
            e.g., for experimental mixes that use only some strands in the plate.
            To enable the string to depend on the well position
            (instead of being the same string in every well), `well_marker` can also be a function
            that takes as input a string representing the well (such as ``"B3"`` or ``"E11"``),
            and outputs a string. For example, giving the identity function
            ``mix.to_table(well_marker=lambda x: x)`` puts the well address itself in the well.
        :param title_level:
            The "title" is the first line of the returned string, which contains the plate's name
            and volume to pipette. The `title_level` controls the size, with 1 being the largest size,
            (header level 1, e.g., # title in Markdown or <h1>title</h1> in HTML).
        :param warn_unsupported_title_format:
            If True, prints a warning if `tablefmt` is a currently unsupported option for the title.
            The currently supported formats for the title are 'github', 'html', 'unsafehtml', 'rst',
            'latex', 'latex_raw', 'latex_booktabs', "latex_longtable". If `tablefmt` is another valid
            option, then the title will be the Markdown format, i.e., same as for `tablefmt` = 'github'.
        :param tablefmt:
            By default set to `'github'` to create a Markdown table. For other options see
            https://github.com/astanin/python-tabulate#readme
        :param include_plate_maps:
            If True, include plate maps as part of displayed instructions, otherwise only include the
            more compact mixing table (which is always displayed regardless of this parameter).
        :return:
            pipetting instructions in the form of strings combining results of :meth:`Mix.table` and
            :meth:`Mix.plate_maps`
        """
        from IPython.display import display, HTML

        ins_str = self.instructions(
            plate_type=plate_type,
            raise_failed_validation=raise_failed_validation,
            combine_plate_actions=combine_plate_actions,
            well_marker=well_marker,
            title_level=title_level,
            warn_unsupported_title_format=warn_unsupported_title_format,
            buffer_name=buffer_name,
            tablefmt=tablefmt,
            include_plate_maps=include_plate_maps,
        )
        display(HTML(ins_str))

    def instructions(
        self,
        plate_type: PlateType = PlateType.wells96,
        raise_failed_validation: bool = False,
        combine_plate_actions: bool = True,
        well_marker: None | str | Callable[[str], str] = None,
        title_level: Literal[1, 2, 3, 4, 5, 6] = 3,
        warn_unsupported_title_format: bool = True,
        buffer_name: str = "Buffer",
        tablefmt: str | TableFormat = "pipe",
        include_plate_maps: bool = True,
    ) -> str:
        """
        Returns string combiniing the string results of calling :meth:`Mix.table` and
        :meth:`Mix.plate_maps` (then calling :meth:`PlateMap.to_table` on each :class:`PlateMap`).

        :param plate_type:
            96-well or 384-well plate; default is 96-well.
        :param raise_failed_validation:
            If validation fails (volumes don't make sense), raise an exception.
        :param combine_plate_actions:
            If True, then if multiple actions in the Mix take the same volume from the same plate,
            they will be combined into a single :class:`PlateMap`.
        :param well_marker:
            By default the strand's name is put in the relevant plate entry. If `well_marker` is specified
            and is a string, then that string is put into every well with a strand in the plate map instead.
            This is useful for printing plate maps that just put,
            for instance, an `'X'` in the well to pipette (e.g., specify ``well_marker='X'``),
            e.g., for experimental mixes that use only some strands in the plate.
            To enable the string to depend on the well position
            (instead of being the same string in every well), `well_marker` can also be a function
            that takes as input a string representing the well (such as ``"B3"`` or ``"E11"``),
            and outputs a string. For example, giving the identity function
            ``mix.to_table(well_marker=lambda x: x)`` puts the well address itself in the well.
        :param title_level:
            The "title" is the first line of the returned string, which contains the plate's name
            and volume to pipette. The `title_level` controls the size, with 1 being the largest size,
            (header level 1, e.g., # title in Markdown or <h1>title</h1> in HTML).
        :param warn_unsupported_title_format:
            If True, prints a warning if `tablefmt` is a currently unsupported option for the title.
            The currently supported formats for the title are 'github', 'html', 'unsafehtml', 'rst',
            'latex', 'latex_raw', 'latex_booktabs', "latex_longtable". If `tablefmt` is another valid
            option, then the title will be the Markdown format, i.e., same as for `tablefmt` = 'github'.
        :param tablefmt:
            By default set to `'github'` to create a Markdown table. For other options see
            https://github.com/astanin/python-tabulate#readme
        :param include_plate_maps:
            If True, include plate maps as part of displayed instructions, otherwise only include the
            more compact mixing table (which is always displayed regardless of this parameter).
        :return:
            pipetting instructions in the form of strings combining results of :meth:`Mix.table` and
            :meth:`Mix.plate_maps`
        """
        table_str = self.table(
            raise_failed_validation=raise_failed_validation,
            buffer_name=buffer_name,
            tablefmt=tablefmt,
        )
        plate_map_strs = []

        if include_plate_maps:
            plate_maps = self.plate_maps(
                plate_type=plate_type,
                # validate=validate, # FIXME
                combine_plate_actions=combine_plate_actions,
            )
            for plate_map in plate_maps:
                plate_map_str = plate_map.to_table(
                    well_marker=well_marker,
                    title_level=title_level,
                    warn_unsupported_title_format=warn_unsupported_title_format,
                    tablefmt=tablefmt,
                )
                plate_map_strs.append(plate_map_str)

        # make title for whole instructions a bit bigger, if we can
        table_title_level = title_level if title_level == 1 else title_level - 1
        raw_table_title = f'Mix "{self.name}":'
        if self.test_tube_name is not None:
            raw_table_title += f' (test tube name: "{self.test_tube_name}")'
        table_title = _format_title(
            raw_table_title, level=table_title_level, tablefmt=tablefmt
        )
        return (
            table_title
            + "\n\n"
            + table_str
            + ("\n\n" + "\n\n".join(plate_map_strs) if len(plate_map_strs) > 0 else "")
        )

    def plate_maps(
        self,
        plate_type: PlateType = PlateType.wells96,
        validate: bool = True,
        combine_plate_actions: bool = True,
        # combine_volumes_in_plate: bool = False
    ) -> list[PlateMap]:
        """
        Similar to :meth:`table`, but indicates only the strands to mix from each plate,
        in the form of a :class:`PlateMap`.

        NOTE: this ignores any strands in the :class:`Mix` that are in test tubes. To get a list of strand
        names in test tubes, call :meth:`Mix.vol_to_tube_names` or :meth:`Mix.tubes_markdown`.

        By calling :meth:`PlateMap.to_markdown` on each plate map,
        one can create a Markdown representation of each plate map, for example,

        .. code-block::

            plate 1, 5 uL each
            |     | 1    | 2      | 3      | 4    | 5        | 6   | 7   | 8   | 9   | 10   | 11   | 12   |
            |-----|------|--------|--------|------|----------|-----|-----|-----|-----|------|------|------|
            | A   | mon0 | mon0_F |        | adp0 |          |     |     |     |     |      |      |      |
            | B   | mon1 | mon1_Q | mon1_F | adp1 | adp_sst1 |     |     |     |     |      |      |      |
            | C   | mon2 | mon2_F | mon2_Q | adp2 | adp_sst2 |     |     |     |     |      |      |      |
            | D   | mon3 | mon3_Q | mon3_F | adp3 | adp_sst3 |     |     |     |     |      |      |      |
            | E   | mon4 |        | mon4_Q | adp4 | adp_sst4 |     |     |     |     |      |      |      |
            | F   |      |        |        | adp5 |          |     |     |     |     |      |      |      |
            | G   |      |        |        |      |          |     |     |     |     |      |      |      |
            | H   |      |        |        |      |          |     |     |     |     |      |      |      |

        or, with the `well_marker` parameter of :meth:`PlateMap.to_markdown` set to ``'X'``, for instance
        (in case you don't need to see the strand names and just want to see which wells are marked):

        .. code-block::

            plate 1, 5 uL each
            |     | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10   | 11   | 12   |
            |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|
            | A   | *   | *   |     | *   |     |     |     |     |     |      |      |      |
            | B   | *   | *   | *   | *   | *   |     |     |     |     |      |      |      |
            | C   | *   | *   | *   | *   | *   |     |     |     |     |      |      |      |
            | D   | *   | *   | *   | *   | *   |     |     |     |     |      |      |      |
            | E   | *   |     | *   | *   | *   |     |     |     |     |      |      |      |
            | F   |     |     |     | *   |     |     |     |     |     |      |      |      |
            | G   |     |     |     |     |     |     |     |     |     |      |      |      |
            | H   |     |     |     |     |     |     |     |     |     |      |      |      |

        Parameters
        ----------

        plate_type
            96-well or 384-well plate; default is 96-well.

        validate
            Ensure volumes make sense.

        combine_plate_actions
            If True, then if multiple actions in the Mix take the same volume from the same plate,
            they will be combined into a single :class:`PlateMap`.


        Returns
        -------
            A list of all plate maps.
        """
        """
        not implementing the parameter `combine_volumes_in_plate` for now; eventual docstrings for it below

        If `combine_volumes_in_plate` is False (default), if multiple volumes are needed from a single plate,
        then one plate map is generated for each volume. If True, then in each well that is used,
        in addition to whatever else is written (strand name, or `well_marker` if it is specified),
        a volume is also given the line below (if rendered using a Markdown renderer). For example:

        .. code-block::

            plate 1, NOTE different volumes in each well
            |     | 1          | 2           | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10   | 11   | 12   |
            |-----|------------|-------------|-----|-----|-----|-----|-----|-----|-----|------|------|------|
            | A   | m0<br>1 uL | a<br>2 uL   |     |     |     |     |     |     |     |      |      |      |
            | B   | m1<br>1 uL | b<br>2 uL   |     |     |     |     |     |     |     |      |      |      |
            | C   | m2<br>1 uL | c<br>3.5 uL |     |     |     |     |     |     |     |      |      |      |
            | D   | m3<br>2 uL | d<br>3.5 uL |     |     |     |     |     |     |     |      |      |      |
            | E   | m4<br>2 uL |             |     |     |     |     |     |     |     |      |      |      |
            | F   |            |             |     |     |     |     |     |     |     |      |      |      |
            | G   |            |             |     |     |     |     |     |     |     |      |      |      |
            | H   |            |             |     |     |     |     |     |     |     |      |      |      |

        combine_volumes_in_plate
            If False (default), if multiple volumes are needed from a single plate, then one plate
            map is generated for each volume. If True, then in each well that is used, in addition to
            whatever else is written (strand name, or `well_marker` if it is specified),
            a volume is also given.
        """
        mixlines = list(self.mixlines(tablefmt="pipe"))

        if validate:
            try:
                self.validate(tablefmt="pipe", mixlines=mixlines)
            except ValueError as e:
                e.args = e.args + (
                    self.plate_maps(
                        plate_type=plate_type,
                        validate=False,
                        combine_plate_actions=combine_plate_actions,
                    ),
                )
                raise e

        # not used if combine_plate_actions is False
        plate_maps_dict: dict[Tuple[str, Quantity[Decimal]], PlateMap] = {}
        plate_maps = []
        # each MixLine but the last is a (plate, volume) pair
        for mixline in mixlines:
            if len(mixline.names) == 0 or (
                len(mixline.names) == 1 and mixline.names[0].lower() == "buffer"
            ):
                continue
            if mixline.plate.lower() == "tube":
                continue
            if mixline.plate == "":
                continue
            existing_plate = None
            key = (mixline.plate, mixline.each_tx_vol)
            if combine_plate_actions:
                existing_plate = plate_maps_dict.get(key)
            plate_map = self._plate_map_from_mixline(
                mixline, plate_type, existing_plate
            )
            if combine_plate_actions:
                plate_maps_dict[key] = plate_map
            if existing_plate is None:
                plate_maps.append(plate_map)

        return plate_maps

    def _plate_map_from_mixline(
        self,
        mixline: MixLine,
        plate_type: PlateType,
        existing_plate_map: PlateMap | None,
    ) -> PlateMap:
        # If existing_plate is None, return new plate map; otherwise update existing_plate_map and return it
        assert mixline.plate != "tube"

        well_to_strand_name = {}
        for strand_name, well in zip(mixline.names, mixline.wells):
            well_str = str(well)
            well_to_strand_name[well_str] = strand_name

        if existing_plate_map is None:
            plate_map = PlateMap(
                plate_name=mixline.plate,
                plate_type=plate_type,
                vol_each=mixline.each_tx_vol,
                well_to_strand_name=well_to_strand_name,
            )
            return plate_map
        else:
            assert plate_type == existing_plate_map.plate_type
            assert mixline.plate == existing_plate_map.plate_name
            assert mixline.each_tx_vol == existing_plate_map.vol_each

            for well_str, strand_name in well_to_strand_name.items():
                if well_str in existing_plate_map.well_to_strand_name:
                    raise ValueError(
                        f"a previous mix action already specified well {well_str} "
                        f"with strand {strand_name}, "
                        f"but each strand in a mix must be unique"
                    )
                existing_plate_map.well_to_strand_name[well_str] = strand_name
            return existing_plate_map


cell_with_border_css_class = "cell-with-border"


# https://bitbucket.org/astanin/python-tabulate/issues/57/html-class-options-for-tables
def _html_row_with_attrs(
    celltag: str,
    cell_values: Sequence[str],
    colwidths: Sequence[int],
    colaligns: Sequence[str],
) -> str:
    alignment = {
        "left": "",
        "right": ' style="text-align: right;"',
        "center": ' style="text-align: center;"',
        "decimal": ' style="text-align: right;"',
    }
    values_with_attrs = [
        f"<{celltag}{alignment.get(a, '')} class=\"{cell_with_border_css_class}\">{c}</{celltag}>"
        for c, a in zip(cell_values, colaligns)
    ]
    return "<tr>" + "".join(values_with_attrs).rstrip() + "</tr>"


from functools import partial

html_with_borders_tablefmt = TableFormat(
    lineabove=Line(
        f"""\
<style>
th.{cell_with_border_css_class}, td.{cell_with_border_css_class} {{
    border: 1px solid black;
}}
</style>
<table>\
""",
        "",
        "",
        "",
    ),
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=Line("</table>", "", "", ""),
    headerrow=partial(_html_row_with_attrs, "th"),  # type: ignore
    datarow=partial(_html_row_with_attrs, "td"),  # type: ignore
    padding=0,
    with_header_hide=None,
)
"""
Pass this as the parameter `tablefmt` in any method that accepts that parameter to have the table
be an HTML table with borders around each cell.
"""

_ALL_TABLEFMTS = [
    "plain",
    "simple",
    "github",
    "grid",
    "fancy_grid",
    "pipe",
    "orgtbl",
    "jira",
    "presto",
    "pretty",
    "psql",
    "rst",
    "mediawiki",
    "moinmoin",
    "youtrack",
    "html",
    "unsafehtml",
    "latex",
    "latex_raw",
    "latex_booktabs",
    "latex_longtable",
    "textile",
    "tsv",
    html_with_borders_tablefmt,
]

# cast is to shut mypy up; should always be a str if not == html_with_borders_tablefmt
_ALL_TABLEFMTS_NAMES: list[str] = [
    (
        cast(str, fmt)
        if fmt != html_with_borders_tablefmt
        else "html_with_borders_tablefmt"
    )
    for fmt in _ALL_TABLEFMTS
]

_SUPPORTED_TABLEFMTS_TITLE = [
    "github",
    "pipe",
    "simple",
    "grid",
    "html",
    "unsafehtml",
    "rst",
    "latex",
    "latex_raw",
    "latex_booktabs",
    "latex_longtable",
    html_with_borders_tablefmt,
]


@attrs.define()
class PlateMap:
    """
    Represents a "plate map", i.e., a drawing of a 96-well or 384-well plate, indicating which subset
    of wells in the plate have strands. It is an intermediate representation of structured data about
    the plate map that is converted to a visual form, such as Markdown, via the export_* methods.
    """

    plate_name: str
    """Name of this plate."""

    plate_type: PlateType
    """Type of this plate (96-well or 384-well)."""

    well_to_strand_name: dict[str, str]
    """dictionary mapping the name of each well (e.g., "C4") to the name of the strand in that well.

    Wells with no strand in the PlateMap are not keys in the dictionary."""

    vol_each: Quantity[Decimal] | None = None
    """Volume to pipette of each strand listed in this plate. (optional in case you simply want
    to create a plate map listing the strand names without instructions to pipette)"""

    def __str__(self) -> str:
        return self.to_table()

    def _repr_html_(self) -> str:
        return self.to_table(tablefmt="unsafehtml")

    def to_table(
        self,
        well_marker: None | str | Callable[[str], str] = None,
        title_level: Literal[1, 2, 3, 4, 5, 6] = 3,
        warn_unsupported_title_format: bool = True,
        tablefmt: str | TableFormat = "pipe",
        stralign="default",
        missingval="",
        showindex="default",
        disable_numparse=False,
        colalign=None,
    ) -> str:
        """
        Exports this plate map to string format, with a header indicating information such as the
        plate's name and volume to pipette. By default the text format is Markdown, which can be
        rendered in a jupyter notebook using ``display`` and ``Markdown`` from the package
        IPython.display:

        .. code-block:: python

            plate_maps = mix.plate_maps()
            maps_strs = '\n\n'.join(plate_map.to_table())
            from IPython.display import display, Markdown
            display(Markdown(maps_strs))

        It uses the Python tabulate package (https://pypi.org/project/tabulate/).
        The parameters are identical to that of the `tabulate` function and are passed along to it,
        except for `tabular_data` and `headers`, which are computed from this plate map.
        In particular, the parameter `tablefmt` has default value `'github'`,
        which creates a Markdown format. To create other formats such as HTML, change the value of
        `tablefmt`; see https://github.com/astanin/python-tabulate#readme for other possible formats.

        :param well_marker:
            By default the strand's name is put in the relevant plate entry. If `well_marker` is specified
            and is a string, then that string is put into every well with a strand in the plate map instead.
            This is useful for printing plate maps that just put,
            for instance, an `'X'` in the well to pipette (e.g., specify ``well_marker='X'``),
            e.g., for experimental mixes that use only some strands in the plate.
            To enable the string to depend on the well position
            (instead of being the same string in every well), `well_marker` can also be a function
            that takes as input a string representing the well (such as ``"B3"`` or ``"E11"``),
            and outputs a string. For example, giving the identity function
            ``mix.to_table(well_marker=lambda x: x)`` puts the well address itself in the well.
        :param title_level:
            The "title" is the first line of the returned string, which contains the plate's name
            and volume to pipette. The `title_level` controls the size, with 1 being the largest size,
            (header level 1, e.g., # title in Markdown or <h1>title</h1> in HTML).
        :param warn_unsupported_title_format:
            If True, prints a warning if `tablefmt` is a currently unsupported option for the title.
            The currently supported formats for the title are 'github', 'html', 'unsafehtml', 'rst',
            'latex', 'latex_raw', 'latex_booktabs', "latex_longtable". If `tablefmt` is another valid
            option, then the title will be the Markdown format, i.e., same as for `tablefmt` = 'github'.
        :param tablefmt:
            By default set to `'github'` to create a Markdown table. For other options see
            https://github.com/astanin/python-tabulate#readme
        :param stralign:
            See https://github.com/astanin/python-tabulate#readme
        :param missingval:
            See https://github.com/astanin/python-tabulate#readme
        :param showindex:
            See https://github.com/astanin/python-tabulate#readme
        :param disable_numparse:
            See https://github.com/astanin/python-tabulate#readme
        :param colalign:
            See https://github.com/astanin/python-tabulate#readme
        :return:
            a string representation of this plate map
        """
        if title_level not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"title_level must be integer from 1 to 6 but is {title_level}"
            )

        if tablefmt not in _ALL_TABLEFMTS:
            raise ValueError(
                f"tablefmt {tablefmt} not recognized; "
                f'choose one of {", ".join(_ALL_TABLEFMTS_NAMES)}'
            )
        elif (
            tablefmt not in _SUPPORTED_TABLEFMTS_TITLE and warn_unsupported_title_format
        ):
            print(
                f'{"*" * 99}\n* WARNING: title formatting not supported for tablefmt = {tablefmt}; '
                f'using Markdown format\n{"*" * 99}'
            )

        num_rows = len(self.plate_type.rows())
        num_cols = len(self.plate_type.cols())
        table = [[" " for _ in range(num_cols + 1)] for _ in range(num_rows)]

        for r in range(num_rows):
            table[r][0] = self.plate_type.rows()[r]

        if self.plate_type is PlateType.wells96:
            well_pos = WellPos(1, 1, platesize=96)
        else:
            well_pos = WellPos(1, 1, platesize=384)
        for c in range(1, num_cols + 1):
            for r in range(num_rows):
                well_str = str(well_pos)
                if well_str in self.well_to_strand_name:
                    strand_name = self.well_to_strand_name[well_str]
                    well_marker_to_use = strand_name
                    if isinstance(well_marker, str):
                        well_marker_to_use = well_marker
                    elif callable(well_marker):
                        well_marker_to_use = well_marker(well_str)
                    table[r][c] = well_marker_to_use
                if not well_pos.is_last():
                    well_pos = well_pos.advance()

        from alhambra_mixes.quantitate import normalize

        raw_title = f'plate "{self.plate_name}"' + (
            f", {normalize(self.vol_each)} each" if self.vol_each is not None else ""
        )
        title = _format_title(raw_title, title_level, tablefmt)

        header = [" "] + [str(col) for col in self.plate_type.cols()]

        out_table = tabulate(
            tabular_data=table,
            headers=header,
            tablefmt=tablefmt,
            stralign=stralign,
            missingval=missingval,
            showindex=showindex,
            disable_numparse=disable_numparse,
            colalign=colalign,
        )
        table_with_title = f"{title}\n{out_table}"
        return table_with_title


def _format_title(
    raw_title: str,
    level: int,
    tablefmt: str | TableFormat,
) -> str:
    # formats a title for a table produced using tabulate,
    # in the formats tabulate understands
    if tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:
        title = f"<h{level}>{raw_title}</h{level}>"
    elif tablefmt == "rst":
        # https://draft-edx-style-guide.readthedocs.io/en/latest/ExampleRSTFile.html#heading-levels
        # #############
        # Heading 1
        # #############
        #
        # *************
        # Heading 2
        # *************
        #
        # ===========
        # Heading 3
        # ===========
        #
        # Heading 4
        # ************
        #
        # Heading 5
        # ===========
        #
        # Heading 6
        # ~~~~~~~~~~~
        raw_title_width = len(raw_title)
        if level == 1:
            line = "#" * raw_title_width
        elif level in [2, 4]:
            line = "*" * raw_title_width
        elif level in [3, 5]:
            line = "=" * raw_title_width
        else:
            line = "~" * raw_title_width

        if level in [1, 2, 3]:
            title = f"{line}\n{raw_title}\n{line}"
        else:
            title = f"{raw_title}\n{line}"
    elif tablefmt in ["latex", "latex_raw", "latex_booktabs", "latex_longtable"]:
        if level == 1:
            size = r"\Huge"
        elif level == 2:
            size = r"\huge"
        elif level == 3:
            size = r"\LARGE"
        elif level == 4:
            size = r"\Large"
        elif level == 5:
            size = r"\large"
        elif level == 6:
            size = r"\normalsize"
        else:
            assert False
        newline = r"\\"
        noindent = r"\noindent"
        title = f"{noindent} {{ {size} {raw_title} }} {newline}"
    else:  # use the title for tablefmt == "pipe"
        hashes = "#" * level
        title = f"{hashes} {raw_title}"
    return title


def _format_errors(errors: list[VolumeError], tablefmt: TableFormat | str) -> str:
    if tablefmt in ["latex"]:
        raise NotImplementedError
    elif tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:
        if not errors:
            return ""
        s = "<div style='color: red'><ul>\n"
        s += "\n".join(f"<li>{e.args[0]}</li>" for e in errors)
        s += "</ul></div>\n"
        return s
    else:
        return "".join(f"- {e.args[0]}\n" for e in errors)


def _format_location(loc: tuple[str | None, WellPos | None]) -> str:
    p, w = loc
    if isinstance(p, str) and isinstance(w, WellPos):
        return f"{p}: {w}"
    elif isinstance(p, str) and (w is None):
        return p
    elif (p is None) and (w is None):
        return ""
    raise ValueError


_REF_COLUMNS = ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"]
_REF_DTYPES = [object, object, object, np.float64, object]

RefFile: TypeAlias = "str | tuple[str, pint.Quantity | str | dict[str, pint.Quantity]]"


def _new_ref_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=_REF_COLUMNS)
    df["Concentration (nM)"] = df["Concentration (nM)"].astype("float")
    return df


@attrs.define()
class Reference:
    df: pd.DataFrame = attrs.field(factory=_new_ref_df)

    @property
    def loc(self) -> _LocIndexer:
        return self.df.loc

    def __getitem__(self, key: Any) -> Any:
        return self.df.__getitem__(key)

    def __eq__(self: Reference, other: object) -> bool:
        if isinstance(other, Reference):
            return (
                ((other.df == self.df) | (other.df.isna() & self.df.isna())).all().all()
            )
        elif isinstance(other, pd.DataFrame):
            return ((other == self.df) | (other.isna() & self.df.isna())).all().all()
        return False

    def __len__(self) -> int:
        return len(self.df)

    def plate_map(
        self,
        name: str,
        plate_type: PlateType = PlateType.wells96,
    ) -> PlateMap:
        """
        :param name:
            Name of plate to make a :class:`PlateMap` for.
        :param plate_type:
            Either :data:`PlateType.wells96` or :data:`PlateType.wells384`;
            default is :data:`PlateType.wells96`.
        :return:
            a :class:`PlateMap` consisting of all strands in this Reference object from plate named
            `name`. Currently always makes a 96-well plate.
        """
        well_to_strand_name = {}
        for row in self.df.itertuples():
            if row.Plate == name:  # type: ignore
                well = row.Well  # type: ignore
                sequence = row.Sequence  # type: ignore
                strand = Strand(name=row.Name, sequence=sequence)  # type: ignore
                well_to_strand_name[well] = strand.name

        plate_map = PlateMap(
            plate_name=name,
            plate_type=plate_type,
            well_to_strand_name=well_to_strand_name,
        )
        return plate_map

    def search(
        self,
        name: str | None = None,
        plate: str | None = None,
        well: str | WellPos | None = None,
        concentration: str | pint.Quantity[Decimal] | None = None,
        sequence: str | None = None,
    ) -> Reference:
        well = _parse_wellpos_optional(well)
        concentration = _parse_conc_optional(concentration)
        cdf = self.df

        if name is not None:
            cdf = cdf.loc[cdf["Name"] == name, :]
        if plate is not None:
            cdf = cdf.loc[cdf["Plate"] == plate, :]
        if well is not None:
            cdf = cdf.loc[cdf["Well"] == str(well), :]
        if not math.isnan(concentration.m):
            conc = concentration.m_as("nM")
            cdf = cdf.loc[cdf["Concentration (nM)"] == conc, :]
        if sequence is not None:
            cdf = cdf.loc[cdf["Sequence"] == sequence, :]
        return Reference(cdf)

    def get_concentration(
        self,
        name: str | None = None,
        plate: str | None = None,
        well: str | WellPos | None = None,
        concentration: str | pint.Quantity | None = None,
        sequence: str | None = None,
    ) -> pint.Quantity[Decimal]:
        valref = self.search(name, plate, well, concentration, sequence)

        if len(valref) == 1:
            return Q_(valref.df["Concentration (nM)"].iloc[0], nM)
        elif len(valref) > 1:
            raise ValueError(
                f"Found multiple possible components: {str(valref)}", valref
            )

        raise ValueError("Did not find any matching components.")

    @classmethod
    def from_csv(
        cls, filename_or_file: str | io.TextIOBase | PathLike[str]
    ) -> Reference:
        """
        Load reference information from a CSV file.

        The reference information loaded by this function should be compiled manually, fitting the :ref:`mix reference` format, or
        be loaded with :func:`compile_reference` or :func:`update_reference`.
        """
        df = pd.read_csv(filename_or_file, converters={"Concentration (nM)": Decimal})

        df = df.reindex(
            ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"], axis="columns"
        )

        return cls(df)

    def to_csv(self, filename: str | PathLike[str]) -> None:
        self.df.to_csv(filename, index=None, float_format="%.6f")

    def update(
        self: Reference, files: Sequence[RefFile] | RefFile, round: int = -1
    ) -> Reference:
        """
        Update reference information.

        This updates an existing reference dataframe with new files, with the same methods as :func:`compile_reference`.
        """
        if isinstance(files, str) or (
            len(files) == 2
            and isinstance(files[1], str)
            and not Path(files[1]).exists()
        ):
            files_list: Sequence[RefFile] = [cast(RefFile, files)]
        else:
            files_list = cast(Sequence[RefFile], files)

        # FIXME: how to deal with repeats?
        for filename in files_list:
            filetype = None
            all_conc = None
            conc_dict: dict[str, pint.Quantity] = {}

            if isinstance(filename, tuple):
                conc_info = filename[1]
                filepath = Path(filename[0])

                if isinstance(conc_info, Mapping):
                    conc_dict = {
                        k: _parse_conc_required(v) for k, v in conc_info.values()
                    }
                    if "default" in conc_dict:
                        all_conc = _parse_conc_required(conc_dict["default"])
                        del conc_dict["default"]
                else:
                    all_conc = _parse_conc_required(conc_info)
            else:
                filepath = Path(filename)

            if filepath.suffix in (".xls", ".xlsx"):
                data: dict[str, pd.DataFrame] = pd.read_excel(filepath, sheet_name=None)
                if "Plate Specs" in data:
                    if len(data) > 1:
                        raise ValueError(
                            f"Plate specs file {filepath} should only have one sheet, but has {len(data)}."
                        )
                    sheet: pd.DataFrame = data["Plate Specs"]
                    filetype = "plate-specs"

                    sheet.loc[:, "Concentration (nM)"] = 1000 * sheet.loc[
                        :, "Measured Concentration µM "
                    ].round(round)
                    sheet.loc[:, "Sequence"] = [
                        x.replace(" ", "") for x in sheet.loc[:, "Sequence"]
                    ]
                    sheet.rename(lambda x: x.lower(), inplace=True, axis="columns")
                    sheet.rename(
                        {
                            "plate name": "Plate",
                            "well position": "Well",
                            "sequence name": "Name",
                        },
                        axis="columns",
                        inplace=True,
                    )

                    self.df = pd.concat(
                        (self.df, sheet.loc[:, _REF_COLUMNS]), ignore_index=True
                    )

                    continue

                else:
                    # FIXME: need better check here
                    # if not all(
                    #    next(iter(data.values())).columns
                    #    == ["Well Position", "Name", "Sequence"]
                    # ):
                    #    raise ValueError
                    filetype = "plates-order"
                    for k, v in data.items():
                        if "Plate" in v.columns:
                            # There's already a plate column.  That's problematic.  Let's check,
                            # then delete it.
                            if not all(v["Plate"] == k):
                                raise ValueError(
                                    "Not all rows in sheet {k} have same plate value (normal IDT order files do not have a plate column)."
                                )
                            del v["Plate"]
                        v["Concentration (nM)"] = conc_dict.get(
                            k, all_conc if all_conc is not None else Q_(DNAN, nM)
                        ).m_as(nM)
                    all_seqs = (
                        pd.concat(
                            data.values(), keys=data.keys(), names=["Plate"], copy=False
                        )
                        .reset_index()
                        .drop(columns=["level_1"])
                    )
                    all_seqs.rename(
                        {"Well Position": "Well", "Well position": "Well"},
                        axis="columns",
                        inplace=True,
                    )

                    self.df = pd.concat((self.df, all_seqs), ignore_index=True)
                    continue

            if filepath.suffix == ".csv":
                tubedata = pd.read_csv(filepath)
                filetype = "idt-bulk"

            if filepath.suffix == ".txt":
                tubedata = pd.read_table(filepath)
                filetype = "idt-bulk"

            if filetype == "idt-bulk":
                tubedata["Plate"] = "tube"
                tubedata["Well"] = None
                tubedata["Concentration (nM)"] = (
                    all_conc.m_as(nM) if all_conc is not None else DNAN
                )
                self.df = pd.concat(
                    (self.df, tubedata.loc[:, _REF_COLUMNS]), ignore_index=True
                )
                continue

            raise NotImplementedError

        # FIXME: validation

        return self

    @classmethod
    def compile(cls, files: Sequence[RefFile] | RefFile, round: int = -1) -> Reference:
        """
        Compile reference information.

        This loads information from the following sources:

        - An IDT plate order spreadsheet.  This does not include concentration.  To add concentration information, list it as a tuple of
        :code:`(file, concentration)`.
        - An IDT bulk order entry text file.
        - An IDT plate spec sheet.
        """
        return cls().update(files, round=round)


def load_reference(filename_or_file: str | io.TextIOBase) -> Reference:
    return Reference.from_csv(filename_or_file)


_MIXES_CLASSES = {
    c.__name__: c for c in [FixedVolume, FixedConcentration, Mix, Strand, Component]
}


def _unstructure(x):
    if isinstance(x, ureg.Quantity):
        return str(x)
    elif isinstance(x, list):
        return [_unstructure(y) for y in x]
    elif isinstance(x, WellPos):
        return str(x)
    elif hasattr(x, "__attrs_attrs__"):
        d = {}
        d["class"] = x.__class__.__name__
        for att in x.__attrs_attrs__:
            if att.name in ["reference"]:
                continue
            val = getattr(x, att.name)
            if val is att.default:
                continue
            d[att.name] = _unstructure(val)
        return d
    else:
        return x


def _structure(x):
    if isinstance(x, dict) and ("class" in x):
        c = _MIXES_CLASSES[x["class"]]
        del x["class"]
        for k in x.keys():
            x[k] = _structure(x[k])
        return c(**x)
    elif isinstance(x, list):
        return [_structure(y) for y in x]
    else:
        return x


def load_mixes(file_or_stream: str | PathLike | IO):
    if isinstance(file_or_stream, (str, PathLike)):
        p = Path(file_or_stream)
        if not p.suffix:
            p = p.with_suffix(".json")
        s = open(file_or_stream, "r")
    else:
        s = file_or_stream

    d = json.load(s)

    return {k: _structure(v) for k, v in d.items()}


def save_mixes(mixes: Sequence | Mapping | Mix, file_or_stream: str | PathLike | IO):
    if isinstance(file_or_stream, (str, PathLike)):
        p = Path(file_or_stream)
        if not p.suffix:
            p = p.with_suffix(".json")
        s = open(p, "w")
    else:
        s = file_or_stream

    if isinstance(mixes, Mix):
        d = {mixes.name: _unstructure(mixes)}
    elif isinstance(mixes, Sequence):
        d = {x.name: _unstructure(x) for x in mixes}
    elif isinstance(mixes, Mapping):
        d = {x.name: _unstructure(x) for x in mixes.values()}  # FIXME: check mapping

    json.dump(d, s)
