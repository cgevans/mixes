"""
A module for handling mixes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import logging
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)
from pathlib import Path

import numpy as np
import pandas as pd
import pint
import pint_pandas
from pint.quantity import Quantity
from tabulate import tabulate

from alhambra.seeds import Seed

from .tiles import TileList
from .tilesets import TileSet

import attrs

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
    "load_reference",
    "compile_reference",
    "update_reference",
)

log = logging.getLogger("alhambra")

ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg
ureg.default_format = "~"

MOLARITY_TOLERANCE = ureg.Quantity(1.0, "fM")
VOLUME_TOLERANCE = ureg.Quantity(1.0, "pl")


uL = ureg("uL")
uM = ureg("uM")
nM = ureg("nM")
Q_ = ureg.Quantity

ROW_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWX"

MIXHEAD_EA = (
    "Comp",
    "Src []",
    "Dest []",
    "#",
    "Ea Tx Vol",
    "Tot Tx Vol",
    "Loc",
    "Note",
)
MIXHEAD_NO_EA = ("Comp", "Src []", "Dest []", "Tx Vol", "Loc", "Note")


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
    def _validate_row(self, v: int):
        rmax = 8 if self.platesize == 96 else 16
        if (v <= 0) or (v > rmax):
            raise ValueError(
                f"Row {ROW_ALPHABET[v-1]} ({v}) out of bounds for plate size {self.platesize}"
            )

    @col.validator
    def _validate_col(self, v: int):
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
        match (ref_or_row, col):
            case (str(x), None):
                row: int = ROW_ALPHABET.index(x[0]) + 1
                col = int(x[1:])
            case (WellPos() as x, None):
                row = x.row
                col = x.col
                platesize = x.platesize
            case (int(x), int(y)):
                row = x
                col = y
            case _:
                raise TypeError

        if platesize not in (96, 384):
            raise ValueError(f"Plate size {platesize} not supported.")
        object.__setattr__(self, "platesize", platesize)

        self._validate_col(cast(int, col))
        self._validate_row(row)

        object.__setattr__(self, "row", row)
        object.__setattr__(self, "col", col)

    def __str__(self) -> str:
        return f"{ROW_ALPHABET[self.row-1]}{self.col}"

    def __repr__(self) -> str:
        return f'WellPos("{self}")'

    def __eq__(self, other: object) -> bool:
        match other:
            case WellPos(row, col, platesize):  # type: ignore
                return (row == self.row) and (col == self.col)
            case str(ws):
                return self == WellPos(other, platesize=self.platesize)
            case _:
                return False
        return False

    def key_byrow(self) -> tuple[int, int]:
        "Get a tuple (row, col) key that can be used for ordering by row."
        return (self.row, self.col)

    def key_bycol(self) -> tuple[int, int]:
        "Get a tuple (col, row) key that can be used for ordering by column."
        return (self.col, self.row)

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


@attrs.define(eq=True)
class _MixLine:
    """Class for handling a line of a (processed) mix recipe."""

    name: str | None
    source_conc: Quantity[float] | str | None
    dest_conc: Quantity[float] | str | None
    total_tx_vol: Quantity[float] | None
    number: int = 1
    each_tx_vol: Quantity[float] | None = None
    location: str | None = None
    note: str | None = None

    def toline(self, incea: bool):
        if incea:
            return [
                _formatter(getattr(self, x), x)
                for x in [
                    "name",
                    "source_conc",
                    "dest_conc",
                    "number",
                    "each_tx_vol",
                    "total_tx_vol",
                    "location",
                    "note",
                ]
            ]
        else:
            return [
                _formatter(getattr(self, x))
                for x in [
                    "name",
                    "source_conc",
                    "dest_conc",
                    "total_tx_vol",
                    "location",
                    "note",
                ]
            ]


def _formatter(x: int | float | str | None, t: str = "") -> str:
    match x:
        case y if isinstance(y, (int, str)):
            if t == "number" and x == 1:
                return ""
            return str(y)
        case None:
            return ""
        case y if isinstance(y, (float, Quantity)):
            return f"{y:.2f}"
        case _:
            raise TypeError
    raise TypeError


T = TypeVar("T")


class AbstractComponent(ABC):
    """Abstract class for a component in a mix."""

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        "Name of the component."
        ...

    @property
    @abstractmethod
    def location(self) -> tuple[None | str, WellPos | None]:
        ...

    @property
    @abstractmethod
    def concentration(self) -> Quantity[float]:  # pragma: no cover
        "(Source) concentration of the component as a pint Quantity.  NaN if undefined."
        ...

    @abstractmethod
    def all_components(self) -> pd.DataFrame:  # pragma: no cover
        "A dataframe of all components."
        ...

    @abstractmethod
    def with_reference(self: T, reference: pd.DataFrame) -> T:  # pragma: no cover
        ...


def _parse_conc_optional(v: str | pint.Quantity | None) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check("nM"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check("nM"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return x.to_compact()
        case None:
            return Q_(np.nan, "nM")
    raise ValueError


def _parse_conc_required(v: str | pint.Quantity) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check("nM"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check("nM"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return x.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be molarity).")


def _parse_vol_optional(v: str | pint.Quantity | None) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check("uL"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check("uL"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return x.to_compact()
        case None:
            return Q_(np.nan, "uL")
    raise ValueError


def _parse_vol_required(v: str | pint.Quantity) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check("uL"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check("uL"):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return x.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be volume).")


def _parse_wellpos_optional(v: str | WellPos | None) -> WellPos | None:
    match v:
        case str(x):
            return WellPos(x)
        case WellPos() as x:
            return x
        case None:
            return None
    try:
        if np.isnan(v): # type: ignore
            return None
    except:
        pass
    raise ValueError(f"Can't interpret {v} as well position or None.")


@attrs.define()
class Component(AbstractComponent):
    """A single named component, potentially with a concentration and location."""

    name: str
    concentration: Quantity[float] = attrs.field(
        converter=_parse_conc_optional, default=None
    )
    place: str | None = attrs.field(default=None, kw_only=True)
    well: WellPos | None = attrs.field(
        converter=_parse_wellpos_optional, default=None, kw_only=True
    )

    def __eq__(self, other: Any) -> bool:
        if not other.__class__ == Component:
            return False
        if self.name != other.name:
            return False
        match (self.concentration, other.concentration):
            case (Quantity() as x, Quantity() as y):
                if np.isnan(x) and np.isnan(y):
                    return True
                return abs(x - y) <= MOLARITY_TOLERANCE
            case x, y:
                return x == y
        return False

    @property
    def location(self) -> tuple[str | None, WellPos | None]:
        return (self.place, self.well)

    def all_components(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "concentration_nM": [self.concentration.to("nM").magnitude],
                "component": [self],
            },
            index=pd.Index([self.name], name="name"),
        )
        return df

    def with_reference(self: Component, reference: pd.DataFrame) -> Component:
        if reference.index.name == "Name":
            ref_by_name = reference
        else:
            ref_by_name = reference.set_index("Name")
        ref_comp = ref_by_name.loc[self.name]

        if len(ref_comp.shape) > 1:  # Is this a series or a dataframe
            log.warning(
                "Component %s has more than one location: %s.  Choosing last.",
                self.name,
                ref_comp,
            )
            ref_comp = ref_comp.iloc[-1, :]

        ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], "nM")

        if not np.isnan(self.concentration) and not np.allclose(
            ref_conc, self.concentration
        ):
            raise ValueError(
                f"Component {self.name}: concentration {self.concentration} does not match reference {ref_conc} of {ref_comp}."
            )

        ref_place = ref_comp["Plate"]
        if self.place and ref_place != self.place:
            raise ValueError(
                f"Component {self.name}: plate {self.place} does not match reference {ref_place} of {ref_comp}."
            )

        ref_well = _parse_wellpos_optional(ref_comp["Well"])
        if self.well and self.well != ref_well:
            raise ValueError(
                f"Component {self.name}: well {self.well} does not match reference {ref_well} of {ref_comp}."
            )

        return Component(self.name, ref_conc, place=ref_place, well=ref_well)


@attrs.define()
class Strand(Component):
    """A single named strand, potentially with a concentration, location and sequence."""

    sequence: str | None = None

    def with_reference(self: Strand, reference: pd.DataFrame) -> Strand:
        if reference.index.name == "Name":
            ref_by_name = reference
        else:
            ref_by_name = reference.set_index("Name")
        ref_comp = ref_by_name.loc[self.name]

        if len(ref_comp.shape) > 1:  # Is this a series or a dataframe
            log.warning(
                "Component %s has more than one location: %s.  Choosing last.",
                self.name,
                ref_comp,
            )
            ref_comp = ref_comp.iloc[-1, :]

        ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], "nM")

        if not np.isnan(self.concentration) and not np.allclose(
            ref_conc, self.concentration
        ):
            raise ValueError(
                f"Strand {self.name}: concentration {self.concentration} does not match reference {ref_conc} of {ref_comp}."
            )

        match (self.sequence, ref_comp["Sequence"]):
            case (None, None):
                seq = None
            case (str(x), None) | (str(x), "") | (None, str(x)):
                seq = x
            case (str(x), str(y)):
                if x != y:
                    raise ValueError(
                        f"Sequence {self.name}: sequence {x} does not match reference {y} of {ref_comp}."
                    )
                seq = x

        ref_place = ref_comp["Plate"]
        if self.place and ref_place != self.place:
            raise ValueError(
                f"Sequence {self.name}: plate {self.place} does not match reference {ref_place} of {ref_comp}."
            )

        ref_well = _parse_wellpos_optional(ref_comp["Well"])
        if self.well and self.well != ref_well:
            raise ValueError(
                f"Sequence {self.name}: well {self.well} does not match reference {ref_well} of {ref_comp}."
            )

        return Strand(self.name, ref_conc, sequence=x, well=ref_well, place=ref_place)


class AbstractAction(ABC):
    """
    Abstract class defining an action in a mix recipe.
    """

    @property
    def name(self) -> str:  # pragma: no cover
        ...

    @abstractmethod
    def tx_volume(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> Quantity[float]:  # pragma: no cover
        ...

    @abstractmethod
    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[_MixLine]:  # pragma: no cover
        ...

    @abstractmethod
    def all_components(
        self, mix_vol: Quantity[float]
    ) -> pd.DataFrame:  # pragma: no cover
        ...

    @abstractmethod
    def with_reference(self: T, reference: pd.DataFrame) -> T:  # pragma: no cover
        ...

    def dest_concentration(self, mix_vol: pint.Quantity) -> pint.Quantity:
        raise ValueError


def findloc(locations: pd.DataFrame | None, name: str) -> str | None:
    match findloc_tuples(locations, name):
        case (_, plate, well):
            if well:
                return f"{plate}: {well}"
            else:
                return f"{plate}"
        case None:
            return None
    return None


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


@attrs.define()
class FixedConcentration(AbstractAction):
    """A mix action adding one component at a fixed destination concentration."""

    component: AbstractComponent
    fixed_concentration: Quantity[float] = attrs.field(converter=_parse_conc_required)

    def dest_concentration(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> Quantity[float]:
        return self.fixed_concentration

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, "nM")) -> Quantity[float]:
        retval: Quantity[float] = (
            mix_vol * self.fixed_concentration / self.component.concentration
        ).to_compact()
        retval.check("M")
        return retval

    def all_components(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> pd.DataFrame:
        comps = self.component.all_components()
        comps.concentration_nM *= (
            (self.fixed_concentration / self.component.concentration).to("").magnitude
        )
        return comps

    def _mixlines(
        self,
        mix_vol: Quantity[float] = Q_(np.nan, "nM"),
        locations: pd.DataFrame | None = None,
    ) -> Sequence[_MixLine]:

        return [
            _MixLine(
                name=self.component.name,
                source_conc=self.component.concentration,
                dest_conc=self.dest_concentration(mix_vol),
                total_tx_vol=self.tx_volume(mix_vol),
                location=_format_location(self.component.location),
            )
        ]

    def with_reference(self, reference: pd.DataFrame) -> FixedConcentration:
        return FixedConcentration(
            self.component.with_reference(reference), self.fixed_concentration
        )

    @property
    def name(self) -> str:
        return self.component.name


@attrs.define()
class FixedVolume(AbstractAction):
    """A mix action adding one component, at a fixed destination volume."""

    component: AbstractComponent
    fixed_volume: Quantity[float] = attrs.field(converter=_parse_vol_required)

    def dest_concentration(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> Quantity[float]:
        return (self.component.concentration * self.fixed_volume / mix_vol).to_compact()

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, "nM")) -> Quantity[float]:
        return self.fixed_volume

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        comps = self.component.all_components()
        comps.concentration_nM *= (
            (self.dest_concentration(mix_vol) / self.component.concentration)
            .to("")
            .magnitude
        )
        return comps

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[_MixLine]:

        return [
            _MixLine(
                name=self.component.name,
                source_conc=self.component.concentration,
                dest_conc=self.dest_concentration(mix_vol),
                total_tx_vol=self.tx_volume(mix_vol),
                location=_format_location(self.component.location),
            )
        ]

    def with_reference(self, reference: pd.DataFrame) -> FixedVolume:
        return FixedVolume(self.component.with_reference(reference), self.fixed_volume)

    @property
    def name(self) -> str:
        return self.component.name


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
    cps["concentration_nM"] = pd.Series([], dtype=float)
    cps["component"] = pd.Series([], dtype=object)
    return cps


@attrs.define()
class MultiFixedVolume(AbstractAction):
    """A action adding multiple components, with a set destination volume (potentially keeping equal concentration).
    
    MultiFixedVolume adds a selection of components, with a specified transfer volume.  Depending on the setting of

    """

    components: Sequence[AbstractComponent]
    fixed_volume: Quantity[float] = attrs.field(converter=_parse_vol_required)
    set_name: str | None = None
    compact_display: bool = True
    equal_conc: bool | Literal['max_vol', 'min_vol'] = True

    def with_reference(self, reference: pd.DataFrame) -> MultiFixedVolume:
        return MultiFixedVolume(
            [c.with_reference(reference) for c in self.components],
            self.fixed_volume,
            self.set_name,
            self.compact_display,
        )

    @property
    def source_concentration(self):
        conc = self.components[0].concentration

        if not all(c.concentration == conc for c in self.components):
            raise ValueError("Not all components have equal concentration.")

        return self.components[0].concentration

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp in self.components:
            comps = comp.all_components()
            comps.concentration_nM *= (
                (self.dest_concentration(mix_vol) / self.source_concentration)
                .to("")
                .magnitude
            )

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=0.0)
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentration(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> Quantity[float]:
        return (self.source_concentration * self.fixed_volume / mix_vol).to_compact()

    def each_volume(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> Quantity[float]:
        return self.fixed_volume

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, "nM")) -> Quantity[float]:
        return self.fixed_volume * self.number

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[_MixLine]:
        if not self.compact_display:
            return [
                _MixLine(
                    comp.name,
                    comp.concentration,
                    self.dest_concentration(mix_vol),
                    self.each_volume(mix_vol),
                    location=_format_location(comp.location),
                )
                for comp in self.components
            ]
        else:

            name, locs = self._compactstrs(locations=locations)

            return [
                _MixLine(
                    name,
                    self.source_concentration,
                    self.dest_concentration(mix_vol),
                    self.tx_volume(mix_vol),
                    self.number,
                    self.each_volume(mix_vol),
                    location=locs,
                )
            ]

    @property
    def number(self) -> int:
        return len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return ", ".join(c.name for c in self.components)
        else:
            return self.set_name

    def _compactstrs(self, locations: pd.DataFrame | None) -> tuple[str, str | None]:
        if locations is None:
            return ", ".join(c.name for c in self.components), None
        else:
            locs = [(c.name,) + c.location for c in self.components]
            names = [c.name for c in self.components]

            if all(x is None for x in locs):
                return ", ".join(names), None

            if any(x is None for x in locs):
                raise ValueError(
                    [name for name, loc in zip(names, locs) if loc is None]
                )

            locdf = pd.DataFrame(locs, columns=("Name", "Plate", "Well"))

            locdf.sort_values(by=["Plate", "Well"])

            ns, ls = [], []

            for p, ll in locdf.groupby("Plate"):
                names = list(ll["Name"])
                wells: list[WellPos] = list(ll["Well"])

                byrow = mixgaps(sorted(wells, key=WellPos.key_byrow), by="row")
                bycol = mixgaps(sorted(wells, key=WellPos.key_bycol), by="col")

                sortkey = WellPos.key_bycol if bycol <= byrow else WellPos.key_byrow
                sortnext = WellPos.next_bycol if bycol <= byrow else WellPos.next_byrow

                nw = sorted(
                    [(name, well) for name, well in zip(names, wells, strict=True)],
                    key=(lambda nwitem: sortkey(nwitem[1])),
                )

                wellsf = []
                nwi = iter(nw)
                prevpos = next(nwi)[1]
                wellsf.append(f"**{prevpos}**")
                for _, w in nwi:
                    if sortnext(prevpos) != w:
                        wellsf.append(f"**{w}**")
                    else:
                        wellsf.append(f"{w}")
                    prevpos = w

                ns.append(", ".join(n for n, _ in nw))
                ls.append(p + ": " + ", ".join(wellsf))

            return "\n".join(ns), "\n".join(ls)


@attrs.define()
class MultiFixedConcentration(AbstractAction):
    """A action adding multiple components, each with the same destination concentration (NOT IMPLEMENTED)."""

    components: Sequence[AbstractComponent]
    fixed_concentration: Quantity[float] = attrs.field(converter=_parse_conc_required)
    set_name: str | None = None
    compact_display: bool = True

    def with_reference(self, reference: pd.DataFrame) -> MultiFixedConcentration:
        return MultiFixedConcentration(
            [c.with_reference(reference) for c in self.components],
            self.fixed_concentration,
            self.set_name,
            self.compact_display,
        )

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp in self.components:
            comps = comp.all_components()
            comps.concentration_nM *= (
                (self.dest_concentration(mix_vol) / comp.concentration).to("").magnitude
            )

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=0.0)
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentration(
        self, mix_vol: Quantity[float] = Q_(np.nan, "nM")
    ) -> Quantity[float]:
        return self.fixed_concentration

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, "nM")) -> Quantity[float]:
        v = Q_(0.0, "uL")
        for comp in self.components:
            v += (mix_vol * self.fixed_concentration / comp.concentration).to(uL)
        return v

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[_MixLine]:
        raise NotImplementedError
        if not self.compact_display:
            return [
                _MixLine(
                    comp.name,
                    comp.concentration,
                    self.dest_concentration(mix_vol),
                    (
                        mix_vol * self.fixed_concentration / comp.concentration
                    ).to_compact(),
                    location=findloc(locations, comp.name),
                )
                for comp in self.components
            ]
        else:

            name, scs, vol, locs = self._compactstrs(
                mix_vol=mix_vol, reference=locations
            )

            return [
                _MixLine(
                    name,
                    scs,
                    self.dest_concentration(mix_vol),
                    self.tx_volume(mix_vol),
                    self.number,
                    vol,
                    location=locs,
                )
            ]

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
        self, mix_vol: pint.Quantity, reference: pd.DataFrame | None
    ) -> tuple[str, pint.Quantity, pint.Quantity, str | None]:
        raise NotImplementedError
        locs = [findloc_tuples(reference, c.name) for c in self.components]
        names = [c.name for c in self.components]
        scs = [c.concentration for c in self.components]
        vols = [
            (mix_vol * self.dest_concentration / c.concentration).to_compact()
            for c in self.components
        ]

        # if all(x is None for x in locs):
        #     return ", ".join(names), None

        # if any(x is None for x in locs):
        #     raise ValueError([name for name, loc in zip(names, locs) if loc is None])

        # locdf = pd.DataFrame(locs, columns=("Name", "Plate", "Well"))

        # locdf.sort_values(by=["Plate", "Well"])

        # ns, ls = [], []

        # for p, ll in locdf.groupby("Plate"):
        #     names = list(ll["Name"])
        #     wells: list[WellPos] = list(ll["Well"])

        #     byrow = mixgaps(sorted(wells, key=WellPos.key_byrow), by="row")
        #     bycol = mixgaps(sorted(wells, key=WellPos.key_bycol), by="col")

        #     sortkey = WellPos.key_bycol if bycol <= byrow else WellPos.key_byrow
        #     sortnext = WellPos.next_bycol if bycol <= byrow else WellPos.next_byrow

        #     nw = sorted(
        #         [(name, well) for name, well in zip(names, wells, strict=True)],
        #         key=(lambda nwitem: sortkey(nwitem[1])),
        #     )

        #     wellsf = []
        #     nwi = iter(nw)
        #     prevpos = next(nwi)[1]
        #     wellsf.append(f"**{prevpos}**")
        #     for _, w in nwi:
        #         if sortnext(prevpos) != w:
        #             wellsf.append(f"**{w}**")
        #         else:
        #             wellsf.append(f"{w}")
        #         prevpos = w

        #     ns.append(", ".join(n for n, _ in nw))
        #     ls.append(p + ": " + ", ".join(wellsf))

        # return "\n".join(ns), "\n".join(vs), "\n".join(ls)


@attrs.define()
class FixedRatio(AbstractAction):
    """A mix action adding a component at some fixed concentration ratio, regardless of concentration.
    Useful, for, eg, concentrated buffers."""

    component: AbstractComponent
    source_value: float
    dest_value: float

    @property
    def name(self) -> str:
        return self.component.name

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, "nM")) -> Quantity[float]:
        return mix_vol * self.dest_value / self.source_value

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        v = self.component.all_components()
        v.loc[:, "concentration_nM"] *= self.dest_value / self.source_value
        return v

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[_MixLine]:
        return [
            _MixLine(
                self.name,
                str(self.source_value) + "x",
                str(self.dest_value) + "x",
                self.tx_volume(mix_vol),
                location=_format_location(self.component.location),
            )
        ]

    def with_reference(self, reference: pd.DataFrame) -> FixedRatio:
        return FixedRatio(
            self.component.with_reference(reference), self.source_value, self.dest_value
        )


@attrs.define()
class Mix(AbstractComponent):
    """Class denoting a Mix, a collection of source components mixed to
    some volume or concentration.
    """

    actions: Sequence[AbstractAction]
    name: str
    fixed_total_volume: Optional[Quantity[float]] = None
    fixed_concentration: Union[str, Quantity[float], None] = None
    buffer_name: Optional[str] = None
    reference: pd.DataFrame | None = None

    def __attrs_post_init__(self) -> None:
        if self.reference is not None:
            self.actions = [
                action.with_reference(self.reference) for action in self.actions
            ]

    @property
    def concentration(self) -> Quantity[float]:
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
                ac.loc[self.fixed_concentration, "concentration_nM"], "nM"
            )
        elif self.fixed_concentration is None:
            return self.actions[0].dest_concentration(self.total_volume)
        else:
            raise NotImplemented

    @property
    def total_volume(self) -> Quantity[float]:
        """
        Total volume of the the mix.  If the mix has a fixed total volume, then that,
        otherwise, the sum of the transfer volumes of each component.
        """
        if self.fixed_total_volume is not None:
            return self.fixed_total_volume
        else:
            return sum([c.tx_volume() for c in self.actions], 0 * ureg("µL"))

    @property
    def buffer_volume(self) -> Quantity[float]:
        """
        The volume of buffer to be added to the mix, in addition to the components.
        """
        mvol = sum(c.tx_volume(self.total_volume) for c in self.actions)
        return self.total_volume - mvol

    def mdtable(self):
        tv = self.total_volume

        _mixlines = []

        for action in self.actions:
            _mixlines += action._mixlines(tv, locations=self.reference)

        if self.fixed_total_volume is not None:
            _mixlines.append(_MixLine("Buffer", None, None, self.buffer_volume))

        include_numbers = any(ml.number != 1 for ml in _mixlines)

        return tabulate(
            [ml.toline(include_numbers) for ml in _mixlines],
            MIXHEAD_EA if include_numbers else MIXHEAD_NO_EA,
            "pipe",
        )

    def all_components(self) -> pd.DataFrame:
        """
        Return a Series of all component names, and their concentrations (as pint nM).
        """
        cps = _empty_components()

        for action in self.actions:
            mcomp = action.all_components(self.total_volume)
            cps, _ = cps.align(mcomp)
            cps.loc[:, "concentration_nM"].fillna(0.0, inplace=True)
            cps.loc[mcomp.index, "concentration_nM"] += mcomp.concentration_nM
            cps.loc[mcomp.index, "component"] = mcomp.component
        return cps

    def _repr_markdown_(self):
        return str(self)

    def __str__(self):
        return (
            f"Table: Mix: {self.name}, Conc: {self.concentration:.2f}, Total Vol: {self.total_volume:.2f}\n\n"
            + self.mdtable()
        )

    def to_tileset(
        self,
        tilesets_or_lists: TileSet | TileList | Iterable[TileSet | TileList],
        *,
        seed: bool | Seed = False,
        base_conc=100 * ureg("nM"),
    ) -> TileSet:
        """
        Given some :any:`TileSet`\ s, or lists of :any:`Tile`\ s from which to
        take tiles, generate an TileSet from the mix.
        """
        newts = TileSet()

        if isinstance(tilesets_or_lists, (TileList, TileSet)):
            tilesets_or_lists = [tilesets_or_lists]

        for name, row in self.all_components().iterrows():
            new_tile = None
            for tl_or_ts in tilesets_or_lists:
                try:
                    if isinstance(tl_or_ts, TileSet):
                        tile = tl_or_ts.tiles[name]
                    else:
                        tile = tl_or_ts[name]
                    new_tile = tile.copy()
                    new_tile.stoic = float(
                        Q_(row["concentration_nM"], "nM") / base_conc
                    )
                    newts.tiles.add(new_tile)
                    break
                except KeyError:
                    pass
            if new_tile is None:
                log.warn(f"Component {name} not found in tile lists.")

        match seed:
            case True:
                firstts = next(iter(tilesets_or_lists))
                assert isinstance(firstts, TileSet)
                newts.seeds["default"] = firstts.seeds["default"]
            case False:
                pass
            case Seed() as x:
                newts.seeds["default"] = x

        if len(newts.tiles) == 0:
            raise ValueError("No mix components match tiles.")

        return newts

    def with_reference(self: Mix, reference: pd.DataFrame) -> Mix:
        new = Mix(
            name=self.name,
            actions=[action.with_reference(reference) for action in self.actions],
            fixed_total_volume=self.fixed_total_volume,
            fixed_concentration=self.fixed_concentration,
            buffer_name=self.buffer_name,
        )
        new.reference = reference
        return new

    @property
    def location(self) -> tuple[None | str, WellPos | None]:
        return (None, None)


def _format_location(loc: tuple[str | None, WellPos | None]):
    match loc:
        case str(p), WellPos() as w:
            return f"{p}: {w}"
        case str(p), None:
            return p
        case None, None:
            return ""
    raise ValueError


def load_reference(filename_or_file):
    """
    Load reference information from a CSV file.

    The reference information loaded by this function should be compiled manually, fitting the :ref:`mix reference` format, or
    be loaded with :func:`compile_reference` or :func:`update_reference`.
    """
    df = pd.read_csv(filename_or_file)

    return df.reindex(
        ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"],
        axis="columns",
    )


RefFile: TypeAlias = "str | tuple[str, pint.Quantity | str | dict[str, pint.Quantity]]"

REF_COLUMNS = ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"]


def update_reference(
    reference: pd.DataFrame | None, files: Sequence[RefFile] | RefFile, round: int = -1
) -> pd.DataFrame:
    """
    Update reference information.

    This updates an existing reference dataframe with new files, with the same methods as :func:`compile_reference`.
    """
    if reference is None:
        reference = pd.DataFrame(columns=REF_COLUMNS)

    if isinstance(files, str) or (
        len(files) == 2 and isinstance(files[1], str) and not Path(files[1]).exists()
    ):
        files = [cast(RefFile, files)]

    # FIXME: how to deal with repeats?
    for filename in files:
        filetype = None
        all_conc = None
        conc_dict: dict[str, pint.Quantity] = {}

        if isinstance(filename, tuple):
            conc_info = filename[1]
            filename = filename[0]

            if isinstance(conc_info, Mapping):
                conc_dict = {k: _parse_conc_required(v) for k, v in conc_info.values()}
                if "default" in conc_dict:
                    all_conc = _parse_conc_required(conc_dict["default"])
                    del conc_dict["default"]
            else:
                _parse_conc_required(filename[1])

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
                sheet.rename(
                    {
                        "Plate Name": "Plate",
                        "Well Position": "Well",
                        "Sequence Name": "Name",
                    },
                    axis="columns",
                    inplace=True,
                )

                reference = pd.concat(
                    (reference, sheet.loc[:, REF_COLUMNS]), ignore_index=True
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
                        k, all_conc if all_conc is not None else Q_(np.nan, "nM")
                    ).m_as("nM")
                all_seqs = (
                    pd.concat(
                        data.values(), keys=data.keys(), names=["Plate"], copy=False
                    )
                    .reset_index()
                    .drop(columns=["level_1"])
                )
                all_seqs.rename({"Well Position": "Well"}, axis="columns", inplace=True)

                reference = pd.concat((reference, all_seqs), ignore_index=True)
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
                all_conc.m_as("nM") if all_conc is not None else np.nan
            )
            reference = pd.concat(
                (reference, tubedata.loc[:, REF_COLUMNS]), ignore_index=True
            )
            continue

        raise NotImplementedError

    # FIXME: validation

    return reference


def compile_reference(files: Sequence[RefFile] | RefFile) -> pd.DataFrame:
    """
    Compile reference information.

    This loads information from the following sources:

    - An IDT plate order spreadsheet.  This does not include concentration.  To add concentration information, list it as a tuple of
      :code:`(file, concentration)`.
    - An IDT bulk order entry text file.
    - An IDT plate spec sheet.
    """
    return update_reference(None, files)
