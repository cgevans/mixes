"""
A module for handling mixes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import logging
from multiprocessing.sharedctypes import Value
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
from tabulate import tabulate, TableFormat, tabulate_formats

from alhambra.seeds import Seed

from .tiles import TileList
from .tilesets import TileSet

import attrs

import warnings

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
    "load_reference",
    "compile_reference",
    "update_reference",
)

log = logging.getLogger("alhambra")

ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg
ureg.default_format = "~#P"

uL = ureg.uL
uM = ureg.uM
nM = ureg.nM

Q_ = ureg.Quantity
"Convenient constructor for units, eg, :code:`Q_(5.0, 'nM')`"

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
class MixLine:
    """Class for handling a line of a (processed) mix recipe."""

    name: str | None
    source_conc: Quantity[float] | str | None
    dest_conc: Quantity[float] | str | None
    total_tx_vol: Quantity[float] | None
    number: int = 1
    each_tx_vol: Quantity[float] | str | None = None
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
        case int(y) | str(y):
            if t == "number" and x == 1:
                return ""
            return str(y)
        case None:
            return ""
        case float(y):
            return f"{y:,.2f}"
        case Quantity() as y:
            return f"{y:,.2f~#P}"
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
    def location(self) -> tuple[None | str, WellPos | None]:
        return (None, None)

    @property
    def plate(self) -> None | str:
        return None

    @property
    def well(self) -> WellPos | None:
        return None

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
            if not q.check(nM):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check(nM):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return x.to_compact()
        case None:
            return Q_(np.nan, nM)
    raise ValueError


def _parse_conc_required(v: str | pint.Quantity) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check(nM):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check(nM):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be molarity)."
                )
            return x.to_compact()
    raise ValueError(f"{v} is not a valid quantity here (should be molarity).")


def _parse_vol_optional(v: str | pint.Quantity | None) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check(uL):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check(uL):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return x.to_compact()
        case None:
            return Q_(np.nan, uL)
    raise ValueError


def _parse_vol_required(v: str | pint.Quantity) -> pint.Quantity:
    match v:
        case str(x):
            q = ureg(x)
            if not q.check(uL):
                raise ValueError(
                    f"{x} is not a valid quantity here (should be volume)."
                )
            return q
        case pint.Quantity() as x:
            if not x.check(uL):
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
        if np.isnan(v):  # type: ignore
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
    plate: str | None = attrs.field(default=None, kw_only=True)
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
                return np.allclose(x, y)
            case x, y:
                return x == y
        return False

    @property
    def location(self) -> tuple[str | None, WellPos | None]:
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

    def with_reference(self: Component, reference: pd.DataFrame) -> Component:
        if reference.index.name == "Name":
            ref_by_name = reference
        else:
            ref_by_name = reference.set_index("Name")
        ref_comps = ref_by_name.loc[
            [self.name], :
        ]  # using this format to force a dataframe result

        mismatches = []
        matches = []
        for _, ref_comp in ref_comps.iterrows():
            ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], nM)
            if not np.isnan(self.concentration) and not np.allclose(
                ref_conc, self.concentration
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

    def with_reference(self: Strand, reference: pd.DataFrame) -> Strand:
        if reference.index.name == "Name":
            ref_by_name = reference
        else:
            ref_by_name = reference.set_index("Name")
        ref_comps = ref_by_name.loc[
            [self.name], :
        ]  # using this format to force a dataframe result

        mismatches = []
        matches = []
        for _, ref_comp in ref_comps.iterrows():
            ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], nM)
            if not np.isnan(self.concentration) and not np.allclose(
                ref_conc, self.concentration
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

            match (self.sequence, ref_comp["Sequence"]):
                case (str(x), str(y)):
                    x = x.replace(" ", "").replace("-", "")
                    y = y.replace(" ", "").replace("-", "")
                    if x != y:
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
        match (self.sequence, m["Sequence"]):
            case (None, None):
                seq = None
            case (str(x), None) | (str(x), "") | (None, str(x)) | (str(_), str(x)):
                seq = x
            case _:
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
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> Quantity[float]:  # pragma: no cover
        """The total volume transferred by the action to the sample.  May depend on the total mix volume.

        Parameters
        ----------

        mix_vol
            The mix volume.  Does not accept strings.
        """
        ...

    @abstractmethod
    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:  # pragma: no cover
        ...

    @abstractmethod
    def all_components(
        self, mix_vol: Quantity[float]
    ) -> pd.DataFrame:  # pragma: no cover
        """A dataframe containing all base components added by the action.

        Parameters
        ----------

        mix_vol
            The mix volume.  Does not accept strings.
        """
        ...

    @abstractmethod
    def with_reference(self: T, reference: pd.DataFrame) -> T:  # pragma: no cover
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

    def dest_concentrations(self, mix_vol: pint.Quantity) -> pd.Series:
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
    """A mix action adding one component, at a fixed destination concentration.

    Parameters
    ----------

    component
        The component to add.

    fixed_concentration
        The concentration of the component that should be added to the mix, as a string (eg, "50.0 nM")
        or a pint Quantity).  Note that this increases the concentration of the component by this amount;
        if the component is also added to the mix by another action, the final, total concentration of the
        component in the mix may be higher.
    """

    component: AbstractComponent
    fixed_concentration: Quantity[float] = attrs.field(converter=_parse_conc_required)

    def dest_concentration(
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> Quantity[float]:
        return self.fixed_concentration

    def dest_concentrations(
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> Quantity[float]:
        return pd.Series([self.dest_concentration(mix_vol)], dtype="pint[nM]")

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> Quantity[float]:
        retval: Quantity[float] = (
            mix_vol * self.fixed_concentration / self.component.concentration
        ).to_compact()
        retval.check("L")
        return retval

    def all_components(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> pd.DataFrame:
        comps = self.component.all_components()
        comps.concentration_nM *= (
            (self.fixed_concentration / self.component.concentration).to("").magnitude
        )
        return comps

    def _mixlines(
        self,
        mix_vol: Quantity[float] = Q_(np.nan, uL),
        locations: pd.DataFrame | None = None,
    ) -> Sequence[MixLine]:

        return [
            MixLine(
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
    """A mix action adding one component, at a fixed destination volume.

    Parameters
    ----------

    component
        The component to add.

    fixed_volume
        The volume of the component to add, as a string (eg, "5 µL") or a pint Quantity)
    """

    component: AbstractComponent
    fixed_volume: Quantity[float] = attrs.field(converter=_parse_vol_required)

    def dest_concentration(
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> Quantity[float]:
        return (self.component.concentration * self.fixed_volume / mix_vol).to_compact()

    def dest_concentrations(
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> pd.Series:
        return pd.Series([self.dest_concentration(mix_vol)], dtype="pint[nM]")

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> Quantity[float]:
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
    ) -> Sequence[MixLine]:

        return [
            MixLine(
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
    """An action adding multiple components, with a set destination volume (potentially keeping equal concentration).

    MultiFixedVolume adds a selection of components, with a specified transfer volume.  Depending on the setting of
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

    >>> print(Mix([MultiFixedVolume(components, "5 uL")], name="example"))
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

    >>> print(Mix([MultiFixedVolume(components, "5 uL", equal_conc="min_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 25.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 5.00 µl     | 15.00 µl     |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 10.00 µl    | 10.00 µl     |       |        |

    >>> print(Mix([MultiFixedVolume(components, "5 uL", equal_conc="max_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 12.50 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 2.50 µl     | 7.50 µl      |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 5.00 µl     | 5.00 µl      |       |        |

    """

    components: Sequence[AbstractComponent]
    fixed_volume: Quantity[float] = attrs.field(converter=_parse_vol_required)
    set_name: str | None = None
    compact_display: bool = True
    equal_conc: bool | Literal["max_volume", "min_volume"] | tuple[
        Literal["max_fill"], str
    ] = True

    def with_reference(self, reference: pd.DataFrame) -> MultiFixedVolume:
        return MultiFixedVolume(
            [c.with_reference(reference) for c in self.components],
            self.fixed_volume,
            self.set_name,
            self.compact_display,
            self.equal_conc,
        )

    @property
    def source_concentrations(self):
        concs = pd.Series(
            [c.concentration.m_as(nM) for c in self.components], dtype="pint[nM]"
        )
        if not (concs == concs[0]).all() and not self.equal_conc:
            raise ValueError("Not all components have equal concentration.")
        return concs

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp, dc, sc in zip(
            self.components,
            self.dest_concentrations(mix_vol),
            self.source_concentrations,
        ):
            comps = comp.all_components()
            comps.concentration_nM *= (dc / sc).to("").magnitude

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=0.0)
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentrations(
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> pd.Series:
        return self.source_concentrations * self.each_volumes(mix_vol) / mix_vol

    def each_volumes(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> pd.Series:
        match self.equal_conc:
            case str("min_volume"):
                return (
                    self.fixed_volume
                    * self.source_concentrations.max()
                    / self.source_concentrations
                )
            case str("max_volume") | ("max_fill", _):
                return (
                    self.fixed_volume
                    * self.source_concentrations.min()
                    / self.source_concentrations
                )
            case bool(True):
                sc = self.source_concentrations
                if not (sc == sc[0]).all():
                    raise ValueError("Concentrations")
                return pd.Series(
                    [self.fixed_volume.m_as(uL)] * len(self.components),
                    dtype="pint[uL]",
                )
            case bool(False):
                return pd.Series(
                    [self.fixed_volume.m_as(uL)] * len(self.components),
                    dtype="pint[uL]",
                )
        raise ValueError(f"equal_conc={repr(self.equal_conc)} not understood")

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> Quantity[float]:
        match self.equal_conc:
            case ("max_fill", str(buffername)):
                return self.fixed_volume * len(self.components)
        return self.each_volumes(mix_vol).sum()

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:
        if not self.compact_display:
            ml = [
                MixLine(
                    comp.name,
                    comp.concentration,
                    dc,
                    ev,
                    location=_format_location(comp.location),
                )
                for dc, ev, comp in zip(
                    self.dest_concentrations(mix_vol),
                    self.each_volumes(mix_vol),
                    self.components,
                )
            ]
        else:
            ml = list(self._compactstrs(mix_vol))

        match self.equal_conc:
            case ("max_fill", str(buffername)):
                fv = (
                    self.fixed_volume * len(self.components) - self.each_volumes().sum()
                )
                if not np.allclose(fv, Q_(0.0, uL)):
                    ml.append(MixLine(buffername, None, None, fv))

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

    def _compactstrs(self, mix_vol: pint.Quantity) -> Sequence[MixLine]:
        # locs = [(c.name,) + c.location for c in self.components]
        # names = [c.name for c in self.components]

        # if any(x is None for x in locs):
        #     raise ValueError(
        #         [name for name, loc in zip(names, locs) if loc is None]
        #     )

        locdf = pd.DataFrame(
            {
                "names": [c.name for c in self.components],
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

        names, source_concs, dest_concs, numbres, ea_vols, tot_vols, locations = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for plate, plate_comps in locdf.groupby("plate"):
            for vol, plate_vol_comps in plate_comps.groupby("ea_vols"):
                if plate == "":
                    if not pd.isna(plate_vol_comps["well"]).all():
                        raise ValueError
                    names += [", ".join(n for n in plate_vol_comps["names"])]
                    ea_vols += [(vol)]
                    tot_vols += [(vol * len(plate_vol_comps))]
                    numbres += [(len(plate_vol_comps))]
                    source_concs += [(plate_vol_comps["source_concs"].iloc[0])]
                    dest_concs += [(plate_vol_comps["dest_concs"].iloc[0])]
                    locations += [""]
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
                sortnext = WellPos.next_bycol if bycol <= byrow else WellPos.next_byrow

                plate_vol_comps["sortkey"] = [
                    sortkey(c) for c in plate_vol_comps["well"]
                ]

                plate_vol_comps.sort_values(by="sortkey", inplace=True)

                wells_formatted = []
                next_well_iter = iter(plate_vol_comps["well"])
                prevpos = next(next_well_iter)
                wells_formatted.append(f"**{prevpos}**")
                for well in next_well_iter:
                    if sortnext(prevpos) != well:
                        wells_formatted.append(f"**{well}**")
                    else:
                        wells_formatted.append(f"{well}")
                    prevpos = well

                names.append(", ".join(n for n in plate_vol_comps["names"]))
                ea_vols.append((vol))
                numbres.append((len(plate_vol_comps)))
                tot_vols.append((vol * len(plate_vol_comps)))
                source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                locations.append(
                    plate + ": " + ", ".join(str(w) for w in wells_formatted)
                )

        return [
            MixLine(
                name,
                source_conc=source_conc,
                dest_conc=dest_conc,
                number=number,
                each_tx_vol=each_tx_vol,
                total_tx_vol=total_tx_vol,
                location=location,
            )
            for name, source_conc, dest_conc, number, each_tx_vol, total_tx_vol, location in zip(
                names,
                source_concs,
                dest_concs,
                numbres,
                ea_vols,
                tot_vols,
                locations,
                strict=True,
            )
        ]


@attrs.define()
class MultiFixedConcentration(AbstractAction):
    """An action adding multiple components, with a set destination concentration per component (adjusting volumes).

    MultiFixedConcentration adds a selection of components, with a specified destination concentration.

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

    Examples
    --------

    >>> from alhambra.mixes import *
    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ... ]

    >>> print(Mix([MultiFixedConcentration(components, "66.67 uL")], name="example", ))
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

    >>> print(Mix([MultiFixedConcentration(components, "40 nM")], name="example", fixed_total_volume="25 uL"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 25.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 5.00 µl     | 15.00 µl     |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 10.00 µl    | 10.00 µl     |       |        |

    >>> print(Mix([MultiFixedConcentration(components, "5 uL", equal_conc="max_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 12.50 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 2.50 µl     | 7.50 µl      |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 5.00 µl     | 5.00 µl      |       |        |

    """

    components: Sequence[AbstractComponent]
    fixed_concentration: Quantity[float] = attrs.field(converter=_parse_conc_required)
    set_name: str | None = None
    compact_display: bool = True

    def with_reference(self, reference: pd.DataFrame) -> MultiFixedConcentration:
        return attrs.evolve(
            self, components=[c.with_reference(reference) for c in self.components]
        )

    @property
    def source_concentrations(self):
        concs = pd.Series(
            [c.concentration.m_as(nM) for c in self.components], dtype="pint[nM]"
        )
        return concs

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp, dc, sc in zip(
            self.components,
            self.dest_concentrations(mix_vol),
            self.source_concentrations,
        ):
            comps = comp.all_components()
            comps.concentration_nM *= (dc / sc).to("").magnitude

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=0.0)
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentrations(
        self, mix_vol: Quantity[float] = Q_(np.nan, uL)
    ) -> pd.Series:
        return self.source_concentrations * self.each_volumes(mix_vol) / mix_vol
        # FIXME: THIS IS SILLY

    def each_volumes(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> pd.Series:
        return mix_vol * self.fixed_concentration / self.source_concentrations

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> Quantity[float]:
        return self.each_volumes(mix_vol).sum()

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:
        if not self.compact_display:
            ml = [
                MixLine(
                    comp.name,
                    comp.concentration,
                    dc,
                    ev,
                    location=_format_location(comp.location),
                )
                for dc, ev, comp in zip(
                    self.dest_concentrations(mix_vol),
                    self.each_volumes(mix_vol),
                    self.components,
                )
            ]
        else:
            ml = list(self._compactstrs(mix_vol))

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

    def _compactstrs(self, mix_vol: pint.Quantity) -> Sequence[MixLine]:
        # locs = [(c.name,) + c.location for c in self.components]
        # names = [c.name for c in self.components]

        # if any(x is None for x in locs):
        #     raise ValueError(
        #         [name for name, loc in zip(names, locs) if loc is None]
        #     )

        locdf = pd.DataFrame(
            {
                "names": [c.name for c in self.components],
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

        names, source_concs, dest_concs, numbres, ea_vols, tot_vols, locations = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for plate, plate_comps in locdf.groupby("plate"):
            for vol, plate_vol_comps in plate_comps.groupby("ea_vols"):
                if plate == "":
                    if not pd.isna(plate_vol_comps["well"]).all():
                        raise ValueError
                    names += [", ".join(n for n in plate_vol_comps["names"])]
                    ea_vols += [(vol)]
                    tot_vols += [(vol * len(plate_vol_comps))]
                    numbres += [(len(plate_vol_comps))]
                    source_concs += [(plate_vol_comps["source_concs"].iloc[0])]
                    dest_concs += [(plate_vol_comps["dest_concs"].iloc[0])]
                    locations += [""]
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
                sortnext = WellPos.next_bycol if bycol <= byrow else WellPos.next_byrow

                plate_vol_comps["sortkey"] = [
                    sortkey(c) for c in plate_vol_comps["well"]
                ]

                plate_vol_comps.sort_values(by="sortkey", inplace=True)

                wells_formatted = []
                next_well_iter = iter(plate_vol_comps["well"])
                prevpos = next(next_well_iter)
                wells_formatted.append(f"**{prevpos}**")
                for well in next_well_iter:
                    if sortnext(prevpos) != well:
                        wells_formatted.append(f"**{well}**")
                    else:
                        wells_formatted.append(f"{well}")
                    prevpos = well

                names.append(", ".join(n for n in plate_vol_comps["names"]))
                ea_vols.append((vol))
                numbres.append((len(plate_vol_comps)))
                tot_vols.append((vol * len(plate_vol_comps)))
                source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                locations.append(
                    plate + ": " + ", ".join(str(w) for w in plate_vol_comps["well"])
                )

        return [
            MixLine(
                name,
                source_conc=source_conc,
                dest_conc=dest_conc,
                number=number,
                each_tx_vol=each_tx_vol,
                total_tx_vol=total_tx_vol,
                location=location,
            )
            for name, source_conc, dest_conc, number, each_tx_vol, total_tx_vol, location in zip(
                names,
                source_concs,
                dest_concs,
                numbres,
                ea_vols,
                tot_vols,
                locations,
                strict=True,
            )
        ]


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

    def tx_volume(self, mix_vol: Quantity[float] = Q_(np.nan, uL)) -> Quantity[float]:
        return mix_vol * self.dest_value / self.source_value

    def all_components(self, mix_vol: Quantity[float]) -> pd.DataFrame:
        v = self.component.all_components()
        v.loc[:, "concentration_nM"] *= self.dest_value / self.source_value
        return v

    def _mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:
        return [
            MixLine(
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
    fixed_total_volume: Optional[Quantity[float]] = attrs.field(
        converter=_parse_vol_optional, default=None, kw_only=True
    )
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
                ac.loc[self.fixed_concentration, "concentration_nM"], ureg.nM
            )
        elif self.fixed_concentration is None:
            return self.actions[0].dest_concentrations(self.total_volume)[0]
        else:
            raise NotImplemented

    @property
    def total_volume(self) -> Quantity[float]:
        """
        Total volume of the the mix.  If the mix has a fixed total volume, then that,
        otherwise, the sum of the transfer volumes of each component.
        """
        if self.fixed_total_volume is not None and not np.isnan(
            self.fixed_total_volume.magnitude
        ):
            return self.fixed_total_volume
        else:
            return sum(
                [
                    c.tx_volume(self.fixed_total_volume or Q_(np.nan, ureg.uL))
                    for c in self.actions
                ],
                Q_(0.0, ureg.uL),
            )

    @property
    def buffer_volume(self) -> Quantity[float]:
        """
        The volume of buffer to be added to the mix, in addition to the components.
        """
        mvol = sum(c.tx_volume(self.total_volume) for c in self.actions)
        return self.total_volume - mvol

    def table(self, tablefmt: TableFormat | str = "pipe", validate: bool = True):
        """Generate a table describing the mix.

        Parameters
        ----------

        tablefmt
            The output format for the table.

        validate
            Ensure volumes make sense.
        """
        mixlines = list(self.mixlines())

        if validate:
            try:
                self.validate(mixlines=mixlines)
            except ValueError as e:
                e.args = e.args + self.table(validate=False)
                raise e

        mixlines.append(
            MixLine("*Total:*", None, self.concentration, self.total_volume)
        )

        include_numbers = any(ml.number != 1 for ml in mixlines)

        return tabulate(
            [ml.toline(include_numbers) for ml in mixlines],
            MIXHEAD_EA if include_numbers else MIXHEAD_NO_EA,
            tablefmt=tablefmt,
        )

    def mixlines(self) -> Sequence[MixLine]:
        tv = self.total_volume

        mixlines: list[MixLine] = []

        for action in self.actions:
            mixlines += action._mixlines(tv, locations=self.reference)

        if self.fixed_total_volume is not None:
            mixlines.append(MixLine("Buffer", None, None, self.buffer_volume))
        return mixlines

    def validate(self, mixlines: Sequence[MixLine] | None = None):
        if mixlines is None:
            mixlines = self.mixlines()
        tx_vols = [m.total_tx_vol for m in mixlines if m.total_tx_vol is not None]
        if any(np.isnan(x.magnitude) for x in tx_vols):
            raise ValueError("Some volumes are undefined.")
        if any(x < Q_(0.0, uL) for x in tx_vols):
            raise ValueError("Some volumes are negative.")

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
            f"Table: Mix: {self.name}, Conc: {self.concentration:,.2f~#P}, Total Vol: {self.total_volume:,.2f~#P}\n\n"
            + self.table()
        )

    def to_tileset(
        self,
        tilesets_or_lists: TileSet | TileList | Iterable[TileSet | TileList],
        *,
        seed: bool | Seed = False,
        base_conc=Q_(100.0, nM),
    ) -> TileSet:
        """
        Given some :any:`TileSet`\ s, or lists of :any:`Tile`\ s from which to
        take tiles, generate an TileSet from the mix.
        """
        from .flatish import BaseSSTile

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
                    if isinstance(new_tile, BaseSSTile) and (
                        (seq := getattr(row["component"], "sequence", None)) is not None
                    ):
                        new_tile.sequence |= seq
                    new_tile.stoic = float(Q_(row["concentration_nM"], nM) / base_conc)
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
            filepath = Path(filename[0])

            if isinstance(conc_info, Mapping):
                conc_dict = {k: _parse_conc_required(v) for k, v in conc_info.values()}
                if "default" in conc_dict:
                    all_conc = _parse_conc_required(conc_dict["default"])
                    del conc_dict["default"]
            else:
                all_conc = _parse_conc_required(conc_info)

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
                        k, all_conc if all_conc is not None else Q_(np.nan, nM)
                    ).m_as(nM)
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
                all_conc.m_as(nM) if all_conc is not None else np.nan
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
