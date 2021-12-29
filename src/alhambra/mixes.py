from __future__ import annotations

import logging
from dataclasses import dataclass
from lib2to3 import refactor
from re import M
from typing import (
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import pint
import pint_pandas
from pint.quantity import Quantity
from tabulate import tabulate

from alhambra.seeds import Seed

from .tiles import TileList
from .tilesets import TileSet

log = logging.getLogger("alhambra")

UR = pint.UnitRegistry()

pint_pandas.PintType.ureg = UR


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


@dataclass(init=False, frozen=True, order=True)
class WellPos:
    row: int
    col: int
    platesize: Literal[96, 384] = 96

    @overload
    def __init__(self, ref_or_row: int, col: int) -> None:
        ...

    @overload
    def __init__(self, ref_or_row: str, col: int | None = None) -> None:
        ...

    def __init__(self, ref_or_row: str | int, col: int | None = None) -> None:
        if col is None:
            col = int(ref_or_row[1:]) - 1
            ref_or_row = ref_or_row[0]
            assert len(ref_or_row) == 1

        assert col in range(0, 12)

        match ref_or_row:
            case str(x):
                row: int = "ABCDEFGH".index(x)
            case int(x):
                row = ref_or_row

        super().__setattr__("row", row)
        super().__setattr__("col", col)

    def __str__(self) -> str:
        return f"{'ABCDEFGH'[self.row]}{self.col + 1}"

    def __repr__(self) -> str:
        return f"WellPos({self})"

    def key_byrow(self) -> tuple[int, int]:
        return (self.row, self.col)

    def key_bycol(self) -> tuple[int, int]:
        return (self.col, self.row)

    def next_byrow(self) -> WellPos:
        return WellPos(self.row + (self.col + 1) // 12, (self.col + 1) % 12)

    def next_bycol(self) -> WellPos:
        return WellPos((self.row + 1) % 8, self.col + (self.row + 1) // 8)


@dataclass(eq=True)
class MixLine:
    name: str | None
    source_conc: Quantity[float] | None
    dest_conc: Quantity[float] | None
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


class MixStep(Protocol):
    """
    Defines a step (eg, one line) of a mix recipe.
    """

    @property
    def name(self) -> str:
        ...

    def dest_conc(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        ...

    def tx_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        ...

    def mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:
        ...

    @property
    def number(self) -> int:
        ...

    def all_comps(self, mix_vol: Quantity[float]) -> pd.Series[float]:
        """
        DataFrame should be {'name': str, 'conc': Quantity[float, 'nM'], 'location': str}
        """
        ...


class BaseComponent(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def conc(self) -> Quantity[float]:
        ...

    def all_comps(self) -> pd.Series[float]:
        ...


def findloc(locations: pd.DataFrame | None, name: str) -> str | None:
    match findloc_tuples(locations, name):
        case (name, plate, well):
            if well:
                return f"{plate}: {well}"
            else:
                return f"{plate}"
        case None:
            return None


def findloc_tuples(
    locations: pd.DataFrame | None, name: str
) -> tuple[str, str, str] | None:
    if locations is None:
        return None
    locs = locations.loc[locations["Name"] == name]

    if len(locs) > 1:
        log.warning(f"Found multiple locations for {name}, using first.")
    elif len(locs) == 0:
        return None

    loc = locs.iloc[0]

    try:
        well = WellPos(loc["Well Position"])
    except Exception:
        well = loc["Well Position"]

    return (loc["Name"], loc["Plate"], well)


@dataclass
class FixedConc(MixStep):
    comp: BaseComponent
    set_dest_conc: Quantity[float]

    def dest_conc(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return self.set_dest_conc

    def tx_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        if mix_vol is None:
            raise ValueError
        retval: Quantity[float] = (
            mix_vol * self.set_dest_conc / self.comp.conc
        ).to_compact()
        retval.check("M")
        return retval

    def all_comps(self, mix_vol: Quantity[float]) -> pd.Series[float]:
        return (
            self.comp.all_comps() * (self.set_dest_conc / self.comp.conc).to_compact()
        )

    def mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:

        return [
            MixLine(
                name=self.comp.name,
                source_conc=self.comp.conc,
                dest_conc=self.dest_conc(mix_vol),
                total_tx_vol=self.tx_vol(mix_vol),
                location=findloc(locations, self.comp.name),
            )
        ]

    @property
    def number(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return self.comp.name


@dataclass
class FixedVol(MixStep):
    comp: BaseComponent
    set_dest_vol: Quantity[float]

    def dest_conc(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return (self.comp.conc * self.set_dest_vol / mix_vol).to_compact()

    def tx_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return self.set_dest_vol

    def all_comps(self, mix_vol: Quantity[float]) -> pd.Series[float]:
        return self.comp.all_comps() * (self.dest_conc(mix_vol) / self.comp.conc)

    def mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:

        return [
            MixLine(
                name=self.comp.name,
                source_conc=self.comp.conc,
                dest_conc=self.dest_conc(mix_vol),
                total_tx_vol=self.tx_vol(mix_vol),
                location=findloc(locations, self.comp.name),
            )
        ]

    @property
    def number(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return self.comp.name


@dataclass
class NFixedVol(MixStep):
    comp: BaseComponent
    set_number: int
    set_dest_vol: Quantity[float]

    def dest_conc(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return (self.comp.conc * self.set_dest_vol / mix_vol).to_compact()

    def all_comps(self, mix_vol: Quantity[float]) -> pd.Series[float]:
        return (
            self.comp.all_comps
            * (self.dest_conc(mix_vol) / self.comp.conc).to_compact()
        )

    def ea_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return self.set_dest_vol

    def tx_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return self.set_dest_vol * self.number

    def locs(self, locations: pd.DataFrame | None) -> str:
        ...

    def mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None
    ) -> Sequence[MixLine]:

        return [
            MixLine(
                name=self.comp.name,
                source_conc=self.comp.conc,
                dest_conc=self.dest_conc(mix_vol),
                total_tx_vol=self.tx_vol(mix_vol),
                location=findloc(locations, self.comp.name),
                each_tx_vol=self.ea_vol(mix_vol),
                number=self.number,
            )
        ]

    @property
    def number(self) -> int:
        return self.set_number

    @property
    def name(self) -> str:
        return self.comp.name


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


@dataclass
class MultiFixedVol(MixStep):
    comps: Sequence[BaseComponent]
    set_dest_vol: Quantity[float]
    set_name: str | None = None
    compact: bool = False

    @property  # FIXME: this assumes all equal...
    def comp_conc(self):
        return self.comps[0].conc

    def all_comps(self, mix_vol: Quantity[float]) -> pd.Series[float]:
        newdf = pd.Series(dtype="pint[nM]")

        for comp in self.comps:
            newdf = newdf.add(
                comp.all_comps()
                * (self.dest_conc(mix_vol) / self.comp_conc).to_compact(),
                fill_value=0 * UR("nM"),
            )

        return newdf

    def dest_conc(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return (self.comp_conc * self.set_dest_vol / mix_vol).to_compact()

    def ea_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return self.set_dest_vol

    def tx_vol(self, mix_vol: Optional[Quantity[float]]) -> Quantity[float]:
        return self.set_dest_vol * self.number

    def mixlines(
        self, mix_vol: Quantity[float], locations: pd.DataFrame | None = None
    ) -> Sequence[MixLine]:
        if not self.compact:
            return [
                MixLine(
                    comp.name,
                    comp.conc,
                    self.dest_conc(mix_vol),
                    self.ea_vol(mix_vol),
                    location=findloc(locations, comp.name),
                )
                for comp in self.comps
            ]
        else:

            name, locs = self.compactstrs(locations=locations)

            return [
                MixLine(
                    name,
                    self.comp_conc,
                    self.dest_conc(mix_vol),
                    self.tx_vol(mix_vol),
                    self.number,
                    self.ea_vol(mix_vol),
                    location=locs,
                )
            ]

    @property
    def number(self) -> int:
        return len(self.comps)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return ", ".join(c.name for c in self.comps)
        else:
            return self.set_name

    def compactstrs(self, locations: pd.DataFrame | None) -> tuple[str, str | None]:
        if locations is None:
            return ", ".join(c.name for c in self.comps), None
        else:
            locs = [findloc_tuples(locations, c.name) for c in self.comps]
            names = [c.name for c in self.comps]

            if all(x is None for x in locs):
                return ", ".join(names), None

            if any(x is None for x in locs):
                raise ValueError(
                    [name for name, loc in zip(names, locs) if loc is None]
                )

            locdf = pd.DataFrame(locs, columns=("Name", "Plate", "Well Position"))

            locdf.sort_values(by=["Plate", "Well Position"])

            ns, ls = [], []

            for p, ll in locdf.groupby("Plate"):
                names: list[str] = list(ll["Name"])
                wells: list[WellPos] = list(ll["Well Position"])

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


@dataclass
class FixedRatio(MixStep):
    comp: BaseComponent
    source_val: float
    dest_val: float

    @property
    def number(self) -> int:
        return 1


@dataclass(frozen=True)
class Component(BaseComponent):
    set_name: str
    set_conc: Quantity[float]

    @property
    def name(self) -> str:
        return self.set_name

    @property
    def conc(self) -> Quantity[float]:
        return self.set_conc

    def all_comps(self) -> pd.Series[float]:
        return pd.Series([self.conc], index=[self.name], dtype="pint[nM]")


@dataclass
class Mix:
    name: str
    mixcomps: Sequence[MixStep]
    set_total_vol: Optional[Quantity[float]] = None
    set_conc: Union[str, Quantity[float], None] = None
    buffer: Optional[str] = None
    locations: pd.DataFrame | None = None

    @property
    def conc(self) -> Quantity[float]:
        if isinstance(self.set_conc, pint.Quantity):
            return self.set_conc
        elif isinstance(self.set_conc, str):
            for mc in self.mixcomps:
                if mc.name == self.set_conc:
                    return mc.dest_conc(self.total_volume)
            raise ValueError
        elif self.set_conc is None:
            return self.mixcomps[0].dest_conc(self.total_volume)
        else:
            raise NotImplemented

    @property
    def total_volume(self) -> Quantity[float]:
        if self.set_total_vol is not None:
            return self.set_total_vol
        else:
            return sum([c.tx_vol(None) for c in self.mixcomps], 0 * UR("µL"))

    @property
    def buffer_volume(self) -> Quantity[float]:
        mvol = sum(c.tx_vol(self.total_volume) for c in self.mixcomps)
        return self.total_volume - mvol

    def mdtable(self):
        tv = self.total_volume

        mixlines = []

        allnums1 = [mc.number == 1 for mc in self.mixcomps]
        include_ea = not all(allnums1)

        for mixcomp in self.mixcomps:
            mixlines += mixcomp.mixlines(tv, locations=self.locations)

        if self.set_total_vol is not None:
            mixlines.append(MixLine("Buffer", None, None, self.buffer_volume))

        incea = any(ml.number != 1 for ml in mixlines)

        return tabulate(
            [ml.toline(incea) for ml in mixlines],
            MIXHEAD_EA if include_ea else MIXHEAD_NO_EA,
            "pipe",
        )

    def all_comps(self) -> pd.Series:
        cps = pd.Series(dtype="pint[nM]")
        for mcomp in self.mixcomps:
            cps = cps.add(mcomp.all_comps(self.total_volume), fill_value=0 * UR("nM"))
        return cps

    def _repr_markdown_(self):
        return (
            f"Table: Mix: {self.name}, Conc: {self.conc:.2f}, Total Vol: {self.total_volume:.2f}\n\n"
            + self.mdtable()
        )

    def __str__(self):
        return self.mdtable()

    def to_tileset(
        self,
        tilesets_or_lists: TileSet | TileList | Iterable[TileSet | TileList],
        *,
        seed: bool | Seed = False,
        base_conc=100 * UR("nM"),
    ) -> TileSet:
        newts = TileSet()

        if isinstance(tilesets_or_lists, (TileList, TileSet)):
            tilesets_or_lists = [tilesets_or_lists]

        for comp, conc in self.all_comps().items():
            new_tile = None
            for tl_or_ts in tilesets_or_lists:
                try:
                    if isinstance(tl_or_ts, TileSet):
                        tile = tl_or_ts.tiles[comp]
                    else:
                        tile = tl_or_ts[comp]
                    new_tile = tile.copy()
                    new_tile.stoic = float(conc / base_conc)
                    newts.tiles.add(new_tile)
                    break
                except KeyError:
                    pass
            if new_tile is None:
                log.warn(f"Component {comp} not found in tile lists.")

        match seed:
            case True:
                firstts = next(iter(tilesets_or_lists))
                assert isinstance(firstts, TileSet)
                newts.seeds["default"] = firstts.seeds["default"]
            case False:
                pass
            case Seed as x:
                newts.seeds["default"] = x

        if len(newts.tiles) == 0:
            raise ValueError("No mix components match tiles.")

        return newts
