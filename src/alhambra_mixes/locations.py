from __future__ import annotations

import enum
from math import isnan
from typing import Iterable, Literal, cast, overload
from math import isnan

import attrs

__all__ = ["WellPos", "PlateType"]


ROW_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWX"


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
        if isnan(v):  # type: ignore
            return None
    except:
        pass
    raise ValueError(f"Can't interpret {v} as well position or None.")
