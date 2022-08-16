"""
A module for handling mixes.
"""

from __future__ import annotations

import decimal
import enum
import json
import logging
import warnings
from abc import ABC, abstractmethod
from decimal import Decimal
from math import isnan
from os import PathLike
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import attrs
import numpy as np
import pandas as pd
import pint
from pint import Quantity
from tabulate import TableFormat, tabulate
from typing_extensions import TypeAlias

from .actions import AbstractAction  # Fixme: should not need special cases
from .actions import FixedConcentration, FixedVolume
from .components import AbstractComponent, Component, Strand, _empty_components
from .locations import PlateType, WellPos
from .logging import log
from .printing import (
    _ALL_TABLEFMTS,
    _ALL_TABLEFMTS_NAMES,
    _SUPPORTED_TABLEFMTS_TITLE,
    MixLine,
    _format_errors,
    _format_title,
    emphasize,
    html_with_borders_tablefmt,
)
from .references import Reference
from .units import *
from .units import VolumeError, _parse_vol_optional
from .util import _maybesequence

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
    "Mix",
    "_format_title",
    "save_mixes",
    "load_mixes",
)

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


T = TypeVar("T")


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

    @property
    def is_mix(self) -> bool:
        return True

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
            return self.actions[0].dest_concentrations(self.total_volume, self.actions)[
                0
            ]
        else:
            raise NotImplemented

    @property
    def total_volume(self) -> Quantity[Decimal]:
        """
        Total volume of the mix.  If the mix has a fixed total volume, then that,
        otherwise, the sum of the transfer volumes of each component.
        """
        if self.fixed_total_volume is not None and not (
            isnan(self.fixed_total_volume.m)
        ):
            return self.fixed_total_volume
        else:
            return sum(
                [
                    c.tx_volume(
                        self.fixed_total_volume or Q_(DNAN, ureg.uL), self.actions
                    )
                    for c in self.actions
                ],
                Q_(Decimal(0.0), ureg.uL),
            )

    @property
    def buffer_volume(self) -> Quantity[Decimal]:
        """
        The volume of buffer to be added to the mix, in addition to the components.
        """
        mvol = sum(c.tx_volume(self.total_volume, self.actions) for c in self.actions)
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
        self, tablefmt: str | TableFormat = "pipe", buffer_name: str = "Buffer"
    ) -> Sequence[MixLine]:
        mixlines: list[MixLine] = []

        for action in self.actions:
            mixlines += action._mixlines(
                tablefmt=tablefmt, mix_vol=self.total_volume, actions=self.actions
            )

        if self.has_fixed_total_volume():
            mixlines.append(MixLine([buffer_name], None, None, self.buffer_volume))
        return mixlines

    def has_fixed_concentration_action(self) -> bool:
        return any(isinstance(action, FixedConcentration) for action in self.actions)

    def has_fixed_total_volume(self) -> bool:
        return not isnan(self.fixed_total_volume.m)

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

        nan_vols = [", ".join(n) for n, x in ntx if isnan(x.m)]
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
                not isnan(mixline.each_tx_vol.m)
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
                action.components, action.each_volumes(self.total_volume, self.actions)
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
            mcomp = action.all_components(self.total_volume, self.actions)
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
        from IPython.display import HTML, display

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


def load_mixes(file_or_stream: str | PathLike | TextIO):
    if isinstance(file_or_stream, (str, PathLike)):
        p = Path(file_or_stream)
        if not p.suffix:
            p = p.with_suffix(".json")
        s: TextIO = open(file_or_stream, "r")
    else:
        s = file_or_stream

    d = json.load(s)

    return {k: _structure(v) for k, v in d.items()}


def save_mixes(
    mixes: Sequence | Mapping | Mix, file_or_stream: str | PathLike | TextIO
):
    if isinstance(file_or_stream, (str, PathLike)):
        p = Path(file_or_stream)
        if not p.suffix:
            p = p.with_suffix(".json")
        s: TextIO = open(p, "w")
    else:
        s = file_or_stream

    if isinstance(mixes, Mix):
        d = {mixes.name: _unstructure(mixes)}
    elif isinstance(mixes, Sequence):
        d = {x.name: _unstructure(x) for x in mixes}
    elif isinstance(mixes, Mapping):
        d = {x.name: _unstructure(x) for x in mixes.values()}  # FIXME: check mapping

    json.dump(d, s)
