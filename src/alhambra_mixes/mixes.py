"""
A module for handling mixes.
"""

from __future__ import annotations

import warnings
from math import isnan
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
    cast,
    Iterable,
    List,
)

import attrs
import pandas as pd
import pint
from tabulate import TableFormat, tabulate

from .actions import AbstractAction  # Fixme: should not need special cases
from .actions import FixedConcentration, FixedVolume
from .components import AbstractComponent, Component, Strand, _empty_components
from .dictstructure import _STRUCTURE_CLASSES, _structure, _unstructure
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

if TYPE_CHECKING:  # pragma: no cover
    from .references import Reference
    from .experiments import Experiment
    from attrs import Attribute

from .units import *
from .units import VolumeError, _parse_vol_optional, normalize

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
    "split_mix",
    "master_mix",
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

    return loc["Name"], loc["Plate"], well


def _maybesequence_action(
    object_or_sequence: Sequence[AbstractAction] | AbstractAction,
) -> list[AbstractAction]:
    if isinstance(object_or_sequence, Sequence):
        return list(object_or_sequence)
    return [object_or_sequence]


def remove_buffer_mixline_if_absent(mixlines: list[MixLine], buffer_name: str) -> None:
    idx_to_remove = -1
    for idx, mixline in enumerate(mixlines):
        if (
            len(mixline.names) == 1
            and mixline.names[0] == buffer_name
            and mixline.each_tx_vol == ureg("0 uL")
        ):
            idx_to_remove = idx
            break

    if idx_to_remove >= 0:
        del mixlines[idx_to_remove]


@attrs.define(eq=False)
class Mix(AbstractComponent):
    """Class denoting a Mix, a collection of source components mixed to
    some volume or concentration.
    """

    actions: Sequence[AbstractAction] = attrs.field(
        converter=_maybesequence_action, on_setattr=attrs.setters.convert
    )
    name: str = ""
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
        default=Q_("0.5", uL),
        kw_only=True,
        on_setattr=attrs.setters.convert,
    )

    @property
    def is_mix(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        for a in self.__attrs_attrs__:  # type: ignore
            a = cast("Attribute", a)
            v1 = getattr(self, a.name)
            v2 = getattr(other, a.name)
            if isinstance(v1, Quantity):
                if isnan(v1.m) and isnan(v2.m) and (v1.units == v2.units):
                    continue
            if v1 != v2:
                return False
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
                Q_("0.0", ureg.uL),
            )

    @property
    def buffer_volume(self) -> Quantity:
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
        buffer_line_if_absent=False,
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

        buffer_line_if_absent
            If True and the buffer volume is 0, include an explicit line for buffer anyway that says 0 uL.
        """
        mixlines = list(self.mixlines(buffer_name=buffer_name, tablefmt=tablefmt))

        if not buffer_line_if_absent:
            remove_buffer_mixline_if_absent(mixlines, buffer_name)

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
    ) -> list[MixLine]:
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

    def with_experiment(self: Mix, experiment: Experiment, inplace: bool = True) -> Mix:
        newactions = [
            action.with_experiment(experiment, inplace) for action in self.actions
        ]
        if inplace:
            self.actions = newactions
            return self
        else:
            return attrs.evolve(self, actions=newactions)

    def with_reference(self: Mix, reference: Reference, inplace: bool = True) -> Mix:
        if inplace:
            self.reference = reference
            for action in self.actions:
                action.with_reference(reference, inplace=True)
            return self
        else:
            new = attrs.evolve(
                self,
                actions=[action.with_reference(reference) for action in self.actions],
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
            + "\n"
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

    def _update_volumes(
        self,
        consumed_volumes: Dict[str, Quantity] = {},
        made_volumes: Dict[str, Quantity] = {},
    ) -> Tuple[Dict[str, Quantity], Dict[str, Quantity]]:
        """
        Given a
        """
        if self.name in made_volumes:
            # We've already been seen.  Ignore our components.
            return consumed_volumes, made_volumes

        made_volumes[self.name] = self.total_volume
        consumed_volumes[self.name] = ZERO_VOL

        for action in self.actions:
            for component, volume in zip(
                action.components, action.each_volumes(self.total_volume, self.actions)
            ):
                consumed_volumes[component.name] = (
                    consumed_volumes.get(component.name, ZERO_VOL) + volume
                )
                component._update_volumes(consumed_volumes, made_volumes)

        # Potentially deal with buffer...
        if self.buffer_volume.m > 0:
            made_volumes[self.buffer_name] = made_volumes.get(
                self.buffer_name, 0 * ureg.ul
            )
            consumed_volumes[self.buffer_name] = (
                consumed_volumes.get(self.buffer_name, 0 * ureg.ul) + self.buffer_volume
            )

        return consumed_volumes, made_volumes

    def _unstructure(self, experiment: "Experiment" | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {}
        d["class"] = self.__class__.__name__
        for a in cast("Sequence[Attribute]", self.__attrs_attrs__):
            if a.name == "actions":
                d[a.name] = [a._unstructure(experiment) for a in self.actions]
            elif a.name == "reference":
                continue
            else:
                val = getattr(self, a.name)
                if val == a.default:
                    continue
                # FIXME: nan quantities are always default, and pint handles them poorly
                if isinstance(val, Quantity) and isnan(val.m):
                    continue
                d[a.name] = _unstructure(val)
        return d

    @classmethod
    def _structure(
        cls, d: dict[str, Any], experiment: "Experiment" | None = None
    ) -> "Mix":
        for k, v in d.items():
            d[k] = _structure(v, experiment)
        return cls(**d)


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


# define subclass with overridden instructions method that prints final instruction for splitting.
@attrs.define(eq=False)
class _SplitMix(Mix):
    num_tubes: int = -1

    small_mix_volume: Quantity[Decimal] = Q_(Decimal(0), "uL")

    names: None | list[str] = None

    def __attrs_post_init__(self) -> None:
        if self.num_tubes < 1:
            raise ValueError("num_tubes must be positive")
        if self.small_mix_volume == Q_(Decimal(0), "uL"):
            raise ValueError("small_mix_volume must be positive")

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
        super_instructions = super().instructions(
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
        names = [f"*{name}*" for name in self.names] if self.names is not None else None
        # below is a bit redundant but prevents mypy error since names could be None
        if names is None:
            names_of_tubes = "."
        elif isinstance(names, list):
            names_of_tubes = ": " + ", ".join(names)
        else:
            raise AssertionError("unreachable")
        self.small_mix_volume = normalize(self.small_mix_volume)
        super_instructions += (
            f"\n\nAliquot {self.small_mix_volume} from this mix "
            + f"into {self.num_tubes} different test tubes{names_of_tubes}"
        )
        return super_instructions


def split_mix(
    mix: Mix,
    num_tubes: int | None = None,
    names: Iterable[str] | None = None,
    excess: int | float | Decimal = Decimal(0.05),
) -> Mix:
    """
    A "split mix" is a :any:`Mix` that involves creating a large volume mix and splitting it into several
    test tubes with identical contents. The advantage of specifying a split mix is that one can give
    the desired volumes/concentrations in the individual test tubes (post splitting) and the number of
    test tubes, and the correct amounts in the larger mix will automatically be calculated.

    The :meth:`Mix.instructions` method of a split mix includes the additional instruction at the end
    to aliquot from the larger mix.

    Parameters
    ----------

    mix
        The :any:`Mix` object describing what each
        individual smaller test tube should contain after the split.

    num_tubes
        The number of test tubes into which to split the large mix. Should not be specified if `names`
        is specified; in that case `num_tubes` is assumed to be the number of strings in `names`.

    excess
        A fraction (between 0 and 1) indicating how much extra of the large mix to make. This is useful
        when `num_tubes` is large, since the aliquots prior to the last test tube may take a small amount
        of extra volume, resulting in the final test tube receiving significantly less volume if the
        large mix contained only just enough total volume.

        For example, if the total volume is 100 uL and `num_tubes` is 20, then each aliquot
        from the large mix to test tubes would be 100/20 = 5 uL. But if due to pipetting imprecision 5.05 uL
        is actually taken, then the first 19 aliquots will total to 19*5.05 = 95.95 uL, so there will only be
        100 - 95.95 = 4.05 uL left for the last test tube. But by setting `excess` to 0.05,
        then to make 20 test tubes of 5 uL each, we would have 5*20*1.05 = 105 uL total, and in this case
        even assuming pipetting error resulting in taking 95.95 uL for the first 19 samples, there is still
        105 - 95.95 = 9.05 uL left, more than enough for the 20'th test tube.

        Note: using `excess` > 0 means than the test tube with the large mix should *not* be
        reused as one of the final test tubes, since it will have too much volume at the end.

    names
        Names of smaller individual test tubes (will be printed in instructions).

    Returns
    -------
        A "large" mix, from which `num_tubes` aliquots can be made to create each of the identical
        "small" mixes.
    """
    if (
        names is None
        and num_tubes is None
        or names is not None
        and num_tubes is not None
    ):
        raise ValueError("exactly one of `names` or `num_tubes` should be specified")

    if names is not None:
        names = list(names)
        num_tubes = len(names)

    # should be true because of checks above, but need explicit assertion to assure mypy num_tubes is int
    assert isinstance(num_tubes, int)

    if isinstance(excess, (float, int)):
        excess = Decimal(excess)
    elif not isinstance(excess, Decimal):
        raise TypeError(
            f"parameter `excess` = {excess} must be a float or Decimal but is {type(excess)}"
        )

    # create new action with large fixed total volume if specified
    volume_multiplier = num_tubes * (1 + excess)
    large_volume = mix.total_volume * volume_multiplier
    actions = list(mix.actions)

    # replace FixedVolume actions in `large_mix` with larger volumes
    new_fixed_volume_actions = {}
    for i, action in enumerate(actions):
        if isinstance(action, FixedVolume):
            large_fixed_volume_action = FixedVolume(
                components=action.components,
                fixed_volume=action.fixed_volume * volume_multiplier,
                set_name=action.set_name,
                compact_display=action.compact_display,
            )
            new_fixed_volume_actions[i] = large_fixed_volume_action

    for i, large_fixed_volume_action in new_fixed_volume_actions.items():
        actions[i] = large_fixed_volume_action

    large_mix = _SplitMix(
        num_tubes=num_tubes,
        small_mix_volume=mix.total_volume,
        names=names,
        actions=actions,
        name=mix.name,
        test_tube_name=mix.test_tube_name,
        fixed_total_volume=large_volume if mix.fixed_total_volume is not None else None,
        fixed_concentration=mix.fixed_concentration,
        buffer_name=mix.buffer_name,
        reference=mix.reference,
        min_volume=mix.min_volume,
    )

    return large_mix


def intersection(s1: Iterable[T], s2: Iterable[T]) -> list[T]:
    """
    Interprets s1 and s2 as "sets" (with unhashable elements that implement ==) and
    computes a list of their intersection s1 \cap s2.

    Parameters
    ----------

    s1
        first set (as an iterable)

    s2
        second set (as an iterable)

    Returns
    -------
       list of elements in both `s1` and `s2`
    """
    return [elt for elt in s1 if elt in s2]


def difference(s1: Iterable[T], s2: Iterable[T]) -> list[T]:
    """
    Interprets s1 and s2 as "sets" (with unhashable elements that implement ==) and
    computes a list of their difference s1 \ s2.

    Parameters
    ----------

    s1
        first set (as an iterable)

    s2
        second set (as an iterable)

    Returns
    -------
        list of elements in `s1` but not `s2`
    """
    return [elt for elt in s1 if elt not in s2]


def compute_shared_actions(
    mixes: Iterable[Mix],
    exclude_shared_components: Iterable[str | Component] = (),
) -> tuple[list[AbstractAction], list[list[AbstractAction]]]:
    """
    Compute the components (identified by Actions) shared by every mix in `mixes`, as well as those
    that are unique to each mix.

    Parameters
    ----------

    mixes
        the list of :any:`Mix`'s of which to determine shared and unique actions

    exclude_shared_components
        components appearing in actions to exclude from the return value `shared_actions`,
        even if those actions appear in every mix in `mixes` (note if an action has many components,
        if at least one of them is in `exclude_shared_components`, then the entire action will be excluded)

    Returns
    -------
        pair `(shared_actions, unique_actions)`, where
        `shared_actions` is a list of Actions shared by each :any:`Mix` in `mixes`,
        `unique_actions` is a list of lists of Actions; `unique_actions[i]` are the actions of `mixes[i]`
        that are not part of `shared_actions`.
    """
    exclude_shared_components = list(exclude_shared_components)
    # normalize exclude_shared_components to have string names only
    for idx, component in enumerate(exclude_shared_components):
        if isinstance(component, Component):
            exclude_shared_components[idx] = component.name
    # now that we set them all to be strings, cast the variable so mypy doesn't complain below
    # for some reason cannot cast to list[str] (causes runtime error), but can cast to List[str]
    exclude_shared_components = cast(List[str], exclude_shared_components)

    action_sets = [mix.actions for mix in mixes]
    if len(action_sets) == 0:
        raise ValueError("mixes cannot be empty")

    # compute actions shared among ALL mixes
    shared_actions = list(action_sets[0])
    for action_set in action_sets[1:]:
        shared_actions = intersection(shared_actions, action_set)

    # exclude actions that contain components in exclude_shared_components
    shared_actions_excluded = []
    for action in shared_actions:
        contains_excluded_components = False
        for comp in action.components:
            if comp.name in exclude_shared_components:
                contains_excluded_components = True
                break
        if not contains_excluded_components:
            shared_actions_excluded.append(action)
    shared_actions = shared_actions_excluded

    # for each mix, compute its actions that are not shared as found above
    unique_action_lists = []
    at_least_one_unique_action = False
    for action_set in action_sets:
        unique_actions = difference(action_set, shared_actions)
        if len(unique_actions) > 0:
            at_least_one_unique_action = True
        unique_action_lists.append(unique_actions)
    if not at_least_one_unique_action:
        raise ValueError(
            "None of the mixes has any actions unique to it, so it does not make sense "
            "to create a master mix.\nSee the function `split_mix` for a simpler function "
            "that achieves the goal of making a large mix that can be split into identical "
            "test tubes."
        )

    return shared_actions, unique_action_lists


def verify_mixes_for_master_mix(mixes: Iterable[Mix]) -> None:
    # check that mixes satisfy constraints for using in a master mix

    mixes = list(mixes)

    # must have at least two mixes
    if len(mixes) < 2:
        raise ValueError(
            f"must have at least two mixes, but found {len(mixes)}:\nmixes = {mixes}"
        )

    # all should have same total volume and buffer name
    first_mix = mixes[0]
    for mix in mixes[1:]:
        if mix.total_volume != first_mix.total_volume:
            raise ValueError(
                f"must have same total volume in all mixes, but mix {mix.name} has "
                f"total volume {mix.total_volume} whereas mix {first_mix.name} has "
                f"total volume {first_mix.total_volume}"
            )
        if mix.buffer_name != first_mix.buffer_name:
            raise ValueError(
                f"must have same buffer name in all mixes, but mix {mix.name} has "
                f'buffer name "{mix.buffer_name}" whereas mix {first_mix.name} has '
                f"buffer name {first_mix.buffer_name}"
            )

    # only handling FixedVolume and FixedConcentration actions for now
    for mix in mixes:
        for action in mix.actions:
            if not isinstance(action, (FixedVolume, FixedConcentration)):
                raise ValueError(
                    f"master_mix can only handle mixes with FixedVolume and FixedConcentration "
                    f"actions, but mix {mix.name} contains a {type(action)} action: "
                    f"{action}"
                )


def master_mix(
    mixes: Iterable[Mix],
    name: str = "master mix",
    excess: float | int | Decimal = Decimal(0.05),
    exclude_shared_components: Iterable[str | Component] = (),
) -> tuple[Mix, list[Mix]]:
    """
    Create a "master mix" useful for saving pipetting steps when creating :any:`Mix`'s in `mixes`
    by grouping components shared among each :any:`Mix`'s in `mixes` into a single large master mix
    from which the shared components can be pipetted to create the downstream mixes.

    Components are considered "shared" if they appear in *all* :any:`Mix`'s in `mixes`.

    To ensure sufficient volume for the last mix when the number of mixes is large (due to slight pipetting
    error from the master mix adding up over many steps), the parameter `excess`
    can be used to control how much of a slight excess of necessary volume is included in the master mix.

    Shared Components may be excluded from the master mix by putting them or their names in the parameter
    `exclude_shared_components`.

    Example:

    .. code-block:: python

        # staple mix to be shared in all mixes
        staples = [Strand(f"stap{i}", concentration="1uM") for i in range(5)]
        staple_mix = Mix(
            actions=[FixedConcentration(components=staples, fixed_concentration="100 nM")],
            name="staple mix",
        )

        # "adapter" mixes that are different between mixes
        num_variants = 3
        adapter_mixes = {}
        for adp_idx in range(num_variants):
            adapters = [Strand(f'adp_{adp_idx}_{i}', concentration="1uM") for i in range(5)]
            adapter_mix = Mix(
                actions=[FixedConcentration(components=adapters, fixed_concentration="50 nM")],
                name=f"adapters {adp_idx} mix",
            )
            adapter_mixes[adp_idx] = adapter_mix

        m13 = Strand("m13 100nM", concentration="100 nM")
        mixes = [Mix(
            actions=[
                FixedConcentration(components=[m13], fixed_concentration=f"1 nM"),
                FixedConcentration(components=[staple_mix], fixed_concentration=f"10 nM"),
                FixedConcentration(components=[adapter_mixes[adp_idx]], fixed_concentration=f"10 nM"),
            ],
            name="mm",
            fixed_total_volume=f"100 uL",
        ) for adp_idx, adapter_mix in adapter_mixes.items()]
        mm, final_mixes = master_mix(mixes=mixes, name='origami master mix', excess=0.1)

        print(mm.instructions())
        for mix in final_mixes:
            print(mix.instructions())

    This should print the following. Note that only 63 uL of master mix are strictly required, but
    the total master mix volume is 10% higher (69.3 uL) due to the parameter `excess` = 0.1.

    .. code-block::

        ## Mix "origami master mix":
        | Component   | [Src]     | [Dest]     | #   | Ea Tx Vol   | Tot Tx Vol   | Location  | Note  |
        |:------------|:----------|:-----------|:----|:------------|:-------------|:----------|:------|
        | staple mix  | 100.00 nM | 47.62 nM   |     | 33.00 µl    | 33.00 µl     |           |       |
        | m13 100nM   | 100.00 nM | 4.76 nM    |     | 3.30 µl     | 3.30 µl      |           |       |
        | 10x buffer  | 100.00 mM | 47.62 mM   |     | 33.00 µl    | 33.00 µl     |           |       |
        | Buffer      |           |            |     | 0.00 µl     | 0.00 µl      |           |       |
        | *Total:*    |           | *47.62 nM* | *4* |             | *69.30 µl*   |           |       |

        Aliquot 21.00 µl from this mix into 3 different test tubes.

        ## Mix "mix0":
        | Component          | [Src]     | [Dest]     | #   | Ea Tx Vol   | Tot Tx Vol   | Location  | Note  |
        |:-------------------|:----------|:-----------|:----|:------------|:-------------|:----------|:------|
        | origami master mix | 47.62 nM  | 10.00 nM   |     | 21.00 µl    | 21.00 µl     |           |       |
        | Mg++               | 125.00 mM | 12.50 mM   |     | 10.00 µl    | 10.00 µl     |           |       |
        | adapters 0 mix     | 50.00 nM  | 20.00 nM   |     | 40.00 µl    | 40.00 µl     |           |       |
        | Buffer             |           |            |     | 29.00 µl    | 29.00 µl     |           |       |
        | *Total:*           |           | *10.00 nM* | *4* |             | *100.00 µl*  |           |       |

        ## Mix "mix1":
        | Component          | [Src]     | [Dest]     | #   | Ea Tx Vol   | Tot Tx Vol   | Location  | Note  |
        |:-------------------|:----------|:-----------|:----|:------------|:-------------|:----------|:------|
        | origami master mix | 47.62 nM  | 10.00 nM   |     | 21.00 µl    | 21.00 µl     |           |       |
        | Mg++               | 125.00 mM | 12.50 mM   |     | 10.00 µl    | 10.00 µl     |           |       |
        | adapters 1 mix     | 55.00 nM  | 20.00 nM   |     | 36.36 µl    | 36.36 µl     |           |       |
        | Buffer             |           |            |     | 32.64 µl    | 32.64 µl     |           |       |
        | *Total:*           |           | *10.00 nM* | *4* |             | *100.00 µl*  |           |       |

        ## Mix "mix2":
        | Component          | [Src]     | [Dest]     | #   | Ea Tx Vol   | Tot Tx Vol   | Location  | Note  |
        |:-------------------|:----------|:-----------|:----|:------------|:-------------|:----------|:------|
        | origami master mix | 47.62 nM  | 10.00 nM   |     | 21.00 µl    | 21.00 µl     |           |       |
        | Mg++               | 125.00 mM | 12.50 mM   |     | 10.00 µl    | 10.00 µl     |           |       |
        | adapters 2 mix     | 60.00 nM  | 20.00 nM   |     | 33.33 µl    | 33.33 µl     |           |       |
        | Buffer             |           |            |     | 35.67 µl    | 35.67 µl     |           |       |
        | *Total:*           |           | *10.00 nM* | *4* |             | *100.00 µl*  |           |       |


    Parameters
    ----------

    mixes
        the list of :any:`Mix`'s of which to calculate a shared master mix

    name
        name of the master mix

    excess
        fraction of "excess" volume to include in master mix to ensure sufficient volume in all downstream
        mixes; see parameter `excess` of :func:`split_mix` for explanation

    exclude_shared_components
        names of shared components (or Components themselves) to exclude from master mix;
        raises exception if any element of `exclude_shared_components` is not shared by all :any:`Mix`'s
        in the parameter `mixes`

    Returns
    -------
        pair `(master_mix, final_mixes)`, where `master_mix` is the master mix to use in
        downstream `final_mixes`. Length of `final_mixes` is the same as parameter `mixes`, and
        they use the same names, but each :any:`Mix` in `final_mixes` will be created by a single
        pipetting step from `master_mix` rather than individual pipetting steps for each shared component.

    """
    if isinstance(exclude_shared_components, str):
        raise TypeError(
            f"parameter `exclude_shared_components` must be Iterable of strings or "
            f"components, but cannot be a string itself: exclude_shared_components = "
            f'"{exclude_shared_components}"'
        )

    verify_mixes_for_master_mix(mixes)

    shared_actions, unique_actions_list = compute_shared_actions(
        mixes, exclude_shared_components
    )

    num_shared_actions = len(shared_actions)
    if num_shared_actions <= 1:
        raise ValueError(
            f"master_mix can only be used when mixes have at least two actions shared "
            f"among all of them, but I only found {num_shared_actions}"
            f", which is {shared_actions[0]}"
            if num_shared_actions == 1
            else ""
        )

    mixes = list(mixes)
    # We have already required total volumes be the same.
    first_mix = mixes[0]
    total_small_mix_volume = first_mix.total_volume
    volume_shared_actions = sum(
        shared_action.tx_volume(total_small_mix_volume)
        for shared_action in shared_actions
    )

    # We care about the *minimum* buffer volume; then mixes that have more buffer
    # can have buffer added individually.
    volume_buffer = min(x.buffer_volume for x in mixes)
    volume_shared_actions_and_buffer = volume_shared_actions + volume_buffer
    concentration_multiplier = total_small_mix_volume / volume_shared_actions_and_buffer

    # replace FixedConcentration actions in `large_mix` with larger concentrations
    # to account for subsequent dilution when pipetting master mix to final small mix
    # FixedVolume actions that require larger volume are handled by the call to `split_mix` below
    new_fixed_concentration_actions = {}
    for i, action in enumerate(shared_actions):
        if isinstance(action, FixedConcentration):
            new_fixed_concentration_action = FixedConcentration(
                components=action.components,
                fixed_concentration=action.fixed_concentration
                * concentration_multiplier,
                set_name=action.set_name,
                compact_display=action.compact_display,
            )
            new_fixed_concentration_actions[i] = new_fixed_concentration_action

    for i, new_fixed_concentration_action in new_fixed_concentration_actions.items():
        shared_actions[i] = new_fixed_concentration_action

    # `small_shared_mix` describes how much of the master mix will go into each smaller downstream mix
    small_shared_mix = Mix(
        actions=shared_actions,
        name=name,
        fixed_total_volume=volume_shared_actions_and_buffer,
        buffer_name=first_mix.buffer_name,
        reference=first_mix.reference,
        min_volume=first_mix.min_volume,
    )

    names = [mix.name for mix in mixes]
    mas_mix = split_mix(mix=small_shared_mix, names=names, excess=excess)

    # create new mixes using master mix and unique actions of each mix
    new_mixes = []
    master_mix_action = FixedVolume(
        components=[mas_mix], fixed_volume=volume_shared_actions_and_buffer
    )
    for orig_mix, unique_actions in zip(mixes, unique_actions_list):
        # `[master_mix] + unique_actions` causes mypy error here, so we get explicit about variable types
        all_actions: list[AbstractAction] = [master_mix_action]
        all_actions.extend(unique_actions)
        new_mix = Mix(
            actions=all_actions,
            name=orig_mix.name,
            fixed_total_volume=orig_mix.total_volume,
            buffer_name=orig_mix.buffer_name,
            reference=orig_mix.reference,
            min_volume=orig_mix.min_volume,
        )
        new_mixes.append(new_mix)

    return mas_mix, new_mixes


_STRUCTURE_CLASSES["Mix"] = Mix
_STRUCTURE_CLASSES["_SplitMix"] = _SplitMix
